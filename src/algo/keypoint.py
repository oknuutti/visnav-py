
import math

import numpy as np
import quaternion
import cv2

from settings import *
from algo import tools
from algo.tools import PositioningException

# TODO:
#  - call solve_pnp from visnav.py (& testloop.py)
# 

class KeypointAlgo():
    (
        ORB,
        AKAZE,
    ) = range(2)
    
    def __init__(self, system_model, glWidget):
        self.system_model = system_model
        self.glWidget = glWidget
        self.debug_filebase = None
        
        self.DEBUG_IMG_POSTFIX = 'k'   # fi batch mode, save result image in a file ending like this
        
        self.MIN_FEATURES = 12         # fail if less inliers at the end
        self.LOWE_METHOD_COEF = 0.75   # default 0.7
        self.RANSAC_ITERATIONS = 1000  # default 100
        self.RANSAC_ERROR = 8.0        # default 8.0
        self.SCENE_SCALE_STEP = 1.4142 # sqrt(2) scale scene image by this amount if fail
        self.MAX_SCENE_SCALE_STEPS = 5 # from mid range 64km to near range 16km (64/sqrt(2)**(5-1) => 16)


    def solve_pnp(self, orig_sce_img, feat=ORB, near_range=True, **kwargs):
        # maybe load scene image
        if isinstance(orig_sce_img, str):
            self.debug_filebase = orig_sce_img[0:-4]+self.DEBUG_IMG_POSTFIX
            self.glWidget.loadTargetImage(orig_sce_img, remove_bg=False)
            orig_sce_img = self.glWidget.full_image

        # render model image
        render_z = -MED_DISTANCE
        orig_z = self.system_model.z_off.value
        self.system_model.z_off.value = render_z
        ref_img, depth_result = self.glWidget.render(depth=True)
        self.system_model.z_off.value = orig_z
        
        # scale to match scene image asteroid extent in pixels
        init_z = kwargs.get('init_z', render_z)
        ref_img_sc = min(1,render_z/init_z) * CAMERA_WIDTH/VIEW_WIDTH
        ref_img = cv2.resize(ref_img, None, fx=ref_img_sc, fy=ref_img_sc, 
                interpolation=cv2.INTER_CUBIC)
        
        # get keypoints and descriptors
        ref_kp, ref_desc = self._detect_features(ref_img, feat, nfeats=4500)
        
        ok = False
        for i in range(self.MAX_SCENE_SCALE_STEPS):
            try:
                sce_img_sc = 1/self.SCENE_SCALE_STEP**i
                if np.isclose(sce_img_sc, 1):
                    sce_img = orig_sce_img
                else:
                    sce_img = cv2.resize(orig_sce_img, None,
                            fx=sce_img_sc, fy=sce_img_sc, 
                            interpolation=cv2.INTER_CUBIC)
                
                sce_kp, sce_desc = self._detect_features(sce_img, feat, nfeats=1500)

                # match descriptors
                try:
                    matches = self._match_features(sce_desc, ref_desc, method='brute')
                    error = None
                except PositioningException as e:
                    matches = []
                    error = e

                # debug by drawing matches
                if not BATCH_MODE or DEBUG:
                    print('matches: %s/%s'%(len(matches), min(len(sce_kp), len(ref_kp))), flush=True, end=", ")
                self._draw_matches(sce_img, sce_img_sc, sce_kp, ref_img, ref_img_sc, ref_kp, matches, pause=False, show=DEBUG)

                if error is not None:
                    raise error

                # get feature 3d points using 3d model
                ref_kp_3d = self._inverse_project([ref_kp[m.trainIdx].pt for m in matches], depth_result, ref_img_sc)
                sce_kp_2d = np.array([tuple(np.divide(sce_kp[m.queryIdx].pt, sce_img_sc)) for m in matches], dtype='float')

                if DEBUG:
                    print('3d z-range: %s'%(ref_kp_3d.ptp(axis=0),), flush=True)

                # solve pnp with ransac
                rvec, tvec, inliers = self._solve_pnp_ransac(sce_kp_2d, ref_kp_3d)

                # debug by drawing inlier matches
                if not BATCH_MODE or DEBUG:
                    print('inliers: %s/%s'%(len(inliers), len(matches)), flush=True)
                self._draw_matches(sce_img, sce_img_sc, sce_kp, ref_img, ref_img_sc, ref_kp,
                                   [matches[i[0]] for i in inliers], label='inliers', pause=False)
                
                # dont try again if found enough inliers
                ok = True
                break
            
            except PositioningException as e:
                if not near_range:
                    raise e
                # maybe try again using scaled down scene image
                
        if not ok:
            raise PositioningException('Not enough inliers even if tried scaling scene image down x%.1f'%(1/sce_img_sc))
        else:
            print('success at x%.1f'%(1/sce_img_sc))
        
        # set model params to solved pose & pos
        self._set_sc_from_ast_rot_and_trans(rvec, tvec)
        
        if BATCH_MODE and self.debug_filebase:
            self.glWidget.saveViewToFile(self.debug_filebase+'r.png')
        
    
    def _detect_features(self, img, feat, **kwargs):
        if feat == KeypointAlgo.ORB:
            kp, desc = self._orb_features(img, **kwargs)
        elif feat == KeypointAlgo.AKAZE:
            kp, desc = self._akaze_features(img, **kwargs)
        else:
            assert False, 'invalid feature: %s'%(feat,)
        return kp, desc
        
    
    def _orb_features(self, img, nfeats=1000):
        params = {
            'edgeThreshold':31,  # default: 31
            'fastThreshold':20,  # default: 20
            'firstLevel':0,      # always 0
            'nfeatures':nfeats,  # default: 500
            'nlevels':8,         # default: 8
            'patchSize':31,      # default: 31
            'scaleFactor':1.2,   # default: 1.2
            'scoreType':cv2.ORB_HARRIS_SCORE,  # default ORB_HARRIS_SCORE, other: ORB_FAST_SCORE
            'WTA_K':2,           # default: 2
        }
        
        # orb features from scene image
        orb = cv2.ORB_create(**params)
        kp, desc = orb.detectAndCompute(img, None)
        return kp, desc
    
    
    def _akaze_features(ref_img, nfeats=1000):
        params = {
            # TODO
        }
        akaze = cv2.AKAZE_create(**params)
        kp, desc = akaze.detectAndCompute(img, None)
        
        
    
    def _match_features(self, desc1, desc2, method='flann', symmetry_test=False, ratio_test=True, norm=cv2.NORM_HAMMING):
        
        if desc1 is None or desc2 is None or len(desc1)<self.MIN_FEATURES or len(desc2)<self.MIN_FEATURES:
            raise PositioningException('Not enough features found')
        
        if method == 'flann':
            FLANN_INDEX_LSH = 6
    #        FLANN_INDEX_KDTREE = 0 # for SIFT
            index_params = {
    #            'algorithm':FLANN_INDEX_KDTREE, # for SIFT
    #            'trees':5, # for SIFT
                'algorithm':FLANN_INDEX_LSH,
                'table_number':6,      # 12
                'key_size':12,         # 20
                'multi_probe_level':1, #2
            }
            search_params = {
                'checks':100,
            }

            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif method == 'brute':
            matcher = cv2.BFMatcher(norm, symmetry_test) # for ORB
        else:
            assert False, 'unknown method %s'%mathod
            
        if ratio_test:
            matches = matcher.knnMatch(desc1, desc2, k=2)
        else:
            matches = matcher.match(desc1, desc2)

        if len(matches)<self.MIN_FEATURES:
            raise PositioningException('Not enough features matched')

        if ratio_test:
            # ratio test as per "Lowe's paper"
            matches = list(
                m[0]
                for m in matches
                if len(m)>1 and m[0].distance < self.LOWE_METHOD_COEF*m[1].distance
            )
            if len(matches)<self.MIN_FEATURES:
                raise PositioningException('Too many features discarded')
        
        return matches

    
    def _draw_matches(self, img1, img1_sc, kp1, img2, img2_sc, kp2, matches, pause=True, show=True, label='matches'):
        matches = list([m] for m in matches)
        draw_params = {
#            matchColor: (88, 88, 88),
            'singlePointColor': (0, 0, 255),
#            'flags': cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        }
        
        # scale keypoint positions
        for kp in kp1:
            kp.pt = tuple(np.divide(kp.pt, img1_sc*CAMERA_WIDTH/VIEW_WIDTH))
        for kp in kp2:
            kp.pt = tuple(np.divide(kp.pt, img2_sc))
        
        # scale image
        img1sc = cv2.cvtColor(cv2.resize(img1, None,
                            fx=1/img1_sc*VIEW_WIDTH/CAMERA_WIDTH,
                            fy=1/img1_sc*VIEW_HEIGHT/CAMERA_HEIGHT,
                            interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
        img2sc = cv2.cvtColor(cv2.resize(img2, None,
                            fx=1/img2_sc,
                            fy=1/img2_sc,
                            interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)

        img3 = cv2.drawMatchesKnn(img1sc, kp1, img2sc, kp2, matches, None, **draw_params)

        # restore original keypoint positions
        for kp in kp1:
            kp.pt = tuple(np.multiply(kp.pt, img1_sc*CAMERA_WIDTH/VIEW_WIDTH))
        for kp in kp2:
            kp.pt = tuple(np.multiply(kp.pt, img2_sc))

        if BATCH_MODE and self.debug_filebase:
            cv2.imwrite(self.debug_filebase+label[:1]+'.png', img3)
        
        if show:
            cv2.imshow(label, img3)
        if pause:
            cv2.waitKey()
    
    
    def _solve_pnp_ransac(self, sce_kp_2d, ref_kp_3d):
        
        # assuming no lens distortion
        dist_coeffs = None
        ref_kp_3d = np.reshape(ref_kp_3d, (len(ref_kp_3d),1,3))
        sce_kp_2d = np.reshape(sce_kp_2d, (len(sce_kp_2d),1,2))
        
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                ref_kp_3d, sce_kp_2d, tools.intrinsic_camera_mx(), dist_coeffs,
                iterationsCount = self.RANSAC_ITERATIONS,
                reprojectionError = self.RANSAC_ERROR)
        
        if not retval:
            raise PositioningException('RANSAC algorithm returned False')
        if len(inliers) < self.MIN_FEATURES:
            raise PositioningException('RANSAC algorithm was left with too few inliers')

        return rvec, tvec, inliers

    
    def _inverse_project(self, points_2d, depths, img_sc):
        z0 = self.system_model.z_off.value # translate to object origin
        sc = img_sc * VIEW_WIDTH/CAMERA_WIDTH
        
        def invproj(xi, yi):
            z = -tools.interp2(depths, xi/img_sc, yi/img_sc)
            x, y = tools.calc_xy(xi/sc, yi/sc, z, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
            return x, -y, -(z-z0) # same as rotate using cv2gl_q
        
        points_3d = np.array([invproj(pt[0], pt[1]) for pt in points_2d])
        return points_3d
    
    
    def _set_sc_from_ast_rot_and_trans(self, rvec, tvec, rotate_sc=False):
        sm = self.system_model

        # rotate to gl frame from opencv camera frame
        gl2cv_q = sm.frm_conv_q(sm.OPENGL_FRAME, sm.OPENCV_FRAME)
        sm.spacecraft_pos = tools.q_times_v(gl2cv_q, tvec)
        
        # camera rotation in opencv frame
        cv_cam_delta_q = tools.angleaxis_to_q(rvec)
        
        if rotate_sc:
            # from opencv cam frame to spacecraft cam frame
            sc2cv_q = sm.frm_conv_q(sm.SPACECRAFT_FRAME, sm.OPENCV_FRAME)
            sc_delta_q =  sc2cv_q * cv_cam_delta_q * sc2cv_q.conj()
            sm.rotate_spacecraft(sc_delta_q.conj())
        else:
            # from asteroid frame to opencv cam frame
            ast2sc_q = sm.frm_conv_q(sm.ASTEROID_FRAME, sm.SPACECRAFT_FRAME)
            sc2cv_q = sm.frm_conv_q(sm.SPACECRAFT_FRAME, sm.OPENCV_FRAME)
            sc_q = sm.spacecraft_q()
            ast_q = sm.asteroid_q()
            
            # -- arrived to this frame rotation formula by experimetation!
            frame_q = ast_q.conj() * ast2sc_q * sc_q * sc2cv_q
            
            ast_delta_q = frame_q * cv_cam_delta_q * frame_q.conj()
            sm.rotate_asteroid(ast_delta_q)
