
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
    def __init__(self, system_model, glWidget):
        self.system_model = system_model
        self.glWidget = glWidget
        
        self.debug_filebase = None
        
        self.DEBUG_IMG_POSTFIX = 'k'   # fi batch mode, save result image in a file ending like this
        
        self.MIN_FEATURES = 8          # fail if less inliers at the end
        self.MODEL_DISTANCE_COEF = 1.6 # default 1.6
        self.LOWE_METHOD_COEF = 0.75   # default 0.7
        self.RANSAC_ITERATIONS = 1000  # default 100
        self.RANSAC_ERROR = 8.0        # default 8.0


    def solve_pnp(self, sce_img, **kwargs):
        # maybe load scene image
        if isinstance(sce_img, str):
            self.debug_filebase = sce_img[0:-4]+self.DEBUG_IMG_POSTFIX
            self.glWidget.loadTargetImage(sce_img, remove_bg=False)
            sce_img = self.glWidget.full_image

        # render model image
        init_dist = -kwargs.get('init_dist', MIN_DISTANCE * self.MODEL_DISTANCE_COEF)
        self.system_model.z_off.value = init_dist
        ref_img, depth_result = self.glWidget.render(depth=True)
        
        # get keypoints and descriptors for both images
        sce_kp, sce_desc, ref_kp, ref_desc = self._orb_features(sce_img, ref_img)
        
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
        self._draw_matches(sce_img, sce_kp, ref_img, ref_kp, matches, pause=False, show=DEBUG)
        
        if error is not None:
            raise error
        
        # get feature 3d points using 3d model
        ref_kp_3d = self._inverse_project([ref_kp[m.trainIdx].pt for m in matches], depth_result)
        # TODO: scale coordinates from scene resolution to render resolution?
        sce_kp_2d = np.array([sce_kp[m.queryIdx].pt for m in matches], dtype='float')

        if DEBUG:
            print('3d z-range: %s'%(ref_kp_3d.ptp(axis=0),), flush=True)
        
        # solve pnp with ransac
        rvec, tvec, inliers = self._solve_pnp_ransac(sce_kp_2d, ref_kp_3d)
        
        # debug by drawing inlier matches
        if not BATCH_MODE or DEBUG:
            print('inliers: %s/%s'%(len(inliers), len(matches)), flush=True)
        self._draw_matches(sce_img, sce_kp, ref_img, ref_kp,
                           [matches[i[0]] for i in inliers], label='inliers', pause=False)
        
        # set model params to solved pose & pos
        self._set_sc_from_ast_rot_and_trans(rvec, tvec)
        
        if BATCH_MODE and self.debug_filebase:
            self.glWidget.saveViewToFile(self.debug_filebase+'r.png')
    
    
    def _orb_features(self, scene_img, model_img):
        params = {
            'edgeThreshold':31,  # default: 31
            'fastThreshold':20,  # default: 20
            'firstLevel':0,      # always 0
            'nfeatures':500,     # default: 500
            'nlevels':8,         # default: 8
            'patchSize':31,      # default: 31
            'scaleFactor':1.2,   # default: 1.2
            'scoreType':cv2.ORB_HARRIS_SCORE,  # default ORB_HARRIS_SCORE, other: ORB_FAST_SCORE
            'WTA_K':2,           # default: 2
        }
        
        # orb features from scene image
        params['nfeatures'] = 1000
        orb = cv2.ORB_create(**params)
        sce_kp, sce_desc = orb.detectAndCompute(scene_img, None)

        # orb features from model image
        params['nfeatures'] = 1000
        orb = cv2.ORB_create(**params)
        ref_kp, ref_desc = orb.detectAndCompute(model_img, None)
        
        return sce_kp, sce_desc, ref_kp, ref_desc
    
    
    def _match_features(self, desc1, desc2, method='flann', symmetry_test=False, use_lowe=True, orb_wta_k_gt2=False):
        
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
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING2 if orb_wta_k_gt2 else cv2.NORM_HAMMING, symmetry_test) # for ORB
        else:
            assert False, 'unknown method %s'%mathod
            
        if use_lowe:
            matches = matcher.knnMatch(desc1, desc2, k=2)
        else:
            matches = matcher.match(desc1, desc2)

        if len(matches)<self.MIN_FEATURES:
            raise PositioningException('Not enough features matched')

        if use_lowe:
            # ratio test as per Lowe's paper
            matches = list(
                m[0]
                for m in matches
                if len(m)>1 and m[0].distance < self.LOWE_METHOD_COEF*m[1].distance
            )
            if len(matches)<self.MIN_FEATURES:
                raise PositioningException('Too many features discarded')
        
        return matches

    
    def _draw_matches(self, img1, kp1, img2, kp2, matches, pause=True, show=True, label='matches'):
        matches = list([m] for m in matches)
        draw_params = {
#            matchColor: (88, 88, 88),
            'singlePointColor': (0, 0, 255),
#            'flags': cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        }
        
        # scale keypoint positions
        for kp in kp1:
            kp.pt = (kp.pt[0]*VIEW_WIDTH/CAMERA_WIDTH, kp.pt[1]*VIEW_HEIGHT/CAMERA_HEIGHT)
        
        # scale image
        img1sc = cv2.cvtColor(cv2.resize(img1, None,
                            fx=VIEW_WIDTH/CAMERA_WIDTH,
                            fy=VIEW_HEIGHT/CAMERA_HEIGHT,
                            interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
        img2c = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        img3 = cv2.drawMatchesKnn(img1sc, kp1, img2c, kp2, matches, None, **draw_params)

        # restore original keypoint positions
        for kp in kp1:
            kp.pt = (kp.pt[0]*CAMERA_WIDTH/VIEW_WIDTH, kp.pt[1]*CAMERA_HEIGHT/VIEW_HEIGHT)

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
        
        try:
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                    ref_kp_3d, sce_kp_2d, tools.intrinsic_camera_mx(), dist_coeffs,
                    iterationsCount = self.RANSAC_ITERATIONS,
                    reprojectionError = self.RANSAC_ERROR)
            if not retval:
                raise PositioningException('RANSAC algorithm returned False')
            if len(inliers) < self.MIN_FEATURES:
                raise PositioningException('RANSAC algorithm was left with too few inliers')
        except Exception as e:
            if DEBUG:
                print(str(e))
            raise PositioningException('RANSAC algorithm ran into problems') from e

        return rvec, tvec, inliers

    
    def _inverse_project(self, points_2d, depths):
        z0 = self.system_model.z_off.value # translate to object origin
        def invproj(xi, yi):
            z = -depths[int(yi)][int(xi)]
            x, y = tools.calc_xy(xi, yi, z, width=VIEW_WIDTH, height=VIEW_HEIGHT)
            return x, -y, -(z-z0) # same as rotate using cv2gl_q
        
        points_3d = np.array([invproj(pt[0], pt[1]) for pt in points_2d])
        return points_3d
    
    
    def _set_sc_from_ast_rot_and_trans(self, rvec, tvec):
        # from opencv cam frame (axis: +z, up: -y) to opengl (axis -z, up: +y)
        # by rotating 180deg around x-axis
        cv2gl_q = np.quaternion(0, 1, 0, 0)
        self.system_model.spacecraft_pos = tools.q_times_v(cv2gl_q, tvec)#[tvec[0], -tvec[1], -tvec[2]]
        
        # from opencv cam frame to spacecraft cam frame
        cv2sc_q = cv2gl_q * self.system_model.sc2gl_q.conj()
        
        cv_cam_delta_q = tools.angleaxis_to_q(rvec)
        sc_delta_q =  cv2sc_q.conj() * cv_cam_delta_q.conj() * cv2sc_q
        
        self.system_model.rotate_spacecraft(sc_delta_q)

        # for some reason rotating asteroid instead of spacecraft 
        # doesnt work as simply as below
        #self.system_model.rotate_asteroid(sc_delta_q.conj())
        