
import math
import pickle

import numpy as np
import quaternion
import cv2

from algo.base import AlgorithmBase
from settings import *
from algo import tools
from algo.tools import PositioningException, Stopwatch


class KeypointAlgo(AlgorithmBase):
    (
        ORB,
        AKAZE,
        SIFT,
        SURF,
    ) = range(4)

    BYTES_PER_FEATURE = {
        ORB: 32,
        AKAZE: 61,
        SIFT: 128,
        SURF: 64,
    }

    MAX_WORK_MEM = 0*1024       # in bytes, usable for both ref and scene features, default 512kB
    FDB_MAX_MEM = 128*1024      # in bytes per scene, default 192kB
    FDB_TOL = math.radians(10)  # features from db never more than 7 deg off
    FDB_REAL = True
    
    def __init__(self, system_model, render_engine, obj_idx):
        super(KeypointAlgo, self).__init__(system_model, render_engine, obj_idx)

        self.sm_noise = 0

        self._pause = False
        self._shape_model_rng = None
        self._latest_detector = None
        self._ransac_err = None

        self._fdb_feat = None
        self._fdb = None
        self._fdb_sc_ast_perms = None
        self._fdb_light_perms = None

        self.DEBUG_IMG_POSTFIX = 'k'    # if batch mode, save result image in a file ending like this

        self.SCENE_SCALE_STEP = 1.4142  # sqrt(2) scale scene image by this amount if fail
        self.MAX_SCENE_SCALE_STEPS = 5  # from mid range 64km to near range 16km (64/sqrt(2)**(5-1) => 16)

        self.RENDER_SHADOWS = True
        self.MAX_FEATURES = 2000
        self.MIN_FEATURES = 10          # fail if less inliers at the end

        # if want that mask radius is x when res 1024x1024 and ast dist 64km => coef = 64*x/1024
        self.MATCH_MASK_COEF = 6.25     # 6.25, used only when running centroid algo before this one
        self.LOWE_METHOD_COEF = 0.85    # default 0.7 (0.8)  # 0.825 vs 0.8 => less fails, worse accuracy

        self.RANSAC_ITERATIONS = 10000  # default 100
        self.DEF_RANSAC_ERROR = 16      # default 8.0, for ORB & if fdb: use half the error given here
        # (SOLVEPNP_ITERATIVE), SOLVEPNP_P3P, SOLVEPNP_AP3P, SOLVEPNP_EPNP, ?SOLVEPNP_UPNP, ?SOLVEPNP_DLS
        self.RANSAC_KERNEL = cv2.SOLVEPNP_AP3P


    def solve_pnp(self, orig_sce_img, outfile, feat=ORB, use_feature_db=False, 
            add_noise=False, scale_cam_img=False, vary_scale=False, match_mask_params=None, **kwargs):

        # set max mem usable by reference features, scene features use rest of MAX_WORK_MEM
        ref_max_mem = KeypointAlgo.FDB_MAX_MEM if use_feature_db else KeypointAlgo.MAX_WORK_MEM/2

        # maybe load scene image
        if isinstance(orig_sce_img, str):
            self.debug_filebase = outfile+self.DEBUG_IMG_POSTFIX
            orig_sce_img = self.load_target_image(orig_sce_img)

        if add_noise:
            self._shape_model_rng = np.max(np.ptp(self.system_model.asteroid.real_shape_model.vertices, axis=0))

        self._ransac_err = self.DEF_RANSAC_ERROR * (0.5 if feat == KeypointAlgo.ORB or use_feature_db else 1)
        render_z = -self.system_model.min_med_distance
        init_z = kwargs.get('init_z', render_z)
        ref_img_sc = min(1, render_z / init_z) * (VIEW_WIDTH if scale_cam_img else self._cam.width) / VIEW_WIDTH

        self.timer = Stopwatch()
        self.timer.start()

        if use_feature_db and KeypointAlgo.FDB_REAL:
            # find correct set of keypoints & descriptors from features db
            ref_desc, ref_kp_3d, ref_kp, ref_img = self._query_fdb(feat)
        else:
            # render model image
            orig_z = self.system_model.z_off.value
            self.system_model.z_off.value = render_z
            ref_img, depth_result = self.render(center=True, depth=True,
                                                discretize_tol=KeypointAlgo.FDB_TOL if use_feature_db else False,
                                                shadows=self.RENDER_SHADOWS)
            #ref_img = ImageProc.equalize_brightness(ref_img, orig_sce_img)
            #orig_sce_img = ImageProc.adjust_gamma(orig_sce_img, 0.5)
            #ref_img = ImageProc.adjust_gamma(ref_img, 0.5)
            self.system_model.z_off.value = orig_z

        # scale to match scene image asteroid extent in pixels
        ref_img = cv2.resize(ref_img, None, fx=ref_img_sc, fy=ref_img_sc, interpolation=cv2.INTER_CUBIC)

        if not (use_feature_db and KeypointAlgo.FDB_REAL):
            # get keypoints and descriptors
            ref_kp, ref_desc = self.detect_features(ref_img, feat, maxmem=ref_max_mem, for_ref=True)
            if DEBUG:
                sz = self._latest_detector.descriptorSize() # in bytes
                print('Descriptor mem use: %.0f x %.0fB => %.1f kB'%(len(ref_kp), sz, len(ref_kp)*sz/1024))
        
        if BATCH_MODE and self.debug_filebase:
            # save start situation in log archive
            self.timer.stop()
            sce = cv2.resize(orig_sce_img, ref_img.shape)
            cv2.imwrite(self.debug_filebase+'a.png', np.concatenate((sce, ref_img), axis=1))
            if DEBUG:
                cv2.imshow('compare', np.concatenate((sce, ref_img), axis=1))
                if self._pause:
                    cv2.waitKey()
            self.timer.start()

        # AKAZE, SIFT, SURF are truly scale invariant, couldnt get ORB to work as good
        vary_scale = vary_scale if feat==self.ORB else False
        
        ok = False
        for i in range(self.MAX_SCENE_SCALE_STEPS):
            try:
                sce_img_sc = (VIEW_WIDTH if scale_cam_img else self._cam.width)/self._cam.width/self.SCENE_SCALE_STEP**i
                if np.isclose(sce_img_sc, 1):
                    sce_img = orig_sce_img
                else:
                    sce_img = cv2.resize(orig_sce_img, None,
                                         fx=sce_img_sc, fy=sce_img_sc,
                                         interpolation=cv2.INTER_CUBIC)

                sce_max_mem = KeypointAlgo.MAX_WORK_MEM - (KeypointAlgo.BYTES_PER_FEATURE[feat] + 12)*len(ref_desc)
                sce_kp, sce_desc = self.detect_features(sce_img, feat, maxmem=sce_max_mem)

                # match descriptors
                try:
                    mask = None
                    if match_mask_params is not None:
                        mask = self._calc_match_mask(sce_kp, ref_kp, render_z, sce_img_sc, ref_img_sc, *match_mask_params)
                    matches = self._match_features(sce_desc, ref_desc, mask=mask, method='brute')
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

                if use_feature_db and KeypointAlgo.FDB_REAL:
                    ref_kp_3d = ref_kp_3d[[m.trainIdx for m in matches], :]
                    if add_noise:
                        # add noise to noiseless 3d ref points from fdb
                        self.timer.stop()
                        ref_kp_3d, self.sm_noise, _ = tools.points_with_noise(ref_kp_3d, only_z=True, max_rng=self._shape_model_rng)
                        self.timer.start()
                else:
                    # get feature 3d points using 3d model
                    ref_kp_3d = self._inverse_project([ref_kp[m.trainIdx].pt for m in matches], depth_result, render_z, ref_img_sc)

                sce_kp_2d = np.array([tuple(np.divide(sce_kp[m.queryIdx].pt, sce_img_sc)) for m in matches], dtype='float')

                # solve pnp with ransac
                rvec, tvec, inliers = self._solve_pnp_ransac(sce_kp_2d, ref_kp_3d)

                # debug by drawing inlier matches
                self._draw_matches(sce_img, sce_img_sc, sce_kp, ref_img, ref_img_sc, ref_kp,
                                   [matches[i[0]] for i in inliers], label='c) inliers', pause=self._pause)

                if len(inliers) < self.MIN_FEATURES:
                    raise PositioningException('RANSAC algorithm was left with too few inliers')

                # dont try again if found enough inliers
                ok = True
                break
            
            except PositioningException as e:
                if not vary_scale:
                    raise e
                # maybe try again using scaled down scene image
                
        if not ok:
            raise PositioningException('Not enough inliers even if tried scaling scene image down x%.1f'%(1/sce_img_sc))
        elif vary_scale:
            print('success at x%.1f'%(1/sce_img_sc))
        
        self.timer.stop()
        
        # set model params to solved pose & pos
        self._set_sc_from_ast_rot_and_trans(rvec, tvec, self.latest_discretization_err_q)
        if not BATCH_MODE or DEBUG:
            rp_err = self._reprojection_error(sce_kp_2d, ref_kp_3d, inliers, rvec, tvec)
            sh_err = self.system_model.calc_shift_err()

            print('inliers: %s/%s, repr-err: %.2f, rel-rot-err: %.2f°, dist-err: %.2f%%, lat-err: %.2f%%, shift-err: %.1fm'%(
                len(inliers), len(matches), rp_err,
                math.degrees(self.system_model.rel_rot_err()),
                self.system_model.dist_pos_err()*100,
                self.system_model.lat_pos_err()*100,
                sh_err*1000,
            ), flush=True)
        
        if BATCH_MODE and self.debug_filebase:
            # save result in log archive
            res_img = self.render(shadows=self.RENDER_SHADOWS)
            sce_img = cv2.resize(orig_sce_img, res_img.shape)
            cv2.imwrite(self.debug_filebase+'d.png', np.concatenate((sce_img, res_img), axis=1))


    def _get_detector(self, feat, maxmem, for_ref=False, **kwargs):
        # extra bytes needed for keypoints (only coordinates (float32), the rest not used)
        if maxmem > 0:
            nb = 4 * (3 if for_ref else 2)
            bytes_per_feat = KeypointAlgo.BYTES_PER_FEATURE[feat] + nb
            nfeats = kwargs.pop('nfeats', np.clip(int(maxmem / bytes_per_feat), 0, self.MAX_FEATURES))
        else:
            nfeats = self.MAX_FEATURES

        if feat == KeypointAlgo.ORB:
            detector = self._orb_detector(nfeats=nfeats, **kwargs)
        elif feat == KeypointAlgo.AKAZE:
            detector = self._akaze_detector(nfeats=nfeats, **kwargs)
        elif feat == KeypointAlgo.SIFT:
            detector = self._sift_detector(nfeats=nfeats, **kwargs)
        elif feat == KeypointAlgo.SURF:
            detector = self._surf_detector(nfeats=nfeats, **kwargs)
        else:
            assert False, 'invalid feature: %s' % (feat,)

        return detector, nfeats


    def detect_features(self, img, feat, maxmem, for_ref=False, **kwargs):
        self._latest_detector, nfeats = self._get_detector(feat, maxmem, for_ref, **kwargs)
        kp, dc = self._latest_detector.detectAndCompute(img, None)
        if kp is None:
            return None

        idxs = np.argsort([-p.response for p in kp])
        return [kp[i] for i in idxs[:nfeats]], [dc[i] for i in idxs[:nfeats]]
        
    
    def _orb_detector(self, nfeats=1000):
        params = {
            'nfeatures':nfeats,         # default: 500
            'edgeThreshold':31,         # default: 31
            'fastThreshold':20,         # default: 20
            'firstLevel':0,             # always 0
            'nlevels':8,                # default: 8
            'patchSize':31,             # default: 31
            'scaleFactor':1.2,          # default: 1.2
            'scoreType':cv2.ORB_HARRIS_SCORE,  # default ORB_HARRIS_SCORE, other: ORB_FAST_SCORE
            'WTA_K':2,                  # default: 2
        }
        return cv2.ORB_create(**params)
    
    def _akaze_detector(self, nfeats=1000):
        params = {
            'descriptor_type':cv2.AKAZE_DESCRIPTOR_MLDB, # default: cv2.AKAZE_DESCRIPTOR_MLDB
            'descriptor_channels':3,    # default: 3
            'descriptor_size':0,        # default: 0
            'diffusivity':cv2.KAZE_DIFF_PM_G2, # default: cv2.KAZE_DIFF_PM_G2
            'threshold':0.001,          # default: 0.001
            'nOctaves':4,               # default: 4
            'nOctaveLayers':4,          # default: 4
        }
        return cv2.AKAZE_create(**params)
        
    def _sift_detector(self, nfeats=1000):
        params = {
            'nfeatures':nfeats,
            'nOctaveLayers':3,          # default: 3
            'contrastThreshold':0.04,   # default: 0.04
            'edgeThreshold':10,         # default: 10
            'sigma':1.6,                # default: 1.6
        }
        return cv2.xfeatures2d.SIFT_create(**params)
    
    def _surf_detector(self, nfeats=1000):
        params = {
            'hessianThreshold':100.0,   # default: 100.0
            'nOctaves':4,               # default: 4
            'nOctaveLayers':3,          # default: 3
            'extended':False,           # default: False
            'upright':False,            # default: False
        }
        return cv2.xfeatures2d.SURF_create(**params)

    def _calc_match_mask(self, sce_kp, ref_kp, render_z, sce_img_sc, ref_img_sc, ix_off, iy_off, ast_z):
        n_sce_kp = (np.array([kp.pt for kp in sce_kp])
                        - np.array([ix_off + self._cam.width/2, iy_off + self._cam.height/2])*sce_img_sc
                   ).reshape((-1, 1, 2)) * abs(ast_z)
        n_ref_kp = (np.array([kp.pt for kp in ref_kp])
                        - np.array([VIEW_WIDTH/2, VIEW_HEIGHT/2])*ref_img_sc
                   ).reshape((1, -1, 2)) * abs(render_z)
        O = np.repeat(n_sce_kp, n_ref_kp.shape[1], axis=1) - np.repeat(n_ref_kp, n_sce_kp.shape[0], axis=0)
        D = np.linalg.norm(O, axis=2)

        # in px*km, if coef=6.25, cam res 1024x1024 and asteroid distance 64km => mask circle radius is 100px
        return (D < self.MATCH_MASK_COEF * min(self._cam.width, self._cam.height)).astype('uint8')
    
    def _match_features(self, desc1, desc2, mask=None, method='brute', symmetry_test=False, ratio_test=True):
        
        if desc1 is None or desc2 is None or len(desc1)<self.MIN_FEATURES or len(desc2)<self.MIN_FEATURES:
            raise PositioningException('Not enough features found')
        
        if method == 'flann':
            ss = self._latest_detector.defaultNorm() != cv2.NORM_HAMMING

            FLANN_INDEX_LSH = 6     # for ORB, AKAZE
            FLANN_INDEX_KDTREE = 0  # for SIFT, SURF

            index_params = {
                'algorithm':    FLANN_INDEX_KDTREE if ss else FLANN_INDEX_LSH,
                'table_number': 6,       # 12
                'key_size':     12,      # 20
                'multi_probe_level': 1,  # 2
            }
            if ss:
                index_params['trees'] = 5

            search_params = {
                'checks': 100,
            }

            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif method == 'brute':
            matcher = cv2.BFMatcher(self._latest_detector.defaultNorm(), symmetry_test)
        else:
            assert False, 'unknown method %s'%method

        if ratio_test:
            matches = matcher.knnMatch(np.array(desc1), np.array(desc2), 2, mask=mask)
        else:
            matches = matcher.match(np.array(desc1), np.array(desc2), mask=mask)

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

    
    def _draw_matches(self, img1, img1_sc, kp1, img2, img2_sc, kp2, matches, pause=True, show=True, label='b) matches'):
        self.timer.stop()
        
        matches = list([m] for m in matches)
        draw_params = {
#            matchColor: (88, 88, 88),
            'singlePointColor': (0, 0, 255),
#            'flags': cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        }
        
        # scale keypoint positions
        for kp in kp1:
            kp.pt = tuple(np.divide(kp.pt, img1_sc*self._cam.width/VIEW_WIDTH))
        for kp in kp2:
            kp.pt = tuple(np.divide(kp.pt, img2_sc))
        
        # scale image
        img1sc = cv2.cvtColor(cv2.resize(img1, None,
                            fx=1/img1_sc*VIEW_WIDTH/self._cam.width,
                            fy=1/img1_sc*VIEW_HEIGHT/self._cam.height,
                            interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
        img2sc = cv2.cvtColor(cv2.resize(img2, None,
                            fx=1/img2_sc,
                            fy=1/img2_sc,
                            interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)

        img3 = cv2.drawMatchesKnn(img1sc, kp1, img2sc, kp2, matches, None, **draw_params)

        # restore original keypoint positions
        for kp in kp1:
            kp.pt = tuple(np.multiply(kp.pt, img1_sc*self._cam.width/VIEW_WIDTH))
        for kp in kp2:
            kp.pt = tuple(np.multiply(kp.pt, img2_sc))

        if BATCH_MODE and self.debug_filebase:
            cv2.imwrite(self.debug_filebase+label[:1]+'.png', img3)
        
        if show:
            cv2.imshow(label, img3)
        cv2.waitKey(0 if pause else 25)
        
        self.timer.start()
    
    def _solve_pnp_ransac(self, sce_kp_2d, ref_kp_3d):
        
        # assuming no lens distortion
        dist_coeffs = None
        sm = self.system_model
        cam_mx = self._cam.intrinsic_camera_mx()
        ref_kp_3d = np.reshape(ref_kp_3d, (len(ref_kp_3d),1,3))
        sce_kp_2d = np.reshape(sce_kp_2d, (len(sce_kp_2d),1,2))

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                ref_kp_3d, sce_kp_2d, cam_mx, dist_coeffs,
                iterationsCount = self.RANSAC_ITERATIONS,
                reprojectionError = self._ransac_err,
                flags=self.RANSAC_KERNEL)
        
        if not retval:
            raise PositioningException('RANSAC algorithm returned False')
        if np.linalg.norm(tvec) > 1280:
            raise PositioningException('RANSAC estimated that asteroid at %s km'%(tvec,))

        return rvec, tvec, inliers

    
    def _reprojection_error(self, sce_kp_2d, ref_kp_3d, inliers, rvec, tvec):
        dist_coeffs = None
        cam_mx = self._cam.intrinsic_camera_mx()
        ref_kp_3d = np.reshape(ref_kp_3d, (len(ref_kp_3d),1,3))
        sce_kp_2d = np.reshape(sce_kp_2d, (len(sce_kp_2d),1,2))
        prj_kp_2d, _ = cv2.projectPoints(ref_kp_3d, rvec, tvec, cam_mx, dist_coeffs)
        return np.sqrt(np.mean((sce_kp_2d[inliers] - prj_kp_2d[inliers])**2))
    
    def _inverse_project(self, points_2d, depths, render_z, img_sc, max_dist=30):
        sc = img_sc * VIEW_WIDTH/self._cam.width
        max_val = self.system_model.max_distance-3
        cam = self._cam
        idxs = tools.foreground_idxs(depths, max_val=max_val)

        def invproj(xi, yi):
            z = -tools.interp2(depths, xi/img_sc, yi/img_sc, idxs=idxs, max_val=max_val, max_dist=max_dist)
            x, y = cam.calc_xy(xi/sc, yi/sc, z)
            return x, -y, -(z-render_z) # same as rotate using cv2gl_q
        
        points_3d = np.array([invproj(pt[0], pt[1]) for pt in points_2d], dtype='float32')
        return points_3d
    
    def _set_sc_from_ast_rot_and_trans(self, rvec, tvec, discretization_err_q, rotate_sc=False):
        sm = self.system_model
        
        # rotate to gl frame from opencv camera frame
        gl2cv_q = sm.frm_conv_q(sm.OPENGL_FRAME, sm.OPENCV_FRAME)
        new_sc_pos = tools.q_times_v(gl2cv_q, tvec)

        # camera rotation in opencv frame
        cv_cam_delta_q = tools.angleaxis_to_q(rvec)

        # solvePnPRansac has some bug that apparently randomly gives 180deg wrong answer
        if new_sc_pos[2]>0:
            tpos = -new_sc_pos
            tdelta_q = cv_cam_delta_q * tools.ypr_to_q(0,math.pi,0)
            # print('Bug with solvePnPRansac, correcting:\n\t%s => %s\n\t%s => %s'%(
            #     new_sc_pos, tpos, tools.q_to_ypr(cv_cam_delta_q), tools.q_to_ypr(tdelta_q)))
            new_sc_pos = tpos
            cv_cam_delta_q = tdelta_q
        
        sm.spacecraft_pos = new_sc_pos
        
        err_q = discretization_err_q or np.quaternion(1, 0, 0, 0)
        if rotate_sc:
            sc2cv_q = sm.frm_conv_q(sm.SPACECRAFT_FRAME, sm.OPENCV_FRAME)
            sc_delta_q = err_q * sc2cv_q * cv_cam_delta_q.conj() * sc2cv_q.conj()
            sm.rotate_spacecraft(sc_delta_q)
        else:
            sc2cv_q = sm.frm_conv_q(sm.SPACECRAFT_FRAME, sm.OPENCV_FRAME)
            sc_q = sm.spacecraft_q()
            
            frame_q = sc_q * err_q * sc2cv_q
            ast_delta_q = frame_q * cv_cam_delta_q * frame_q.conj()
            
            err_corr_q = sc_q * err_q.conj() * sc_q.conj()
            sm.rotate_asteroid(err_corr_q * ast_delta_q)

    def _query_fdb(self, feat):
        self._ensure_fdb_loaded(feat)

        sc_ast_q, _ = self.system_model.gl_sc_asteroid_rel_q()
        light_v, _ = self.system_model.gl_light_rel_dir()
        i1, i2, d_sc_ast_q, d_light_v, err_q, err_angle = self._closest_scene(sc_ast_q, light_v)

        sc2gl_q = self.system_model.frm_conv_q(self.system_model.SPACECRAFT_FRAME, self.system_model.OPENGL_FRAME)
        self.latest_discretization_err_q = sc2gl_q * err_q * sc2gl_q.conj()

        if True:
            ref_desc = self._fdb[0][i1, i2, self._fdb[3][i1, i2, :], :]
            ref_kp_3d = self._fdb[2][i1, i2, self._fdb[3][i1, i2, :], :]
        else:
            nf = self._fdb[4][i1, i2]
            ref_desc = self._fdb[0][i1, i2, 0:nf, :]
            ref_kp_3d = self._fdb[2][i1, i2, 0:nf, :]

        self.timer.stop()

        # construct necessary parts of ref_kp for plotting
        # render feature db scene for plotting
        pos = (0, 0, -self.system_model.min_med_distance)
        ref_kp = [cv2.KeyPoint(*self._cam.calc_img_xy(x, -y, -z+pos[2]), 1) for x, y, z in ref_kp_3d]

        ref_img = self.render_engine.render(self.obj_idx, pos, d_sc_ast_q, d_light_v)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)

        self.timer.start()
        return ref_desc, ref_kp_3d, ref_kp, ref_img

    def _fdb_fname(self, feat):
        return os.path.join(CACHE_DIR, self.system_model.mission_id, 'fdb_%s_w%d_m%d_t%d.pickle'%(
            feat,
            VIEW_WIDTH,
            KeypointAlgo.FDB_MAX_MEM/1024,
            10*math.degrees(KeypointAlgo.FDB_TOL)
        ))

    def _ensure_fdb_loaded(self, feat):
        if self._fdb is None or self._fdb_feat != feat:
            fname = self._fdb_fname(feat)
            try:
                with open(fname, 'rb') as fh:
                    tmp = pickle.load(fh)
                if len(tmp) == 3:
                    # backwards compatibility
                    self._fdb_sc_ast_perms = np.array(tools.bf2_lat_lon(KeypointAlgo.FDB_TOL))
                    self._fdb_light_perms = np.array(tools.bf2_lat_lon(KeypointAlgo.FDB_TOL,
                                                     lat_range=(-math.pi/2, math.radians(90 - 45))))
                    n1, n2 = len(self._fdb_sc_ast_perms), len(self._fdb_light_perms)
                    status, scenes, self._fdb = tmp
                    assert len(scenes) == n1 * n2, \
                        'Wrong amount of scenes in loaded fdb: %d vs %d' % (len(scenes), n1 * n2)
                    scenes = None
                else:
                    status, self._fdb_sc_ast_perms, self._fdb_light_perms, self._fdb = tmp
                assert status['stage'] >= 3, 'Incomplete FDB status: %s'%(status,)
            except (FileNotFoundError, EOFError):
                assert False, 'Couldn\'t find feature db file %s'%fname
            self._fdb_feat = feat

    def _closest_scene(self, sc_ast_q, light_v):
        """ in opengl frame """

        d_sc_ast_q, i1 = tools.discretize_q(sc_ast_q, points=self._fdb_sc_ast_perms)
        err_q = sc_ast_q * d_sc_ast_q.conj()

        c_light_v = tools.q_times_v(err_q.conj(), light_v)
        d_light_v, i2 = tools.discretize_v(c_light_v, points=self._fdb_light_perms)
        err_angle = tools.angle_between_v(light_v, d_light_v)

        return i1, i2, d_sc_ast_q, d_light_v, err_q, err_angle
