
import math
import pickle

import numpy as np
import quaternion
import cv2

from visnav.algo.base import AlgorithmBase
from visnav.algo.bundleadj import bundle_adj
from visnav.algo.image import ImageProc
from visnav.iotools import lblloader
from visnav.missions.rosetta import RosettaSystemModel
from visnav.render.render import RenderEngine
from visnav.settings import *
from visnav.algo import tools
from visnav.algo.tools import PositioningException, Stopwatch


class KeypointAlgo(AlgorithmBase):
    (
        ORB,
        AKAZE,
        SIFT,
        SURF,
        FLOW,
    ) = range(5)

    BYTES_PER_FEATURE = {
        ORB: 32,
        AKAZE: 61,
        SIFT: 128,
        SURF: 64,
        FLOW: 32,   # just to put something, actually no descriptor here
    }

    MAX_FEATURES = 1200         # upper limit of features to calculate from an image
    MAX_WORK_MEM = 0*1024       # in bytes, usable for both ref and scene features, default 512kB
    FDB_MAX_MEM = 512*1024      # in bytes per scene, default 192kB
    FDB_TOL = math.radians(12)  # features from db never more than FDB_TOL off
    FDB_REAL = True
    FDB_USE_ALL_FEATS = False

    MIN_FEATURES = 12  # fail if less inliers at the end

    # grid size in expected asteroid diameter fraction for discarding features too close to each other
    FEATURE_FILTERING_RELATIVE_GRID_SIZE = 0.015
    FEATURE_FILTERING_FALLBACK_GRID_SIZE = 2  # grid in pixels for use if don't know extent
    (FFS_NONE, FFS_SIMPLE_GRID, FFS_OVERLAPPING_GRID) = range(3)
    FEATURE_FILTERING_SCHEME = FFS_OVERLAPPING_GRID

    DISCARD_OFF_OBJECT_FEATURES = False

    # if want that mask radius is x when res 1024x1024 and ast dist 64km => coef = 64*x/1024
    MATCH_MASK_RADIUS = 0.15  # ratio to max asteroid diameter
    LOWE_METHOD_COEF = 0.85  # default 0.7 (0.8)  # 0.825 vs 0.8 => less fails, worse accuracy

    RANSAC_ITERATIONS = 20000  # default 100
    DEF_RANSAC_ERROR = 15  # default 8.0, for ORB: use half the error given here
    # (SOLVEPNP_ITERATIVE), SOLVEPNP_P3P, SOLVEPNP_AP3P, SOLVEPNP_EPNP, ?SOLVEPNP_UPNP, ?SOLVEPNP_DLS
    # AP3P seemed best, related paper: https://arxiv.org/pdf/1701.08237.pdf
    #  - related c-code: https://git.codingcafe.org/Mirrors/opencv/opencv/blob/master/modules/calib3d/src/ap3p.cpp
    RANSAC_KERNEL = cv2.SOLVEPNP_AP3P

    def __init__(self, system_model, render_engine, obj_idx, est_real_ast_orient=False):
        super(KeypointAlgo, self).__init__(system_model, render_engine, obj_idx)

        self.sm_noise = 0
        self._noise_lv = False
        self.extra_values = None
        self.est_real_ast_orient = est_real_ast_orient

        self._pause = DEBUG == 2
        self._shape_model_rng = None
        self._latest_detector = None
        self._ransac_err = None
        self._render_z = None

        self._fdb_helper = None
        self._fdb_feat = None
        self._fdb = None
        self._fdb_sc_ast_perms = None
        self._fdb_light_perms = None

        self.DEBUG_IMG_POSTFIX = 'k'    # if batch mode, save result image in a file ending like this

        self.SCENE_SCALE_STEP = 1.4142  # sqrt(2) scale scene image by this amount if fail
        self.MAX_SCENE_SCALE_STEPS = 5  # from mid range 64km to near range 16km (64/sqrt(2)**(5-1) => 16)

        self.REFINE_RANSAC_RESULT = False

        self.RENDER_SHADOWS = True
        self.RENDER_TEXTURES = True
        self.RENDER_REFLECTION_MODEL = RenderEngine.REFLMOD_LUNAR_LAMBERT
        self.RENDER_PSF_PARAMS = None

    def solve_pnp(self, orig_sce_img, outfile=None, feat=ORB, use_feature_db=False, adjust_sc_rot=False,
                  add_noise=False, scale_cam_img=False, vary_scale=False, match_mask_params=None, verbose=1, **kwargs):

        # set max mem usable by reference features, scene features use rest of MAX_WORK_MEM
        ref_max_mem = KeypointAlgo.FDB_MAX_MEM if use_feature_db else KeypointAlgo.MAX_WORK_MEM/2
        sm = self.system_model
        self._ransac_err = KeypointAlgo.DEF_RANSAC_ERROR
        self._render_z = kwargs.get('init_z', -sm.min_med_distance)

        ref_img_sc = (sm.view_width if scale_cam_img else self._cam.width) / sm.view_width
        self.extra_values = None

        if outfile is not None:
            self.debug_filebase = outfile + (self.DEBUG_IMG_POSTFIX if isinstance(orig_sce_img, str) else '')

        if self.est_real_ast_orient:
            # so that can track rotation of 67P
            sm.reset_to_real_vals()

        if use_feature_db and self._fdb_helper is None:
            from visnav.algo.fdbgen import FeatureDatabaseGenerator
            self._fdb_helper = FeatureDatabaseGenerator(self.system_model, self.render_engine, self.obj_idx)

        # maybe load scene image
        if isinstance(orig_sce_img, str):
            orig_sce_img = self.load_target_image(orig_sce_img)

        if add_noise:
            self._shape_model_rng = np.max(np.ptp(sm.asteroid.real_shape_model.vertices, axis=0))

        self.timer = Stopwatch()
        self.timer.start()

        if use_feature_db:
            if KeypointAlgo.FDB_REAL:
                # find correct set of keypoints & descriptors from features db
                ref_desc, ref_kp_3d, ref_kp, ref_img = self._query_fdb(feat)
            else:
                # calculate on-the-fly exactly the same features that would be returned from a feature db
                ref_desc, ref_kp_3d, ref_kp, ref_img = self._fake_fdb(feat)
        else:
            # render model image
            ref_img, depth_result = self.render_ref_img(ref_img_sc)

            if False:
                # normalize ref_img to match sce_img
                ref_img = ImageProc.equalize_brightness(ref_img, orig_sce_img, percentile=99.999, image_gamma=1.8)

            if False:
                gamma = 1.0/1.8
                ref_img = ImageProc.adjust_gamma(ref_img, gamma)
                orig_sce_img = ImageProc.adjust_gamma(orig_sce_img, gamma)

            # get keypoints and descriptors
            ee = sm.pixel_extent(abs(self._render_z))
            ref_kp, ref_desc, self._latest_detector = \
                    KeypointAlgo.detect_features(ref_img, feat, maxmem=ref_max_mem, max_feats=KeypointAlgo.MAX_FEATURES,
                                                 for_ref=True, expected_pixel_extent=ee,
                                                 detector_params=kwargs.get('detector_params', None))

        if BATCH_MODE and self.debug_filebase:
            # save start situation in log archive
            self.timer.stop()
            img1 = cv2.resize(orig_sce_img, (sm.view_width, sm.view_height))
            img2 = cv2.resize(ref_img, (sm.view_width, sm.view_height))
            cv2.imwrite(self.debug_filebase+'a.png', np.concatenate((img1, img2), axis=1))
            if DEBUG:
                cv2.imshow('compare', np.concatenate((img1, img2), axis=1))
            self.timer.start()

        # AKAZE, SIFT, SURF are truly scale invariant, couldnt get ORB to work as good
        vary_scale = vary_scale if feat==self.ORB else False
        
        if len(ref_kp) < KeypointAlgo.MIN_FEATURES:
            raise PositioningException('Too few (%d) reference features found' % (len(ref_kp),))

        ok = False
        for i in range(self.MAX_SCENE_SCALE_STEPS):
            try:
                # resize scene image if necessary
                sce_img_sc = (sm.view_width if scale_cam_img else self._cam.width)/self._cam.width/self.SCENE_SCALE_STEP**i
                if np.isclose(sce_img_sc, 1):
                    sce_img = orig_sce_img
                else:
                    sce_img = cv2.resize(orig_sce_img, None, fx=sce_img_sc, fy=sce_img_sc, interpolation=cv2.INTER_CUBIC)

                # detect features in scene image
                if feat != KeypointAlgo.FLOW:
                    sce_max_mem = KeypointAlgo.MAX_WORK_MEM - (KeypointAlgo.BYTES_PER_FEATURE[feat] + 12)*len(ref_desc)
                    ee = sm.pixel_extent(abs(match_mask_params[2])) if match_mask_params is not None else 0
                    sce_kp, sce_desc, self._latest_detector = \
                            KeypointAlgo.detect_features(sce_img, feat, maxmem=sce_max_mem,
                                                         max_feats=KeypointAlgo.MAX_FEATURES, expected_pixel_extent=ee,
                                                         detector_params=kwargs.get('detector_params', None))

                    if len(sce_kp) < KeypointAlgo.MIN_FEATURES:
                        raise PositioningException('Too few (%d) scene features found' % (len(sce_kp),))

                    # match descriptors
                    try:
                        mask = None
                        if match_mask_params is not None:
                            mask = KeypointAlgo.calc_match_mask(sm, sce_kp, ref_kp, self._render_z,
                                                                sce_img_sc, ref_img_sc, *match_mask_params)
                        matches = KeypointAlgo.match_features(sce_desc, ref_desc, self._latest_detector.defaultNorm(),
                                                              mask=mask, method='brute')
                        error = None
                    except PositioningException as e:
                        matches = []
                        error = e
                else:
                    params = {
                        'winSize': (32, 32),  # was (32, 32)
                        'maxLevel': 3,  # was 4
                        'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.05),
                        'minEigThreshold': 0.0003,
                    }
                    ref_kp_2d = np.array([kp.pt for kp in ref_kp], dtype=np.float32)
                    sce_kp_2d, mask, err = cv2.calcOpticalFlowPyrLK(ref_img, sce_img, ref_kp_2d, None, **params)
                    sce_kp = [cv2.KeyPoint(*kp, 1) for kp in sce_kp_2d]
                    matches = [cv2.DMatch(s, s, 0) for s in np.where(mask)[0]]
                    error = None

                # debug by drawing matches
                if not BATCH_MODE or DEBUG:
                    print('matches: %s/%s'%(len(matches), min(len(sce_kp), len(ref_kp))), flush=True, end=", ")
                if verbose > 0:
                    self._draw_matches(sce_img, sce_img_sc, sce_kp, ref_img, ref_img_sc,
                                       ref_kp, matches, pause=False, show=DEBUG)

                if error is not None:
                    raise error

                # select matched scene feature image coordinates
                sce_kp_2d = np.array([tuple(np.divide(sce_kp[m.queryIdx].pt, sce_img_sc)) for m in matches], dtype='float')

                # prepare reference feature 3d coordinates (for only matched features)
                if use_feature_db:
                    ref_kp_3d = ref_kp_3d[[m.trainIdx for m in matches], :]
                    if add_noise:
                        # add noise to noiseless 3d ref points from fdb
                        self.timer.stop()
                        ref_kp_3d, self.sm_noise, _ = tools.points_with_noise(ref_kp_3d, only_z=True,
                                                                              noise_lv=SHAPE_MODEL_NOISE_LV[add_noise],
                                                                              max_rng=self._shape_model_rng)
                        self.timer.start()
                else:
                    # get feature 3d points using 3d model
                    ref_kp_3d = KeypointAlgo.inverse_project(sm, [ref_kp[m.trainIdx].pt for m in matches], depth_result,
                                                             self._render_z, ref_img_sc)

                if KeypointAlgo.DISCARD_OFF_OBJECT_FEATURES:
                    I = np.where(np.logical_not(np.isnan(ref_kp_3d[:, 0])))[0]
                    if len(I) < self.MIN_FEATURES:
                        raise PositioningException('Too few matches found')
                    sce_kp_2d = sce_kp_2d[I, :]
                    ref_kp_3d = ref_kp_3d[I, :]
                    matches = [matches[i] for i in I]

                # finally solve pnp with ransac
                rvec, tvec, inliers = KeypointAlgo.solve_pnp_ransac(sm, sce_kp_2d, ref_kp_3d, self._ransac_err)

                # debug by drawing inlier matches
                if verbose > 0:
                    self._draw_matches(sce_img, sce_img_sc, sce_kp, ref_img, ref_img_sc, ref_kp,
                                       [matches[i[0]] for i in inliers], label='c) inliers', pause=self._pause)

                inlier_count = self.count_inliers(sce_kp, ref_kp, matches, inliers)
                if DEBUG:
                    print('inliers: %s/%s, ' % (inlier_count, len(matches)), end='', flush=True)
                if inlier_count < KeypointAlgo.MIN_FEATURES:
                    raise PositioningException('RANSAC algorithm was left with too few inliers')

                if self.REFINE_RANSAC_RESULT:
                    rvec, tvec = KeypointAlgo.optimize_pose(sm, sce_kp_2d[inliers, :], ref_kp_3d[inliers, :], rvec, tvec)

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
        self._set_sc_from_ast_rot_and_trans(rvec, tvec, self.latest_discretization_err_q, rotate_sc=adjust_sc_rot)

        # debugging
        if not BATCH_MODE or DEBUG:
            rp_err = KeypointAlgo.reprojection_error(self._cam, sce_kp_2d, ref_kp_3d, inliers, rvec, tvec, sce_img=sce_img)
            sh_err = sm.calc_shift_err()

            print('repr-err: %.2f, rel-rot-err: %.2f°, dist-err: %.2f%%, lat-err: %.2f%%, shift-err: %.1fm'%(
                rp_err,
                math.degrees(sm.rel_rot_err()),
                sm.dist_pos_err()*100,
                sm.lat_pos_err()*100,
                sh_err*1000,
            ), flush=True)

        # save result image
        if BATCH_MODE and self.debug_filebase:
            # save result in log archive
            res_img = self.render(shadows=self.RENDER_SHADOWS)
            sce_img = cv2.resize(orig_sce_img, tuple(np.flipud(res_img.shape)))
            cv2.imwrite(self.debug_filebase+'d.png', np.concatenate((sce_img, res_img), axis=1))

    def count_inliers(self, sce_kp, ref_kp, matches, inliers, div=2):
        ref_i = [ref_kp[matches[i[0]].trainIdx] for i in inliers]
        sce_i = [sce_kp[matches[i[0]].queryIdx] for i in inliers]

        ref_count = len({(int(p.pt[0])//div, int(p.pt[1])//div) for p in ref_i})
        sce_count = len({(int(p.pt[0])//div, int(p.pt[1])//div) for p in sce_i})

        return min(ref_count, sce_count)

    def render_ref_img(self, ref_img_sc):
        sm = self.system_model
        orig_z = sm.z_off.value
        sm.z_off.value = self._render_z
        ref_img, depth = self.render(center=True, depth=True, shadows=self.RENDER_SHADOWS, textures=self.RENDER_TEXTURES,
                                     reflection=self.RENDER_REFLECTION_MODEL)
        sm.z_off.value = orig_z

        if self.RENDER_PSF_PARAMS is not None:
            ref_img = ImageProc.apply_point_spread_fn(ref_img, **self.RENDER_PSF_PARAMS)

        # scale to match scene image asteroid extent in pixels
        ref_img = cv2.resize(ref_img, None, fx=ref_img_sc, fy=ref_img_sc, interpolation=cv2.INTER_CUBIC)
        return ref_img, depth

    @classmethod
    def detect_features(cls, img, feat, maxmem, max_feats=MAX_FEATURES, for_ref=False, detector_params=None, **kwargs):
        expected_pixel_extent = abs(kwargs.pop('expected_pixel_extent', 0))

        t_max_feats = max_feats
        if cls.FEATURE_FILTERING_SCHEME is not cls.FFS_NONE and feat == KeypointAlgo.ORB:
            # in opencv orb it's impossible to control detection threshold directly,
            # it's set based on target feature count, TODO: implement better
            t_max_feats = 30000

        detector, nfeats = cls.get_detector(feat, maxmem, t_max_feats, for_ref, detector_params, **kwargs)
        nfeats = min(max_feats, nfeats)

        if feat != KeypointAlgo.FLOW:
            kp, dc = detector.detectAndCompute(img, None)
        else:
            kp = detector.detect(img, None)
            dc = np.zeros((len(kp),))
        if kp is None:
            return None

        if cls.FEATURE_FILTERING_RELATIVE_GRID_SIZE > 0 and expected_pixel_extent > 0:
            grid = cls.FEATURE_FILTERING_RELATIVE_GRID_SIZE * expected_pixel_extent
        else:
            grid = cls.FEATURE_FILTERING_FALLBACK_GRID_SIZE

        if cls.FEATURE_FILTERING_SCHEME is not cls.FFS_NONE and grid > 0:
            ng = img.shape[0] // grid * (img.shape[1] // grid + 1) + img.shape[1] // grid + 1
            offsets = [(0, 0)]
            if cls.FEATURE_FILTERING_SCHEME == cls.FFS_OVERLAPPING_GRID:
                offsets.append((grid // 2, grid // 2))
            groups = {}
            for i, p in enumerate(kp):
                for j, (xoff, yoff) in enumerate(offsets):
                    group = (p.pt[0] + xoff) // grid \
                            + ((p.pt[1]+yoff) // grid) * (img.shape[1] // grid + 1) \
                            + j * ng
                    if group not in groups or groups[group][0] < p.response:
                        groups[group] = (p.response, i)
            idxs = {i for r, i in groups.values()}
            kp = [kp[i] for i in idxs]
            dc = [dc[i] for i in idxs]

        idxs = np.argsort([-p.response for p in kp])
        return [kp[i] for i in idxs[:nfeats]], [dc[i] for i in idxs[:nfeats]], detector

    @staticmethod
    def get_detector(feat, maxmem, max_feats=MAX_FEATURES, for_ref=False, detector_params=None, **kwargs):
        # extra bytes needed for keypoints (only coordinates (float32), the rest not used)
        if maxmem > 0:
            nb = 4 * (3 if for_ref else 2)
            bytes_per_feat = KeypointAlgo.BYTES_PER_FEATURE[feat] + nb
            nfeats = kwargs.pop('nfeats', np.clip(int(maxmem / bytes_per_feat), 0, max_feats))
        else:
            nfeats = max_feats

        if feat == KeypointAlgo.ORB:
            detector = KeypointAlgo.orb_detector(nfeats, detector_params)
        elif feat == KeypointAlgo.AKAZE:
            detector = KeypointAlgo.akaze_detector(nfeats, detector_params)
        elif feat == KeypointAlgo.SIFT:
            detector = KeypointAlgo.sift_detector(nfeats, detector_params)
        elif feat == KeypointAlgo.SURF:
            detector = KeypointAlgo.surf_detector(nfeats, detector_params)
        elif feat == KeypointAlgo.FLOW:
            detector = KeypointAlgo.gftt_detector(nfeats, detector_params)
        else:
            assert False, 'invalid feature: %s' % (feat,)

        return detector, nfeats

    @staticmethod
    def orb_detector(nfeats=1000, detector_params=None):
        params = {
            'nfeatures':nfeats,         # default: 500
            'edgeThreshold':31,         # default: 31
            'fastThreshold':10,         # default: 20
            'firstLevel':0,             # always 0
            'nlevels':8,                # default: 8
            'patchSize':31,             # default: 31
            'scaleFactor':1.2,          # default: 1.2
            'scoreType':cv2.ORB_HARRIS_SCORE,  # default ORB_HARRIS_SCORE, other: ORB_FAST_SCORE
            'WTA_K':2,                  # default: 2
        }
        params.update({} if detector_params is None else detector_params)
        return cv2.ORB_create(**params)

    @staticmethod
    def akaze_detector(nfeats=1000, detector_params=None):
        params = {
            'descriptor_type': cv2.AKAZE_DESCRIPTOR_MLDB, # default: cv2.AKAZE_DESCRIPTOR_MLDB
            'descriptor_channels': 3,    # default: 3
            'descriptor_size': 0,        # default: 0
            'diffusivity': cv2.KAZE_DIFF_CHARBONNIER, # default: cv2.KAZE_DIFF_PM_G2
            'threshold': 0.00005,         # default: 0.001
            'nOctaves': 4,               # default: 4
            'nOctaveLayers': 4,          # default: 4
        }
        params.update({} if detector_params is None else detector_params)
        return cv2.AKAZE_create(**params)

    @staticmethod
    def sift_detector(nfeats=1000, detector_params=None):
        params = {
            'nfeatures':nfeats,
            'nOctaveLayers':3,          # default: 3
            'contrastThreshold':0.01,   # default: 0.04
            'edgeThreshold':25,         # default: 10
            'sigma':1.6,                # default: 1.6
        }
        params.update({} if detector_params is None else detector_params)
        return cv2.xfeatures2d.SIFT_create(**params)

    @staticmethod
    def surf_detector(nfeats=1000, detector_params=None):
        params = {
            'hessianThreshold':100.0,   # default: 100.0
            'nOctaves':4,               # default: 4
            'nOctaveLayers':3,          # default: 3
            'extended':False,           # default: False
            'upright':False,            # default: False
        }
        params.update({} if detector_params is None else detector_params)
        return cv2.xfeatures2d.SURF_create(**params)

    @staticmethod
    def gftt_detector(nfeats=1000, detector_params=None):
        params = {
            'maxCorners': nfeats,
            'qualityLevel': 0.05,  # default around 0.05?
            'minDistance': 10,
            'blockSize': 4,
        }
        params.update({} if detector_params is None else detector_params)
        return cv2.GFTTDetector_create(**params)

    @staticmethod
    def calc_match_mask(sm, sce_kp, ref_kp, render_z, sce_img_sc, ref_img_sc,
                        ix_off, iy_off, ast_z, uncertainty_radius):

        n_sce_kp = np.tan((np.array([kp.pt for kp in sce_kp])
                        - np.array([ix_off + sm.cam.width/2, iy_off + sm.cam.height/2])*sce_img_sc
                   ) * math.radians(sm.cam.x_fov) / sm.cam.width).reshape((-1, 2)) * abs(ast_z)
        n_ref_kp = np.tan((np.array([kp.pt for kp in ref_kp])
                        - np.array([sm.view_width/2, sm.view_height/2])*ref_img_sc
                   ) * math.radians(sm.cam.x_fov) / sm.cam.width).reshape((-1, 2)) * abs(render_z)
        D = tools.distance_mx(n_sce_kp, n_ref_kp)

        match_radius = uncertainty_radius + KeypointAlgo.MATCH_MASK_RADIUS * sm.asteroid.max_radius * 2 / 1000
        return (D < match_radius).astype('uint8')

    @staticmethod
    def match_features(desc1, desc2, norm, mask=None, method='brute', symmetry_test=False, ratio_test=True):
        
        if desc1 is None or desc2 is None or len(desc1) < KeypointAlgo.MIN_FEATURES or len(desc2) < KeypointAlgo.MIN_FEATURES:
            raise PositioningException('Not enough features found')
        
        if method == 'flann':
            ss = norm != cv2.NORM_HAMMING

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
            matcher = cv2.BFMatcher(norm, symmetry_test)
        else:
            assert False, 'unknown method %s' % method

        if ratio_test:
            matches = matcher.knnMatch(np.array(desc1), np.array(desc2), 2, mask=mask)
        else:
            matches = matcher.match(np.array(desc1), np.array(desc2), mask=mask)

        if len(matches) < KeypointAlgo.MIN_FEATURES:
            raise PositioningException('Not enough features matched')

        if ratio_test:
            # ratio test as per "Lowe's paper"
            matches = list(
                m[0]
                for m in matches
                if len(m) > 1 and m[0].distance < KeypointAlgo.LOWE_METHOD_COEF*m[1].distance
            )
            if len(matches) < KeypointAlgo.MIN_FEATURES:
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
            kp.pt = tuple(np.divide(kp.pt, img1_sc*self._cam.width/self.system_model.view_width))
        for kp in kp2:
            kp.pt = tuple(np.divide(kp.pt, img2_sc))
        
        # scale image
        img1sc = cv2.cvtColor(cv2.resize(img1, None,
                            fx=1/img1_sc*self.system_model.view_width/self._cam.width,
                            fy=1/img1_sc*self.system_model.view_height/self._cam.height,
                            interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
        img2sc = cv2.cvtColor(cv2.resize(img2, None,
                            fx=1/img2_sc,
                            fy=1/img2_sc,
                            interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)

        img3 = cv2.drawMatchesKnn(img1sc, kp1, img2sc, kp2, matches, None, **draw_params)

        if BATCH_MODE and self.debug_filebase:
            cv2.imwrite(self.debug_filebase+label[:1]+'.png', img3)

        if show:
            if 0:
                cv2.imshow(label, img3)
                cv2.waitKey(0 if pause else 25)
            else:
                import matplotlib.pyplot as plt
                if 1:
                    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 5))
                    axs[0].imshow(img3[:, :img3.shape[1]//2, :] * 0.8 / 255)
                    axs[1].imshow(img3[:, img3.shape[1]//2:, :] * 0.8 / 255)
                else:
                    idx1 = np.array([m[0].queryIdx for m in matches], dtype=int)
                    idx2 = np.array([m[0].trainIdx for m in matches], dtype=int)
                    for p1, p2 in zip(np.array(kp1)[idx1], np.array(kp2)[idx2]):
                        pt1, pt2 = map(lambda x: tuple(np.round(x.pt).astype(int)), (p1, p2))
                        img2sc = cv2.line(img2sc, pt1, pt2, (0, 255, 0), 1)
                        img2sc = cv2.circle(img2sc, pt2, 5, (0, 255, 0), 1)
                    plt.imshow(img2sc)
                plt.tight_layout()
                plt.show()

        # restore original keypoint positions
        for kp in kp1:
            kp.pt = tuple(np.multiply(kp.pt, img1_sc*self._cam.width/self.system_model.view_width))
        for kp in kp2:
            kp.pt = tuple(np.multiply(kp.pt, img2_sc))

        self.timer.start()

    @staticmethod
    def solve_pnp_ransac(sm, sce_kp_2d, ref_kp_3d, ransac_err=DEF_RANSAC_ERROR,
                         n_iter=RANSAC_ITERATIONS, kernel=RANSAC_KERNEL):
        
        # assuming no lens distortion
        dist_coeffs = None
        cam_mx = sm.cam.intrinsic_camera_mx()
        ref_kp_3d = np.reshape(ref_kp_3d, (len(ref_kp_3d), 1, 3))
        sce_kp_2d = np.reshape(sce_kp_2d, (len(sce_kp_2d), 1, 2))

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                ref_kp_3d, sce_kp_2d, cam_mx, dist_coeffs,
                iterationsCount=n_iter,
                reprojectionError=ransac_err,
                flags=kernel)
        
        if not retval:
            raise PositioningException('RANSAC algorithm returned False')
        if len(inliers) >= KeypointAlgo.MIN_FEATURES and np.linalg.norm(tvec) > sm.max_distance * 1.1:
            # BUG in solvePnPRansac: sometimes estimates object to be very far even though enough good inliers
            # happens with all kernels, reprojectionErrors and iterationsCounts
            raise PositioningException('RANSAC estimated that asteroid at %s km' % (tvec.T,))

        return rvec, tvec, inliers

    @staticmethod
    def optimize_pose(sm, sce_kp_2d, ref_kp_3d, rvec0, tvec0):
        poses = np.array([[*rvec0.flatten(), *tvec0.flatten()]])
        cam_idxs = np.zeros((len(sce_kp_2d),), dtype=int)
        pt3d_idxs = np.arange(len(sce_kp_2d))
        ba_poses, _ = bundle_adj(poses, ref_kp_3d.squeeze(), sce_kp_2d.squeeze(), cam_idxs=cam_idxs, pt3d_idxs=pt3d_idxs,
                                 K=sm.cam.intrinsic_camera_mx(), max_nfev=100, skip_pose_n=0,
                                 poses_only=True, huber_coef=False)

        return ba_poses[0][:3, None], ba_poses[0][3:6, None]

    @staticmethod
    def reprojection_error(cam, sce_kp_2d, ref_kp_3d, inliers, rvec, tvec, sce_img=None):
        dist_coeffs = None
        cam_mx = cam.intrinsic_camera_mx()
        sce_kp_2d = sce_kp_2d[inliers.squeeze(), :]
        prj_kp_2d, _ = cv2.projectPoints(ref_kp_3d[inliers.squeeze(), None, :], rvec, tvec, cam_mx, dist_coeffs)
        prj_kp_2d = prj_kp_2d.squeeze()

        if sce_img is not None:
            import matplotlib.pyplot as plt
            plt.imshow(sce_img, cmap='gray')
            plt.plot(prj_kp_2d[:, 0], prj_kp_2d[:, 1], 'xC0', mfc='none')
            plt.plot(sce_kp_2d[:, 0], sce_kp_2d[:, 1], 'oC0', mfc='none')
            for s, e in zip(prj_kp_2d, sce_kp_2d):
                plt.plot([s[0], e[0]], [s[1], e[1]], 'C0')
            plt.tight_layout()
            plt.show()

        return np.sqrt(np.mean((sce_kp_2d - prj_kp_2d)**2))

    @staticmethod
    def inverse_project(system_model, points_2d, depths, render_z, img_sc, max_dist=30):
        cam = system_model.cam
        sc = img_sc * system_model.view_width/cam.width
        max_val = system_model.max_distance-3
        discard_bg = KeypointAlgo.DISCARD_OFF_OBJECT_FEATURES
        idxs = None if discard_bg else tools.foreground_idxs(depths, max_val=max_val)

        def invproj(xi, yi):
            z = -tools.interp2(depths, xi/img_sc, yi/img_sc, idxs=idxs, max_val=max_val, max_dist=max_dist,
                               discard_bg=discard_bg)
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
            sm.rotate_spacecraft(sc_delta_q)  # TODO: check that rotation ok, changed from sc_q = sc_q * dq to sc_q = dq * sc_q
        else:
            sc2cv_q = sm.frm_conv_q(sm.SPACECRAFT_FRAME, sm.OPENCV_FRAME)
            sc_q = sm.spacecraft_q

            frame_q = sc_q * err_q * sc2cv_q
            ast_delta_q = frame_q * cv_cam_delta_q * frame_q.conj()
            
            err_corr_q = sc_q * err_q.conj() * sc_q.conj()
            ast_q0 = sm.asteroid_q
            sm.rotate_asteroid(err_corr_q * ast_delta_q)

            if self.est_real_ast_orient:
                # so that can track rotation of 67P
                ast_q = sm.asteroid_q
                err_deg = math.degrees(tools.angle_between_q(ast_q0, ast_q))
                self.extra_values = list(quaternion.as_float_array(ast_q)) + [sm.time.value, err_deg]

    def _fake_fdb(self, feat):
        if self._fdb_sc_ast_perms is None or self._fdb_light_perms is None:
            a, b = self._fdb_helper.calc_mesh(KeypointAlgo.FDB_TOL)
            self._fdb_helper.set_mesh(a, b)
            self._fdb_sc_ast_perms, self._fdb_light_perms = a, b

        # get closest scene in fdb
        i1, i2, d_sc_ast_q, d_light_v, err_q, err_angle = self._fdb_helper.closest_scene()

        sc2gl_q = self.system_model.frm_conv_q(self.system_model.SPACECRAFT_FRAME, self.system_model.OPENGL_FRAME)
        self.latest_discretization_err_q = sc2gl_q * err_q * sc2gl_q.conj()

        self.timer.stop()

        # render it and extract features
        ref_img, depth = self._fdb_helper.render_scene(i1, i2)
        ref_kp, ref_desc, detector = KeypointAlgo.detect_features(ref_img, feat, KeypointAlgo.FDB_MAX_MEM,
                                                                  self._fdb_helper.MAX_FEATURES, for_ref=True)

        # get 3d coordinates
        ref_img_sc = self._cam.width / self.system_model.view_width
        ref_kp_2d = np.array([p.pt for p in ref_kp], dtype='float32')
        ref_kp_3d = KeypointAlgo.inverse_project(self.system_model, ref_kp_2d, depth, self._render_z, ref_img_sc)

        if KeypointAlgo.FDB_USE_ALL_FEATS:
            self.timer.start()
            return ref_desc, ref_kp_3d, ref_kp, ref_img

        # get closest neighbours
        visit = self._fdb_helper.get_neighbours(KeypointAlgo.FDB_TOL, i1, i2)
        matched_feats = np.zeros((len(ref_kp_3d),), dtype='bool')
        for i1, i2, j1, j2 in visit:
            try:
                # render neighbour, get features
                n_img, _ = self._fdb_helper.render_scene(j1, j2, get_depth=False)
                n_kp, n_desc, _ = KeypointAlgo.detect_features(n_img, feat, KeypointAlgo.FDB_MAX_MEM,
                                                               self._fdb_helper.MAX_FEATURES, for_ref=True)

                # match features
                matches = KeypointAlgo.match_features(n_desc, ref_desc, detector.defaultNorm(), method='brute')

                # solve pnp with ransac
                kp_3d = ref_kp_3d[[m.trainIdx for m in matches], :]
                kp_2d = np.array([n_kp[m.queryIdx].pt for m in matches])
                rvec, tvec, inliers = KeypointAlgo.solve_pnp_ransac(self.system_model, kp_2d, kp_3d, self._ransac_err)

                # check that solution is correct
                ok, err1, err2 = self._fdb_helper.calc_err(rvec, tvec, j1, i1, warn=len(inliers) > 30)

                if ok:
                    # update matches
                    matched_feats[[matches[i[0]].trainIdx for i in inliers]] = True

            except PositioningException:
                pass

        ref_desc = np.array(ref_desc)[matched_feats]
        ref_kp_3d = ref_kp_3d[matched_feats]
        ref_kp = [k for i, k in enumerate(ref_kp) if matched_feats[i]]

        self.timer.start()
        return ref_desc, ref_kp_3d, ref_kp, ref_img

    def _query_fdb(self, feat):
        self._ensure_fdb_loaded(feat)

        i1, i2, d_sc_ast_q, d_light_v, err_q, err_angle = self._fdb_helper.closest_scene()

        sc2gl_q = self.system_model.frm_conv_q(self.system_model.SPACECRAFT_FRAME, self.system_model.OPENGL_FRAME)
        self.latest_discretization_err_q = sc2gl_q * err_q * sc2gl_q.conj()

        if KeypointAlgo.FDB_USE_ALL_FEATS:
            # all features
            nf = self._fdb[4][i1, i2]
            ref_desc = self._fdb[0][i1, i2, 0:nf, :]
            ref_kp_3d = self._fdb[2][i1, i2, 0:nf, :]
        else:
            # only features that have matched with a neighbor mesh node
            ref_desc = self._fdb[0][i1, i2, self._fdb[3][i1, i2, :], :]
            ref_kp_3d = self._fdb[2][i1, i2, self._fdb[3][i1, i2, :], :]

        self.timer.stop()

        # render feature db scene for plotting
        ref_img, depth = self._fdb_helper.render_scene(i1, i2)
        # pos = (0, 0, -self.system_model.min_med_distance)
        # render_engine.render(self.obj_idx, pos, d_sc_ast_q, d_light_v)
        # ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)

        # construct necessary parts of ref_kp for plotting
        ref_kp = [cv2.KeyPoint(*self._cam.calc_img_xy(x, -y, -z+self._render_z), 1) for x, y, z in ref_kp_3d]

        self.timer.start()
        return ref_desc, ref_kp_3d, ref_kp, ref_img

    def _ensure_fdb_loaded(self, feat):
        self.timer.stop()
        if self._fdb is None or self._fdb_feat != feat:
            fname = self._fdb_helper.fdb_fname(feat)
            try:
                status, self._fdb_sc_ast_perms, self._fdb_light_perms, self._fdb = self._fdb_helper.load_fdb(fname)
                self._fdb_helper.set_mesh(self._fdb_sc_ast_perms, self._fdb_light_perms)
                self._fdb_feat = feat
            except (FileNotFoundError, EOFError):
                assert False, 'Couldn\'t find feature db file %s' % fname
        self.timer.start()


if __name__ == '__main__':
    """
    This code here tries to probe the accuracy of the system model "real" values
    """

    sm = RosettaSystemModel(rosetta_batch='mtp025')
    #img = 'ROS_CAM1_20140823T021833'
    #img = 'ROS_CAM1_20140822T020718'
    img = 'ROS_CAM1_20160208T070515'

    lblloader.load_image_meta(os.path.join(sm.asteroid.image_db_path, img + '.LBL'), sm)

    re = RenderEngine(sm.view_width, sm.view_height, antialias_samples=0)
    obj_idx = re.load_object(sm.asteroid.target_model_file, smooth=False)

    algo = KeypointAlgo(sm, re, obj_idx)
    model = RenderEngine.REFLMOD_LUNAR_LAMBERT
    size = (1024, 1024)  # (256, 256)

    orig_real = cv2.imread(os.path.join(sm.asteroid.image_db_path, img + '_P.png'), cv2.IMREAD_GRAYSCALE)
    real = cv2.resize(orig_real, size)
    bar = 255*np.ones((real.shape[0], 1), dtype='uint8')

    sm.reset_to_real_vals()
    orig_pose = cv2.resize(algo.render(shadows=True, reflection=model), size)

    # adjust s/c pose based on algo
    algo_sc_res = []
    for i in range(1):
        algo.solve_pnp(orig_real, None, algo.AKAZE, adjust_sc_rot=True)
        algo_sc_res.extend([bar, cv2.resize(algo.render(shadows=True, reflection=model), size)])
    print('s/c pose adjusted shift error: %f' % sm.calc_shift_err())

    algo_ast_res = []
    sm.reset_to_real_vals()
    for i in range(1):
        algo.solve_pnp(orig_real, None, algo.AKAZE, adjust_sc_rot=False)
        algo_ast_res.extend([bar, cv2.resize(algo.render(shadows=True, reflection=model), size)])
    print('ast pose adjusted shift error: %f' % sm.calc_shift_err())

    cv2.imshow('real img, rosetta pose based, adj s/c pose, adj ast pose',
               np.concatenate([real, bar, orig_pose] + algo_sc_res + algo_ast_res, axis=1))
    cv2.waitKey()
