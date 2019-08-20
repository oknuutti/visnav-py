import numpy as np
import quaternion
import cv2

from algo import tools
from algo.model import SystemModel


class VisualOdometry:
    (
        KEYPOINT_SHI_TOMASI,
        KEYPOINT_FAST,
        KEYPOINT_ORB,
        KEYPOINT_AKAZE,
    ) = range(4)

    (
        METHOD_FLOW_PRVREF_2D,  # always use previous frame keypoints as reference
        METHOD_FLOW_PRVREF_3D,  # same as above but after first solution, triangulate and use next time for solving pnp
        METHOD_FLOW_INIREF_2D,  # first frame keypoints used as reference for all subsequent frames until next init
        METHOD_FLOW_INIREF_3D,  # same as above but after solving pnp for subsequent frames, always re-triangulate first frame keypoints
        METHOD_MATCH_3D,      # always detect keypoints, use descriptors to match to local map, maintain it, save everything for N iterations
    ) = range(5)

    # TODO: try out interleaved keyframes for INIREF methods, i.e. detect new keypoints but continue to also track old ones
    INTERLEAVED_KEYFRAMES = False

    # TODO: try out matching triangulated 3d points to 3d model using icp for all methods with 3d points
    # TODO: try to init pose globally (before ICP) based on some 3d point-cloud descriptor
    USE_ICP = False

    # TODO: (1) try out bundle adjustment (for all methods?)
    USE_BA = False

    # TODO: (2) try out detection that leaves existing keypoints intact
    # TODO: (3) maintain a local map, project and somehow match
    # TODO: get better distributed keypoints by using grids

    DEF_TIME_THRESHOLD = 60  # in seconds
    MIN_TRACKED_KP = 100  # find new keypoints when tracked keypoint count less than this
    UNCERTAINTY_LIMIT_RATIO = 30  # current solution has to be this much more uncertain as new initial pose to change base pose
    MIN_INLIERS = 20

    def __init__(self, sm, img_width=None, feat=KEYPOINT_AKAZE, method=METHOD_FLOW_PRVREF_2D, time_threshold=DEF_TIME_THRESHOLD):
        self.sm = sm
        self.img_width = img_width
        self.feat = feat
        self.method = method
        assert method in (VisualOdometry.METHOD_FLOW_PRVREF_2D, VisualOdometry.METHOD_FLOW_PRVREF_3D), 'method not implemented'
        self.time_threshold = time_threshold
        self.lk_params = {
            'winSize': (32, 32),
            'maxLevel': 4,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.05),
        }
        if feat == VisualOdometry.KEYPOINT_SHI_TOMASI:
            self.kp_params = {
                'maxCorners': 2000,
                'qualityLevel': 0.003,  # default around 0.25?
                'minDistance': 7,
                'blockSize': 7,
            }
        elif feat == VisualOdometry.KEYPOINT_FAST:
            self.kp_params = {
                'threshold': 5,         # default around 25?
                'nonmaxSuppression': True,
            }
        elif feat == VisualOdometry.KEYPOINT_AKAZE:
            self.kp_params = {
                'diffusivity': cv2.KAZE_DIFF_PM_G2, # default: cv2.KAZE_DIFF_PM_G2, KAZE_DIFF_CHARBONNIER is not as stable
                'threshold': 0.0001,         # default: 0.001
                'nOctaves': 4,               # default: 4
                'nOctaveLayers': 4,          # default: 4
            }

        # state
        self.old_time = None
        self.old_img = None
        self.old_kp2d = None
        self.old_kp3d = None
        self.old_q = None
        self.old_t = None
        self.old_Sr = None
        self.old_St = None

    def __reduce__(self):
        return (VisualOdometry, (
            # init params, state
        ))

    def process(self, new_img, new_time, ini_t, ini_q, ini_St, ini_Sr, verbose=1):
        # maybe scale image
        img_sc = 1
        if self.img_width is not None:
            img_sc = self.img_width / new_img.shape[1]
            new_img = cv2.resize(new_img, None, fx=img_sc, fy=img_sc)

        # maybe force initialization
        if self.old_time is None or new_time - self.old_time > self.time_threshold:
            print('reset state because too long time')
            self.old_img = None
            self.old_kp2d = None
            self.old_kp3d = None
            self.old_q = ini_q
            self.old_t = ini_t
            self.old_Sr = ini_Sr
            self.old_St = ini_St

        K = self.sm.cam.intrinsic_camera_mx()
        scale = np.linalg.norm(self.old_t - ini_t)  # diff between prev pose and current init pose
        mask = None
        kp2d, kp3d = None, None
        res_q, res_t = None, None
        res_Sr, res_St = None, None

        if self.old_img is not None and self.old_kp2d is not None:
            # track keypoints using Lukas-Kanade method
            kp2d, mask, err = cv2.calcOpticalFlowPyrLK(self.old_img, new_img, self.old_kp2d, None, **self.lk_params)
            print('Tracking: %d/%d' % (np.sum(mask), len(self.old_kp2d)))

            # do different things depending on state
            if self.old_kp3d is not None and self.method == VisualOdometry.METHOD_FLOW_PRVREF_3D:
                # do 3d-2d matching
                # solve pose using ransac & ap3p
                inliers = np.where(mask.flatten() == 1)[0].reshape((-1, 1)).astype('int32')
                ok, r, t, inliers = cv2.solvePnPRansac(self.old_kp3d, kp2d/img_sc, K, None, inliers=inliers,
                                                       iterationsCount=10000, reprojectionError=8, flags=cv2.SOLVEPNP_AP3P)
                if ok:
                    print('PnP: %d/%d' % (len(inliers), np.sum(mask)))
                    mask = np.zeros((len(mask),))
                    mask[inliers] = 1
                    q = tools.angleaxis_to_q(r)

                    # solvePnPRansac has some bug that apparently randomly gives 180deg wrong answer
                    if abs(tools.q_to_ypr(q)[2]) > math.pi*0.9:
                        print('rotated pnp-ransac solution by 180deg around x-axis')
                        q = q * tools.ypr_to_q(0, 0, math.pi)
                        t = -t

                else:
                    print('PnP Failed')
                    mask = np.zeros((len(mask),))

            elif self.old_kp2d is not None:
                # no need to do 2d-2d matching?
                # solve pose using ransac & 5-point algo
                E, mask2 = cv2.findEssentialMat(kp2d/img_sc, self.old_kp2d/img_sc, K, mask=mask.copy(),
                                                method=cv2.RANSAC, prob=0.999, threshold=1.0)
                print('E-mat: %d/%d' % (np.sum(mask2), np.sum(mask)))

                _, R, ut, mask = cv2.recoverPose(E, kp2d/img_sc, self.old_kp2d/img_sc, K, mask=mask2.copy())
                print('E=>R: %d/%d' % (np.sum(mask), np.sum(mask2)))

                t = scale * ut
                q = quaternion.from_rotation_matrix(R)

                # methods with interleaved keyframes or descriptor matching:
                #  - TODO: how to get epipolar line angles so that can detect keyframes/ok initialization?

                # if use bundle adjustment:
                #  - TODO: maintain poses, keypoint associations?
            else:
                assert False, 'invalid state!'

            mask = mask.flatten() == 1
            if verbose and kp2d is not None and self.old_kp2d is not None:
                self._draw_matches(self.old_img, new_img, self.old_kp2d, kp2d, mask)

            # only leave inliers
            kp2d = kp2d[mask, :, :]

        # require enough inliers
        kp2d = None if kp2d is None or len(kp2d) < VisualOdometry.MIN_INLIERS else kp2d

        if kp2d is not None:
            # TODO: calculate uncertainty / error var
            St = np.array([1, 1, 1]) * 0.1
            Sr = np.array([1, 1, 1]) * 0.01

            # triangulate matched 2d keypoints to get 3d keypoints
            if self.method == VisualOdometry.METHOD_FLOW_PRVREF_3D:
                T0 = np.hstack((quaternion.as_rotation_matrix(q.conj()),
                                tools.q_times_v(q.conj(), -t).reshape((-1, 1))))
                                #-t))
                T1 = np.hstack((np.identity(3), np.zeros((3, 1))))
                kp4d = cv2.triangulatePoints(K.dot(T0), K.dot(T1), self.old_kp2d[mask, :, :]/img_sc, kp2d/img_sc)
                kp3d = kp4d.T[:, :3]/kp4d.T[:, 3].reshape((-1, 1))

            print('delta q: ' + ' '.join(['%.3fdeg' % math.degrees(a) for a in tools.q_to_ypr(q)]))
            print('delta v: ' + ' '.join(['%.3fm' % a for a in t*1000]))

            # update pose and uncertainty
            self.old_t += tools.q_times_v(self.old_q, t)
            self.old_St += tools.q_times_v(self.old_q, St)
            self.old_Sr += tools.q_times_v(self.old_q, Sr)
            self.old_q = q * self.old_q
            res_t = self.old_t
            res_q = self.old_q
            res_St = self.old_St
            res_Sr = self.old_Sr

            if np.linalg.norm(res_St) / VisualOdometry.UNCERTAINTY_LIMIT_RATIO > np.linalg.norm(ini_St):
                # new initial pose is more certain than the result pose by a certain margin, reset pose
                print('new initial pose is more certain than the result pose, resetting base pose')
                self.old_q = ini_q
                self.old_t = ini_t
                self.old_Sr = ini_Sr
                self.old_St = ini_St

        # maybe detect a new set of 2d keypoints for next time
        if kp2d is None or len(kp2d) < VisualOdometry.MIN_TRACKED_KP:
            if self.feat == VisualOdometry.KEYPOINT_SHI_TOMASI:
                # detect Shi-Tomasi keypoints
                kp2d = cv2.goodFeaturesToTrack(new_img, mask=None, **self.kp_params)
            elif self.feat == VisualOdometry.KEYPOINT_FAST:
                # detect FAST keypoints
                det = cv2.FastFeatureDetector_create(**self.kp_params)
                kp2d = det.detect(new_img)
                kp2d = self.kp2arr(kp2d)
            elif self.feat == VisualOdometry.KEYPOINT_AKAZE:
                # detect AKAZE keypoints
                det = cv2.AKAZE_create(**self.kp_params)
                kp2d = det.detect(new_img)
                kp2d = self.kp2arr(kp2d)
            else:
                #TODO: try same keypoint detection than in AKAZE
                assert False, 'unknown keypoint detection algorithm: %s' % self.feat
            kp3d = None

        # update state if valid 2d keypoints found
        if kp2d is not None:
            self.old_time = new_time
            self.old_img = new_img
            self.old_kp2d = kp2d
            self.old_kp3d = kp3d

        return res_q, res_t, res_Sr, res_St

    def arr2kp(self, arr, size=7):
        return [cv2.KeyPoint(p[0, 0], p[0, 1], size) for p in arr]

    def kp2arr(self, kp):
        return np.array([k.pt for k in kp], dtype='f4').reshape((-1, 1, 2))

    def _draw_matches(self, img1, img2, kp1, kp2, mask, pause=True, label='matches'):
        idxs = np.array(list(range(len(kp1))))[mask]
        matches = [[cv2.DMatch(i, i, 0)] for i in idxs]
        draw_params = {
            #            matchColor: (88, 88, 88),
            'singlePointColor': (0, 0, 255),
            #            'flags': cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        }
        # scale images, show
        img_sc = 768/img1.shape[0]
        sc_img1 = cv2.cvtColor(cv2.resize(img1, None, fx=img_sc, fy=img_sc, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
        sc_img2 = cv2.cvtColor(cv2.resize(img2, None, fx=img_sc, fy=img_sc, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
        img3 = cv2.drawMatchesKnn(sc_img1, self.arr2kp(kp1*img_sc), sc_img2, self.arr2kp(kp2*img_sc), matches, None, **draw_params)
        cv2.imshow(label, img3)
        cv2.waitKey(0 if pause else 25)


if __name__ == '__main__':
    import math
    import cv2
    from missions.didymos import DidymosSystemModel
    from render.render import RenderEngine
    from settings import *

    sm = DidymosSystemModel(use_narrow_cam=False, target_primary=False, hi_res_shape_model=False)
    re = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=0)
    re.set_frustum(sm.cam.x_fov, sm.cam.y_fov, 0.05, 2)
    ast_v = np.array([0, 0, -sm.min_med_distance * 1])
    #q = tools.angleaxis_to_q((math.radians(2), 0, 1, 0))
    q = tools.angleaxis_to_q((math.radians(2), 1, 0, 0))
    #q = tools.rand_q(math.radians(2))
    obj_idx = re.load_object(sm.asteroid.real_shape_model)
    # obj_idx = re.load_object(sm.asteroid.hires_target_model_file)

    odo = VisualOdometry(sm, sm.cam.width)

    for i in range(30):
        ast_q = q**i
        image = re.render(obj_idx, ast_v, ast_q, np.array([1, 0, 0])/math.sqrt(1), get_depth=False)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        #n_ast_q = ast_q * tools.rand_q(math.radians(.3))
        n_ast_q = ast_q
        cam_q = SystemModel.cv2gl_q * n_ast_q.conj() * SystemModel.cv2gl_q.conj()
        cam_v = tools.q_times_v(cam_q * SystemModel.cv2gl_q, -ast_v)
        res_q, res_v, res_Sr, res_St = odo.process(image, i, cam_v, cam_q, np.ones((3,))*0.1, np.ones((3,))*0.01, verbose=True)

        if res_q is not None:
            tq = res_q * SystemModel.cv2gl_q
            est_q = tq.conj() * res_q.conj() * tq
            err_q = ast_q.conj() * est_q
            err_angle = tools.angle_between_q(ast_q, est_q)
            est_v = -tools.q_times_v(tq.conj(), res_v)
            err_v = est_v - ast_v

            print('\n')
            # print('rea ypr: %s' % ' '.join('%.1fdeg' % math.degrees(a) for a in tools.q_to_ypr(cam_q)))
            # print('est ypr: %s' % ' '.join('%.1fdeg' % math.degrees(a) for a in tools.q_to_ypr(res_q)))
            print('rea ypr: %s' % ' '.join('%.1fdeg' % math.degrees(a) for a in tools.q_to_ypr(ast_q)))
            print('est ypr: %s' % ' '.join('%.1fdeg' % math.degrees(a) for a in tools.q_to_ypr(est_q)))
            # print('err ypr: %s' % ' '.join('%.2fdeg' % math.degrees(a) for a in tools.q_to_ypr(err_q)))
            print('err angle: %.2fdeg' % math.degrees(err_angle))
            # print('rea v: %s' % ' '.join('%.1fm' % a for a in cam_v*1000))
            # print('est v: %s' % ' '.join('%.1fm' % a for a in res_v*1000))
            print('rea v: %s' % ' '.join('%.1fm' % a for a in ast_v*1000))
            print('est v: %s' % ' '.join('%.1fm' % a for a in est_v*1000))
            # print('err v: %s' % ' '.join('%.2fm' % a for a in err_v*1000))
            print('err norm: %.2fm\n' % np.linalg.norm(err_v*1000))
            # print('\n')
        else:
            print('no solution\n')

        #cv2.imshow('image', image)
        #cv2.waitKey()
