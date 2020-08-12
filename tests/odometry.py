import unittest
import pickle
import tempfile
import os
import math
from datetime import datetime

import numpy as np
import quaternion
import cv2

from visnav.algo.model import Camera
from visnav.algo.odometry import VisualOdometry, Pose
from visnav.algo import tools


class TestOdometry(unittest.TestCase):
    def setUp(self, verbose=False):
        self.cam = get_cam()
        params = {
            'min_keypoint_dist': 10,
            'min_inliers': 12,
            'min_2d2d_inliers': 24,
        }
        self.odo = VisualOdometry(self.cam, self.cam.width/4, verbose=verbose, pause=False,
                                  use_scale_correction=False, est_cam_pose=False, **params)

    def tearDown(self):
        pass

    def assertQuatAlmostEqual(self, quat0, quat1, delta=1e-4, msg=None):
        if quat0 is None and quat1 is None:
            return
        diff = math.degrees(tools.angle_between_q(quat0, quat1))
        self.assertAlmostEqual(0, diff, delta=delta,
                               msg=None if msg is None else (msg + ': angle[deg] %f > %f' % (diff, delta)))

    def assertArrayAlmostEqual(self, arr0, arr1, delta=1e-7, ord=np.inf, msg=None):
        if arr0 is None and arr1 is None:
            return
        norm = np.linalg.norm(np.array(arr0)-np.array(arr1), ord=ord)
        self.assertAlmostEqual(0, norm, delta=delta,
                               msg=None if msg is None else (msg + ': norm(%s) %f > %f' % (ord, norm, delta)))

    def assertPoseAlmostEqual(self, pose0: Pose, pose1: Pose, delta_v=1e-7, delta_q=1e-4, msg=None):
        if pose0 is None and pose1 is None:
            return
        self.assertArrayAlmostEqual(pose0.loc, pose1.loc, delta=delta_v, ord=2,
                                    msg=None if msg is None else (msg + ': loc %s vs %s'%(pose0.loc, pose1.loc)))
        self.assertQuatAlmostEqual(pose0.quat, pose1.quat, delta=delta_q,
                                   msg=None if msg is None else (msg + ': quat %s vs %s'%(pose0.quat, pose1.quat)))

    def assertOdomResultAlmostEqual(self, result0, result1):
        pose0, bias_sds0, scale_sd0 = result0
        pose1, bias_sds1, scale_sd1 = result1
        msg = '%s deviate(s) too much from the expected value(s)'
        self.assertPoseAlmostEqual(pose0, pose1, delta_v=0.02, delta_q=1, msg=msg%'estimated poses')
        self.assertArrayAlmostEqual(bias_sds0, bias_sds1, delta=0.1, ord=np.inf, msg=msg%'error estimates')
        self.assertAlmostEqual(scale_sd0, scale_sd1, delta=0.01, msg=msg%'scale error estimate')

    def test_rotating_object(self, inputs=None, results=None):
        pickle_file = os.path.join(os.path.dirname(__file__), 'data', 'test_rotating_object.pickle')
        record = inputs is not None and results is None
        if not record and results is None:
            inputs, results = self._load_recording(pickle_file)
        else:
            results = []

        cam_q = quaternion.one
        orig_time = datetime.strptime('2020-07-01 15:42:00', '%Y-%m-%d %H:%M:%S').timestamp()

        for i, (img, cam_obj_v, cam_obj_q) in enumerate(inputs):
            time = datetime.fromtimestamp(orig_time + i*60)
            prior = Pose(cam_obj_v, cam_obj_q, np.ones((3,)) * 0.1, np.ones((3,)) * 0.01)
            res = self.odo.process(img, time, prior, cam_q)
            if record:
                results.append(res)
            elif 0:
                self.assertOdomResultAlmostEqual(results[i], res)

            if i > 1 and 0:
                self.assertIsNotNone(res[0], msg='failed to get pose estimate')
                self.assertPoseAlmostEqual(prior, res[0], delta_v=0.1, delta_q=10,
                                           msg='estimated pose deviates too much from the real one')

        if record:
            self._save_recording(pickle_file, inputs, results)

    def _save_recording(self, fname, inputs, results):
        tf = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tf.close()
        for i in range(len(inputs)):
            cv2.imwrite(tf.name, inputs[i][0], (cv2.IMWRITE_PNG_COMPRESSION, 9))
            with open(tf.name, 'br') as fh:
                inputs[i][0] = fh.read()
        os.unlink(tf.name)

        with open(fname, 'wb') as fh:
            pickle.dump((inputs, results), fh)

    def _load_recording(self, fname):
        with open(fname, 'rb') as fh:
            inputs, results = pickle.load(fh)

        tf = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tf.close()
        for i in range(len(inputs)):
            with open(tf.name, 'wb') as fh:
                fh.write(inputs[i][0])
                inputs[i][0] = cv2.imread(tf.name, cv2.IMREAD_GRAYSCALE)
        os.unlink(tf.name)

        return inputs, results

def get_rot_imgs():
    pass


def get_cam():
    common_kwargs_worst = {
        'sensor_size': (2048 * 0.0022, 1944 * 0.0022),
        'quantum_eff': 0.30,
        'px_saturation_e': 2200,  # snr_max = 20*log10(sqrt(sat_e)) dB
        'lambda_min': 350e-9, 'lambda_eff': 580e-9, 'lambda_max': 800e-9,
        'dark_noise_mu': 40, 'dark_noise_sd': 6.32, 'readout_noise_sd': 15,
        # dark_noise_sd should be sqrt(dark_noise_mu)
        'emp_coef': 1,  # dynamic range = 20*log10(sat_e/readout_noise))
        'exclusion_angle_x': 55,
        'exclusion_angle_y': 90,
    }
    common_kwargs_best = dict(common_kwargs_worst)
    common_kwargs_best.update({
        'quantum_eff': 0.4,
        'px_saturation_e': 3500,
        'dark_noise_mu': 25, 'dark_noise_sd': 5, 'readout_noise_sd': 5,
    })
    common_kwargs = common_kwargs_best

    return Camera(
        2048,  # width in pixels
        1944,  # height in pixels
        7.7,  # x fov in degrees  (could be 6 & 5.695, 5.15 & 4.89, 7.7 & 7.309)
        7.309,  # y fov in degrees
        f_stop=5,  # TODO: put better value here
        point_spread_fn=0.50,  # ratio of brightness in center pixel
        scattering_coef=2e-10,  # affects strength of haze/veil when sun shines on the lens
        **common_kwargs
    )


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'record':
        from visnav.algo.model import SystemModel
        from visnav.missions.didymos import DidymosSystemModel
        from visnav.render.render import RenderEngine
        from visnav.settings import *

        sm = DidymosSystemModel(use_narrow_cam=False, target_primary=False, hi_res_shape_model=True)
        re = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=0)
        re.set_frustum(sm.cam.x_fov, sm.cam.y_fov, 0.05, 2)
        obj = sm.asteroid.real_shape_model
        obj_idx = re.load_object(obj)

        light = np.array([1, 0, -0.5])
        light /= np.linalg.norm(light)
        cam_ast_v0 = np.array([0, 0, -sm.min_med_distance * 0.7])
        cam_ast_q0 = quaternion.one
        dq = tools.angleaxis_to_q((math.radians(1), 0, 1, 0))

        inputs = []
        for i in range(60):
            cam_ast_v = cam_ast_v0
            cam_ast_q = dq**i * cam_ast_q0
            image = re.render(obj_idx, cam_ast_v, cam_ast_q, light, gamma=1.8, get_depth=False)
            cam_ast_cv_v = tools.q_times_v(SystemModel.cv2gl_q, cam_ast_v)
            cam_ast_cv_q = SystemModel.cv2gl_q * cam_ast_q * SystemModel.cv2gl_q.conj()
            inputs.append([image, cam_ast_cv_v, cam_ast_cv_q])

        if 0:
            for image, _, _ in inputs:
                cv2.imshow('t', cv2.resize(image, None, fx=0.5, fy=0.5))
                cv2.waitKey()
        else:
            t = TestOdometry()
            t.setUp(verbose=True)
            t.test_rotating_object(inputs=inputs)
    else:
        unittest.main()
