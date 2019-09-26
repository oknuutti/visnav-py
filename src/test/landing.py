import math
from datetime import datetime

import numpy as np
import quaternion
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from algo import tools
from algo.model import Camera, SystemModel
from algo.odometry import VisualOdometry, Pose
from missions.didymos import DidymosSystemModel
from render.render import RenderEngine
from settings import *


if __name__ == '__main__':
    sm = DidymosSystemModel(use_narrow_cam=False, target_primary=False, hi_res_shape_model=False)
    sm.cam = Camera(1024, 1024, 20, 20)

    re = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=0)
    re.set_frustum(sm.cam.x_fov, sm.cam.y_fov, 0.002, 2)
    ast_v = np.array([0, 0, -0.3])
    #q = tools.angleaxis_to_q((math.radians(2), 0, 1, 0))
    #q = tools.angleaxis_to_q((math.radians(1), 1, 0, 0))
    #q = tools.rand_q(math.radians(2))
    #lowq_obj = sm.asteroid.load_noisy_shape_model(Asteroid.SM_NOISE_HIGH)
    obj = sm.asteroid.real_shape_model
    obj_idx = re.load_object(obj)
    # obj_idx = re.load_object(sm.asteroid.hires_target_model_file)
    t0 = datetime.now().timestamp()

    odo = VisualOdometry(sm, sm.cam.width, verbose=True, pause=False, use_scale_correction=True, use_ba=True,
                         )#keypoint_algo=VisualOdometry.KEYPOINT_FAST)
    # ast_q = quaternion.one
    ast_q = tools.rand_q(math.radians(15))

    current_sc = 1/np.linalg.norm(ast_v)
    sc_threshold = 3

    for t in range(100):
        ast_v[2] += 0.001
        #ast_q = q**t
        #ast_q = tools.rand_q(math.radians(0.1)) * ast_q
        #n_ast_q = ast_q * tools.rand_q(math.radians(.3))
        n_ast_q = ast_q
        cam_q = SystemModel.cv2gl_q * n_ast_q.conj() * SystemModel.cv2gl_q.conj()
        cam_v = tools.q_times_v(cam_q * SystemModel.cv2gl_q, -ast_v)
        prior = Pose(cam_v, cam_q, np.ones((3,))*0.1, np.ones((3,))*0.01)

        if False and 1/np.linalg.norm(ast_v) > current_sc * sc_threshold:
            # TODO: implement crop_model and augment_model
            obj = tools.crop_model(obj, cam_v, cam_q, sm.cam.x_fov, sm.cam.y_fov)
            obj = tools.augment_model(obj, multiplier=sc_threshold)
            obj_idx = re.load_object(obj)
            current_sc = 1/np.linalg.norm(ast_v)

        image = re.render(obj_idx, ast_v, ast_q, np.array([1, 0, -1])/math.sqrt(2), gamma=1.8, get_depth=False)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # estimate pose
        res, bias_sds, scale_sd = odo.process(image, datetime.fromtimestamp(t0 + t), prior, quaternion.one)

        if res is not None:
            tq = res.quat * SystemModel.cv2gl_q
            est_q = tq.conj() * res.quat.conj() * tq
            err_q = ast_q.conj() * est_q
            err_angle = tools.angle_between_q(ast_q, est_q)
            est_v = -tools.q_times_v(tq.conj(), res.loc)
            err_v = est_v - ast_v

            #print('\n')
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

            if False and len(odo.state.map3d) > 0:
                pts3d = tools.q_times_mx(SystemModel.cv2gl_q.conj(), odo.get_3d_map_pts())
                tools.plot_vectors(pts3d)
                errs = tools.point_cloud_vs_model_err(pts3d, obj)
                print('\n3d map err mean=%.3f, sd=%.3f, n=%d' % (
                    np.mean(errs),
                    np.std(errs),
                    len(odo.state.map3d),
                ))

            # print('\n')
        else:
            print('no solution\n')

        #cv2.imshow('image', image)
        #cv2.waitKey()
