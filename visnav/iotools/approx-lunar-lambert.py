import math
import os

import cv2
import numpy as np
import quaternion

from scipy.interpolate import interp1d, interp2d
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt

from visnav.algo import tools
from visnav.algo.base import AlgorithmBase
from visnav.algo.image import ImageProc
from visnav.algo.keypoint import KeypointAlgo
from visnav.algo.model import SystemModel
from visnav.iotools import lblloader
from visnav.iotools.objloader import ShapeModel
from visnav.missions.didymos import DidymosPrimary
from visnav.missions.itokawa import ItokawaSystemModel
from visnav.missions.rosetta import RosettaSystemModel
from visnav.render.render import RenderEngine

from visnav.settings import *


def Lfun(x, p):
    L = x[0]
    for i in range(len(x)-1):
        L += x[i+1]*p**(i+1)
    return L


def est_itokawa_model(refl_model, use_ls=False, hapke=True, adjust_pose=False, iters=1, init_noise=0.0, verbose=True):
    sm = ItokawaSystemModel(hi_res_shape_model=True)
    sm.view_width = sm.cam.width
    img_file = 'st_2422895458_v_vf'

    imgs = {    # phase angle, exposure
        img_file: 0.087,   # 16.7, 0.087s
    }

    if 0:
        # initial pose from SISPO config json
        rel_p = np.array([146226188132.0, -68812322442.0, -477863711.0]) \
                - np.array([146226194732.0, -68812326932.0, -477863381.0])
        sm.spacecraft_q = tools.angleaxis_to_q([1.847799, -0.929873, 0.266931, -0.253146])
        sm.spacecraft_pos = tools.q_times_v(SystemModel.sc2gl_q.conj() * sm.spacecraft_q.conj(), -rel_p * 0.001)
        sm.asteroid.real_position = np.array([146226194732.0, -68812326932.0, -477863381.0])
        sm.asteroid_q = tools.eul_to_q((math.radians(90),), 'z')
        sm.real_asteroid_axis = [1, 0, 0]
        sm.real_asteroid_q = quaternion.one
        sm.real_spacecraft_q = quaternion.one
        sm.real_spacecraft_pos = [0, 0, -100]
        sm.real_spacecraft_q = quaternion.one
        sm.time.real_value = 0
        sm.swap_values_with_real_vals()
        sm.save_state(os.path.join(sm.asteroid.image_db_path, img_file + '.lbl'))
        sm.swap_values_with_real_vals()

        re = RenderEngine(sm.view_width, sm.view_height, antialias_samples=0)
        obj_idx = re.load_object(sm.asteroid.real_shape_model, smooth=False)
        ab = AlgorithmBase(sm, re, obj_idx)
        synth = ab.render(shadows=True, reflection=RenderEngine.REFLMOD_HAPKE, gamma=1, textures=False)
        real = cv2.imread(os.path.join(sm.asteroid.image_db_path, img_file + '.png'), cv2.IMREAD_GRAYSCALE)
        cv2.imshow('real vs fake', np.concatenate((cv2.resize(real, synth.shape), synth), axis=1))
        cv2.waitKey()
        quit()
    if 0:
        # PSF estimation
        #   extracted from fig 18, v-band, https://www.sciencedirect.com/science/article/abs/pii/S0019103510000023
        #   using https://apps.automeris.io/wpd/
        psf = np.array([0.43052, 0.85239,
                        0.67939, 0.63180,
                        0.96927, 0.41543,
                        1.2806, 0.26510,
                        1.5518, 0.16749,
                        1.8095, 0.10582,
                        2.2352, 0.051574,
                        2.7611, 0.020587,
                        3.1737, 0.0083832,
                        3.6655, 0.0046986
                        ]).reshape((-1, 2))

        def gauss(x, sig):
            return np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))

        def cost(x):
            return np.log(psf[:, 1]) - np.log(x[0] * gauss(psf[:, 0], x[1]) + (1 - x[0]) * gauss(psf[:, 0], x[2]))

        from scipy.optimize import least_squares
        res = least_squares(cost, (0.9, 0.8, 2.0))
        print('res: ' % (res.x,))

        import matplotlib.pyplot as plt
        plt.scatter(psf[:, 0], psf[:, 1])
        plt.gca().loglog()
        plt.xlim((0.1, 10))
        plt.ylim((1e-3, 1))
        x = np.exp(np.linspace(np.log(1e-2), np.log(10), 100))
        plt.plot(x, res.x[0] * gauss(x, res.x[1]) + (1 - res.x[0]) * gauss(x, res.x[2]))
        plt.show()

    init_poses = {
        # 'st_2422895458_v_vf': [2.02553e-02, -4.04985e-03, -7.98449e+00, 5.53892e+00, 2.06534e-02, -2.72958e+00,
        #                        -4.38323e-03, -4.39473e-01],  # => 10.48100,
        # 'st_2422895458_v_vf': [2.03997e-02, -4.54192e-03, -7.97903e+00, 5.53755e+00, 2.04618e-02, -2.73315e+00,
        #                        -4.18826e-03, -4.34453e-01],  # => 9.66055 => 9.378393
        # 'st_2422895458_v_vf': [2.03239e-02, -4.51303e-03, -7.97817e+00, 5.51440e+00, 2.41751e-02, -2.77959e+00,
        #                        -2.43157e-03, -4.33176e-01],  # => 9.24684,  9.094556
        'st_2422895458_v_vf': [2.03795e-02, -4.56515e-03, -7.97761e+00, 5.51429e+00, 2.48090e-02, -2.78167e+00,
                               -2.39040e-03, -4.31989e-01],  # => 9.04088 => ,
    }

    if refl_model:
        x = est_model(sm, imgs, refl_model, use_ls, hapke=True, init_poses=init_poses, iters=iters,
                      init_noise=init_noise, verbose=True)
    else:
        x = []
        for file, exp in imgs.items():
            imgs = {file: exp}
            x_, err = est_model(sm, imgs, refl_model, use_ls, hapke=True, init_poses=init_poses, iters=iters,
                                adjust_pose=adjust_pose, init_noise=init_noise, verbose=verbose)
            print("'%s': [%s] => %.5f," % (file, ', '.join(['%.5e' % f for f in x_[0]]), err))
            x.append(x_[0])
    return x


def est_rosetta_model(refl_model, use_ls=False, hapke=True, iters=1, init_noise=0.0, verbose=True):
    sm = RosettaSystemModel()
    imgs = {
        'ROS_CAM1_20140831T104353': 3.2,   # 60, 3.2s
        'ROS_CAM1_20140831T140853': 3.2,   # 62, 3.2s
        'ROS_CAM1_20140831T103933': 3.2,   # 65, 3.2s
        'ROS_CAM1_20140831T022253': 3.2,   # 70, 3.2s
        'ROS_CAM1_20140821T100719': 2.8,   # 75, 2.8s
        'ROS_CAM1_20140821T200718': 2.0,   # 80, 2.0s
        'ROS_CAM1_20140822T113854': 2.0,   # 85, 2.0s
        'ROS_CAM1_20140823T021833': 2.0,   # 90, 2.0s
        'ROS_CAM1_20140819T120719': 2.0,   # 95, 2.0s
        'ROS_CAM1_20140824T021833': 2.8,   # 100, 2.8s
        'ROS_CAM1_20140824T020853': 2.8,   # 105, 2.8s
        'ROS_CAM1_20140824T103934': 2.8,   # 110, 2.8s
        'ROS_CAM1_20140818T230718': 2.0,   # 113, 2.0s
        'ROS_CAM1_20140824T220434': 2.8,   # 120, 2.8s
        'ROS_CAM1_20140828T020434': 2.8,   # 137, 2.8s
        'ROS_CAM1_20140827T140434': 3.2,   # 145, 3.2s
        'ROS_CAM1_20140827T141834': 3.2,   # 150, 3.2s
        'ROS_CAM1_20140827T061834': 3.2,   # 155, 3.2s
        'ROS_CAM1_20140827T021834': 3.2,   # 157, 3.2s
        'ROS_CAM1_20140826T221834': 2.8,   # 160, 2.8s
    }
    init_poses = {
        'ROS_CAM1_20140831T104353': [2.86728e+00, -3.29380e+00, -6.93154e+01, 8.50184e-01, 7.69297e-01, 4.18987e+00,
                                     -4.26772e-01, -1.30608e+00],
        'ROS_CAM1_20140831T140853': [2.82040e+00, -3.16094e+00, -6.60739e+01, 5.10328e+00, -1.26288e+00, 2.59983e+00,
                                     -4.96884e-01, -1.07037e+00],
        'ROS_CAM1_20140831T103933': [2.87641e+00, 2.20805e+00, -6.98097e+01, 9.29692e-01, 6.71018e-01, 3.89043e+00,
                                     -5.05890e-01, -1.15714e+00],
        'ROS_CAM1_20140831T022253': [-2.37373e+00, 2.17973e+00, -6.68176e+01, 5.19087e+00, -2.52479e+00, -1.17818e+00,
                                     -4.91720e-01, -1.17365e+00],
        'ROS_CAM1_20140821T100719': [1.73006e-01, -2.38229e-01, -7.25163e+01, -1.78245e-01, 5.16713e-01, 1.46140e+00,
                                     -4.96074e-01, -1.22225e+00],
        'ROS_CAM1_20140821T200718': [7.88705e-02, -4.66685e-01, -6.68539e+01, -5.51790e-01, -1.81700e+00, -8.97430e-01,
                                     -4.30595e-01, -1.11678e+00],
        'ROS_CAM1_20140822T113854': [2.50132e+00, -2.41133e+00, -6.01961e+01, -7.01139e-02, 5.66697e-01, 2.17266e+00,
                                     -4.85382e-01, -1.09542e+00],
        'ROS_CAM1_20140823T021833': [-2.31681e+00, -2.36285e+00, -5.95774e+01, 3.27741e-01, 6.79246e-01, 3.19829e+00,
                                     -5.03423e-01, -1.17515e+00],
        'ROS_CAM1_20140819T120719': [5.51455e-02, -1.46443e-02, -7.83410e+01, 5.27990e-01, 8.24076e-01, 3.91783e+00,
                                     -4.77252e-01, -1.13768e+00],
        'ROS_CAM1_20140824T021833': [-2.58337e+00, -2.73790e+00, -6.70307e+01, 1.68083e-01, 6.58562e-01, 2.91220e+00,
                                     -4.63205e-01, -1.16574e+00],
        'ROS_CAM1_20140824T020853': [2.71176e+00, -2.70252e+00, -6.67518e+01, 1.39304e-01, 6.78840e-01, 2.53165e+00,
                                     -4.68647e-01, -1.13073e+00],
        'ROS_CAM1_20140824T103934': [2.71632e+00, 2.58758e+00, -6.91670e+01, 1.93910e+00, -1.89599e+00, -5.11456e+00,
                                     -3.80947e-01, -1.12699e+00],
        'ROS_CAM1_20140818T230718': [-7.51678e-01, -3.42611e-01, -7.87677e+01, 3.21518e-01, 6.73784e-01, 2.80083e+00,
                                     -5.26223e-01, -1.10787e+00],
        'ROS_CAM1_20140824T220434': [2.24894e+00, 2.28264e+00, -6.03868e+01, 3.98722e+00, -2.55235e+00, -3.74485e+00,
                                     -4.08233e-01, -1.21924e+00],
        'ROS_CAM1_20140828T020434': [2.26508e+00, 2.18112e+00, -6.09602e+01, 1.14413e+00, -1.57632e+00, -4.69158e+00,
                                     -4.78622e-01, -1.13032e+00],
        'ROS_CAM1_20140827T140434': [2.64506e+00, 2.46440e+00, -6.75340e+01, -2.51789e-01, 4.34034e-01, 1.36745e+00,
                                     -4.82309e-01, -1.10132e+00],
        'ROS_CAM1_20140827T141834': [-2.73206e+00, -2.89021e+00, -6.78508e+01, -2.28200e-01, 4.44550e-01, 1.52755e+00,
                                     -4.85886e-01, -1.10526e+00],
        'ROS_CAM1_20140827T061834': [-2.61793e+00, -2.99875e+00, -6.82736e+01, 5.26060e-01, 6.02144e-01, 3.69438e+00,
                                     -5.19115e-01, -1.05481e+00],
        'ROS_CAM1_20140827T021834': [-2.57562e+00, -2.76749e+00, -6.51906e+01, -1.89664e-01, 4.56566e-01, 1.69570e+00,
                                     -5.10384e-01, -1.09044e+00],
        'ROS_CAM1_20140826T221834': [-2.38290e+00, -2.65574e+00, -6.27370e+01, 4.71862e+00, -9.08792e-01, 3.43537e+00,
                                     -4.60109e-01, -1.12087e+00],
    }
    if refl_model:
        x, err = est_model(sm, imgs, refl_model, use_ls, hapke=True, init_poses=init_poses, iters=1, init_noise=0.0,
                           verbose=True)
    else:
        x = []
        for file, exp in imgs.items():
            imgs = {file: exp}
            x_, err = est_model(sm, imgs, refl_model, use_ls, hapke=True, init_poses=init_poses, iters=1,
                                init_noise=0.0, verbose=False)
            print("'%s': [%s] => %.5f," % (file, ', '.join(['%.5e' % f for f in x_[0]]), err))
            x.append(x_[0])
    return x


def est_model(sm, imgs, refl_model=True, use_ls=False, hapke=True, init_poses=None, adjust_pose=False, iters=1, init_noise=0.0, verbose=True):
    map_u = None
    if sm.cam.dist_coefs is not None and np.any(np.array(sm.cam.dist_coefs) != 0):
        cam_mx = sm.cam.intrinsic_camera_mx()
        map_u, map_v = cv2.initUndistortRectifyMap(cam_mx, np.array(sm.cam.dist_coefs), None,
                                                   cam_mx, (sm.cam.width, sm.cam.height), cv2.CV_16SC2)

    target_exposure = np.min(list(imgs.values()))
    is_ros = True
    for img, exposure in imgs.items():
        imgfile = os.path.join(sm.asteroid.image_db_path, img + '_P.png')
        is_ros = os.path.exists(imgfile)    # TODO: remove quick hack to support itokawa instead of only 67p

        real = cv2.imread(imgfile if is_ros else imgfile[:-6]+'.png', cv2.IMREAD_GRAYSCALE)
        if is_ros:
            real = ImageProc.adjust_gamma(real, 1/1.8)

        if map_u is not None:
            real = cv2.remap(real, map_u, map_v, interpolation=cv2.INTER_CUBIC)
            if 1:
                cv2.imwrite(imgfile[:-4]+'_undist.png', real)

        #dark_px_lim = np.percentile(real, 0.1)
        #dark_px = np.mean(real[real<=dark_px_lim])
        real = cv2.resize(real, (sm.view_width, sm.view_height))
        # remove dark pixel intensity and normalize based on exposure
        #real = real - dark_px
        #real *= (target_exposure / exposure)
        imgs[img] = real

    model = RenderEngine.REFLMOD_HAPKE if hapke else RenderEngine.REFLMOD_LUNAR_LAMBERT

    re = RenderEngine(sm.view_width, sm.view_height, antialias_samples=16)
    obj_idx = re.load_object(sm.asteroid.hires_target_model_file, smooth=False)
    ab = KeypointAlgo(sm, re, obj_idx)
    KeypointAlgo.MAX_FEATURES = 1000
    KeypointAlgo.DEF_RANSAC_ERROR = 2
    KeypointAlgo.DISCARD_OFF_OBJECT_FEATURES = True
    ab.RENDER_TEXTURES = False
    ab.RENDER_REFLECTION_MODEL = model
    ab.RENDER_PSF_PARAMS = sm.cam.point_spread_fn
    ab.REFINE_RANSAC_RESULT = True

    defs = sm.asteroid.reflmod_params[model]

    def set_refl_params(sm, x):
        if hapke:
            # optimize J, th, w, b, (c), B_SH0, hs
            # 8.41668e+02, 9.16458e+00, -1.01511e-01, 1.06246e-04
            # 1.1001e+03, 1.5932e+01, 5.6303e-03, 2.8213e-04
            # 8.78012e+02, 2.33807e+01, -1.56845e-01
            xsc = np.abs(x)
            if sppf_n == 1:
                xsc[2] = x[2]
            xsc = list(xsc)
            vals = xsc[:2] + [defs[2]] + xsc[2:] + defs[len(xsc)+1:]
        else:
            vals = [1] + list(np.array(x)[:-1] * scales[:-1]) + [0]*(5-ll_poly) + [x[-1]*scales[-1], 0, 0, 0]
        sm.asteroid.reflmod_params[model] = vals

    def set_pose_params(sm, x_all, i, file):
        # get sc orientation
        lbl_file = os.path.join(sm.asteroid.image_db_path, file + ('.LBL' if is_ros else '.lbl'))
        lblloader.load_image_meta(lbl_file, sm)
        sm.swap_values_with_real_vals()

        # optimize rel_loc, rel_rot, light
        xsc = list(x_all)
        x = xsc[9*i: 9*i + 9]
        sm.spacecraft_pos = x[0:3]
        sm.asteroid_q = tools.angleaxis_to_q(x[3:6])
        sm.asteroid.real_position = tools.spherical2cartesian(*x[6:8], np.linalg.norm(sm.asteroid.real_position))

    if refl_model:
        if hapke:
            # L, th, w, b (scattering anisotropy), c (scattering direction from forward to back), B0, hs
            sppf_n = 1
            if 0:
                real_ini_x = [715, 16.42, 0.3057, 0.8746]
            else:
                real_ini_x = defs[:2] + defs[3:3+sppf_n]
            scales = np.array((500, 20, 3e-1, 3e-1))[:2+sppf_n]
        else:
            ll_poly = 5
            #real_ini_x = np.array(defs[:7])
            real_ini_x = np.array((9.95120e-01, -6.64840e-03, 3.96267e-05, -2.16773e-06, 2.08297e-08, -5.48768e-11, 1))  # theta=20
            real_ini_x = np.hstack((real_ini_x[0:ll_poly+1], (real_ini_x[-1],)))
            scales = np.array((3e-03, 2e-05, 1e-06, 1e-08, 5e-11, 1))
            scales = np.hstack((scales[0:ll_poly], (scales[-1],)))
    else:
        def params_from_sm(sm):
            loc = sm.spacecraft_pos
            rot = tools.q_to_angleaxis(sm.asteroid_q, compact=True)
            light = tools.cartesian2spherical(*sm.asteroid.real_position)
            return [*loc, *rot, *light[:2]]

        x, s = [], []
        for i, (file, real) in enumerate(imgs.items()):
            if file not in init_poses:
                lbl_file = os.path.join(sm.asteroid.image_db_path, file + ('.LBL' if is_ros else '.lbl'))
                lblloader.load_image_meta(lbl_file, sm)
                sm.swap_values_with_real_vals()
                x_ = params_from_sm(sm)
            else:
                x_ = init_poses[file]
                if 1:
                    set_pose_params(sm, x_, 0, file)
                    to_sispo(sm, ab)
                    quit()

            if adjust_pose:
                set_pose_params(sm, x_, 0, file)
                # ab._pause = True
                ab.solve_pnp(real, feat=KeypointAlgo.AKAZE, init_z=x_[2],
                            #detector_params={'qualityLevel': 0.05, 'minDistance': 10})
                             detector_params={'nOctaves': 1, 'nOctaveLayers': 1, 'threshold': 0.0001})
                x_ = params_from_sm(sm)
                print('first: %s' % (x_,))
            x.append(x_)
            s.append([1]*3 + [np.pi/180]*3 + [3*np.pi/180]*2)
        real_ini_x = np.array(x).flatten()
        scales = np.array(s).flatten()

    # debug 1: real vs synth, 2: err img, 3: both
    def costfun(x, refl_model, ls=False, debug=0, verbose=True):
        if refl_model:
            set_refl_params(sm, np.array(x) * scales)

        err = []
        for i, (file, real) in enumerate(imgs.items()):
            if refl_model:
                if file not in init_poses:
                    lbl_file = os.path.join(sm.asteroid.image_db_path, file + ('.LBL' if is_ros else '.lbl'))
                    lblloader.load_image_meta(lbl_file, sm)
                    sm.swap_values_with_real_vals()
                else:
                    set_pose_params(sm, init_poses[file], 0, file)
            else:
                set_pose_params(sm, np.array(x) * scales, i, file)

            synth2 = ab.render(shadows=True, reflection=model, gamma=1, textures=False)
            synth2 = ImageProc.apply_point_spread_fn(synth2, **sm.cam.point_spread_fn)
            err_img = synth2.astype('float') - real
            if not ls:
                if 0:
                    err_img = err_img ** 2
                    lim = np.percentile(err_img, 99)
                    err_img[err_img > lim] = 0
                else:
                    err_img = tools.pseudo_huber_loss(err_img, 10)  # 10 DN error still considered normal
            err.append(err_img)

            if debug:
                if debug % 2:
                    # cv2.imshow('real vs synthetic', np.concatenate((real.astype('uint8'), 255*np.ones((real.shape[0], 1), dtype='uint8'), synth2), axis=1))
                    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
                    axs[0].imshow(real.astype('uint8'), cmap='gray', vmin=0, vmax=255)
                    axs[1].imshow(synth2, cmap='gray', vmin=0, vmax=255)
                    axs[2].imshow(np.abs(synth2.astype('float') - real))
                elif debug > 1:
                    # err_img = (err_img ** 2) ** 0.2
                    # cv2.imshow('err', err_img/np.max(err_img))
                    plt.figure()
                    plt.imshow(np.abs(synth2.astype('float') - real))
                # cv2.waitKey()
                plt.tight_layout()
                plt.show()

        err = np.array(err).flatten()
        err_mean = np.mean(err)

        if verbose:
            print('%s => %f' % (', '.join(['%.4e' % i for i in np.array(x) * scales]), err_mean))

        return err if ls else err_mean

    best_x = real_ini_x/scales
    best_err = float('inf')
    init_noise = init_noise if iters > 1 else 0
    for i in range(iters):
        if refl_model:
            if hapke:
                ini_x = tuple(real_ini_x + init_noise*np.random.normal(0, 1, (len(scales),))*scales)
            else:
                ini_x = tuple(real_ini_x[1:-1]/real_ini_x[0] + init_noise*np.random.normal(0, 1, (len(scales)-1,))*scales[:-1]) + (real_ini_x[-1]*real_ini_x[0],)
        else:
            ini_x = tuple(real_ini_x + init_noise*np.random.normal(0, 1, (len(scales),))*scales)

        if verbose:
            print('\n\n\n==== i:%d ====\n'%i)
        if not use_ls:
            res = minimize(costfun, tuple(ini_x/scales), args=(refl_model, False, 0, verbose),
                            #method="BFGS", options={'maxiter': 10, 'eps': 1e-3, 'gtol': 1e-3})
                            method="Nelder-Mead", options={'maxiter': 100, 'xtol': 1e-3, 'ftol': 1e-3})
                            #method="COBYLA", options={'rhobeg': 1.0, 'maxiter': 200, 'disp': False, 'catol': 0.0002})
        else:
            res = least_squares(costfun, tuple(ini_x/scales), args=(refl_model, True, 0, verbose),
                                verbose=2, max_nfev=100, ftol=1e-3, xtol=1e-3, method='trf', # jac_sparsity=A,
                                x_scale='jac', jac='2-point', loss='linear' if 0 else 'huber', f_scale=10.0)  #huber_coef,
                                # tr_solver='lsmr',

        if not verbose:
            print('%s => %f' % (', '.join(['%.5e' % i for i in np.array(res.x)*scales]), res.fun))

        if res.fun < best_err:
            best_err = res.fun
            best_x = res.x

    if verbose:
        best_err = costfun(best_x, refl_model, False, debug=5, verbose=True)

    if refl_model:
        if hapke:
            xsc_ = best_x * scales
            xsc = np.abs(xsc_)
            if sppf_n == 1:
                xsc[2] = xsc_[2]
            x = tuple(xsc)
        else:
            x = (1,) + tuple(best_x * scales)
            if verbose:
                p = np.linspace(0, 160, 100)
                L = get_graph_L(20, p)
                plt.plot(p, L, p, Lfun(x[:-1], p))
                plt.show()
    else:
        xsc = best_x * scales
        x = [xsc[9*i:9*i+9] for i in range(len(imgs))]

    return x, best_err


def match_ll_with_hapke(img_n=20, iters=1, init_noise=0.0, verbose=True, hapke_params=None, ini_params=None):
    m_ll = RenderEngine.REFLMOD_LUNAR_LAMBERT
    m_hapke = RenderEngine.REFLMOD_HAPKE

    if hapke_params is not None:
        RenderEngine.REFLMOD_PARAMS[m_hapke] = hapke_params

    re = RenderEngine(512, 512)
    re.set_frustum(5, 5, 25*0.5, 1250)
    obj = ShapeModel(fname=os.path.join(DATA_DIR, 'test-ball.obj'))
    obj_idx = re.load_object(obj)
    pos = [0, 0, -70 * 0.8 * 2]

    ll_poly = 5
    real_ini_x = np.array(RenderEngine.REFLMOD_PARAMS[m_ll][:7]) if ini_params is None else ini_params[:7]
    real_ini_x = np.hstack((real_ini_x[0:ll_poly + 1], (real_ini_x[-1],)))
    scales = np.array((3e-03, 2e-05, 1e-06, 1e-08, 5e-11, 1))
    scales = np.hstack((scales[0:ll_poly], (scales[-1],)))

    def set_params(x):
        vals = [1] + list(np.array(x)[:-1] * scales[:-1]) + [0]*(5-ll_poly) + [x[-1]*scales[-1], 0, 0, 0]
        RenderEngine.REFLMOD_PARAMS[m_ll] = vals

    # debug 1: real vs synth, 2: err img, 3: both
    def costfun(x, debug=0, verbose=True):
        set_params(x)
        err = 0

        for phase_angle in np.radians(np.linspace(0, 150, img_n)):
            light = tools.q_times_v(tools.ypr_to_q(phase_angle, 0, 0), np.array([0, 0, -1]))
            synth1 = re.render(obj_idx, pos, np.quaternion(1,0,0,0), tools.normalize_v(light), get_depth=False, reflection=m_hapke)
            synth2 = re.render(obj_idx, pos, np.quaternion(1,0,0,0), tools.normalize_v(light), get_depth=False, reflection=m_ll)

            err_img = (synth1.astype('float') - synth2.astype('float'))**2
            err += np.mean(err_img)
            if debug:
                if debug%2:
                    cv2.imshow('hapke vs ll', np.concatenate((synth1.astype('uint8'), 255*np.ones((synth2.shape[0], 1), dtype='uint8'), synth2), axis=1))
                if debug>1:
                    err_img = err_img**0.2
                    cv2.imshow('err', err_img/np.max(err_img))
                cv2.waitKey()
        err /= img_n
        if verbose:
            print('%s => %f' % (', '.join(['%.4e' % i for i in np.array(x)*scales]), err))
        return err

    best_x = None
    best_err = float('inf')
    for i in range(iters):
        ini_x = tuple(real_ini_x[1:-1]/real_ini_x[0] + init_noise*np.random.normal(0, 1, (len(scales)-1,))*scales[:-1]) + (real_ini_x[-1]*real_ini_x[0],)

        if verbose:
            print('\n\n\n==== i:%d ====\n'%i)
        res = minimize(costfun, tuple(ini_x/scales), args=(0, verbose),
                        #method="BFGS", options={'maxiter': 10, 'eps': 1e-3, 'gtol': 1e-3})
                        method="Nelder-Mead", options={'maxiter': 120, 'xtol': 1e-4, 'ftol': 1e-4})
                        #method="COBYLA", options={'rhobeg': 1.0, 'maxiter': 200, 'disp': False, 'catol': 0.0002})
        if not verbose:
            print('%s => %f' % (', '.join(['%.5e' % i for i in np.array(res.x)*scales]), res.fun))

        if res.fun < best_err:
            best_err = res.fun
            best_x = res.x

    if verbose:
        costfun(best_x, 3, verbose=True)

    x = (1,) + tuple(best_x * scales)
    if verbose:
        p = np.linspace(0, 160, 100)
        L = get_graph_L(20, p)
        plt.plot(p, L, p, Lfun(x[:-1], p))
        plt.show()

    return x


def get_graph_L(theta, phase_angles):
    # phase angle in deg, corresponding L value

    Lpoints = False
    if theta == 20:
        # Based on Fig. 17, theta=20deg, w=0.1, g=-0.4:
        Lpoints =  np.array(
            [[ 0.00000000e+00,  1.00000000e+00],
            [ 1.74371749e+01,  8.74987858e-01],
            [ 3.17934829e+01,  7.79791583e-01],
            [ 4.02408174e+01,  7.11111917e-01],
            [ 5.01445890e+01,  6.01751490e-01],
            [ 6.47921843e+01,  4.37710852e-01],
            [ 7.71927052e+01,  3.08586481e-01],
            [ 9.81653974e+01,  1.08311967e-01],
            [ 1.05655646e+02,  4.77025908e-02],
            [ 1.14227816e+02, -1.70725062e-03],
            [ 1.26128988e+02, -4.65054961e-02],
            [ 1.37614036e+02, -5.57286530e-02],
            [ 1.47933931e+02, -5.44110791e-02],
            [ 1.60084780e+02, -3.39883521e-02]])
    elif theta == 10:
        # Based on Fig. 16, theta=10deg, w=0.1, g=-0.4:
        # extracted from an .svg-file with a line segment drawn in inkscape on top of the figure
        line_segs = np.array(((67.35192,18.817487), (16.570711,0.26727), (20.312479,5.87993), (49.97941,8.819895), (50.24667,6.949006), (23.78698,5.34539), (29.66692,12.828939), (44.36673,24.321527), (25.65789,17.105246), (44.09946,29.93419), (39.02135,35.27958), (24.85607,28.33057), (32.60688,49.17759), (60.13564,101.82968), (23.51974,40.89226), (31.27052,56.39384), (27.2615,42.22859), (22.18336,29.39965), (17.63979,17.6398), (21.38156,14.43254), (25.12333,11.22532), (23.78699,4.00904), (14.43257,0.26729)))
        px_width = 667.63928
        px_height = 520.64105
        x_range = 160
        y_range = 1.0
    elif theta == 0:
        # Based on Fig. 14, theta=0deg, w=0.1, g=-0.4:
        line_segs = np.array(((-169.33333,71.726197), (32.50595,4.535707), (52.916666,3.77976), (51.782739,0.377979), (54.050594,5.291666), (41.955356,4.157737), (55.184525,8.315476), (60.47619,10.205358), (50.27083,12.09524), (55.56249,16.25297), (39.30954,12.47322), (36.66369,16.63095), (35.90774,19.65476), (30.61607,18.14286)))
        px_width = 598.71429
        px_height = 464.91071
        x_range = 160
        y_range = 1.0
    else:
        assert False, 'invalid theta=%s'%theta

    if Lpoints is False:
        px_coords =np.cumsum(line_segs, axis=0)
        Lpoints = (px_coords - px_coords[0, :]) / np.array((px_width, px_height)) \
                  * np.array((x_range, -y_range)) + np.array((0, y_range))

    interp = interp1d(Lpoints[:, 0], Lpoints[:, 1])
    p = np.linspace(np.min(Lpoints[:, 0]), np.max(Lpoints[:, 0]), 100)
    L = interp(phase_angles)
    return L


def get_L_from_graph(theta=20):
    p = np.linspace(0, 180, 100)
    L = get_graph_L(theta, p)

    def costfun(x):
        return np.mean((L-Lfun(np.array(x)*np.array(scales), p))**2)

    scales = np.array((1, 5e-03, 2e-04, 3e-06, 1e-08, 1e-10))
    ini_x = np.array((1.0236e+00, -5.6993e-03, 1.8722e-04, -2.8399e-06, 1.0204e-08, 0))
    res = minimize(costfun, tuple(ini_x/scales), args=tuple(), method="BFGS",
                   options={'maxiter': 6000, 'eps': 1e-8, 'gtol': 1e-6})
    print('%s'%res)
    x = np.array(res.x)*scales
    plt.plot(p, L, p, Lfun(x, p))
    plt.show()

    return x


def estimate_hapke_k_polynomial():
    """
    cant seem to get a good fit
    """
    g = np.radians(np.hstack(([0, 2, 5], np.linspace(10, 180, 18))))
    th_p = np.radians(np.linspace(0, 60, 7))
    K = np.array([
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        [1.00, 0.997, 0.991, 0.984, 0.974, 0.961, 0.943],
        [1.00, 0.994, 0.981, 0.965, 0.944, 0.918, 0.881],
        [1.00, 0.991, 0.970, 0.943, 0.909, 0.866, 0.809],
        [1.00, 0.988, 0.957, 0.914, 0.861, 0.797, 0.715],
        [1.00, 0.986, 0.947, 0.892, 0.825, 0.744, 0.644],
        [1.00, 0.984, 0.938, 0.871, 0.789, 0.692, 0.577],
        [1.00, 0.982, 0.926, 0.846, 0.748, 0.635, 0.509],
        [1.00, 0.979, 0.911, 0.814, 0.698, 0.570, 0.438],
        [1.00, 0.974, 0.891, 0.772, 0.637, 0.499, 0.366],
        [1.00, 0.968, 0.864, 0.719, 0.566, 0.423, 0.296],
        [1.00, 0.959, 0.827, 0.654, 0.487, 0.346, 0.231],
        [1.00, 0.946, 0.777, 0.575, 0.403, 0.273, 0.175],
        [1.00, 0.926, 0.708, 0.484, 0.320, 0.208, 0.130],
        [1.00, 0.894, 0.617, 0.386, 0.243, 0.153, 0.094],
        [1.00, 0.840, 0.503, 0.290, 0.175, 0.107, 0.064],
        [1.00, 0.747, 0.374, 0.201, 0.117, 0.070, 0.041],
        [1.00, 0.590, 0.244, 0.123, 0.069, 0.040, 0.023],
        [1.00, 0.366, 0.127, 0.060, 0.032, 0.018, 0.010],
        [1.00, 0.128, 0.037, 0.016, 0.0085, 0.0047, 0.0026],
        [1.00, 0, 0, 0, 0, 0, 0]
    ]).T
    g_, thp_ = np.meshgrid(g, th_p)
    # interp = interp2d(g.flatten(), th_p.flatten(), K.flatten())
    # plt.plot(g_[i, :], interp(g_[i, :], thp_[i, :]))
    g0 = 0

    if False:
        def Kfun(x, g, th):
            return x[0] + x[1]*g + x[2]*th + x[4]*g**2 + x[3]*g*th + x[5]*th**2 \
                 + x[6]*g**3 + x[7]*g**2*th + x[8]*g*th**2 + x[9]*th**3 \
                 + x[10]*g**4 + x[11]*g**3*th + x[12]*g**2*th**2 + x[13]*g*th**3 + x[14]*th**4

        scales = np.array((1, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8))
        def costfun(x):
            return np.mean((K[:, g0:] - Kfun(np.array(x)*np.array(scales), g_[:, g0:], thp_[:, g0:]))**2)

        best_x = None
        best_err = float('inf')
        for i in range(100):
            ini_x = np.random.normal(0, .1, 15)
            ini_x[0] = .7
            res = minimize(costfun, tuple(ini_x/scales), args=tuple(), method="BFGS",
                           options={'maxiter': 8000, 'eps': 1e-9, 'gtol': 1e-6})
            print('%s => %f' % (', '.join(['%.4e' % i for i in np.array(res.x)*scales]), res.fun))
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x

        x = np.array(best_x)*scales
        print('best: %s => %f' % (', '.join(['%.4e' % i for i in x]), best_err))
    else:
        x = 0
        def Kfun(x, g, th_p):
            return np.exp(-0.32 * th_p * np.sqrt(np.tan(th_p) * np.tan(g / 2)) - 0.52 * th_p * np.tan(th_p) * np.tan(g / 2))

    for i in range(1, 7):
        plt.plot(np.degrees(g_[i, g0:]), K[i, g0:], '--')

    plt.gca().set_prop_cycle(None)
    for i in range(1, 7):
        plt.plot(np.degrees(g_[i, g0:]), Kfun(x, g_[i, g0:], thp_[i, g0:]))

    plt.show()


def to_sispo(sm, ab=None):
    # FROM:
    # rel_p = np.array([146226188132.0, -68812322442.0, -477863711.0]) \
    #         - np.array([146226194732.0, -68812326932.0, -477863381.0])
    # sm.spacecraft_q = tools.angleaxis_to_q([1.847799, -0.929873, 0.266931, -0.253146])
    # sm.spacecraft_pos = tools.q_times_v(SystemModel.sc2gl_q.conj() * sm.spacecraft_q.conj(), -rel_p * 0.001)
    # sm.asteroid.real_position = np.array([146226194732.0, -68812326932.0, -477863381.0])
    # sm.asteroid_q = tools.eul_to_q((math.radians(90),), 'z')

    # TO:
    ast_q0 = quaternion.one  #tools.eul_to_q((math.radians(90),), 'z')
    ast_qd = ast_q0.conj() * sm.asteroid_q
    ast_pos = tools.q_times_v(ast_qd.conj(), sm.asteroid.real_position)
    sc_q1 = ast_qd.conj() * sm.spacecraft_q
    sc_pos = ast_pos + tools.q_times_v(sc_q1 * SystemModel.sc2gl_q, -np.array(sm.spacecraft_pos) * 1000)

    print('spacecraft:')
    print('    "r": [%s],' % (', '.join(['%.1f' % v for v in sc_pos]),))
    print('    "angleaxis": [%s]' % (tools.q_to_angleaxis(sc_q1),))
    print('sssb:')
    print('    "att": {"RA": 0, "Dec": 90, "ZLRA": 0}')
    print('    "trj": {"r": [%s]}' % (', '.join(['%.1f' % v for v in ast_pos]),))

    if ab is not None:
        sm.asteroid_q = ast_q0
        sm.spacecraft_q = sc_q1
        sm.asteroid.real_position = ast_pos
        img = ab.render(shadows=True, reflection=2, gamma=1, textures=False)
        img = ImageProc.apply_point_spread_fn(img, **sm.cam.point_spread_fn)
        img = ImageProc.distort_image(img, sm.cam)

        cv2.imwrite('synth-img-1.exr', img.astype(np.float32), (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT))
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    if False:
        match_ll_with_hapke(hapke_params=DidymosPrimary.HAPKE_PARAMS, ini_params=DidymosPrimary.LUNAR_LAMBERT_PARAMS)
    elif False:
        estimate_hapke_k_polynomial()
    elif False:
        x = get_L_from_graph()
        # print(' '.join(['%+.4e%s'%(f, '*a'*i) for i, f in enumerate(x)]))
    elif False:
        print("== HAPKE ==")
        x = est_refl_model(hapke=True, iters=5, init_noise=0.3, verbose=False)
        print(', '.join(['%.5e' % f for i, f in enumerate(x)]))
        #print("== LL ==")
        #x = est_model(hapke=False, iters=5, init_noise=0.3, verbose=False)
        #print(', '.join(['%.5e'%f for i, f in enumerate(x)]))
    elif 0:
        x = est_rosetta_model(refl_model=False, hapke=True, use_ls=False)
        print('\n'.join(['[%s],' % ', '.join(['%.5e' % f for f in _x]) for _x in x]))
    elif 0:
        x = est_rosetta_model(refl_model=True, hapke=True, use_ls=False)
        print(', '.join(['%.5e' % f for i, f in enumerate(x)]))
    elif 0:
        x = est_itokawa_model(refl_model=True, hapke=True, use_ls=False, verbose=True,
                              iters=1, init_noise=0.003)
        print(', '.join(['%.5e' % f for f in x]))
    else:
        x = est_itokawa_model(refl_model=False, hapke=True, use_ls=False, verbose=True,
                              adjust_pose=False, iters=1, init_noise=0.003)
        print('\n'.join(['[%s],' % ', '.join(['%.5e' % f for f in _x]) for _x in x]))

