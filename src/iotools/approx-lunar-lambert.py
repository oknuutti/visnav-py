import math
import os

import cv2
import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from algo import tools
from algo.base import AlgorithmBase
from algo.image import ImageProc
from iotools import lblloader
from missions.rosetta import RosettaSystemModel
from render.render import RenderEngine


def Lfun(x, p):
    L = x[0]
    for i in range(len(x)-1):
        L += x[i+1]*p**(i+1)
    return L


def est_refl_model(hapke=True, iters=1, init_noise=0.0, verbose=True):
    sm = RosettaSystemModel()
    imgsize = (512, 512)
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

    target_exposure = np.min(list(imgs.values()))
    for img, exposure in imgs.items():
        real = cv2.imread(os.path.join(sm.asteroid.image_db_path, img + '_P.png'), cv2.IMREAD_GRAYSCALE)
        real = ImageProc.adjust_gamma(real, 1/1.8)
        #dark_px_lim = np.percentile(real, 0.1)
        #dark_px = np.mean(real[real<=dark_px_lim])
        real = cv2.resize(real, imgsize)
        # remove dark pixel intensity and normalize based on exposure
        #real = real - dark_px
        #real *= (target_exposure / exposure)
        imgs[img] = real

    re = RenderEngine(*imgsize, antialias_samples=0)
    obj_idx = re.load_object(sm.asteroid.hires_target_model_file, smooth=False)
    ab = AlgorithmBase(sm, re, obj_idx)

    model = RenderEngine.REFLMOD_HAPKE if hapke else RenderEngine.REFLMOD_LUNAR_LAMBERT
    defs = RenderEngine.REFLMOD_PARAMS[model]

    if hapke:
        # L, th, w, b (scattering anisotropy), c (scattering direction from forward to back), B0, hs
        #real_ini_x = [515, 16.42, 0.3057, 0.8746]
        real_ini_x = defs[:2] + defs[3:5]
        # scales = np.array((1e2, 1, 1e-2, 1e-1))
        scales = np.array((500, 20, 3e-1, 3e-1))
    else:
        real_ini_x = np.array((9.95120e-01, -6.64840e-03, 3.96267e-05, -2.16773e-06, 2.08297e-08, -5.48768e-11, 1))  # theta=20
        #real_ini_x = np.array((1.00218e+00, -1.36698e-03, -6.35077e-06, 4.12667e-07, -1.27338e-08, 5.73643e-11, 0.7))  # theta=10
        #real_ini_x = np.array(defs[:7])
        scales = np.array((3e-03, 2e-05, 1e-06, 1e-08, 5e-11, 1))
    #    scales = np.array((1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1))

    def set_params(x):
        if hapke:
            # optimize J, th, w, b, (c), B_SH0, hs
            xsc = list(np.array(x) * scales)
            vals = xsc[:2] + [defs[2]] + xsc[2:] + defs[len(xsc)+1:]
        else:
            vals = [1] + list(np.array(x) * scales) + [0, 0, 0]
        RenderEngine.REFLMOD_PARAMS[model] = vals

    # debug 1: real vs synth, 2: err img, 3: both
    def costfun(x, debug=0, verbose=True):
        set_params(x)
        err = 0
        for file, real in imgs.items():
            lblloader.load_image_meta(os.path.join(sm.asteroid.image_db_path, file + '.LBL'), sm)
            sm.swap_values_with_real_vals()
            synth2 = ab.render(shadows=True, reflection=model, gamma=1)
            err_img = (synth2.astype('float') - real)**2
            lim = np.percentile(err_img, 99)
            err_img[err_img > lim] = 0
            err += np.mean(err_img)
            if debug:
                if debug%2:
                    cv2.imshow('real vs synthetic', np.concatenate((real.astype('uint8'), 255*np.ones((real.shape[0], 1), dtype='uint8'), synth2), axis=1))
                if debug>1:
                    err_img = err_img**0.2
                    cv2.imshow('err', err_img/np.max(err_img))
                cv2.waitKey()
        err /= len(imgs)
        if verbose:
            print('%s => %f' % (', '.join(['%.4e' % i for i in np.array(x)*scales]), err))
        return err

    best_x = None
    best_err = float('inf')
    for i in range(iters):
        if hapke:
            ini_x = tuple(real_ini_x + init_noise*np.random.normal(0, 1, (len(scales),))*scales)
        else:
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

    if hapke:
        x = tuple(best_x * scales)
    else:
        x = (1,) + tuple(best_x * scales)
        if verbose:
            p = np.linspace(0, 180, 100)
            plt.plot(p, Lfun(real_ini_x[:-1]/real_ini_x[0], p), p, Lfun(x[:-1], p))
            plt.show()

    return x


def get_L_from_graph():
    # phase angle in deg, corresponding L value

    Lpoints = False
    if False:
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
    elif True:
        # Based on Fig. 16, theta=10deg, w=0.1, g=-0.4:
        # extracted from an .svg-file with a line segment drawn in inkscape on top of the figure
        line_segs = np.array(((67.35192,18.817487), (16.570711,0.26727), (20.312479,5.87993), (49.97941,8.819895), (50.24667,6.949006), (23.78698,5.34539), (29.66692,12.828939), (44.36673,24.321527), (25.65789,17.105246), (44.09946,29.93419), (39.02135,35.27958), (24.85607,28.33057), (32.60688,49.17759), (60.13564,101.82968), (23.51974,40.89226), (31.27052,56.39384), (27.2615,42.22859), (22.18336,29.39965), (17.63979,17.6398), (21.38156,14.43254), (25.12333,11.22532), (23.78699,4.00904), (14.43257,0.26729)))
        px_width = 667.63928
        px_height = 520.64105
        x_range = 160
        y_range = 1.0
    else:
        # Based on Fig. 14, theta=0deg, w=0.1, g=-0.4:
        line_segs = np.array(((-169.33333,71.726197), (32.50595,4.535707), (52.916666,3.77976), (51.782739,0.377979), (54.050594,5.291666), (41.955356,4.157737), (55.184525,8.315476), (60.47619,10.205358), (50.27083,12.09524), (55.56249,16.25297), (39.30954,12.47322), (36.66369,16.63095), (35.90774,19.65476), (30.61607,18.14286)))
        px_width = 598.71429
        px_height = 464.91071
        x_range = 160
        y_range = 1.0

    if Lpoints is False:
        px_coords =np.cumsum(line_segs, axis=0)
        Lpoints = (px_coords - px_coords[0, :]) / np.array((px_width, px_height)) \
                  * np.array((x_range, -y_range)) + np.array((0, y_range))

    interp = interp1d(Lpoints[:, 0], Lpoints[:, 1])
    p = np.linspace(np.min(Lpoints[:, 0]), np.max(Lpoints[:, 0]), 100)
    L = interp(p)

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


if __name__ == '__main__':
    if False:
        x = get_L_from_graph()
        # print(' '.join(['%+.4e%s'%(f, '*a'*i) for i, f in enumerate(x)]))
    elif False:
        print("== HAPKE ==")
        x = est_refl_model(hapke=True, iters=5, init_noise=0.3, verbose=False)
        print(', '.join(['%.5e' % f for i, f in enumerate(x)]))
        #print("== LL ==")
        #x = est_refl_model(hapke=False, iters=5, init_noise=0.3, verbose=False)
        #print(', '.join(['%.5e'%f for i, f in enumerate(x)]))
    else:
        x = est_refl_model(hapke=True)
        print(', '.join(['%.5e'%f for i, f in enumerate(x)]))
