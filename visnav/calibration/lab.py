
import pickle
from functools import lru_cache

import math
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

from visnav.algo.image import ImageProc
from visnav.algo.model import Camera
from visnav.calibration.base import Frame, NANOCAM_GAINS

GAMMA_OVERRIDE = False
GAMMA_BREAK_OVERRIDE = False
IGNORE_INVERT_COLOR_CORRECTION = False
IGNORE_SUBTRACT_BACKGROUND = False
SHOW_LAB_MEASURES = False
USE_CACHED_MEAS = False
WRITE_CACHE = False


def use_lab_bg_imgs(folder, get_bgr_cam):
    debug = 1
    override = {
        'gamma_break': 0.1,
        'bg_image': '../lab_bg_img2.png',
        #        'gamma': 1.0,
        #        'ccm_bgr_red': [],
        'gain': NANOCAM_GAINS[2048],  # x128,
    }
    bgr_cam = get_bgr_cam()
    mean_sum = {}
    std_sum = {}
    n_sum = {}
    n1_sum = {}
    for i, file in enumerate(os.listdir(folder)):
        if file[-4:].lower() in ('.png', '.jpg'):
            fullpath = os.path.join(folder, file)
            f = LabFrame.from_file(bgr_cam, fullpath, fullpath[:-4]+'.lbl', override=override, debug=debug)

            if f.exposure not in mean_sum:
                mean_sum[f.exposure] = np.zeros(3)
                std_sum[f.exposure] = np.zeros(3)
                n_sum[f.exposure] = 0
                n1_sum[f.exposure] = 0
            n = np.prod(f.image.shape[:2])
            mean_sum[f.exposure] += n * np.mean(f.image.reshape((-1, 3)), axis=0)
            std_sum[f.exposure] += (n-1) * np.std(f.image.reshape((-1, 3)), axis=0)
            n_sum[f.exposure] += n
            n1_sum[f.exposure] += n-1

    mean = {e: mean_sum[e] / n_sum[e] for e in mean_sum.keys()}
    std = {e: std_sum[e] / n1_sum[e] for e in std_sum.keys()}

    print('exposure [ms]\tred mean\tred sd\tgreen mean\tgreen sd\tblue mean\tblue sd')
    for e in mean.keys():
        print('%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f' % (e*1e3, mean[e][2], std[e][2], mean[e][1],
                                                          std[e][1], mean[e][0], std[e][0]))
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # for e in mean.keys():
    #     ax1.plot(e, mean, 'x')
    #     ax2.plot(e, std/mean, 'x')
    # ax1.set_title('dark current mean')
    # ax2.set_title('dark current std/mean')
    # fig.show()


def use_lab_imgs(folder, get_bgr_cam):
    debug = 0
    override = {
        'gain': 1,
#        'impulse_psfd': 1.657e-18,
        'impulse_psfd': 1565,
        'impulse_spread': 50,
#        'impulse_size': (121, 135),
    }
    if GAMMA_OVERRIDE:
        override['gamma'] = GAMMA_OVERRIDE
    if GAMMA_BREAK_OVERRIDE:
        override['gamma_break'] = GAMMA_BREAK_OVERRIDE
    if IGNORE_INVERT_COLOR_CORRECTION:
        override['ccm_bgr_red'] = []

    if 0:
        override.update({
            'gamma_break': 0,
            'bg_image': '../lab_bg_img_gb00.png',
            'impulse_psfd': 1.446e-18,
        })
    if 0:
        override.update({
            'gamma_break': 0.15,
            'bg_image': '../lab_bg_img_gb15.png',
            'impulse_psfd': 1.862e-18,
        })
    if IGNORE_SUBTRACT_BACKGROUND:
        override['bg_image'] = None

    bgr_cam = get_bgr_cam()

    def gaussian(mean, var, x):
        return np.exp(-(x - mean) ** 2 / (2 * var)) / (2 * math.pi * var) ** .5

    @lru_cache(maxsize=100)
    def get_kernel(peak, fwhm, psfd, size):
        var = (fwhm/np.sqrt(8*np.log(2)))**2
        amp = psfd / gaussian(0, var, 0)
        spectrum_fn = lambda lam: amp * gaussian(peak, var, lam)
        kernel = LabFrame.calc_source_kernel(bgr_cam, spectrum_fn, size, points=(peak,))
        if IGNORE_INVERT_COLOR_CORRECTION:
            kernel = ImageProc.color_correct(kernel, np.array([
                [2.083400, -0.524300, -0.389100],
                [-0.516800, 2.448100, -0.761300],
                [-0.660600, 0.149600, 1.680900],
            ]))

        return kernel

    cache_file = ('bg_' if IGNORE_SUBTRACT_BACKGROUND else '') + ('cc_' if IGNORE_INVERT_COLOR_CORRECTION else '') \
                 + 'lab_meas_cache.pickle'

    if not USE_CACHED_MEAS or not os.path.exists(cache_file):
        off_ccoef = {}
        mean_sum = {}
        var_sum = {}
        n0_sum = {}
        n_sum = {}
        n1_sum = {}
        lines = {}
        for i, file in enumerate(os.listdir(folder)):
            if file[-4:].lower() in ('.png', '.jpg'):
                fullpath = os.path.join(folder, file)
                f = LabFrame.from_file(bgr_cam, fullpath, fullpath[:-4]+'.lbl', override=override, bg_offset=False, debug=debug)
                kernel = get_kernel(f.impulse_peak, f.impulse_fwhm, f.impulse_psfd, f.impulse_size)
                m = f.detect_source(kernel)

                key = (f.exposure, f.impulse_peak, f.impulse_size)
                if key not in mean_sum:
                    off_ccoef[key] = 0
                    mean_sum[key] = np.zeros(3)
                    var_sum[key] = np.zeros(3)
                    n0_sum[key] = 0
                    n_sum[key] = 0
                    n1_sum[key] = 0
                off_ccoef[key] += off_center_coef(m.center, f.cam[0])
                mean_sum[key] += m.n * m.mean
                var_sum[key] += (m.n-1) * m.std**2
                n0_sum[key] += 1
                n_sum[key] += m.n
                n1_sum[key] += m.n-1

                k2 = (f.impulse_peak, f.impulse_size)
                if k2 not in lines:
                    lines[k2] = {}
                if f.exposure not in lines[k2]:
                    lines[k2][f.exposure] = []
                lines[k2][f.exposure].append(m.mean)

        off_ccoef = {e: off_ccoef[e] / n0_sum[e] for e in off_ccoef.keys()}
        mean = {e: mean_sum[e] / n_sum[e] for e in mean_sum.keys()}
        px_std = {e: np.sqrt(var_sum[e] / n1_sum[e]) for e in var_sum.keys()}
        mean_std = {e: px_std[e] / math.sqrt(n_sum[e]) for e in px_std.keys()}

        if WRITE_CACHE:
            with open(cache_file, 'wb') as fh:
                pickle.dump((mean, px_std, mean_std, lines, off_ccoef), fh)
    else:
        with open(cache_file, 'rb') as fh:
            mean, px_std, mean_std, lines, off_ccoef = pickle.load(fh)

    # snr_max = 20*log10(sqrt(sat_e)) dB
    # dynamic range = 20*log10(sat_e/readout_noise))
    # dark_noise_sd should be sqrt(dark_noise_mu)

    print('\tred mean\tred sd\tgreen mean\tgreen sd\tblue mean\tblue sd')
    size_map = {(121, 135): 'center', (135, 135): 'side'}
    for i, key in enumerate(mean.keys()):
        exp, lam, size = key
        lab = '%dms, %.1fnm, %s' % (exp*1e3, lam*1e9, size_map[size])
        print('%s\t%.3f\t%.6f\t%.3f\t%.6f\t%.3f\t%.6f' % (lab, mean[key][2], mean_std[key][2], mean[key][1],
                                                          mean_std[key][1], mean[key][0], mean_std[key][0]))

    occ, occ_n = {}, {}
    for key, c in off_ccoef.items():
        exp, lam, size = key
        cls = size_map[size]
        if cls not in occ:
            occ[cls] = 0
            occ_n[cls] = 0
        occ[cls] += c
        occ_n[cls] += 1

    center_coef = occ['center'] / occ_n['center']
    side_coef = occ['side'] / occ_n['side']
    print("\nexpected side vs center: %.2f%%" % (100 * side_coef / center_coef,))
    print("average side vs center:")

    total_val, total_n = {}, {}
    for key, val in mean.items():
        exp, lam, size = key
        cls = size_map[size]
        if (exp, lam) not in total_val:
            total_val[(exp, lam)] = {}
            total_n[(exp, lam)] = {}
        if cls not in total_val[(exp, lam)]:
            total_val[(exp, lam)][cls] = 0
            total_n[(exp, lam)][cls] = 0
        total_val[(exp, lam)][cls] += val
        total_n[(exp, lam)][cls] += 1
    for key, tot in total_val.items():
        exp, lam = key
        ch = 1 if lam == 5.577e-7 else 2
        lab = '%dms, %.1fnm, %s' % (exp * 1e3, lam * 1e9, {1: 'G', 2: 'R'}[ch])
        mean_side_center_ratio = 100 * (total_val[(exp, lam)]['side'] / total_n[(exp, lam)]['side']) \
                                     / (total_val[(exp, lam)]['center'] / total_n[(exp, lam)]['center'])
        print("%s: %.2f%%" % (lab, mean_side_center_ratio[ch]))

    # plot all measurements
    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()
    for i, (lam, size) in enumerate(lines.keys()):
        xs = sorted(list(lines[(lam, size)].keys()))
        for j, c in enumerate(('b', 'g', 'r')):
            n = len(lines[(lam, size)][xs[0]])
            ys = [[lines[(lam, size)][x][k][j] for x in xs] for k in range(n)]
            for y in ys:
                y.insert(0, (xs[0]*y[1] - xs[1]*y[0]) / (xs[0] - xs[1]))
                axs[i].plot(np.array([0] + xs) * 1e3, y, c + 'x-')
        axs[i].set_title('%.1fnm, %s' % (lam * 1e9, size_map[size]))
    plt.show()

#        for j, c in enumerate(('b', 'g', 'r')):
#            ax1.plot(i*3+j, mean[key][j], 'x'+c, label=lab+(' [%s]'%c))
#            ax2.plot(i*3+j, (std[key]/mean[key])[j], 'x'+c, label=lab+(' [%s]'%c))
#    ax1.set_title('mean')
#    ax1.legend()
#    ax2.set_title('std/mean')
#    ax2.legend()
#    fig.show()


def off_center_coef(center, cam):
    cam_mx = cam.intrinsic_camera_mx()
    pp, fl = cam_mx[0:2, 2], (cam_mx[0, 0] + cam_mx[1, 1]) / 2
    off_angle = np.arctan(np.linalg.norm(center - pp) / fl)
    if 1:
        solid_angle = cam.pixel_solid_angle(*center)
        return solid_angle * np.cos(off_angle)
    else:
        return np.cos(off_angle)**4


class LabMeasure:
    def __init__(self, frame, mean, std, n, center=None):
        self.frame = frame
        self.mean = mean
        self.std = std
        self.n = n
        self.center = center


class LabFrame(Frame):
    def __init__(self, *args, **kwargs):
        super(LabFrame, self).__init__(*args, **kwargs)
        # about calibration images
        self.impulse_psfd = None
        self.impulse_peak = None
        self.impulse_fwhm = None
        self.impulse_size = None
        self.impulse_spread = None

    @classmethod
    def process_metadata(cls, frame, meta):
        if meta.get('impulse_peak', False):
            frame.impulse_size = meta.get('impulse_size')
            frame.impulse_spread = meta.get('impulse_spread')
            frame.impulse_psfd = meta.get('impulse_psfd')
            frame.impulse_peak = meta.get('impulse_peak')
            frame.impulse_fwhm = meta.get('impulse_fwhm')

    @staticmethod
    def calc_source_kernel(cams, spectrum_fn, patch_size, points=None):
        # detect source
        kernel = ImageProc.bsphkern(tuple(map(int, patch_size)) if '__iter__' in dir(patch_size) else int(patch_size))

        expected_bgr = np.zeros(3)
        for i, cam in enumerate(cams):
            ef, _ = Camera.electron_flux_in_sensed_spectrum_fn(cam.qeff_coefs, spectrum_fn, cam.lambda_min,
                                                               cam.lambda_max, fast=False, points=points)
            expected_bgr[i] = cam.gain * cam.aperture_area * cam.emp_coef * ef

        kernel = np.repeat(np.expand_dims(kernel, axis=2), 3, axis=2) * expected_bgr
        return kernel

    def detect_source(self, kernel, total_radiation=False):
        assert kernel.shape[0] % 2 and kernel.shape[1] % 2, 'kernel width and height must be odd numbers'
        kernel = self.gain * self.exposure * kernel
        fkernel = ImageProc.fuzzy_kernel(kernel, self.impulse_spread)
        method = [cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF][0]
        corr = cv2.matchTemplate(self.image.astype(np.float32), fkernel.astype(np.float32), method)
        _, _, minloc, maxloc = cv2.minMaxLoc(corr)   # minval, maxval, minloc, maxloc
        loc = minloc if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED) else maxloc
        loc_i = tuple(np.round(np.array(loc)).astype(np.int))
        center = np.array(loc) + np.flip(fkernel.shape[:2]) / 2

        if SHOW_LAB_MEASURES:
            sc = 1024/self.image.shape[1]
            img = cv2.resize(self.image.astype(np.float), None, fx=sc, fy=sc) / np.max(self.image)
            img = cv2.circle(img, tuple(np.round(center*sc).astype(np.int)),
                             round((kernel.shape[0]-self.impulse_spread)/2*sc), [0, 0, 1.0])
            cv2.imshow('te', img)
            print('waiting...', end='', flush=True)
            cv2.waitKey()
            print('done')

        if True:
            # use result as a mask to calculate mean and variance
            kh, kw = np.array(fkernel.shape[:2]) // 2
            win = self.image[loc_i[1]:loc_i[1]+kh*2+1, loc_i[0]:loc_i[0]+kw*2+1, :].reshape((-1, 3))
            kernel_max = np.max(np.sum(kernel, axis=2))
            mask = np.sum(fkernel, axis=2)/kernel_max > 0.95
            mean = np.median(win[mask.flatten(), :], axis=0)
            std = np.std(win[mask.flatten(), :], axis=0)
            n = np.sum(mask)

            tmp = np.zeros(self.image.shape[:2])
            tmp[loc_i[1]:loc_i[1]+kh*2+1, loc_i[0]:loc_i[0]+kw*2+1] = mask
            img_m = tmp

        else:
            # calculate a correlation channel (over whole image)
            k = kernel.shape[0]//2
            corr = cv2.matchTemplate(self.image.astype(np.float32), kernel[k:k+1, k:k+1, :].astype(np.float32), method)

            # calculate mean & variance of kernel area using corr channel
            win = corr[loc_i[1]-k:loc_i[1]+k+1, loc_i[0]-k:loc_i[0]+k+1]
            corr_mean = np.mean(win)
            corr_std = np.std(win)

            # threshold using mean - sd
            _, mask = cv2.threshold(corr, corr_mean - corr_std, 1, cv2.THRESH_BINARY)

            # dilate & erode to remove inner spots
            krn1 = ImageProc.bsphkern(round(1.5*corr.shape[0]/512)*2 + 1)
            krn2 = ImageProc.bsphkern(round(2*corr.shape[0]/512)*2 + 1)
            mask = cv2.dilate(mask, krn1, iterations=1)   # remove holes
            mask = cv2.erode(mask, krn2, iterations=1)    # same size
            mask = mask.astype(np.bool)

            # use result as a mask to calculate mean and variance
            mean = np.mean(self.image.reshape((-1, 3))[mask.flatten()], axis=0)
            var = np.var(self.image.reshape((-1, 3))[mask.flatten()], axis=0)
            n = np.sum(mask)

        if self.debug:
            sc = 1024/self.image.shape[1]
            img_m = np.repeat(np.atleast_3d(img_m.astype(np.uint8)*127), 3, axis=2)
            merged = ImageProc.merge((self.image.astype(np.float32)/np.max(self.image), img_m.astype(np.float32)/255))
            img = cv2.resize(merged, None, fx=sc, fy=sc)
            cv2.imshow('te', img)
            arr_n, lims, _ = plt.hist(self.image[:, :, 1].flatten(), bins=np.max(self.image)+1, log=True, histtype='step')
            plt.hist(win[mask.flatten(), 1].flatten(), bins=np.max(self.image)+1, log=True, histtype='step')
            x = np.linspace(0, np.max(win), np.max(win)+1)
            i = list(np.logical_and(lims[1:] > mean[1], arr_n > 0)).index(True)
            plt.plot(x, arr_n[i]*np.exp(-((x-mean[1])/std[1])**2))
            plt.ylim(1e-1, 1e6)
            plt.figure()
            plt.imshow(self.image[loc_i[1]:loc_i[1] + kh * 2 + 1, loc_i[0]:loc_i[0] + kw * 2 + 1, :] / np.max(self.image))
            print('waiting (%.1f, %.1f)...' % (mean[1], std[1]), end='', flush=True)
            cv2.waitKey(1)
            plt.show()
            print('done')

        return LabMeasure(self, mean, std, n, center)
