import configparser
import pickle
import os
import glob
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq, fmin_bfgs, minimize
import cv2

from visnav.algo import tools
from visnav.algo.image import ImageProc
from visnav.algo.model import Camera
from visnav.calibration.aurora import analyze_aurora_img
from visnav.calibration.base import merge, Frame, plot_bgr_qeff, RAW_IMG_MAX_VALUE
from visnav.calibration.lab import use_lab_imgs, use_lab_bg_imgs
from visnav.calibration.moon import use_moon_img, MoonFrame
from visnav.calibration.stars import StarFrame, StarMeasure
from visnav.render.stars import Stars


DEBUG_MEASURES = 0
PLOT_INITIAL_QEFF_FN = 0
SHOW_MEASURES = 1
SHOW_STAR_KERNELS = 1

# for moon
STAR_GAIN_ADJUSTMENT = 1.0
MOON_GAIN_ADJ = (2400 / 2000) * 1.0

STAR_GAIN_ADJUSTMENT_TN = 0.7           #
STAR_PSF_COEF_TN = (0.22, 0.15, 0.12)   #

STAR_LAB_DATAPOINT_WEIGHT = 1.0
STAR_MOON_WEIGHT = 1.0
IGNORE_MEASURES_INIT = (('moon', 2), )        # (obj_id, cam channel)
IGNORE_MEASURES_FINETUNE = IGNORE_MEASURES_INIT + ((27919, 0), (37173, 0), (26246, 0))  # alf ori, alf cmi, eps ori
IGNORE_MEASURES = IGNORE_MEASURES_INIT
STAR_CALIB_PRIOR_WEIGHT = 0.2    # 0.5: high, 0.3: med, 0.1: low          # 1st: 0.30, then 0.10;   old: first, 0.1, then 0.0001
STAR_CALIB_HUBER_COEF = np.log10(1 + 0.2)       # %: err_dus/measured_du  # 1st: 0.30, then 0.15;   old: first: 0.25, then 0.10
STAR_PSF_SDS = (1.8768,)  # 1.8, 1.7)   # (1.8, 1.8, 1.8)    # bgr                         # best: (180, 100, 65), (150, 95, 80)
INIT_QEFF_ADJ = (1.0, 1.0, 1.0)  # bgr                         # best: (1.0, 0.95, 0.8),

FIXED_PSF_SDS = 0
USE_ESTIMATED_QEC = 1     # use estimated qec instead of initial one
STAR_OPT_WEIGHTING = 1
STAR_USE_CACHED_MEAS = 1
OPTIMIZER_START_N = 0     # set to zero for skipping optimizing and just show result for the initial solution

(FRAME_GAIN_NONE, FRAME_GAIN_SAME, FRAME_GAIN_STATIC, FRAME_GAIN_INDIVIDUAL) = range(4)
FRAME_GAINS = FRAME_GAIN_STATIC   # 0: dont use, 1: same gain for all, 2: static gains, 3: individual gains
GENERAL_GAIN_ADJUSTMENT = False   # 1.0       #

STAR_IGNORE_IDS = (
        32263,  # Sirius      # brightest star, very saturated
        24378,  # Rigel       # saturated
#        37173,  # Procyon     # saturated
#        27919,  # Betelgeuse  # saturated, highly variable, was experiencing severe dimming

        26142,  # Meissa      # the neck/head, bright open cluster too near (too dim to detect with current threashold)
#        25865,  # Mintaka    # eastern star of belt, very blue, could try still (!)
        26176,  # Hatysa A    # the sword, orion nebula and lesser stars too near
        26132,  # Hatysa B
        26134,  # Hatysa C
    ) if 1 else tuple()

OVERRIDE_STAR_DATA = {
    # Betelgeuse: CTOA observation on 2019-01-24, accessed through https://www.aavso.org database
#    27919: {'mag_v': 0.708, 'mag_b': 2.532, 't_eff': },  # estimated based on b-v: K

    # Betelgeuse: CTOA observation on 2019-01-24, accessed through https://www.aavso.org database
    #  - other details from http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2000ApJ...537..205R
    27919: {'mag_v': 0.708, 't_eff': 3540, 'log_g': 0.0, 'fe_h': 0.05},  # t_eff from pastel: 3626

#     24378: {'mag_v': 0.19, }, #0.19, },
#     37173: {'mag_v': 0.37, }, #0.37, },
#     25273: {'mag_v': 1.61, },
#     26246: {'mag_v': 1.69, },
#     26662: {'mag_v': 1.75, },
#     30251: {'mag_v': 1.88, },
#     27298: {'mag_v': 2.05, },
#     25865: {'mag_v': 2.20, },
#     23820: {'mag_v': 2.76, },
#     36087: {'mag_v': 2.88, },
}


def calibrate(folder, thumbnail=True):
    debug = 1
    bgr_cam = get_bgr_cam(thumbnail=thumbnail, estimated=USE_ESTIMATED_QEC)
    cache_file = '../%s_meas_cache.pickle' % ('th' if thumbnail else ('sm' if len(sys.argv) > 3 else 's'))

    if not STAR_USE_CACHED_MEAS or not os.path.exists(cache_file):
        s_frames = []
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                if file[-4:].lower() in ('.png', '.jpg'):
                    fullpath = os.path.join(folder, file)
                    f = StarFrame.from_file(bgr_cam, fullpath, fullpath[:-4]+'.lbl', bg_offset=21, debug=debug)
                    f.override_star_data = OVERRIDE_STAR_DATA
                    if 0:
                        f.show_image()
                    s_frames.append(f)
        else:
            f = StarFrame.from_file(bgr_cam, folder, folder[:-4] + '.lbl', bg_offset=21, debug=debug)
            f.override_star_data = OVERRIDE_STAR_DATA
            s_frames.append(f)

        if 0 and not thumbnail:
            f.show_image(processed=False, median_filter=False, zero_bg=30, gain=1, save_as='c:/projects/s100imgs/processed-stars-mf.png')

        measures = []
        stars = {}
        for f in s_frames:
            m_tmp, m_str = f.detect_stars(thumbnail=thumbnail)
            measures.extend(m_tmp)
            merge(stars, m_str)

        if len(sys.argv) > 3:
            Frame.MISSING_BG_REMOVE_STRIPES = False
            m_folder = sys.argv[3]
            m_frames = []
            if os.path.isdir(m_folder):
                for file in os.listdir(m_folder):
                    if file[-4:].lower() in ('.png', '.jpg'):
                        fullpath = os.path.join(m_folder, file)
                        m_frames.append(MoonFrame.from_file(bgr_cam, fullpath, fullpath[:-4] + '.lbl', debug=debug))
            else:
                m_frames.append(MoonFrame.from_file(bgr_cam, m_folder, m_folder[:-4] + '.lbl', debug=debug))

            for f in m_frames:
                f.detect_moon()
                measures.extend(f.measures)

        with open(cache_file, 'wb') as fh:
            pickle.dump((measures, stars), fh)
    else:
        with open(cache_file, 'rb') as fh:
            measures, stars = pickle.load(fh)
        for m in measures:
            m.frame.cam = bgr_cam

    # maybe filter out measures of certain stars
    measures = [m for m in measures if m.obj_id[0] not in STAR_IGNORE_IDS]

    # override star params again in case they were changed after caching
    for m in measures:
        if m.obj_id[0] in OVERRIDE_STAR_DATA:
            for f in ('mag_v', 'mag_b', 't_eff', 'log_g', 'fe_h'):
                od = OVERRIDE_STAR_DATA[m.obj_id[0]]
                if f in od:
                    stars[m.obj_id][0][f] = od[f]
                    if getattr(m, f, None) is not None:
                        setattr(m, f, od[f])

    if STAR_OPT_WEIGHTING:
        # set different weights to measures so that various star temperatures equally represented
        star_meas = [m for m in measures if m.obj_id[0] != 'moon']
        temps = np.unique([m.t_eff for m in star_meas])

        if 1:
            b = 2.897771955e-3
            len_sc = 2 * (100e-9 / b) ** 2  # 100 nm
            summed_weights = {temp: np.sum([np.exp(-(1/temp - 1/m.t_eff) ** 2 / len_sc) for m in star_meas]) for temp in temps}
        else:
            len_sc = 2 * np.log10(1.33)**2  # np.log(1.5)**2
            summed_weights = {temp: np.sum([np.exp(-(np.log10(temp) - np.log10(m.t_eff))**2/len_sc) for m in star_meas]) for temp in temps}

        # also weight by magnitude
#        mim = np.max([m.mag_v for m in star_meas]) + 1

        for m in star_meas:
            m.weight = 1/summed_weights[m.t_eff]  #* (mim - m.mag_v)   # weight by magnitude

        # give the moon the same weight as the median star
        # moon_weight = np.median([m.weight for m in star_meas])
        moon_weight = STAR_MOON_WEIGHT
        for m in measures:
            if m.obj_id[0] == 'moon':
                m.weight = moon_weight

    opt = Optimizer({'method': 'leastsq'})
    qeff_coefs, f_gains, gain_adj, psf_sd, err, measured, expected = opt.optimize(measures)

    for i, qec in enumerate(qeff_coefs):
        bgr_cam[i].qeff_coefs = qec

    print('err: %.3f' % np.mean(err))
    target = np.array((24.75, 22.50))
    print('r/g, b/g @ 557.7nm: (%.2f%%, %.2f%%), target = (%.2f%%, %.2f%%)' % (
        *(np.sqrt(err[0:2])/STAR_LAB_DATAPOINT_WEIGHT*100 + target), *target))
    print('queff_coefs: %s' % (qeff_coefs,))
    print('frame gains: %s' % (f_gains,))
    print('gain_adj: %s' % (gain_adj,))
    if len(psf_sd) == 3:
        print('psf SDs [px]: %.3f, %.3f, %.3f' % (*psf_sd,))
    else:
        print('psf SD [px]: %s' % (psf_sd,))

    ## star measurement table
    ##
    sort = 'mag_v'  # 't_eff'
    s_by = np.array([(id, st[0][sort]) for id, st in stars.items()])
    idxs = np.argsort(s_by[:, 1])

    # NOTE: Overwrites multiple measurement expected dus,
    # i.e. wrong results if different exposure time across different measurements of the same star
    star_exp_dus = {}
    star_px_du_sat = {}
    for m in measures:
        if m.obj_id not in star_exp_dus:
            star_exp_dus[m.obj_id] = [None]*3
            if isinstance(m, StarMeasure):
                star_px_du_sat[m.obj_id] = [None, [None]*3, [None]*3, [None]*3]
        star_exp_dus[m.obj_id][m.cam_i] = m.c_expected_du
        if isinstance(m, StarMeasure) and StarFrame.STAR_SATURATION_MODELING == StarFrame.STAR_SATURATION_MODEL_MOTION:
            star_px_du_sat[m.obj_id][0] = m
            star_px_du_sat[m.obj_id][1][m.cam_i] = int(m.du_count)
            star_px_du_sat[m.obj_id][2][m.cam_i] = int(m.c_unsat_du)
            star_px_du_sat[m.obj_id][3][m.cam_i] = m.c_px_du_sat

    if SHOW_STAR_KERNELS and StarFrame.STAR_SATURATION_MODELING == StarFrame.STAR_SATURATION_MODEL_MOTION:
        star_order = [(id, np.mean(d[2])) for id, d in star_px_du_sat.items()]
        star_order = [id for id, _ in sorted(star_order, key=lambda x: -x[1])]
        full = False
        names = {
            'alp_cmi': 'Procyon',
            'alp_ori': 'Betelgeuse',
            'gam_ori': 'Bellatrix',
            'eps_ori': 'Alnilam',
            'zet_ori': 'Alnitak',
            'bet_cma': 'Mirzam',
            'kap_ori': 'Saiph',
            'del_ori': 'Mintaka',
            'bet_eri': 'Cursa',
            'bet_cmi': 'Gomeisa',
        }
        fig, axs = plt.subplots(4 if full else 2, 3, figsize=(9, 5))
        axs = axs.flatten()
        s = np.max([c.shape for t in star_px_du_sat.values() for c in t[3]])

        for i, obj_id in enumerate(star_order):
            if not full and i >= 6:
                break
            m, meas_du, unsat_du, px_du_sat = star_px_du_sat[obj_id]
            px_du_sat = np.flip(pad_and_stack(px_du_sat, shape=(s, s)), axis=2)
            # sat_du = tuple(np.sum(np.flip(px_du_sat,axis=2), axis=(0, 1)).astype(np.int))
            (x, y) = map(int, m.ixy)
            win = np.flip(m.frame.image[y-s//2:y-s//2+s, x-s//2:x-s//2+s], axis=2)
            max_val = RAW_IMG_MAX_VALUE  #max(np.max(win), RAW_IMG_MAX_VALUE)
            img = np.concatenate((px_du_sat, win), axis=1) / max_val

            axs[i].imshow(img)
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
            axs[i].set_title(names[m.bayer], fontsize=16)

        for j in range(i+1, 12 if full else 6):
            axs[j].axis('off')
        fig.tight_layout()
        plt.show()

    tot_std = 0
    tot_std_n = 0
    print('HIP\tVhat\tVmag\tTeff\tlog(g)\tFe/H\tModel Red\tModel Green\tModel Blue\tSamples\tRed\tGreen\tBlue\tRed SD\tGreen SD\tBlue SD')
    for id in s_by[idxs, 0]:
        st = stars[id]
        s = {'meas': np.array([s['meas'] for s in st]), 'm_mag_v': st[0]['m_mag_v'], 'mag_v': st[0]['mag_v'],
             't_eff': st[0]['t_eff'], 'log_g': st[0]['log_g'], 'fe_h': st[0]['fe_h'], 'expected': star_exp_dus[id] if id in star_exp_dus else (0, 0, 0)}
        stars[id] = s
        tyc = '&'.join([Stars.get_catalog_id(i) for i in id])
        modeled = np.flip(s['expected'])
        means = np.flip(np.mean(s['meas'], axis=0))
        std = np.flip(np.std(s['meas'], axis=0))
        n = len(s['meas'])
        tot_std += np.sum(std*(n-1))
        tot_std_n += 3*(n-1)
        #both = np.vstack((means, std)).T.flatten()
#        print('%s\t%.2f\t%.0f\t%d\t%.1f ± %.1f\t%.1f ± %.1f\t%.1f ± %.1f' % (
        print('%s\t%.2f\t%.2f\t%s\t%.3f\t%.3f\t%.0f\t%.0f\t%.0f\t%d\t%.0f\t%.0f\t%.0f\t%.1f\t%.1f\t%.1f' % (
                tyc, s['m_mag_v'], s['mag_v'], s['t_eff'], s['log_g'], s['fe_h'], *modeled, n, *means, *std))
    #print('\ntotal std: %.2f' % (tot_std/tot_std_n))  # v0: 26.12, v1:
    ##

    moon_bgr = {}
    for m in measures:
        if m.obj_id[0] == 'moon':
            moon_bgr[m.cam_i] = m
    if len(moon_bgr) == 3:
        print('Moon\t\t\t\t\t\t%.0f\t%.0f\t%.0f\t1\t%.0f\t%.0f\t%.0f\t0\t0\t0' % (
            moon_bgr[2].c_expected_du, moon_bgr[1].c_expected_du, moon_bgr[0].c_expected_du,
            moon_bgr[2].du_count, moon_bgr[1].du_count, moon_bgr[0].du_count,
        ))

    if SHOW_MEASURES and False:
        x = np.array([(stars[id]['t_eff'] + np.random.uniform(-30, 30, size=None),) + tuple(stars[id]['meas'][j, :])
                    for id in s_by[idxs, 0] for j in range(len(stars[id]['meas']))])

        plt.plot(x[:, 0], x[:, 1], 'bo', fillstyle='none')
        plt.plot(x[:, 0], x[:, 2], 'gx')
        plt.plot(x[:, 0], x[:, 3], 'r+')
        plt.show()

    bgr_cam0 = get_bgr_cam(estimated=False)
    plot_bgr_qeff(bgr_cam0, hold=True, color=('lightblue', 'lightgreen', 'pink'), linestyle='dashed', linewidth=1, marker="")
    plot_bgr_qeff(bgr_cam)


def pad_and_stack(imgs, shape=None):
    if shape is None:
        shape = np.max([img.shape for img in imgs], axis=0)
    else:
        shape = np.array(shape)
    padded_imgs = []
    for img in imgs:
        p_img = np.zeros(shape, dtype=img.dtype)
        oy, ox = (shape - img.shape) // 2
        p_img[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
        padded_imgs.append(p_img)
    return np.stack(padded_imgs, axis=2)


def do_bg_img(input, outfile=None, postfix=''):
    IGNORE_COLOR_CORRECTION = 0

    bits = 10
    max_val = 2 ** bits - 1
    gamma = 2.2
    gamma_break = 0.1
    bgr_cc_mx = None if IGNORE_COLOR_CORRECTION else np.array([
        [2.083400, -0.524300, -0.389100],
        [-0.516800, 2.448100, -0.761300],
        [-0.660600, 0.149600, 1.680900],
    ])

    def write_img(raw_imgs, outfile):
        imgs = []
        for raw in raw_imgs:
            img = ImageProc.change_color_depth(raw.astype('float'), 8, bits)
            img = ImageProc.adjust_gamma(img, gamma, gamma_break=gamma_break, inverse=True, max_val=max_val)
            if bgr_cc_mx is not None:
                img = ImageProc.color_correct(img, bgr_cc_mx, inverse=True, max_val=max_val)
            imgs.append(np.expand_dims(img, axis=0))

        if len(imgs) == 1:
            imgs = imgs[0]

        stacked = np.stack(imgs, axis=0)
        reduced = np.median(stacked, axis=0) if len(imgs) > 2 else np.min(stacked, axis=0)
        bg_img = np.round(reduced).squeeze().astype('uint16')
        cv2.imwrite(outfile, bg_img, (cv2.CV_16U,))

    imgs = []
    for i, path in enumerate(input):
        listing = glob.glob(path)
        for folder in listing:
            if os.path.isdir(folder):
                for file in os.listdir(folder):
                    if file[-4:].lower() in ('.png', '.jpg'):
                        fullpath = os.path.join(folder, file)
                        imgs.append(cv2.imread(fullpath))
                if outfile is None:
                    write_img(imgs, folder + postfix + '.png')
                    imgs.clear()
            elif folder[-4:].lower() in ('.png', '.jpg'):
                imgs.append(cv2.imread(folder))

    if outfile is not None:
        write_img(imgs, outfile)


def get_bgr_cam(thumbnail=False, estimated=False, final=False):
    if 0:
        bgr = (
            {'qeff_coefs': [.05] * 2 + [0.4] + [.05] * 10,
             'lambda_min': 350e-9, 'lambda_eff': 465e-9, 'lambda_max': 1000e-9},
            {'qeff_coefs': [.05] * 4 + [0.4] + [.05] * 8,
             'lambda_min': 350e-9, 'lambda_eff': 540e-9, 'lambda_max': 1000e-9},
            {'qeff_coefs': [.05] * 6 + [0.4] + [.05] * 6,
             'lambda_min': 350e-9, 'lambda_eff': 650e-9, 'lambda_max': 1000e-9},
        )
    elif estimated:
        array = tuple
        if final:
            # CURR RESULT
            tmp = [array([0.04597736, 0.18328294, 0.35886904, 0.20962509, 0.07183794,
       0.06324343, 0.06437495, 0.09206215, 0.07814311, 0.07806824,
       0.06221573, 0.06078359, 0.01155586, 0.00993577]), array([0.0365881 , 0.07068747, 0.07992755, 0.24703731, 0.32896325,
       0.14336659, 0.04652175, 0.08300058, 0.12259696, 0.04717292,
       0.06284823, 0.06651458, 0.06912351, 0.06644719]), array([4.09008330e-02, 6.04753849e-02, 1.09132383e-04, 9.39146573e-03,
       4.33269662e-02, 3.12552300e-01, 3.00956896e-01, 2.20188491e-01,
       1.88448761e-01, 1.38525314e-01, 9.33445846e-02, 5.93350748e-02,
       2.43120789e-02, 3.52109874e-02])]

        else:
            # CURR RESULT
            tmp = [array([0.04597736, 0.18328294, 0.35886904, 0.20962509, 0.07183794,
       0.06324343, 0.06437495, 0.09206215, 0.07814311, 0.07806824,
       0.06221573, 0.06078359, 0.01155586, 0.00993577]), array([0.0365881 , 0.07068747, 0.07992755, 0.24703731, 0.32896325,
       0.14336659, 0.04652175, 0.08300058, 0.12259696, 0.04717292,
       0.06284823, 0.06651458, 0.06912351, 0.06644719]), array([4.09008330e-02, 6.04753849e-02, 1.09132383e-04, 9.39146573e-03,
       4.33269662e-02, 3.12552300e-01, 3.00956896e-01, 2.20188491e-01,
       1.88448761e-01, 1.38525314e-01, 9.33445846e-02, 5.93350748e-02,
       2.43120789e-02, 3.52109874e-02])]

        bgr = (
            {'qeff_coefs': tmp[0], 'lambda_min': 350e-9, 'lambda_eff': 465e-9, 'lambda_max': 1000e-9},
            {'qeff_coefs': tmp[1], 'lambda_min': 350e-9, 'lambda_eff': 540e-9, 'lambda_max': 1000e-9},
            {'qeff_coefs': tmp[2], 'lambda_min': 350e-9, 'lambda_eff': 650e-9, 'lambda_max': 1000e-9},
        )
    elif 0:
        bgr = (
            {'qeff_coefs': list(reversed([.05 * .9, .15 * .9, .33 * .95, .22 * .95, .07 * .95, .05 * .95, .05 * .95, .05 * .95, .05 * .95])),
             'lambda_min': 350e-9, 'lambda_eff': 465e-9, 'lambda_max': 750e-9},
            {'qeff_coefs': list(reversed([.05 * .9, .05 * .95, .23 * .95, .35 * .95, .17 * .95, .07 * .95, .11 * .95, .12 * .95, .05 * .95])),
             'lambda_min': 400e-9, 'lambda_eff': 540e-9, 'lambda_max': 800e-9},
            {'qeff_coefs': list(reversed([.05 * .95, .05 * .95, .35 * .95, .35 * .95, .27 * .95, .23 * .95, .18 * .925, .13 * .9, .09 * .9, .05 * .875])),
             'lambda_min': 500e-9, 'lambda_eff': 650e-9, 'lambda_max': 950e-9},
        )
    elif 1:
        # DEFAULT
        bgr = (
            {'qeff_coefs': np.array(
                [.05 * .05, .15 * .9, .33 * .95, .22 * .95, .07 * .95, .05 * .95, .04 * .95, .05 * .95, .05 * .95, .05 * .925,
                           .05 * .9, .05 * .9, .05 * .875, .05 * .85]) * INIT_QEFF_ADJ[0],
             'lambda_min': 350e-9, 'lambda_eff': 465e-9, 'lambda_max': 1000e-9},
            {'qeff_coefs': np.array(
                [.03 * .05, .03 * .9, .05 * .95, .23 * .95, .35 * .95, .17 * .95, .07 * .95, .11 * .95, .12 * .95, .05 * .925,
                           .05 * .9, .05 * .9, .05 * .875, .05 * .85]) * INIT_QEFF_ADJ[1],
             'lambda_min': 350e-9, 'lambda_eff': 540e-9, 'lambda_max': 1000e-9},
            {'qeff_coefs': np.array(
                [.05 * .05, .05 * .9, .01 * .95, .03 * .95, .05 * .95, .35 * .95, .35 * .95, .27 * .95, .23 * .95, .18 * .925,
                           .13 * .9, .09 * .9, .05 * .875, .05 * .85]) * INIT_QEFF_ADJ[2],
             'lambda_min': 350e-9, 'lambda_eff': 650e-9, 'lambda_max': 1000e-9},
        )
    elif 0:
        bgr = (
            {'qeff_coefs': np.array(
                [.05 * .05, .05 * .05, .15 * .9, .33 * .95, .22 * .95, .07 * .95, .05 * .95, .04 * .95, .05 * .95,
                 .05 * .95, .05 * .925, .05 * .9, .05 * .9, .05 * .875]) * INIT_QEFF_ADJ[0],
             'lambda_min': 300e-9, 'lambda_eff': 465e-9, 'lambda_max': 950e-9},
            {'qeff_coefs': np.array(
                [.03 * .05, .03 * .05, .03 * .9, .05 * .95, .23 * .95, .35 * .95, .17 * .95, .07 * .95, .11 * .95,
                 .12 * .95, .05 * .925, .05 * .9, .05 * .9, .05 * .85]) * INIT_QEFF_ADJ[1],
             'lambda_min': 300e-9, 'lambda_eff': 540e-9, 'lambda_max': 950e-9},
            {'qeff_coefs': np.array(
                [.05 * .05, .05 * .05, .05 * .9, .01 * .95, .03 * .95, .05 * .95, .35 * .95, .35 * .95, .27 * .95,
                 .23 * .95, .18 * .925, .13 * .9, .09 * .9, .05 * .85]) * INIT_QEFF_ADJ[2],
             'lambda_min': 300e-9, 'lambda_eff': 650e-9, 'lambda_max': 950e-9},
        )
    else:
        bgr = (
            {'qeff_coefs': list(reversed(
                [.05 * .9, .15 * .9, .33 * .95, .22 * .95, .07 * .95, .05 * .95, .04 * .95, .05 * .95, .05 * .95, .05 * .925,
                           .05 * .9, .05 * .9])),
             'lambda_min': 350e-9, 'lambda_eff': 465e-9, 'lambda_max': 900e-9},
            {'qeff_coefs': list(reversed(
                [.05 * .9, .05 * .9, .05 * .95, .23 * .95, .35 * .95, .17 * .95, .07 * .95, .11 * .95, .12 * .95, .05 * .925,
                           .05 * .9, .05 * .9])),
             'lambda_min': 350e-9, 'lambda_eff': 540e-9, 'lambda_max': 900e-9},
            {'qeff_coefs': list(reversed(
                [.01 * .9, .01 * .9, .01 * .95, .03 * .95, .05 * .95, .35 * .95, .35 * .95, .27 * .95, .23 * .95, .18 * .925,
                           .13 * .9, .09 * .9])),
             'lambda_min': 350e-9, 'lambda_eff': 650e-9, 'lambda_max': 900e-9},
        )

    bgr_cam = []
    for i in range(3):
        # snr_max = 20*log10(sqrt(sat_e))
        # sat_e = (10**(43/20))**2 => 19952
        bgr_cam.append(Camera(2048, 1536, None, None, sensor_size=(2048 * 3.2e-3, 1536 * 3.2e-3), focal_length=8.2,
                              f_stop=1.4, px_saturation_e=20000, emp_coef=1 / 16 ** 2 if thumbnail else 1, dark_noise_mu=500,
                              readout_noise_sd=15, point_spread_fn=0.5, scattering_coef=5e-9, **bgr[i]))

    if PLOT_INITIAL_QEFF_FN:
        plot_bgr_qeff(bgr_cam)

    return bgr_cam


def copy_lbl_files():
    from shutil import copyfile
    import re
    rx = re.compile(r'[1-9]\d*')
    folder = sys.argv[2]
    src_fp = os.path.join(folder, 'img000381.lbl')
    short_exp = list(range(381, 411)) + list(range(441, 471))
    green = list(range(381, 396)) + list(range(411, 426)) + list(range(441, 456)) + list(range(471, 486))
    side = list(range(441, 501))

    for file in os.listdir(folder):
        if file[-4:].lower() in ('.png', '.jpg'):
            id = int(rx.findall(file)[0])
            dst_fp = os.path.join(folder, file[:-4] + '.lbl')
            if not os.path.exists(dst_fp):
                copyfile(src_fp, dst_fp)
            meta = configparser.ConfigParser()
            meta.read(dst_fp)
            meta.set('main', 'exposure', '0.05' if id in short_exp else '0.15')
            meta.set('main', 'impulse_peak', '557.7e-9' if id in green else '630e-9')
            meta.set('main', 'impulse_size', '(135, 135)' if id in side else '(121, 135)')
            with open(dst_fp, 'w') as fp:
                meta.write(fp)


def test_thumbnail_gamma_effect():
    img = np.zeros((1536, 2048))
    j = 1536//2 + 8
    I = np.array(tuple(range(8, 2048, 32))[:-1], dtype='int')
    mag = I[-1] * 0.85
    img[j-8:j+8, I] = np.array(I) * (mag/I[-1])
    img = ImageProc.apply_point_spread_fn(img, 0.4)
    print('max: %d' % np.max(img))

    img = np.clip(img, 0, 1023).astype('uint16')
    plt.imshow(img)
    plt.show()

    thb_real = cv2.resize(img, None, fx=1/16, fy=1/16, interpolation=cv2.INTER_AREA)
    plt.imshow(thb_real)
    plt.show()

    img_gamma = ImageProc.adjust_gamma(img, 2.2, 0.1, max_val=1023)
    thb_gamma = cv2.resize(img_gamma, None, fx=1/16, fy=1/16, interpolation=cv2.INTER_AREA)
    thb = ImageProc.adjust_gamma(thb_gamma, 2.2, 0.1, inverse=1, max_val=1023)

    x = thb_real[j//16, I//16]
    xf = img[j, I]
    xfg = img_gamma[j, I]
    yg = thb_gamma[j//16, I//16]
    y = thb[j//16, I//16]
    line = np.linspace(0, np.max(x))

    plt.plot(x, y, 'x')
    plt.plot(line, line)
    gamma, gamma_break, max_val, scale = fit_gamma(x, y)
    plt.plot(line, ImageProc.adjust_gamma(line, gamma, gamma_break, max_val=max_val) * scale)
    plt.show()
    quit()


def fit_gamma(x, y, p0=None):
    p0 = p0 or (1.6, 0.03, 900, 1.0)
    _USE_BFGS = 0

    def costfn(p, x, y):
        gamma, gamma_break, max_val, scale = tuple(map(abs, p))
        diff = ImageProc.adjust_gamma(x, gamma, gamma_break, max_val=max_val) * scale - y
        diff = tools.pseudo_huber_loss(120, diff)
        return np.sum(diff) if _USE_BFGS else diff

    #res = fmin(costfn, p0, args=(x, y), full_output=True)
    if _USE_BFGS:
        res = fmin_bfgs(costfn, p0, args=(x.astype('float'), y.astype('float')), full_output=True)
    else:
        res = leastsq(costfn, p0, args=(x, y), full_output=True)
    pr = res[0]

    gamma, gamma_break, max_val, scale = tuple(map(abs, pr))
    print('gamma: %f, gamma_break: %f, max_val: %f, scale: %f' % (gamma, gamma_break, max_val, scale))
    return gamma, gamma_break, max_val, scale


class Optimizer:
    def __init__(self, params=None):
        self.params = params or {}
        self.method = self.params.pop('method', 'leastsq')

    def optimize(self, measures):
        opt_method = self.method
        cams = measures[0].frame.cam
        cn = len(cams)
        qn = [len(cams[i].qeff_coefs) for i in range(cn)]
        fn = {
            FRAME_GAIN_NONE: 0,
            FRAME_GAIN_SAME: 1,
            FRAME_GAIN_STATIC: 0,
            FRAME_GAIN_INDIVIDUAL: np.max([m.frame.id for m in measures]) + 1,
        }[FRAME_GAINS]
        gn = 1 if GENERAL_GAIN_ADJUSTMENT is not False else 0

        STAR_SATURATION_MODELING = StarFrame.STAR_SATURATION_MODELING != StarFrame.STAR_SATURATION_MODEL_IDEAL

        f_gains = np.ones(fn)
        for m in measures:
            if FRAME_GAINS == FRAME_GAIN_SAME:
                f_gains[0] = STAR_GAIN_ADJUSTMENT
            elif FRAME_GAINS == FRAME_GAIN_INDIVIDUAL:
                if m.obj_id[0] == 'moon':
                    f_gains[m.frame.id] = MOON_GAIN_ADJ
                else:
                    f_gains[m.frame.id] = STAR_GAIN_ADJUSTMENT if m.frame.cam[0].emp_coef >= 1 else STAR_GAIN_ADJUSTMENT_TN

        def encode(cams, f_gains, gain_adj, psf_coef):
            # if len(psf_coef) == 3 and not StarFrame.STAR_SATURATION_MULTI_KERNEL:
            #     psf_coef = list(psf_coef)
            #     psf_coef[2] = np.log(psf_coef[1]/psf_coef[2])
            #     psf_coef[1] = np.log(psf_coef[0]/psf_coef[1])

            # parameterize cam spectral responsivity, frame specific exposure correction
            return (*[qec for c in cams for qec in c.qeff_coefs], *f_gains, *((gain_adj,) if gn else tuple()), *psf_coef)

        def decode(xr):
            x = np.abs(xr)
            off1 = len(STAR_PSF_SDS)
            off0 = off1 + (1 if gn else 0)

            if not FIXED_PSF_SDS:
                psf_coef = list(x[-off1:] if STAR_SATURATION_MODELING else (1, 0, 0))

            # if len(STAR_PSF_SDS) == 3 and not StarFrame.STAR_SATURATION_MULTI_KERNEL:
            #     psf_coef[1] = psf_coef[0] * np.exp(-psf_coef[1])
            #     psf_coef[2] = psf_coef[1] * np.exp(-psf_coef[2])

            k = 0
            qeff_coefss = []
            for i in range(cn):
                qeff_coefss.append(x[k:k+qn[i]])
                k += qn[i]

            return (qeff_coefss, x[k:len(x)-off0], (x[-off0] if gn else 1), psf_coef)

        def cost_fun(x, measures, prior_x, return_details=False, plot=False):
            c_qeff_coefs, f_gains, gain_adj, psf_coef = decode(x)

            band = []
            obj_ids = []
            measured_du = []
            expected_du = []
            weights = []
            for m in measures:
                if FRAME_GAINS == FRAME_GAIN_SAME:
                    pre_sat_gain = f_gains[0]
                elif FRAME_GAINS == FRAME_GAIN_INDIVIDUAL:
                    pre_sat_gain = f_gains[m.frame.id]
                elif FRAME_GAINS == FRAME_GAIN_STATIC:
                    if m.obj_id[0] == 'moon':
                        pre_sat_gain = MOON_GAIN_ADJ
                    else:
                        pre_sat_gain = STAR_GAIN_ADJUSTMENT if m.frame.cam[0].emp_coef >= 1 else STAR_GAIN_ADJUSTMENT_TN
                else:
                    pre_sat_gain = 1

                edu = m.expected_du(pre_sat_gain=pre_sat_gain, post_sat_gain=gain_adj,
                                    qeff_coefs=c_qeff_coefs, psf_coef=psf_coef)

                if return_details or (m.obj_id[0], m.cam_i) not in IGNORE_MEASURES:
                    expected_du.append(edu)
                    measured_du.append(m.du_count)
                    weights.append(m.weight)
                    band.append(m.cam_i)
                    obj_ids.append(m.obj_id)

            measured_du, expected_du, band = map(np.array, (measured_du, expected_du, band))

            if plot:
                plt.rcParams.update({'font.size': 16})
                fig, ax = plt.subplots(1, 1, figsize=[6.4, 4.8])
                sb, = ax.plot(expected_du[band == 0]*1e-3, measured_du[band == 0]*1e-3, 'bx')
                sg, = ax.plot(expected_du[band == 1]*1e-3, measured_du[band == 1]*1e-3, 'gx')
                sr, = ax.plot(expected_du[band == 2]*1e-3, measured_du[band == 2]*1e-3, 'rx')
                line = np.linspace(0, np.max(expected_du))
                ax.plot(line*1e-3, line*1e-3, 'k--', linewidth=0.5)
                ax.set_xlabel('Expected [1000 DNs]')
                ax.set_ylabel('Measured [1000 DNs]')
                names = Stars.get_catalog_id(np.unique(list(s[0] for s in obj_ids if s[0] != 'moon')), 'simbad')
                names['moon'] = 'Moon'
                labels = np.array([names[id[0]] for id in obj_ids])
                tools.hover_annotate(fig, ax, sb, labels[band == 0])
                tools.hover_annotate(fig, ax, sg, labels[band == 1])
                tools.hover_annotate(fig, ax, sr, labels[band == 2])
                plt.tight_layout()
                plt.show()

            _, _, gain_adj0, _ = decode(prior_x)
#            err = tuple(tools.pseudo_huber_loss(STAR_CALIB_HUBER_COEF, (measured_du - expected_du) * 2 / (expected_du + measured_du)) * np.array(weights))
            err = tuple(tools.pseudo_huber_loss(np.log10(expected_du) - np.log10(measured_du), STAR_CALIB_HUBER_COEF) * np.array(weights))

            n = 3*len(c_qeff_coefs[0])

            lab_dp = tuple()
            if STAR_LAB_DATAPOINT_WEIGHT > 0:
                c, lam = m.frame.cam, 557.7e-9
                g = Camera.sample_qeff(c_qeff_coefs[1], c[1].lambda_min, c[1].lambda_max, lam)
                eps = 1e-10
                r_g = (Camera.sample_qeff(c_qeff_coefs[2], c[2].lambda_min, c[2].lambda_max, lam) + eps) / (g + eps)
                b_g = (Camera.sample_qeff(c_qeff_coefs[0], c[0].lambda_min, c[0].lambda_max, lam) + eps) / (g + eps)
                lab_dp = tuple(STAR_LAB_DATAPOINT_WEIGHT * (np.log10(r_g) - np.log10(np.array((0.26, 0.25, 0.24, 0.24))))**2) \
                        +tuple(STAR_LAB_DATAPOINT_WEIGHT * (np.log10(b_g) - np.log10(np.array((0.23, 0.24, 0.21, 0.22))))**2)

            prior = tuple(STAR_CALIB_PRIOR_WEIGHT ** 2 * (np.array(x[:n]) - np.array(prior_x[:n])) ** 2) \
                if STAR_CALIB_PRIOR_WEIGHT > 0 else tuple()

            err_tuple = lab_dp + err + prior
            return (err_tuple, measured_du, expected_du) if return_details else \
                    err_tuple if opt_method == 'leastsq' else \
                    np.sum(err_tuple)

        if STAR_SATURATION_MODELING:
            psf_coef = STAR_PSF_COEF_TN if cams[0].emp_coef < 1 else STAR_PSF_SDS
        else:
            psf_coef = tuple()

        x0b = encode(cams, f_gains, GENERAL_GAIN_ADJUSTMENT, psf_coef)
        prior_x = encode(get_bgr_cam(estimated=False), f_gains, GENERAL_GAIN_ADJUSTMENT, psf_coef)

        if DEBUG_MEASURES:
            cost_fun(x0b, measures, x0b, plot=True)

        timer = tools.Stopwatch()
        timer.start()
        results = [None] * OPTIMIZER_START_N
        scores = [None] * OPTIMIZER_START_N

        for i in range(OPTIMIZER_START_N):
            tools.show_progress(OPTIMIZER_START_N, i)
            x0 = tuple(np.array(x0b) * (1 if OPTIMIZER_START_N == 1 else np.random.lognormal(0, 0.05, len(x0b))))

            if opt_method == 'leastsq':
                res = leastsq(cost_fun, x0, args=(measures, prior_x), full_output=True, **self.params)
                x, fval = res[0], np.sum(res[2]['fvec'])
            else:
                res = minimize(cost_fun, x0, args=(measures, prior_x), method=opt_method, **self.params)
                x, fval = res.x, res.fun

            results[i] = (x, x0)
            scores[i] = fval if fval > 0 else float('inf')
        timer.stop()

        if len(scores) > 0:
            best = np.argmin(scores)
            print('\nscores: %s' % sorted(scores))
            res, x0 = results[best]
            print('\nbest prior_x: %s' % (x0,))
            print('best x:  %s' % (res,))
            print('time: %.1fs' % timer.elapsed)
        else:
            res = x0b

        qeff_coefs, f_gains, gain_adj, psf_sd = decode(res)
        err, measured, expected = cost_fun(res, measures, x0b, return_details=True, plot=True)
        return qeff_coefs, f_gains, gain_adj, psf_sd, err, measured, expected


if __name__ == '__main__':
    if 0:
        test_thumbnail_gamma_effect()
    if 0:
        copy_lbl_files()
        # Stars.correct_supplement_data()
    else:
        method = sys.argv[1]
        folder = sys.argv[2]
        if method == 'stars':
            calibrate(folder, thumbnail=True)
        elif method == 'fstars':
            calibrate(folder, thumbnail=False)
        elif method == 'moon':
            use_moon_img(folder, get_bgr_cam, MOON_GAIN_ADJ)
        elif method == 'aurora':
            analyze_aurora_img(folder, get_bgr_cam)
        elif method == 'lab':
            use_lab_imgs(folder, get_bgr_cam)
        elif method == 'lab_bg':
            use_lab_bg_imgs(folder, get_bgr_cam)
        elif method == 'bg':
            if len(sys.argv) > 3:
                outfile = folder
                input = sys.argv[3:]
            else:
                outfile = None
                input = [folder]
            do_bg_img(input, outfile, postfix='')

