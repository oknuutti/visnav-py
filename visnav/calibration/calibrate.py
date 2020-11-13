import configparser
import pickle
from abc import ABC
from functools import lru_cache

import math
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from datetime import datetime

from scipy.optimize import leastsq, fmin, fmin_bfgs, fmin_cg
from scipy.interpolate import interp1d

from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture
from scipy.optimize import minimize

from visnav.algo import tools
from visnav.algo.image import ImageProc
from visnav.algo.model import Camera
from visnav.render.stars import Stars
from visnav.calibration import ssi_table

# for star images
from visnav.render.sun import Sun

DEBUG_EXTRACTION = 0
DEBUG_CHANNELS = 0
DEBUG_MATCHING = 0  # show 1=tycho, 2=t_eff, 3=mag_v
DEBUG_MEASURES = 0
MANUAL_ATTITUDE = 0
PLOT_INITIAL_QEFF_FN = 0
SHOW_MEASURES = 1

RAW_IMG_MAX_VALUE = 567

# for moon
MOON_GAIN_ADJ = RAW_IMG_MAX_VALUE/1023   #  was: 506/1023

STAR_GAIN_ADJUSTMENT_TN = 0.7           #
STAR_PSF_COEF_TN = (0.22, 0.15, 0.12)   #

STAR_SATURATION_MODELING = True
STAR_LAB_DATAPOINT_WEIGHT = 1.0
STAR_MOON_WEIGHT = 1.0
STAR_IGNORE_MOON_CH = (2,)        # cam channels
STAR_CALIB_PRIOR_WEIGHT = 0.0001   # 0.5: high, 0.3: med, 0.1: low          # first, 0.1, then 0.0001
STAR_CALIB_HUBER_COEF = np.log10(1 + 0.1)       # %: err_dus/measured_du    # first: 0.25, then 0.10
STAR_GAIN_ADJUSTMENT = RAW_IMG_MAX_VALUE/1023   #                             # was 608/1023
STAR_PSF_COEF = (145, 95, 70)     # bgr                         # best: (180, 100, 65), (150, 95, 80)
INIT_QEFF_ADJ = (1.0, 1.0, 1.0)   # bgr                         # best: (1.0, 0.95, 0.8),

USE_ESTIMATED_QEC = 1             # use estimated qec instead of initial one
STAR_OPT_WEIGHTING = 1
STAR_USE_CACHED_MEAS = 1
OPTIMIZER_START_N = 1          # set to zero for skipping optimizing and just show result for the initial solution

(FRAME_GAIN_NONE, FRAME_GAIN_SAME, FRAME_GAIN_STATIC, FRAME_GAIN_INDIVIDUAL) = range(4)
FRAME_GAINS = FRAME_GAIN_STATIC    # 0: dont use, 1: same gain for all, 2: static gains, 3: individual gains
GENERAL_GAIN_ADJUSTMENT = False   # 1.0       #

STAR_IGNORE_IDS = (
        32263,  # Sirius      # brightest star, very saturated
#        24378,  # Rigel       # saturated
#        37173,  # Procyon     # saturated
#        27919,  # Betelgeuse  # saturated, highly variable, was experiencing severe dimming

        26142,  # Meissa      # the neck/head, bright open cluster too near (too dim to detect with current threashold)
#        25865,  # Mintaka     # eastern star of belt, very blue, could try still (!)
        26176,  # Hatysa      # the sword, orion nebula and lesser stars too near
    ) if 1 else tuple()

OVERRIDE_STAR_DATA = {
    # Betelgeuse: CTOA observations on 2018-12-07 and 18-12-22, accessed through https://www.aavso.org database
#    27919: {'mag_v': 0.8680, 'mag_b': 2.6745, 't_eff': 3546},  # estimated based on b-v: 3546K

    # Betelgeuse: CTOA observation on 2018-12-22, accessed through https://www.aavso.org database
    27919: {'mag_v': 0.882, 'mag_b': 2.680}, #, 't_eff': 3550},  # estimated based on b-v:

    # 37173: {'t_eff': 6500, },
    # 25273: {'t_eff': 18000, },
    # 26246: {'t_eff': 24670, },
    # 30251: {'t_eff': 26630, 'log_g': 3.89, },
    # 27298: {'t_eff': 26500, 'log_g': 2.7, },
    # 25865: {'t_eff': 30100, 'log_g': 3.31, },
    # 23820: {'t_eff': 8000, },
    # 36087: {'t_eff': 13100, 'log_g': 3.611, },
}

# for lab images
SHOW_LAB_MEASURES = 0
IGNORE_COLOR_CORRECTION = 0

# for both
GAMMA_OVERRIDE = 2.2  # 1.92
GAMMA_BREAK_OVERRIDE = 0.1  # 0.072094  # 0.10
GAMMA_ADJUSTMENT = None    # TODO: remove
MISSING_BG_REMOVE_STRIPES = 1


# NANOCAM_GAINS = 65535: x128, 32768: x64, 16384: x32, 2048: x4, 64: x0.125


def use_aurora_img(img_file):
    debug = 1
    n = 0

    override = {
        # 'exposure': 2.4,
        # 'gamma_break': 0.1,
        # 'phase_angle': 2,
        # 'sun_moon_lon': 2,
        # 'gain': NANOCAM_GAINS[65535],  # x128
        # 'bg_image': None,
    }
    Frame.MISSING_BG_REMOVE_STRIPES = 0

    if GAMMA_OVERRIDE:
        override['gamma'] = GAMMA_OVERRIDE
    if GAMMA_BREAK_OVERRIDE:
        override['gamma_break'] = GAMMA_BREAK_OVERRIDE

    bgr_cam = get_bgr_cam(thumbnail=False, estimated=1, final=1)
    f = Frame.from_file(bgr_cam, img_file, img_file[:-4]+'.lbl', override=override, bg_offset=False, debug=debug)

    if 0:
        f.show_image(processed=True, save_as='C:/projects/s100imgs/processed-aurora.png')

    img = f.image.astype('float')

    if 0:
        img = img - np.percentile(img, 5, axis=1).reshape((-1, 1, 3))

    bg1 = (830, 1070), (1180, 1400)
#    bg1 = (0, 900), (660, 1280)
#    bg2 = (1560, 1050), (2048, 1350)
    mean_bg = np.mean(img[bg1[0][1]:bg1[1][1], bg1[0][0]:bg1[1][0], :].reshape((-1, 3)), axis=0)
#    mean_bg = np.mean(np.vstack((img[bg1[0][1]:bg1[1][1], bg1[0][0]:bg1[1][0], :].reshape((-1, 3)),
#                                 img[bg2[0][1]:bg2[1][1], bg2[0][0]:bg2[1][0], :].reshape((-1, 3)))), axis=0)
    img = img - mean_bg
    # img = ImageProc.apply_point_spread_fn(img - mean_bg, 0.01)
    # img = np.clip(img, 0, 1023).astype('uint16')
    # img = cv2.medianBlur(img, 31)
    # img = np.clip(img, 0, RAW_IMG_MAX_VALUE) / RAW_IMG_MAX_VALUE

    n += 1
    plt.figure(n)
    imsh = (np.clip(img * 2 + RAW_IMG_MAX_VALUE * 0.3, 0, RAW_IMG_MAX_VALUE) / RAW_IMG_MAX_VALUE * 255).astype('uint8')
    rd_y = (720, 700), (890, 720)
    rd_r1 = (720, 830), (890, 870)
    rd_r2 = (1080, 820), (1250, 860)
    gr_r = (1280, 770), (1450, 795)
    cv2.rectangle(imsh, bg1[0], bg1[1], (255, 0, 0), 2)       # bg1
#    cv2.rectangle(imsh, bg2[0], bg2[1], (255, 0, 0), 2)       # bg2
    cv2.rectangle(imsh, rd_y[0], rd_y[1], (0, 200, 200), 2)       # yellow
    cv2.rectangle(imsh, rd_r1[0], rd_r1[1], (0, 0, 255), 2)     # red1
    cv2.rectangle(imsh, rd_r2[0], rd_r2[1], (0, 0, 255), 2)     # red2
    cv2.rectangle(imsh, gr_r[0], gr_r[1], (0, 255, 0), 2)     # green
    plt.imshow(np.flip(imsh, axis=2))
    plt.show()

    def electrons(lam):
        h = 6.626e-34  # planck constant (m2kg/s)
        c = 3e8  # speed of light
        return h * c / lam  # energy per photon

    blue = 427.8e-9
    green = 557.7e-9
    yellow = 589.3e-9
    red = 630.0e-9
    colors = (blue, green, yellow, red)
    dus_per_rad = dict(zip(colors, ([], [], [], [])))   # DUs per 1 W/m2/sr of radiance
    coef = f.exposure * f.gain * RAW_IMG_MAX_VALUE

    for wl in dus_per_rad.keys():
        for cam in bgr_cam:
            cgain = cam.gain * cam.emp_coef * cam.aperture_area
            fn, _ = Camera.qeff_fn(tuple(cam.qeff_coefs), 350e-9, 1000e-9)

            # W/m2/sr => phot/s/m2/sr => elec/s/m2/sr => DUs/sr
            dus_per_rad[wl].append(1/electrons(wl) * fn(wl) * coef * cgain)
    for wl in dus_per_rad.keys():
        dus_per_rad[wl] = np.array(dus_per_rad[wl])

    class Patch:
        def __init__(self, name, rect, bands, mean=None, rad=None):
            self.name, self.rect, self.bands, self.mean, self.rad = name, rect, bands, mean, rad
    nt = lambda n, r, b: Patch(name=n, rect=r, bands=b)

    patches = [
        nt('Clean Red', rd_r1, (blue, green, red)),
        nt('Strong Red', rd_r2, (blue, green, red)),
        nt('Green', gr_r, (blue, green, red)),
        nt('Sodium', rd_y, (blue, green, yellow)),
    ]

    # pseudo inverse
    for p in patches:
        p.mean = np.mean(img[p.rect[0][1]:p.rect[1][1], p.rect[0][0]:p.rect[1][0], :].reshape((-1, 3)), axis=0)
        px_sr = cam.pixel_solid_angle((p.rect[0][0]+p.rect[1][0])//2, (p.rect[0][1]+p.rect[1][1])//2)
        E = np.hstack((dus_per_rad[p.bands[0]], dus_per_rad[p.bands[1]], dus_per_rad[p.bands[2]])) * px_sr
        invE = np.linalg.inv(E.T.dot(E)).dot(E.T)
        rad = invE.dot(p.mean)    # radiance in W/m2/sr
        # e = E.dot(rad)
        # diff = (p.mean - e) * 100 / np.linalg.norm(p.mean)
        p.rad = [''] * len(colors)
        for i, b in enumerate(p.bands):
            idx = colors.index(b)
            p.rad[idx] = rad[i]

    sep = '\t' if 1 else ' & '
    le = '\n' if 1 else ' \\\\\n'
    if 0:
        print(sep.join(('Patch', 'Emission at', '', '', 'Red', 'Green', 'Blue')), end=le)
        for name, irr, mean, model, diff in patches:
            print(sep.join((name, '428 nm', ('%.3e' % irr[0]) if irr[0] else 'n/a', 'Mean',
                            *('%.1f' % m for m in np.flip(mean.flatten())))), end=le)
            print(sep.join(('', '557.7 nm', ('%.3e' % irr[1]) if irr[1] else 'n/a', 'Modeled',
                            *('%.1f' % m for m in np.flip(model.flatten())))), end=le)
            print(sep.join(('', '589 nm', ('%.3e' % irr[2]) if irr[2] else 'n/a', 'Diff. [%]',
                            *('%.1f' % m for m in np.flip(diff.flatten())))), end=le)
            print(sep.join(('', '630 nm', ('%.3e' % irr[3]) if irr[3] else 'n/a', *(['']*4))), end=le)
    else:
        print(sep.join(('Patch', 'Red', 'Green', 'Blue', '428 nm', '557.7 nm', '589 nm', '630 nm')), end=le)
        for p in patches:
            # in kilo Rayleigh (kR) == 6330*1e9*lambda * W/m2/sr    or  4*pi*10^(-10)*10^(-3) * photon flux
            print(sep.join((p.name, *('%.1f' % m for m in np.flip(p.mean.flatten())),
                                    *(tools.fixed_precision(r*4*np.pi*1e-13/electrons(colors[i]), 3, True) if r else ''  # r*1e-13*4*np.pi
                                      for i, r in enumerate(p.rad)))), end=le)
    quit()

    aurora = np.zeros_like(img)
    for i, color in enumerate(colors):
        # d/dx[(r-aw)'*(r-aw)] == 0
        #  => w == (r'*a)/(a'*a)
        a = emission[color]
        w = np.sum(img.reshape((-1, 3)) * a.T, axis=1) / sum(a**2)
        e = (w*a).T
        r = img.reshape((-1, 3)) - e
        x = w / np.linalg.norm(r, axis=1)

        # plt.figure(2)
        # plt.imshow(w.reshape(img.shape[:2])/np.max(w))
        # plt.title('weight (max=%f)' % np.max(w))

        # plt.figure(3)
        # plt.imshow(x.reshape(img.shape[:2])/np.max(x))
        # plt.title('x (max=%f)' % np.max(x))

        n += 1
        plt.figure(n)
        x[x < {red: 10, green: 6, yellow: 16}[color]] = 0
        x[w < 100] = 0
        xf = ImageProc.apply_point_spread_fn(x.reshape(img.shape[:2]), 0.03)
        xf = cv2.medianBlur(xf.astype('uint16'), 11)
        plt.imshow(xf / np.max(xf))
        plt.title('Emission detection @ %.1fnm' % (color * 1e9))

        e[xf.flatten() == 0, :] = (0, 0, 0)
        aurora += e.reshape(img.shape)

        # plt.figure(6)
        # plt.imshow(np.flip(e.reshape(img.shape) / np.max(e), axis=2))
        # plt.title('modeled aurora')

        # plt.figure(7)
        # plt.imshow(np.flip(r.reshape(img.shape)/np.max(r), axis=2))
        # plt.title('residual')
        # plt.show()

    plt.figure(8)
    plt.imshow(np.flip(aurora / np.max(aurora), axis=2))
    plt.title('modeled aurora')
    plt.show()

    # TODO: translate rgb values to aurora (ir)radiance
    #  - following uses W/m2/sr for "in-band radiance"
    #  - https://www.osapublishing.org/DirectPDFAccess/A2F3D832-975A-1850-088634AAFCF21258_186134/ETOP-2009-ESB4.pdf?da=1&id=186134&uri=ETOP-2009-ESB4&seq=0&mobile=no
    #  - use pixel sr?

    print('done')


def use_moon_img(img_file):
    debug = 1
    override = {
        #'exposure': 0.01,
#        'gamma_break': 0.1,
        # 'phase_angle': 2,
        # 'sun_moon_lon': 2,
#        'gain': NANOCAM_GAINS[64],  # x0.125
    }
    Frame.MISSING_BG_REMOVE_STRIPES = 0   # force to false

    if GAMMA_OVERRIDE:
        override['gamma'] = GAMMA_OVERRIDE
    if GAMMA_BREAK_OVERRIDE:
        override['gamma_break'] = GAMMA_BREAK_OVERRIDE

    bgr_cam = get_bgr_cam(thumbnail=False, estimated=0)
    f = Frame.from_file(bgr_cam, img_file, img_file[:-4]+'.lbl', override=override, debug=debug)

    measures = f.detect_moon()
    md = [m.du_count for m in measures]
    ed = [m.expected_du(post_sat_gain=MOON_GAIN_ADJ, plot=debug) for m in measures]
    print('Measured DUs (B, G, R): %s => [G/B=%d%%, R/G=%d%%]\nExpected DUs (B, G, R): %s => [G/B=%d%%, R/G=%d%%]' % (
            [round(m) for m in md], md[1]/md[0]*100, md[2]/md[1]*100,
            [round(m) for m in ed], ed[1]/ed[0]*100, ed[2]/ed[1]*100,
    ))


def use_stars(folder, thumbnail=True):
    debug = 1

    override = {
        'exposure': 1.6,
        'gamma_break': 0.1,
        #'bg_image': '../bg_img.png',
        #'bg_image': 'bg_aurora.png',
        #        'ccm_bgr_red': [],
        'gain':  NANOCAM_GAINS[16384],  # x32
    }

    if GAMMA_OVERRIDE:
        override['gamma'] = GAMMA_OVERRIDE
    if GAMMA_BREAK_OVERRIDE:
        override['gamma_break'] = GAMMA_BREAK_OVERRIDE

    bgr_cam = get_bgr_cam(thumbnail=thumbnail, estimated=USE_ESTIMATED_QEC)
    cache_file = '../%s_meas_cache.pickle' % ('th' if thumbnail else ('sm' if len(sys.argv) > 3 else 's'))

    if not STAR_USE_CACHED_MEAS or not os.path.exists(cache_file):
        s_frames = []
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                if file[-4:].lower() in ('.png', '.jpg'):
                    fullpath = os.path.join(folder, file)
                    f = Frame.from_file(bgr_cam, fullpath, fullpath[:-4]+'.lbl', override=override, bg_offset=21, debug=debug)
                    if 0:
                        f.show_image()
                    s_frames.append(f)
        else:
            f = Frame.from_file(bgr_cam, folder, folder[:-4] + '.lbl', override=override, bg_offset=21, debug=debug)
            s_frames.append(f)

        if 0 and not thumbnail:
            f.show_image(processed=True, median_filter=False, zero_bg=30, gain=4, save_as='c:/projects/s100imgs/processed-stars-mf.png')

        measures = []
        stars = {}
        for f in s_frames:
            m_tmp, m_str = f.detect_stars(thumbnail=thumbnail)
            measures.extend(m_tmp)
            merge(stars, m_str)

        if len(sys.argv) > 3:
            override, Frame.MISSING_BG_REMOVE_STRIPES = {}, False
            m_folder = sys.argv[3]
            m_frames = []
            if os.path.isdir(m_folder):
                for file in os.listdir(m_folder):
                    if file[-4:].lower() in ('.png', '.jpg'):
                        fullpath = os.path.join(m_folder, file)
                        m_frames.append(Frame.from_file(bgr_cam, fullpath, fullpath[:-4] + '.lbl', override=override, debug=debug))
            else:
                m_frames.append(Frame.from_file(bgr_cam, m_folder, m_folder[:-4] + '.lbl', override=override, debug=debug))

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
    qeff_coefs, f_gains, gain_adj, psf_coef, err, measured, expected = opt.optimize(measures)

    for i, qec in enumerate(qeff_coefs):
        bgr_cam[i].qeff_coefs = qec

    print('err: %.3f' % np.mean(err))
    target = np.array((24.75, 22.50))
    print('r/g, b/g @ 557.7nm: (%.2f%%, %.2f%%), target = (%.2f%%, %.2f%%)' % (
        *(np.sqrt(err[0:2])/STAR_LAB_DATAPOINT_WEIGHT*100 + target), *target))
    print('queff_coefs: %s' % (qeff_coefs,))
    print('frame gains: %s' % (f_gains,))
    print('gain_adj: %s' % (gain_adj,))
    print('psf_coefs: %s' % (psf_coef,))
    print('psf SDs [px]: %.3f, %.3f, %.3f' % (*np.sqrt(np.array(psf_coef) / np.pi / 2),))

    ## star measurement table
    ##
    sort = 'mag_v'  # 't_eff'
    s_by = np.array([(id, st[0][sort]) for id, st in stars.items()])
    idxs = np.argsort(s_by[:, 1])

    # NOTE: Overwrites multiple measurement expected dus,
    # i.e. wrong results if different exposure time across different measurements of the same star
    star_exp_dus = {}
    for m in measures:
        if m.obj_id not in star_exp_dus:
            star_exp_dus[m.obj_id] = [0]*3
        star_exp_dus[m.obj_id][m.cam_i] = m.c_expected_du

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


def use_lab_bg_imgs(folder):
    debug = 1
    override = {
        'gamma_break': 0,
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
            f = Frame.from_file(bgr_cam, fullpath, fullpath[:-4]+'.lbl', override=override, debug=debug)

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


def use_lab_imgs(folder):
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
    if IGNORE_COLOR_CORRECTION:
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

    bgr_cam = get_bgr_cam()

    def gaussian(mean, var, x):
        return np.exp(-(x - mean) ** 2 / (2 * var)) / (2 * math.pi * var) ** .5

    @lru_cache(maxsize=100)
    def get_kernel(peak, fwhm, psfd, size):
        var = (fwhm/np.sqrt(8*np.log(2)))**2
        amp = psfd / gaussian(0, var, 0)
        spectrum_fn = lambda lam: amp * gaussian(peak, var, lam)
        kernel = Frame.calc_source_kernel(bgr_cam, spectrum_fn, size, points=(peak,))
        if IGNORE_COLOR_CORRECTION:
            kernel = ImageProc.color_correct(kernel, np.array([
                [2.083400, -0.524300, -0.389100],
                [-0.516800, 2.448100, -0.761300],
                [-0.660600, 0.149600, 1.680900],
            ]))

        return kernel

    mean_sum = {}
    std_sum = {}
    n_sum = {}
    n1_sum = {}
    for i, file in enumerate(os.listdir(folder)):
        if file[-4:].lower() in ('.png', '.jpg'):
            fullpath = os.path.join(folder, file)
            f = Frame.from_file(bgr_cam, fullpath, fullpath[:-4]+'.lbl', override=override, bg_offset=False, debug=debug)
            kernel = get_kernel(f.impulse_peak, f.impulse_fwhm, f.impulse_psfd, f.impulse_size)
            m = f.detect_source(kernel)

            key = (f.exposure, f.impulse_peak, f.impulse_size)
            if key not in mean_sum:
                mean_sum[key] = np.zeros(3)
                std_sum[key] = np.zeros(3)
                n_sum[key] = 0
                n1_sum[key] = 0
            mean_sum[key] += m.n * m.mean
            std_sum[key] += (m.n-1) * m.std
            n_sum[key] += m.n
            n1_sum[key] += m.n-1

    mean = {e: mean_sum[e] / n_sum[e] for e in mean_sum.keys()}
    std = {e: std_sum[e] / n1_sum[e] for e in std_sum.keys()}

    # snr_max = 20*log10(sqrt(sat_e)) dB
    # dynamic range = 20*log10(sat_e/readout_noise))
    # dark_noise_sd should be sqrt(dark_noise_mu)

    # mean of all combinations
    # std vs mean of all combinations
#    fig, (ax1, ax2) = plt.subplots(1, 2)
    print('\tred mean\tred sd\tgreen mean\tgreen sd\tblue mean\tblue sd')
    size_map = {(121, 135): 'center', (135, 135): 'side'}
    for i, key in enumerate(mean.keys()):
        exp, lam, size = key
        lab = '%dms, %.1fnm, %s' % (exp*1e3, lam*1e9, size_map[size])
        print('%s\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f' % (lab, mean[key][2], std[key][2], mean[key][1],
                                                          std[key][1], mean[key][0], std[key][0]))
#        for j, c in enumerate(('b', 'g', 'r')):
#            ax1.plot(i*3+j, mean[key][j], 'x'+c, label=lab+(' [%s]'%c))
#            ax2.plot(i*3+j, (std[key]/mean[key])[j], 'x'+c, label=lab+(' [%s]'%c))
#    ax1.set_title('mean')
#    ax1.legend()
#    ax2.set_title('std/mean')
#    ax2.legend()
#    fig.show()


class Frame:
    _bg_imgs = {}
    CURRENT_ID = 0
    MISSING_BG_REMOVE_STRIPES = MISSING_BG_REMOVE_STRIPES

    def __init__(self, cam, gain, exposure, timestamp, raw_image, background_img, bg_offset=0, bits=8, applied_gamma=1.0,
                 applied_gamma_break=0.0, applied_bgr_mx=None, q=None, debug=False):
        self.id = Frame.CURRENT_ID
        Frame.CURRENT_ID += 1

        self.cam = [cam] if isinstance(cam, Camera) else cam
        self.resize_scale = raw_image.shape[1] / self.cam[0].width
        for c in self.cam:
            c.height, c.width = raw_image.shape[:2]
        self.bits = bits = int(bits)
        self.gain = gain
        self.exposure = exposure
        self.timestamp = timestamp
        self.raw_image = raw_image
        self.applied_gamma = applied_gamma
        self.applied_gamma_break = applied_gamma_break
        self.applied_bgr_mx = applied_bgr_mx
        self.debug = debug

        img_bits = int(str(raw_image.dtype)[4:])
        max_val = 2**img_bits-1
        img = raw_image.astype('float')

        # NOTE: NanoCam has this, doesnt make sense in general!
        operation_order = reversed((
            'ex_gamma',
            'depth',
            'color',
            'gamma',
        ))

        for op in operation_order:
            if op == 'depth' and img_bits != bits:
                img = ImageProc.change_color_depth(img, img_bits, bits)
                max_val = 2 ** bits - 1
            if op == 'gamma' and applied_gamma != 1.0:
                img = ImageProc.adjust_gamma(img, applied_gamma, gamma_break=applied_gamma_break, inverse=True, max_val=max_val)
            if op == 'color' and applied_bgr_mx is not None:
                img = ImageProc.color_correct(img, applied_bgr_mx, inverse=True, max_val=max_val)
            if op == 'ex_gamma' and GAMMA_ADJUSTMENT:
                img = ImageProc.adjust_gamma(img, GAMMA_ADJUSTMENT, inverse=True, max_val=max_val)

        self.background_img = background_img
        if background_img is not None:
            self.image = ImageProc.remove_bg(img, background_img, gain=1, offset=bg_offset, max_val=max_val)
        elif self.MISSING_BG_REMOVE_STRIPES:
            for k in range(img.shape[2]):
                img[:, :, k] -= np.percentile(img[:, :, k], 5, axis=0).reshape((1, -1))
                img[:, :, k] -= np.percentile(img[:, :, k], 5, axis=1).reshape((-1, 1))
            img += bg_offset - np.min(img)
            self.image = np.clip(img, 0, max_val)
        else:
            self.image = img

        if bg_offset is not False:
            self.image = np.round(self.image).astype('uint16')

        self.measures = []

        # about stars
        self.stars = []
        self.q = q

        # about calibration images
        self.impulse_psfd = None
        self.impulse_peak = None
        self.impulse_fwhm = None
        self.impulse_size = None
        self.impulse_spread = None

        # about moon images
        self.moon_loc = None
        self.sun_moon_dist = None
        self.cam_moon_dist = None
        self.cam_moon_lat = None
        self.cam_moon_lon = None
        self.sun_moon_lon = None
        self.phase_angle = None

    @property
    def max_val(self):
        return 2**self.bits - 1

    def set_orientation(self, q=None, angleaxis=None, dec_ra_pa=None):
        if q is not None:
            self.q = q
        elif angleaxis is not None:
            self.q = tools.angleaxis_to_q(angleaxis)
        else:
            assert dec_ra_pa is not None, 'no orientation given'
            dec, ra, pa = map(math.radians, dec_ra_pa)
            self.q = tools.ypr_to_q(dec, ra, pa)

    @staticmethod
    def from_file(cam, img_file, lbl_file, section='main', mapping=None, override=None, bg_offset=0, debug=False):
        assert os.path.exists(lbl_file), 'file %s for metadata is missing' % lbl_file

        meta = configparser.ConfigParser()
        meta.read(lbl_file)

        class Mapping:
            def __init__(self, meta, section, mapping, override):
                self.meta = meta
                self.section = section
                self.mapping = mapping or {}
                self.override = override or {}

            def __getitem__(self, param):
                return self.get(param)

            def get(self, param, default=None):
                v = self.meta.get(self.section, self.mapping.get(param, param), vars=override, fallback=default)
                try:
                    return float(v)
                except:
                    pass
                try:
                    if v[0] == '[' and v[-1] == ']' or v[0] == '(' and v[-1] == ')':
                        a = np.fromstring(v.strip('([])'), sep=',')
                        if len(a) > v.count(','):
                            return a if v[0] == '[' else tuple(a)
                except:
                    pass
                try:
                    return datetime.strptime(v, '%Y-%m-%d %H:%M:%S %z')
                except:
                    pass
                return v

        meta = Mapping(meta, section, mapping, override)

        if meta['bg_image']:
            bg_path = os.path.realpath(os.path.join(os.path.dirname(lbl_file), meta['bg_image']))
            if bg_path not in Frame._bg_imgs:
                bg_img = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
                Frame._bg_imgs[bg_path] = bg_img
            background_img = Frame._bg_imgs[bg_path]
        else:
            background_img = None
        raw_image = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

        bgr_mx = None
        if meta['channels'] == 3 and len(meta.get('ccm_bgr_red', [])) == 3:
            bgr_mx = np.array([meta['ccm_bgr_blue'], meta['ccm_bgr_green'], meta['ccm_bgr_red']])

        frame = Frame(cam, meta['gain'], meta['exposure'], meta['timestamp'], raw_image,
                      background_img, bg_offset, meta['bits'], meta['gamma'], meta['gamma_break'], bgr_mx, debug=debug)

        if meta.get('dec', False):
            frame.set_orientation(dec_ra_pa=(meta.get('dec'), meta.get('ra'), meta.get('pa', 0)))
        if meta.get('impulse_peak', False):
            frame.impulse_size = meta.get('impulse_size')
            frame.impulse_spread = meta.get('impulse_spread')
            frame.impulse_psfd = meta.get('impulse_psfd')
            frame.impulse_peak = meta.get('impulse_peak')
            frame.impulse_fwhm = meta.get('impulse_fwhm')
        if meta.get('moon_loc', False):
            frame.moon_loc = meta.get('moon_loc')
            frame.sun_moon_dist = meta.get('sun_moon_dist')
            frame.cam_moon_dist = meta.get('cam_moon_dist')
            frame.cam_moon_lat = math.radians(meta.get('cam_moon_lat'))
            frame.cam_moon_lon = math.radians(meta.get('cam_moon_lon'))
            frame.sun_moon_lon = math.radians(meta.get('sun_moon_lon'))
            frame.phase_angle = math.radians(meta.get('phase_angle'))

        return frame

    def show_image(self, gain=1, processed=False, compare=False, median_filter=False, zero_bg=False, save_as=None):
        if processed:
            img = self.image.astype('float')
            if zero_bg:
                img = np.clip(img - np.min(img) - (0 if zero_bg is True else zero_bg), 0, np.inf)
            img *= gain
            if median_filter:
                img = cv2.medianBlur(img.astype('uint16'), median_filter)
            img = ImageProc.color_correct(img, self.applied_bgr_mx, max_val=self.max_val)
            img = ImageProc.adjust_gamma(img, self.applied_gamma, self.applied_gamma_break, max_val=self.max_val)
        else:
            img = self.image
        img = ImageProc.change_color_depth(img, self.bits, 8).astype('uint8')

        if save_as is not None:
            cv2.imwrite(save_as, img)

        s = self.image.shape
        if compare:
            img = np.hstack((self.raw_image.astype(img.dtype), np.ones((s[0], 1, s[2]), dtype=img.dtype), img))

        sc = 1
        plt.imshow(np.flip(img, axis=2))
        plt.show()
        return img, sc

    def detect_moon(self):
        x, y = tuple(int(v) for v in self.moon_loc)
        win = self.image[y-24:y+25, x-29:x+30, :]

        if 0:
            # TODO: what transformation would make this work?
            win = ImageProc.adjust_gamma(win/662*1023, 2.2, inverse=1, max_val=1023)

        h, w, s, c = *win.shape[0:2], 19, 18
        mask = np.zeros((h, w), dtype='uint8')
        mask[(h//2-s):(h//2+s+1), (w//2-s):(w//2+s+1)] = ImageProc.bsphkern(s*2+1).astype('uint8')
        mask[0:c, :] = 0

        if self.debug:
            mask_img = (mask.reshape((h, w, 1))*np.array([255, 0, 0]).reshape((1, 1, 3))).astype('uint8')
            win_img = np.clip(win, 0, 255).astype('uint8')
            plt.imshow(np.flip(ImageProc.merge((mask_img, win_img)), axis=2))
            plt.show()

        mask = mask.flatten().astype('bool')
        n = np.sum(mask)
        measures = []
        for i, cam in enumerate(self.cam):
            raw_du = np.sum(win[:, :, i].flatten()[mask])
            bg_du = np.mean(win[:, :, i].flatten()[np.logical_not(mask)])
            du_count = raw_du - bg_du * n
            measures.append(MoonMeasure(self, i, du_count))
        self.measures = measures
        return measures

    def detect_stars(self, thumbnail=True):
        stars_detected = self._extract_stars()
        if self.q is None:
            self.determine_orientation(stars_detected)
        self.measures, self.star_measures = self.finetune_orientation(stars_detected, thumbnail=thumbnail)
        return self.measures, self.star_measures

    def determine_orientation(self, stars):
        assert False, 'not implemented'  # use deep learning? or custom features and a bag of words type thing?

    def finetune_orientation(self, stars, iterations=100, thumbnail=True):
        """ match stars based on proximity, fine tune orientation, create measure objects """
        MIN_MATCHES = 3

        # use ICP-like algorithm
        matches = []
        for i in range(iterations):
            matches, cols = self._match_stars(stars, max_dist=max(0.02, 0.05-i*0.01), mag_cutoff=3.0 if thumbnail else 3.0)
            if np.sum([j is not None for j in matches]) < MIN_MATCHES:
                break
            if self._update_ori(matches, cols, stars):
                break   # if update small enough, stop iterating
        matches, cols = self._match_stars(stars, max_dist=0.02, mag_cutoff=3.0 if thumbnail else 3.0, plot=SHOW_MEASURES)

        def ifna(v, d):
            return d if v is None or np.isnan(v) else v

        measures = []
        star_meas = {}
        mag_adj = np.median([stars[i]['mag'] - m[cols['mag_v']] for i, m in enumerate(matches) if m is not None])
        for i, m in enumerate(matches):
            if m is not None:
                cid = '&'.join([Stars.get_catalog_id(id) for id in m[cols['id']]])
                for band, j in enumerate(('b', 'g', 'r') if len(self.cam) == 3 else ('v',)):
                    t_eff = float(ifna(m[cols['t_eff']], -1))
                    fe_h = float(ifna(m[cols['fe_h']], Sun.METALLICITY))
                    log_g = float(ifna(m[cols['log_g']], Sun.LOG_SURFACE_G))
                    t_est = 0
                    if t_eff < 0:
                        t_est = 1
                        mag_v, mag_b = m[cols['mag_v']], m[cols['mag_b']]
                        if mag_b is None or np.isnan(mag_b):
                            print('Both t_eff AND mag_b missing! ID=%s' % (m[cols['id']],))
                            mag_b = mag_v
                        t_eff = Stars.effective_temp(mag_b - mag_v, fe_h, log_g)
                        print('star %s, missing t_eff, estimated as %.1f' % (cid, t_eff))
                    measures.append(StarMeasure(self, band, m[cols['id']], stars[i]['du_' + j],
                                                t_eff, fe_h, log_g, m[cols['mag_v']]))
                merge(star_meas, {m[cols['id']]: [{'meas': (stars[i]['du_b'], stars[i]['du_g'], stars[i]['du_r']),
                                                   'm_mag_v': stars[i]['mag'] - mag_adj,
                                                   't_eff': ('(%.0f)' if t_est else '%.0f') % t_eff,
                                                   'fe_h': m[cols['fe_h']], 'log_g': m[cols['log_g']],
                                                   'mag_v': m[cols['mag_v']]}]})

        return measures, star_meas

    def _extract_stars(self):
        """ extract stars from image, count "digital units" after bg substraction, calc centroid x, y """
        # scaled to 0-1 and in grayscale
        data = np.mean(self.image.astype(np.float64)/(2**self.bits-1), axis=2)

        mean, median, std = sigma_clipped_stats(data, sigma=3.0)

        thumbnail = self.image.shape[1] == 128
        bsize = 4 if thumbnail else 20
        assert self.image.shape[1] in (128, 2048), 'unsupported image size'
        if thumbnail:
            daofind = DAOStarFinder(fwhm=3.5, threshold=5.*std, sharplo=.3, sharphi=1.3, roundlo=-.8, roundhi=1.3)
        else:
            daofind = DAOStarFinder(fwhm=28, threshold=12.*std, sharplo=-3.0, sharphi=3.0, roundlo=-3.0, roundhi=3.0)

        sources = daofind(data - median)
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

        if self.debug and (DEBUG_EXTRACTION or DEBUG_CHANNELS):
            norm = ImageNormalize(stretch=SqrtStretch())
            if DEBUG_CHANNELS:
                if 0:
                    f, (b, g, r) = plt.subplots(1, 3)
                    b.imshow(self.image[:, :, 0].astype(np.float64) / (2 ** self.bits - 1), cmap='Greys', norm=norm)
                    g.imshow(self.image[:, :, 1].astype(np.float64) / (2 ** self.bits - 1), cmap='Greys', norm=norm)
                    r.imshow(self.image[:, :, 2].astype(np.float64) / (2 ** self.bits - 1), cmap='Greys', norm=norm)
                    b.set_title('blue')
                    g.set_title('green')
                    r.set_title('red')
                else:
                    f, (w, b_r, g_r) = plt.subplots(1, 3, sharex=True, sharey=True)
                    w.imshow(data, cmap='Greys_r', norm=norm)
                    br = (self.image[:, :, 0].astype(np.float64) - self.image[:, :, 2].astype(np.float64)) / (2 ** self.bits - 1)
                    gr = (self.image[:, :, 1].astype(np.float64) - self.image[:, :, 2].astype(np.float64)) / (2 ** self.bits - 1)
                    b_r.imshow(br - np.min(br), cmap='Greys_r', norm=norm)
                    g_r.imshow(gr - np.min(gr), cmap='Greys_r', norm=norm)
                    w.set_title('white')
                    b_r.set_title('blue - red')
                    g_r.set_title('green - red')
                    plt.tight_layout()
                plt.show()

            else:
                plt.imshow(data, cmap='Greys', norm=norm)
                apertures = CircularAperture(positions, r=bsize)
                apertures.plot(color='blue', lw=1.5, alpha=0.5)
                plt.show()

        stars = []
        img_median = np.median(self.image.reshape((-1, 3)), axis=0)
        for i, (x, y) in enumerate(positions):
            if thumbnail:
                size = 4
            elif sources[i]['flux'] > 16:
                size = 30
            elif sources[i]['flux'] > 6:
                size = 25
            elif sources[i]['flux'] > 2:
                size = 20
            else:
                size = 17

            (b, b0), (g, g0), (r, r0) = self._count_du(x, y, size=2*size+1, bg=img_median)
            if b is not None and (b-b0 > 0 or g-g0 > 0 or r-r0 > 0):
                # TODO: add black level remove level to .lbl files?
                #   - unknown black level was removed in sensor, from param tables: 168, but that doesnt work for all images
                #   - for now, add something here but then adjust at match_stars based on brightest & dimmest
                #bg = 168/8  # 168
                #b0, g0, r0 = b0 + bg, g0 + bg, r0 + bg
                mag = -2.5 * math.log10((b+b0) * (g+g0) * (r+r0) / b0 / g0 / r0) / 3
                stars.append({"du_b": b, "du_g": g, "du_r": r, "x": x, "y": y, "mag": mag, "size": size})
            else:
                print('discarded [%d, %d]' % (x, y))

        return stars

    def _count_du(self, x, y, size=5, bg=None):
        wmrg = size//4
        mmrg = 1 if bg is None else 0
        mask = ImageProc.bsphkern(size + 2*mmrg)
        if bg is None:
            mask[0, :] = 0
            mask[-1, :] = 0
            mask[:, 0] = 0
            mask[:, -1] = 0
        mask = mask.astype(np.bool)
        mr = size//2 + mmrg
        mn = size + 2*mmrg

        h, w, _ = self.image.shape
        x, y = int(round(x)), int(round(y))
        if h-y+wmrg <= mr or w-x+wmrg <= mr or x+wmrg < mr or y+wmrg < mr:
            return zip([None] * 3, [None] * 3)

        win = self.image[max(0, y-mr):min(h, y+mr+1), max(0, x-mr):min(w, x+mr+1), :].reshape((-1, 3))
        mx0, mx1 = -min(0, x-mr), mn - (max(w, x+mr+1) - w)
        my0, my1 = -min(0, y-mr), mn - (max(h, y+mr+1) - h)

        mask = mask[my0:my1, mx0:mx1].flatten()
        bg = np.mean(win[np.logical_not(mask), :], axis=0) if bg is None else bg

        if False:
            tot = np.sum(win[mask, :], axis=0)
            tot_bg = bg * np.sum(mask)
            tot = np.max(np.array((tot, tot_bg)), axis=0)

            # tried to take into account thumbnail mean resizing after gamma correction,
            # also assuming no saturation of original pixels because of motion blur
            # => better results if tune Camera->emp_coef instead
            resizing_gain = (1/self.resize_scale)**2
            g = self.applied_gamma

            # ([sum over i in n: (bg+s_i)**g] / n) ** (1/g)
            #    => cannot compensate for gamma correction as signal components not summable anymore,
            #       only possible if assume that only a single pixel has signal (or some known distribution of signal?)
            # signal in a single, non-saturating pixel (conflicting assumptions):
            adj_tot = (((tot-tot_bg+bg)**g*resizing_gain) - (resizing_gain-1)*bg**g)**(1/g) - bg
            signal = adj_tot
        else:
            #signal = tot - tot_bg
            signal = np.clip(np.sum(win[mask, :] - bg, axis=0), 0, np.inf)

        return zip(signal, bg)

    def _match_stars(self, stars, max_dist=0.05, max_mag_diff=2.0, mag_cutoff=3.0, plot=False):
        """ match stars based on proximity """
        merge_lim = 4
        all_stars, cols = Stars.flux_density(self.q, self.cam[0], array=True, undistorted=True,
                                             mag_cutoff=mag_cutoff+merge_lim, order_by='mag_v')
        if self.debug:
            db_img = np.sqrt(Stars.flux_density(self.q, self.cam[0], mag_cutoff=10.0))

        # override some star data, change None => nan
        for i, st in enumerate(all_stars):
            for j in range(len(st)):
                st[j] = np.nan if st[j] is None else st[j]
            if st[cols['id']] in OVERRIDE_STAR_DATA:
                for f in ('mag_v', 'mag_b', 't_eff', 'log_g', 'fe_h'):
                    od = OVERRIDE_STAR_DATA[st[cols['id']]]
                    if f in od:
                        all_stars[i][cols[f]] = od[f]

        # merge close stars
        all_stars = np.array(all_stars)
        points = np.array([(s[cols['ix']], s[cols['iy']]) for s in all_stars])
        D = tools.distance_mx(points, points)
        radius = 10 if self.cam[0].width > 300 else 2
        db_stars = []
        added = set()
        for i, s in enumerate(all_stars):
            if i in added:
                continue
            I = tuple(set(np.where(
                    np.logical_and(D[i, :] < radius, all_stars[:, cols['mag_v']]-merge_lim < s[cols['mag_v']])
                )[0]) - added)
            cluster = [None]*(max(cols.values())+1)
            cluster[cols['id']] = tuple(all_stars[I, cols['id']].astype(np.int))
            amag_v = 10**(-all_stars[I, cols['mag_v']]/2.5)
            amag_b = 10**(-all_stars[I, cols['mag_b']]/2.5)
            cluster[cols['mag_v']] = -2.5*np.log10(np.sum(amag_v))
            cluster[cols['mag_b']] = -2.5*np.log10(np.sum(amag_b))
            for c in ('ix', 'iy', 'dec', 'ra', 't_eff', 'fe_h', 'log_g'):
                E = np.where(all_stars[I, cols[c]] != None)[0]
                cluster[cols[c]] = np.sum(amag_v[E] * all_stars[I, cols[c]][E])/np.sum(amag_v[E]) if len(E) else None
            if cluster[cols['mag_v']] < mag_cutoff:
                added.update(I)
                db_stars.append(cluster)

        img_st = np.array([(s['x'], s['y'], s['mag'], s['size']) for s in stars])
        db_st = np.array([(s[cols['ix']], s[cols['iy']], s[cols['mag_v']]) for s in db_stars])

        # adjust mags to match, not easy to make match directly as unknown variable black level removed in image sensor
        #b0, b1 = np.min(img_st[:, 2]), np.min(db_st[:, 2])
        #d0, d1 = np.max(img_st[:, 2]), np.max(db_st[:, 2])
        #img_st[:, 2] = (img_st[:, 2] - b0) * (d1-b1)/(d0-b0) + b1
        #img_st[:, 2] = np.log10((10**img_st[:, 2] - 10**b0) * (10**d1-10**b1)/(10**d0-10**b0) + 10**b1)
        img_st[:, 2] = img_st[:, 2] - np.median(img_st[:, 2]) + np.median(db_st[:, 2])

        if self.cam[0].dist_coefs is not None:
            db_st[:, :2] = Camera.distort(db_st[:, :2], self.cam[0].dist_coefs,
                                   self.cam[0].intrinsic_camera_mx(legacy=False),
                                   self.cam[0].inv_intrinsic_camera_mx(legacy=False))

        M = (np.abs(np.repeat(np.expand_dims(img_st[:, 2:3], axis=0), len(db_st), axis=0) \
            - np.repeat(np.expand_dims(db_st[:, 2:3], axis=1), len(img_st), axis=1))).squeeze()
        D = np.repeat(np.expand_dims(img_st[:, :2], axis=0), len(db_st), axis=0) \
            - np.repeat(np.expand_dims(db_st[:, :2], axis=1), len(img_st), axis=1)
        D = np.sum(D**2, axis=2)
        D = D.flatten()
        D[M.flatten() > max_mag_diff] = np.inf
        D = D.reshape(M.shape)
        idxs = np.argmin(D, axis=0)
        max_dist = (self.image.shape[1] * max_dist)**2

        m_idxs = {}
        for i1, j in enumerate(idxs):
            dist = D[j, i1]
            if dist > max_dist or j in m_idxs and dist > D[j, m_idxs[j]]:
                # discard match if distance too high or better match for same db star available
                continue
            m_idxs[j] = i1

        matches = [None] * len(idxs)
        for j, i in m_idxs.items():
            matches[i] = db_stars[j]

        if self.debug and DEBUG_MATCHING or plot:
            if plot:
                norm = ImageNormalize(stretch=SqrtStretch())
                size = np.median(img_st[:, 3].astype('int'))
                data = np.mean(self.image.astype(np.float64) / (2 ** self.bits - 1), axis=2)

                ud_I = set(range(len(db_st))) - set(m_idxs.keys())
                d_I = set(range(len(img_st))) - set(m_idxs.values())
                # detected_pos = img_st[tuple(d_I), :2].astype('int')
                # matched_pos = img_st[tuple(m_idxs.values()), :2].astype('int')
                # undetected_pos = db_st[tuple(ud_I), :2].astype('int')

                plt.imshow(data, cmap='Greys', norm=norm)
                for i in d_I:   # detected
                    CircularAperture(img_st[i, :2].astype('int'), r=img_st[i, 3].astype('int')).plot(color='blue', lw=1.5, alpha=0.5)
                for i in m_idxs.values():   # matched
                    CircularAperture(img_st[i, :2].astype('int'), r=img_st[i, 3].astype('int')).plot(color='green', lw=1.5, alpha=0.5)
                for i in ud_I:   # undetected
                    CircularAperture(db_st[i, :2].astype('int'), r=size).plot(color='red', lw=1.5, alpha=0.5)
                plt.show()

            else:
                dec, ra, pa = map(math.degrees, tools.q_to_ypr(self.q))
                print('ra: %.1f, dec: %.1f, pa: %.1f' % (ra, dec, pa))

                sc, isc = 1, (1024 if 0 else 2800) / (self.image.shape[1] * 2)
                img = np.sqrt(self.image)
                img = ((img / np.max(img)) * 255).astype('uint8')
                img = cv2.resize(img, None, fx=isc, fy=isc, interpolation=cv2.INTER_AREA)
                cv2.drawKeypoints(img, [cv2.KeyPoint(x*isc, y*isc, 60*sc) for x, y in db_st[:, :2]], img, [0, 255, 0], cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.drawKeypoints(img, [cv2.KeyPoint(x*isc, y*isc, 60*sc) for x, y in img_st[:, :2]], img, [255, 0, 0], cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                for j, i in m_idxs.items():
                    cv2.line(img, tuple(np.round(img_st[i, :2]*isc).astype('int')),
                                  tuple(np.round(db_st[j, :2]*isc).astype('int')), [0, 255, 0])
                if DEBUG_MATCHING != 3:
                    for j, pt in enumerate(db_st[:, :2]):
                        cv2.putText(img, str(j), tuple(np.round(pt*isc).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 3*sc, [0, 255, 0])
                else:
                    for i, pt in enumerate(img_st[:, :2]):
                        text = '%.2f' % img_st[i, 2]
                        #text = str(i)
                        cv2.putText(img, text, tuple(np.round(pt*isc).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 1.6*sc, [0, 0, 255])

                db_img = np.repeat(np.expand_dims(db_img, axis=2), self.image.shape[2], axis=2)
                db_img = ((db_img / np.max(db_img)) * 255).astype('uint8')
                db_img = cv2.resize(db_img, None, fx=isc, fy=isc, interpolation=cv2.INTER_AREA)

                for j, pt in enumerate(db_st[:, :2]):
                    if DEBUG_MATCHING == 1:
                        text = '&'.join([Stars.get_catalog_id(id) for id in db_stars[j][cols['id']]])
                    elif DEBUG_MATCHING == 2:
                        t_eff = db_stars[j][cols['t_eff']]
                        t_eff2 = Stars.effective_temp(db_stars[j][cols['mag_b']] - db_stars[j][cols['mag_v']])
                        #if 1:
                        #    print('%s Teff: %s (%.1f)' % (Stars.get_catalog_id(db_stars[j][cols['id']]), t_eff, t_eff2))
                        text = ('%dK' % t_eff) if t_eff else ('(%dK)' % t_eff2)
                    elif DEBUG_MATCHING == 3:
                        text = '%.2f' % db_stars[j][cols['mag_v']]
                    cv2.putText(db_img, text, tuple(np.round(pt*isc+np.array([5, -5])).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 1.6*sc, [255, 0, 0])

                cv2.drawKeypoints(db_img, [cv2.KeyPoint(x*isc, y*isc, 60*sc) for x, y in db_st[:, :2]], db_img, [0, 255, 0], cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                img = np.hstack((db_img, np.ones((img.shape[0], 1, img.shape[2]), dtype='uint8')*255, img))
                cv2.imshow('test', img)
                cv2.waitKey()
    #            plt.figure(1, (16, 12))
    #            plt.imshow(np.flip(img, axis=2))
    #            plt.tight_layout()
    #            plt.show()

        return matches, cols

    def _update_ori(self, db_stars, cols, stars, tol=2e-4):
        """ fine tune orientation based on matches """
        if not MANUAL_ATTITUDE or not self.debug:
            K = self.cam[0].intrinsic_camera_mx(legacy=False)
            iK = self.cam[0].inv_intrinsic_camera_mx(legacy=False)
            I = np.where(np.array([s is not None for s in db_stars]))[0]
            db_pts = np.array([(db_stars[i][cols['ix']], db_stars[i][cols['iy']]) for i in I])
            img_pts = np.array([(stars[i]['x'], stars[i]['y']) for i in I])
            # def solve(Y, X):
            #     L = np.linalg.cholesky(X.T.dot(X) + 1e-8*np.diag(np.ones(3)))
            #     iL = np.linalg.inv(L)
            #     A = np.dot(iL.T, iL.dot(X.T.dot(Y))).T
            #     dx, dy = A[:, 2]
            #     th0 = math.acos(np.clip((A[0, 0]+A[1, 1])/2, -1, 1))
            #     th1 = math.asin(np.clip((A[1, 0]-A[0, 1])/2, -1, 1))
            #     return dx, dy, (th0+th1)/2
#            dx, dy, th = solve(img_pts, np.hstack((db_pts, np.ones((len(db_pts), 1)))))

            def cost_fn(x, Y, X, K, iK):
                dx, dy, th, *dist_coefs = x
                A = np.array([
                    [math.cos(th), -math.sin(th), dx],
                    [math.sin(th),  math.cos(th), dy],
                ])
                Xdot = Camera.distort(X.dot(A.T), dist_coefs, K, iK)
                return tools.pseudo_huber_loss(np.linalg.norm(Y - Xdot, axis=1), delta=3)

            dn = [0, 4, 5, 8, 12][0]  # select the level of detail of the distortion model
            dist_coefs = np.zeros(dn) if self.cam[0].dist_coefs is None else self.cam[0].dist_coefs[:dn]
            dist_coefs = np.pad(dist_coefs, (0, dn - len(dist_coefs)), 'constant')

            x0 = (0, 0, 0, *dist_coefs)
            x, _ = leastsq(cost_fn, x0, args=(img_pts, np.hstack((db_pts, np.ones((len(db_pts), 1)))), K, iK))
            dx, dy, th, *dist_coefs = x

            for c in self.cam:
                c.dist_coefs = dist_coefs
            delta = (dy * math.radians(c.y_fov)/c.height, dx * math.radians(c.x_fov)/c.width, -th)
            #cv2.waitKey()

        else:
            ctrl = {
                27:  [0, 0, 0],   # esc
                115: [1, 0, 0],   # s
                119: [-1, 0, 0],  # w
                100: [0, 1, 0],   # d
                97: [0, -1, 0],   # a
                113: [0, 0, 1],   # q
                101: [0, 0, -1],  # e
            }
            for i in range(10):
                k = cv2.waitKey()
                if k in ctrl:
                    break
            delta = math.radians(0.25) * np.array(ctrl[k])

        self.q = self.q * tools.ypr_to_q(*delta)
        return np.max(np.abs(delta)) < tol

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

        if SHOW_LAB_MEASURES:
            sc = 1024/self.image.shape[1]
            img = cv2.resize(self.image.astype(np.float), None, fx=sc, fy=sc) / np.max(self.image)
            center = np.array(loc) + np.flip(fkernel.shape[:2]) / 2
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

        return LabMeasure(self, mean, std, n)


class Measure(ABC):
    def __init__(self, frame, cam_i, obj_id, du_count, weight=1):
        self.frame = frame
        self.cam_i = cam_i
        self.obj_id = obj_id
        self.du_count = du_count
        self.weight = weight

        # cached value for expected du
        self.c_expected_du = None

    def expected_du(self, pre_sat_gain=1, post_sat_gain=1, qeff_coefs=None, psf_coef=(1, 1, 1)):
        assert False, 'not implemented'


class StarMeasure(Measure):
    def __init__(self, frame, cam_i, obj_id, du_count, t_eff, fe_h, log_g, mag_v, weight=1):
        super(StarMeasure, self).__init__(frame, cam_i, obj_id, du_count, weight=weight)
        self.t_eff = t_eff
        self.fe_h = fe_h
        self.log_g = log_g
        self.mag_v = mag_v

    def expected_du(self, pre_sat_gain=1, post_sat_gain=1, qeff_coefs=None, psf_coef=(1, 1, 1)):
        cam = self.frame.cam[self.cam_i]
        cgain = cam.gain * cam.aperture_area * cam.emp_coef
        fgain = self.frame.gain * self.frame.exposure
        queff_coefs = tuple(cam.qeff_coefs if qeff_coefs is None else qeff_coefs[self.cam_i])
        electrons, _ = Camera.electron_flux_in_sensed_spectrum(queff_coefs, self.t_eff, self.fe_h, self.log_g,
                                                               self.mag_v, cam.lambda_min, cam.lambda_max)

        saturation_val = pre_sat_gain * self.frame.max_val     # would normally be self.frame.max_val, however there's some earlier saturation value
        du = saturation_val * fgain * cgain * electrons
        center_px_val = du / psf_coef[self.cam_i]

        if center_px_val < saturation_val or not STAR_SATURATION_MODELING:
            du = post_sat_gain * psf_coef[self.cam_i] * center_px_val
        else:
            du = post_sat_gain * psf_coef[self.cam_i] * saturation_val * (1 + np.log(center_px_val / saturation_val))
        # du = du ** (1 / gamma) * gain_adj

        self.c_expected_du = du
        return du


class MoonMeasure(Measure):
    """
    Expected measures based on
    [1] "THE SPECTRAL IRRADIANCE OF THE MOON", Hugh H. Kieffer and Thomas C. Stone, The Astronomical Journal, 2005
        https://pdfs.semanticscholar.org/7bd7/e5f41e1113bd47dd616d71cde1fd2d546546.pdf
    """

    ROLO_ABD = np.array((
        (350.0, -2.67511, -1.78539, 0.50612, -0.25578, 0.03744, 0.00981, -0.00322, 0.34185, 0.01441, -0.01602),
        (355.1, -2.71924, -1.74298, 0.44523, -0.23315, 0.03492, 0.01142, -0.00383, 0.33875, 0.01612, -0.00996),
        (405.0, -2.35754, -1.72134, 0.40337, -0.21105, 0.03505, 0.01043, -0.00341, 0.35235, -0.03818, -0.00006),
        (412.3, -2.34185, -1.74337, 0.42156, -0.21512, 0.03141, 0.01364, -0.00472, 0.36591, -0.05902, 0.00080),
        (414.4, -2.43367, -1.72184, 0.43600, -0.22675, 0.03474, 0.01188, -0.00422, 0.35558, -0.03247, -0.00503),
        (441.6, -2.31964, -1.72114, 0.37286, -0.19304, 0.03736, 0.01545, -0.00559, 0.37935, -0.09562, 0.00970),
        (465.8, -2.35085, -1.66538, 0.41802, -0.22541, 0.04274, 0.01127, -0.00439, 0.33450, -0.02546, -0.00484),
        (475.0, -2.28999, -1.63180, 0.36193, -0.20381, 0.04007, 0.01216, -0.00437, 0.33024, -0.03131, 0.00222),
        (486.9, -2.23351, -1.68573, 0.37632, -0.19877, 0.03881, 0.01566, -0.00555, 0.36590, -0.08945, 0.00678),
        (544.0, -2.13864, -1.60613, 0.27886, -0.16426, 0.03833, 0.01189, -0.00390, 0.37190, -0.10629, 0.01428),
        (549.1, -2.10782, -1.66736, 0.41697, -0.22026, 0.03451, 0.01452, -0.00517, 0.36814, -0.09815, -0.00000),
        (553.8, -2.12504, -1.65970, 0.38409, -0.20655, 0.04052, 0.01009, -0.00388, 0.37206, -0.10745, 0.00347),
        (665.1, -1.88914, -1.58096, 0.30477, -0.17908, 0.04415, 0.00983, -0.00389, 0.37141, -0.13514, 0.01248),
        (693.1, -1.89410, -1.58509, 0.28080, -0.16427, 0.04429, 0.00914, -0.00351, 0.39109, -0.17048, 0.01754),
        (703.6, -1.92103, -1.60151, 0.36924, -0.20567, 0.04494, 0.00987, -0.00386, 0.37155, -0.13989, 0.00412),
        (745.3, -1.86896, -1.57522, 0.33712, -0.19415, 0.03967, 0.01318, -0.00464, 0.36888, -0.14828, 0.00958),
        (763.7, -1.85258, -1.47181, 0.14377, -0.11589, 0.04435, 0.02000, -0.00738, 0.39126, -0.16957, 0.03053),
        (774.8, -1.80271, -1.59357, 0.36351, -0.20326, 0.04710, 0.01196, -0.00476, 0.36908, -0.16182, 0.00830),
        (865.3, -1.74561, -1.58482, 0.35009, -0.19569, 0.04142, 0.01612, -0.00550, 0.39200, -0.18837, 0.00978),
        (872.6, -1.76779, -1.60345, 0.37974, -0.20625, 0.04645, 0.01170, -0.00424, 0.39354, -0.19360, 0.00568),
        (882.0, -1.73011, -1.61156, 0.36115, -0.19576, 0.04847, 0.01065, -0.00404, 0.40714, -0.21499, 0.01146),
        (928.4, -1.75981, -1.45395, 0.13780, -0.11254, 0.05000, 0.01476, -0.00513, 0.41900, -0.19963, 0.02940),
        (939.3, -1.76245, -1.49892, 0.07956, -0.07546, 0.05461, 0.01355, -0.00464, 0.47936, -0.29463, 0.04706),
        (942.1, -1.66473, -1.61875, 0.14630, -0.09216, 0.04533, 0.03010, -0.01166, 0.57275, -0.38204, 0.04902),
        (1059.5, -1.59323, -1.71358, 0.50599, -0.25178, 0.04906, 0.03178, -0.01138, 0.48160, -0.29486, 0.00116),
        (1243.2, -1.53594, -1.55214, 0.31479, -0.18178, 0.03965, 0.03009, -0.01123, 0.49040, -0.30970, 0.01237),
        (1538.7, -1.33802, -1.46208, 0.15784, -0.11712, 0.04674, 0.01471, -0.00656, 0.53831, -0.38432, 0.03473),
        (1633.6, -1.34567, -1.46057, 0.23813, -0.15494, 0.03883, 0.02280, -0.00877, 0.54393, -0.37182, 0.01845),
        (1981.5, -1.26203, -1.25138, -0.06569, -0.04005, 0.04157, 0.02036, -0.00772, 0.49099, -0.36092, 0.04707),
        (2126.3, -1.18946, -2.55069, 2.10026, -0.87285, 0.03819, -0.00685, -0.00200, 0.29239, -0.34784, -0.13444),
        (2250.9, -1.04232, -1.46809, 0.43817, -0.24632, 0.04893, 0.00617, -0.00259, 0.38154, -0.28937, -0.01110),
        (2383.6, -1.08403, -1.31032, 0.20323, -0.15863, 0.05955, -0.00940, 0.00083, 0.36134, -0.28408, 0.01010)
    ))
    ROLO_C = (0.00034115, -0.0013425, 0.00095906, 0.00066229)
    ROLO_P = (0.00066229, 4.06054, 12.8802, -30.5858, 16.7498)

    def __init__(self, frame, cam_i, du_count, weight=1):
        super(MoonMeasure, self).__init__(frame, cam_i, ('moon',), du_count, weight=weight)

    @staticmethod
    def _lunar_disk_refl(wlen, g, clat, clon, slon):
        i = list(MoonMeasure.ROLO_ABD[:, 0]).index(wlen)
        abd = MoonMeasure.ROLO_ABD[i, :]
        a, b, c, d, p = abd[1:5], abd[5:8], MoonMeasure.ROLO_C, abd[8:12], MoonMeasure.ROLO_P

        res = sum(ai * g ** i for i, ai in enumerate(a))
        res += sum(bj * slon ** (2 * (j + 1) - 1) for j, bj in enumerate(b))
        res += c[0] * clat + c[1] * clon + c[2] * clat * slon + c[3] * clon * slon
        res += d[0] * math.exp(-g / p[0]) + d[1] * math.exp(-g / p[1]) + d[2] * math.cos((g - p[2]) / p[3])
        return math.exp(res)

    @staticmethod
    def lunar_disk_refl_fn(g, clat, clon, slon):
        I = [MoonMeasure._lunar_disk_refl(x[0], g, clat, clon, slon) for x in MoonMeasure.ROLO_ABD]
        irr = lambda lam: interp1d(MoonMeasure.ROLO_ABD[:, 0], I, kind='linear')(lam * 1e9)

        # lam = np.linspace(350e-9, 1500e-9, 100)
        # plt.plot(lam * 1e9, irr(lam))
        # plt.show()

        return irr

    @staticmethod
    def lunar_disk_irr_fn(lunar_disk_refl_fn, sun_moon_dist, cam_moon_dist):
        if 1:
            moon_solid_angle = 6.4177e-5 * (384.4e6 / cam_moon_dist) ** 2
        else:
            moon_r = 3474200 / 2
            moon_solid_angle = np.pi * moon_r ** 2 / cam_moon_dist ** 2
        sun_solid_angle = Sun.SOLID_ANGLE_AT_1AU * (Sun.AU / sun_moon_dist)**2

        # eq (8) from [1]
        fn = lambda lam: moon_solid_angle * lunar_disk_refl_fn(lam) / math.pi \
                         * sun_solid_angle * Sun.ssr(lam, ssi_table.SSI_2018_12_14)
        return fn

    def expected_du(self, pre_sat_gain=1, post_sat_gain=1, qeff_coefs=None, psf_coef=(1, 1, 1), plot=False):
        f, c = self.frame, self.frame.cam[self.cam_i]
        g, clat, clon, slon = f.phase_angle, f.cam_moon_lat, f.cam_moon_lon, f.sun_moon_lon
        smd, cmd = f.sun_moon_dist, f.cam_moon_dist
        spectrum_fn = MoonMeasure.lunar_disk_irr_fn(MoonMeasure.lunar_disk_refl_fn(g, clat, clon, slon), smd, cmd)

        if plot and self.cam_i == 0:
            lam = np.linspace(c.lambda_min, c.lambda_max, 1000)
            albedo_fn = MoonMeasure.lunar_disk_refl_fn(g, clat, clon, slon)
            moon_sr = 6.4177e-5 * (384.4e6 / cmd) ** 2
            ssi_fn = lambda lam: spectrum_fn(lam)/albedo_fn(lam)/moon_sr * np.pi

            one_fig = False
            plt.rcParams.update({'font.size': 16})

            for i in range(1 if one_fig else 2):
                fig = plt.figure(figsize=[6.4, 4.8])
                if one_fig:
                    axs = fig.subplots(1, 2)
                else:
                    axs = [None]*2
                    axs[i] = fig.subplots(1, 1)

                if i == 0 or one_fig:
                    ax_da = axs[0].twinx()
                    ax_da.plot(lam * 1e9, albedo_fn(lam), color='orange')
                    ax_da.set_ylabel('Disk equivalent albedo', color='orange')
                    ax_da.tick_params(axis='y', labelcolor='orange')
        #            ax_da.title('ROLO-model 2018-12-14 19:00 UTC')

                    axs[0].plot(lam * 1e9, ssi_fn(lam) * 1e-9, color='tab:blue')  # [W/m2/nm]
                    axs[0].set_xlabel('Wave Length [nm]')
                    axs[0].set_ylabel('Spectral Irradiance [W/m2/nm]', color='tab:blue')
                    axs[0].tick_params(axis='y', labelcolor='tab:blue')
        #            axs[0].title('Sunlight SI on 2018-12-14')

                if i == 1 or one_fig:
                    ax_qe = axs[1].twinx()
                    ax_qe.set_ylim([None, 45])
                    ax_qe.set_ylabel('Quantum efficiency [%]')
                    plot_bgr_qeff(self.frame.cam, ax=ax_qe, color=('lightblue', 'lightgreen', 'pink'),
                                  linestyle='dashed', linewidth=1, marker="")
                    axs[1].plot(lam * 1e9, spectrum_fn(lam) * 1e-9, color='tab:blue')  # [W/m2/nm]
                    axs[1].set_xlabel('Wave Length [nm]')
                    axs[1].set_ylabel('Spectral Irradiance [W/m2/nm]', color='tab:blue')
                    axs[1].tick_params(axis='y', labelcolor='tab:blue')
        #            axs[1].title('Moonlight SI on 2018-12-14 19:00 UTC, ROLO-model + SSI')

                plt.tight_layout()
                plt.show()

        cgain = c.gain * c.aperture_area * c.emp_coef
        fgain = f.gain * f.exposure
        queff_coefs = tuple(c.qeff_coefs if qeff_coefs is None else qeff_coefs[self.cam_i])
        electrons, _ = Camera.electron_flux_in_sensed_spectrum_fn(queff_coefs, spectrum_fn, c.lambda_min, c.lambda_max)
        du = f.max_val * pre_sat_gain * post_sat_gain * fgain * cgain * electrons
        self.c_expected_du = du
        return du


class LabMeasure:
    def __init__(self, frame, mean, std, n):
        self.frame = frame
        self.mean = mean
        self.std = std
        self.n = n


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
            # parameterize cam spectral responsivity, frame specific exposure correction
            return (*[qec for c in cams for qec in c.qeff_coefs], *f_gains, *((gain_adj,) if gn else tuple()), *psf_coef)

        def decode(x):
            x = np.abs(x)

            k = 0
            qeff_coefss = []
            for i in range(cn):
                qeff_coefss.append(x[k:k+qn[i]])
                k += qn[i]

            off0, off1 = (4 if gn else 3), 3
            return (qeff_coefss, x[k:len(x)-off0], (x[-off0] if gn else 1), (x[-off1:] if STAR_SATURATION_MODELING else (1, 1, 1)))

        def cost_fun(x, measures, x0, return_details=False, plot=False):
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

                edu = m.expected_du(pre_sat_gain=pre_sat_gain, post_sat_gain=gain_adj, qeff_coefs=c_qeff_coefs, psf_coef=psf_coef)

                if return_details or m.obj_id[0] != 'moon' or m.cam_i not in STAR_IGNORE_MOON_CH:
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
                ax.set_xlabel('Expected [1000 DUs]')
                ax.set_ylabel('Measured [1000 DUs]')
                names = Stars.get_catalog_id(np.unique(list(s[0] for s in obj_ids if s[0] != 'moon')), 'simbad')
                names['moon'] = 'Moon'
                labels = np.array([names[id[0]] for id in obj_ids])
                tools.hover_annotate(fig, ax, sb, labels[band == 0])
                tools.hover_annotate(fig, ax, sg, labels[band == 1])
                tools.hover_annotate(fig, ax, sr, labels[band == 2])
                plt.tight_layout()
                plt.show()

            _, _, gain_adj0, _ = decode(x0)
#            err = tuple(tools.pseudo_huber_loss(STAR_CALIB_HUBER_COEF, (measured_du - expected_du) * 2 / (expected_du + measured_du)) * np.array(weights))
            err = tuple(tools.pseudo_huber_loss(STAR_CALIB_HUBER_COEF, np.log10(expected_du) - np.log10(measured_du)) * np.array(weights))

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

            prior = tuple(STAR_CALIB_PRIOR_WEIGHT**2 * (np.array(x[:n]) - np.array(x0[:n]))**2) \
                if STAR_CALIB_PRIOR_WEIGHT > 0 else tuple()

            err_tuple = lab_dp + err + prior
            return (err_tuple, measured_du, expected_du) if return_details else \
                    err_tuple if opt_method == 'leastsq' else \
                    np.sum(err_tuple)

        if STAR_SATURATION_MODELING:
            psf_coef = STAR_PSF_COEF_TN if cams[0].emp_coef < 1 else STAR_PSF_COEF
        else:
            psf_coef = tuple()

        x0b = encode(cams, f_gains, GENERAL_GAIN_ADJUSTMENT, psf_coef)

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
                res = leastsq(cost_fun, x0, args=(measures, x0b), full_output=True, **self.params)
                x, fval = res[0], np.sum(res[2]['fvec'])
            else:
                res = minimize(cost_fun, x0, args=(measures, x0b), method=opt_method, **self.params)
                x, fval = res.x, res.fun

            results[i] = (x, x0)
            scores[i] = fval if fval > 0 else float('inf')
        timer.stop()

        if len(scores) > 0:
            best = np.argmin(scores)
            print('\nscores: %s' % sorted(scores))
            res, x0 = results[best]
            print('\nbest x0: %s' % (x0,))
            print('best x:  %s' % (res,))
            print('time: %.1fs' % timer.elapsed)
        else:
            res = x0b

        qeff_coefs, f_gains, gain_adj, psf_coef = decode(res)
        err, measured, expected = cost_fun(res, measures, x0b, return_details=True, plot=True)
        return qeff_coefs, f_gains, gain_adj, psf_coef, err, measured, expected


def plot_bgr_qeff(cams, ax=None, color=None, hold=False, **kwargs):
    color = color or ('b', 'g', 'r')
    for col, cam in zip(color, cams):
        cam.plot_qeff_fn(ax=ax, color=col, **kwargs)
    if ax is None and not hold:
        plt.xlabel('Wave Length [nm]')
        plt.ylabel('Quantum efficiency [%]')
        plt.tight_layout()
        plt.show()
    # while not plt.waitforbuttonpress():
    #    pass


def do_bg_img(input, outfile=None):
    bits = 10
    max_val = 2 ** bits - 1
    gamma = GAMMA_OVERRIDE or 2.2
    gamma_break = GAMMA_BREAK_OVERRIDE or 0.10  # 0  # 0.1
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
            if GAMMA_ADJUSTMENT:
                img = ImageProc.adjust_gamma(img, GAMMA_ADJUSTMENT, inverse=True, max_val=max_val)
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
                    write_img(imgs, folder + '.png')
                    imgs.clear()
            elif folder[-4:].lower() in ('.png', '.jpg'):
                imgs.append(cv2.imread(folder))

    if outfile is not None:
        write_img(imgs, outfile)


def merge(old, new):
    for n, v in new.items():
        if n not in old:
            old[n] = []
        old[n].extend(v)


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
            tmp = [array([7.11406102e-02, 1.88573116e-01, 3.57879372e-01, 2.11712854e-01,
       6.74648094e-02, 5.48490265e-02, 4.12507246e-02, 8.67861076e-02,
       4.63874858e-02, 8.59392458e-02, 3.43248506e-03, 2.42848137e-07,
       6.33677240e-05, 2.58960325e-04]), array([1.62839478e-05, 1.22732653e-01, 1.32569634e-01, 2.87502674e-01,
       3.28414592e-01, 1.38869899e-01, 8.55101974e-02, 3.83200070e-02,
       2.82970362e-02, 3.35710523e-02, 4.36661226e-02, 1.50696186e-02,
       1.79069082e-05, 2.56061220e-05]), array([4.51968337e-02, 9.85000036e-02, 3.38110687e-03, 3.69001606e-02,
       4.66199542e-02, 3.25547433e-01, 3.28265779e-01, 2.07627353e-01,
       1.41951262e-01, 5.77236686e-02, 1.02290914e-02, 7.49534067e-06,
       3.77433362e-06, 2.00245623e-04])]
        elif 1:
            # based on stars, no weighting
            tmp = [array([4.24493739e-02, 1.47335623e-01, 3.40773426e-01, 2.07252471e-01,
       6.74203914e-02, 5.38111110e-02, 3.61301608e-02, 7.89194073e-02,
       3.80751330e-02, 7.42875400e-02, 4.32062302e-03, 1.56921402e-08,
       4.92643131e-09, 1.19116641e-03]), array([0.02268899, 0.15809817, 0.11325005, 0.27037449, 0.32800305,
       0.13028288, 0.05137687, 0.0009704 , 0.00501966, 0.01213003,
       0.04284525, 0.04597516, 0.00989027, 0.03579103]), array([4.59842181e-02, 9.39455055e-02, 1.58438456e-03, 3.65986600e-02,
       4.66162686e-02, 3.25682937e-01, 3.28939163e-01, 2.10404115e-01,
       1.48812737e-01, 7.44198447e-02, 2.83898152e-02, 3.58536901e-04,
       9.55504696e-07, 3.19725342e-04])]
        else:
            # based on stars, teff based weighting
            tmp = [array([8.94290274e-02, 1.67909392e-01, 2.99128113e-01, 1.97634109e-01,
       5.72193313e-02, 2.88243251e-02, 6.30241143e-03, 1.14497415e-04,
       9.77582881e-03, 1.48148313e-02, 1.43845189e-02, 1.20971757e-02,
       1.93605985e-03, 5.95808649e-02]), array([1.28027188e-04, 6.81901818e-02, 7.25542690e-02, 2.22804129e-01,
       3.12559848e-01, 1.34514090e-01, 3.36494058e-02, 5.80130821e-02,
       4.94028186e-02, 1.64882729e-04, 1.03730253e-03, 1.07695031e-03,
       1.15575562e-03, 1.87461779e-03]), array([5.19578516e-02, 6.26687643e-02, 2.77121734e-02, 4.97510361e-02,
       4.75843848e-02, 2.60893460e-01, 2.41618357e-01, 1.57117995e-01,
       7.63064815e-02, 9.15206306e-03, 1.22063619e-04, 1.64422073e-04,
       1.05564525e-04, 2.30244743e-04])]

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


def nanocam_gain(gainval: int, debug=False):
    gain = gainval >> 6
    if gain <= 32:
        val = gain
    elif gain <= 64:
        val = (gain >> 1) | 0x40
    else:
        val = (gain - 63) >> 3 << 8 | 0x60

    g0 = 1 + (val >> 6 & 0x01)
    g1 = float(val & 0x3f) / 8
    g2 = 1 + float(val >> 8 & 0x7f) / 8
    actual_gain = g0 * g1 * g2

    if debug:
        print('gain value: %d\nregister value: %d (0x%x)\nactual gain: %.3f (%.3f x %.3f x %.3f)' % (
            gainval, val, val, actual_gain, g0, g1, g2
        ))
    return actual_gain


NANOCAM_GAINS = {g: nanocam_gain(g) for g in (65535, 32768, 16384, 2048, 64)}


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
            use_stars(folder, thumbnail=True)
        elif method == 'fstars':
            use_stars(folder, thumbnail=False)
        elif method == 'moon':
            use_moon_img(folder)
        elif method == 'aurora':
            use_aurora_img(folder)
        elif method == 'lab':
            use_lab_imgs(folder)
        elif method == 'lab_bg':
            use_lab_bg_imgs(folder)
        elif method == 'bg':
            if len(sys.argv) > 3:
                outfile = folder
                input = sys.argv[3:]
            else:
                outfile = None
                input = [folder]
            do_bg_img(input, outfile)

