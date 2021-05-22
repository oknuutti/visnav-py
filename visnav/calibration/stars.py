import math
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.optimize import leastsq

from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture
from scipy import stats

from visnav.algo import tools
from visnav.algo.image import ImageProc
from visnav.algo.model import Camera
from visnav.calibration.base import Measure, Frame, merge, RAW_IMG_MAX_VALUE
from visnav.calibration.spectrum import get_star_spectrum, sensed_electron_flux_star_spectrum
from visnav.render.stars import Stars
from visnav.render.sun import Sun


DEBUG_EXTRACTION = 0
DEBUG_CHANNELS = 0
DEBUG_MATCHING = 0  # show 1=tycho, 2=t_eff, 3=mag_v
MANUAL_ATTITUDE = 0
SHOW_MEASURES = 0
STAR_SPECTRA_PATH = r'C:\projects\s100imgs\spectra'


class StarMeasure(Measure):
    def __init__(self, frame, cam_i, obj_id, du_count, t_eff, fe_h, log_g, mag_v, ixy, weight=1):
        super(StarMeasure, self).__init__(frame, cam_i, obj_id, du_count, weight=weight)
        self.t_eff = t_eff
        self.fe_h = fe_h
        self.log_g = log_g
        self.mag_v = mag_v
        self.ixy = ixy
        simbad = Stars.get_property_by_id(self.obj_id[0], 'simbad')
        self.bayer = simbad.strip(' *').lower().replace(' ', '_').replace('alf_', 'alp_')

        self.c_unsat_du = None
        self.c_px_du_sat = None

    def expected_du(self, pre_sat_gain=1, post_sat_gain=1, qeff_coefs=None, psf_coef=(1, 1, 1)):
        cam = self.frame.cam[self.cam_i]
        cgain = cam.gain * cam.aperture_area * cam.emp_coef
        fgain = self.frame.gain * self.frame.exposure
        queff_coefs = tuple(cam.qeff_coefs if qeff_coefs is None else qeff_coefs[self.cam_i])

        if 0:
            p_elec, _ = Camera.electron_flux_in_sensed_spectrum(queff_coefs, self.t_eff, self.fe_h, self.log_g,
                                                                   self.mag_v, cam.lambda_min, cam.lambda_max)
        if 1:
            gomos_mag_v = self.mag_v  # if self.bayer == 'alp_ori' else None
            electrons = sensed_electron_flux_star_spectrum(STAR_SPECTRA_PATH, self.bayer, self.mag_v, self.t_eff,
                                                           self.log_g, self.fe_h, cam.lambda_min, cam.lambda_max,
                                                           queff_coefs, gomos_mag_v)

        if 0:  #self.bayer == 'alp_ori':
            spectrum_fn0 = Stars.synthetic_radiation_fn(self.t_eff, self.fe_h, self.log_g, mag_v=self.mag_v)
            spectrum_fn0b = Stars.synthetic_radiation_fn(self.t_eff, self.fe_h, self.log_g, mag_v=self.mag_v,
                                                         model='ck04models',
                                                         lam_min=cam.lambda_min - 10e-9, lam_max=cam.lambda_max + 10e-9)
            spectrum_fn1 = get_star_spectrum(STAR_SPECTRA_PATH, self.bayer, self.mag_v, self.t_eff, self.log_g, self.fe_h,
                                             cam.lambda_min, cam.lambda_max, gomos_mag_v)
            lams = np.linspace(cam.lambda_min, cam.lambda_max, 3000)
            plt.plot(lams, spectrum_fn0(lams))
            plt.plot(lams, spectrum_fn0b(lams))
            plt.plot(lams, spectrum_fn1(lams))
            plt.title(self.bayer)
            plt.show()

        du = pre_sat_gain * RAW_IMG_MAX_VALUE * fgain * cgain * electrons
        self.c_unsat_du = du

        if StarFrame.STAR_SATURATION_MODELING == StarFrame.STAR_SATURATION_MODEL_MOTION:
            psf_coef = tuple(psf_coef) if StarFrame.STAR_SATURATION_MULTI_KERNEL else \
                ((psf_coef[self.cam_i],) if len(psf_coef) == 3 else tuple(psf_coef))
            du, self.c_px_du_sat = self._motion_kernel_psf_saturation(du, psf_coef, True)
        elif StarFrame.STAR_SATURATION_MODELING == StarFrame.STAR_SATURATION_MODEL_ANALYTICAL:
            du = self._analytical_psf_saturation(du, psf_coef[self.cam_i])
        else:
            assert StarFrame.STAR_SATURATION_MODELING == StarFrame.STAR_SATURATION_MODEL_IDEAL
            # do nothing

        du *= post_sat_gain
        self.c_expected_du = du
        return du

    def _analytical_psf_saturation(self, du, psf_sd):
        psf_coef = psf_sd**2 * 2 * np.pi
        center_px_val = du / psf_coef
        if center_px_val < self.frame.max_signal:
            sat_du = psf_coef * self.frame.max_signal * (1 + np.log(center_px_val / self.frame.max_signal))
        else:
            sat_du = du
        return sat_du

    def _motion_kernel_psf_saturation(self, du, psf_sd, get_px_du_sat=False):
        read_sd = None
        if len(psf_sd) in (2, 4):
            psf_sd, read_sd = psf_sd[:-1], psf_sd[-1]

        line_xy = self.frame.motion_in_px(self.ixy)
        mb_psf = self._get_motion_kernel(psf_sd, line_xy)
        px_du_sat = np.clip(mb_psf * du, 0, self.frame.max_signal)
        if read_sd:
            noise = trunc_gaussian_shift(px_du_sat, read_sd * self.frame.max_signal, self.frame.max_signal)
            if 1:
                px_du_sat = np.clip(px_du_sat - noise, 0, self.frame.max_signal)
                du_sat = np.sum(px_du_sat)
            else:
                noise = np.random.normal(0, read_sd * saturation_val, px_du_sat.shape)
                noise = cv2.filter2D(noise, cv2.CV_64F, ImageProc.gkern2d(5, 1.0))
                px_du_sat = np.clip(px_du_sat + noise, 0, saturation_val)
        else:
            du_sat = np.sum(px_du_sat)
        return (du_sat,) + ((px_du_sat,) if get_px_du_sat else tuple())

    @staticmethod
    @lru_cache(maxsize=20)
    def _get_motion_kernel(psf_sd, line_xy):
        if len(psf_sd) == 3:
            sd1, w, sd2 = psf_sd
        else:
            sd1, w, sd2 = psf_sd[0], 0, 0

        psf_hw = math.ceil(max(sd1 * 3, sd2 * 2))
        psf_fw = 1 + 2 * psf_hw
        psf = ImageProc.gkern2d(psf_fw, sd1) + (0 if w == 0 else w * ImageProc.gkern2d(psf_fw, sd2))

        line_xy = np.array(line_xy)
        line = np.zeros(np.ceil(np.abs(np.flip(line_xy))).astype(np.int) + psf_fw)

        cnt = np.flip(line.shape)/2
        start = tuple(np.round(cnt - line_xy/2).astype(np.int))
        end = tuple(np.round(cnt + line_xy/2).astype(np.int))
        cv2.line(line, start, end, color=1.0, thickness=1, lineType=cv2.LINE_AA)

        mb_psf = cv2.filter2D(line, cv2.CV_64F, psf)
        mb_psf /= np.sum(mb_psf)  # normalize to one
        return mb_psf


def trunc_gaussian_shift(mean, sd, upper_limit):
    # from https://en.wikipedia.org/wiki/Truncated_normal_distribution
    beta = (upper_limit - mean) / sd
    shift = sd * stats.norm.pdf(beta) / stats.norm.cdf(beta)
    return shift


class StarFrame(Frame):
    (
        STAR_SATURATION_MODEL_IDEAL,
        STAR_SATURATION_MODEL_ANALYTICAL,
        STAR_SATURATION_MODEL_MOTION,
    ) = range(3)
    STAR_SATURATION_MODELING = STAR_SATURATION_MODEL_MOTION
    STAR_SATURATION_MULTI_KERNEL = False

    def __init__(self, *args, q=None, override_star_data=None, **kwargs):
        super(StarFrame, self).__init__(*args, **kwargs)

        def detect(imgc):
            _, imgc = cv2.threshold(imgc, 560, 255, type=cv2.THRESH_BINARY)
            imgc = cv2.dilate(imgc, np.ones((3, 3)))
            imgc = cv2.erode(imgc, np.ones((3, 3)), iterations=2)
            imgc = cv2.dilate(imgc, np.ones((3, 3)))
            return imgc

        b_mask = detect(self.image[:, :, 0])
        g_mask = detect(self.image[:, :, 1])
        r_mask = detect(self.image[:, :, 2])

        b_mean = np.mean(self.image[:, :, 0][b_mask > 0])
        g_mean = np.mean(self.image[:, :, 1][g_mask > 0])
        r_mean = np.mean(self.image[:, :, 2][r_mask > 0])

        bg_mean = np.median(self.image)

        bn, gn, rn = np.sum(b_mask > 0), np.sum(g_mask > 0), np.sum(r_mask > 0)
        sat_mean = (bn * b_mean + gn * g_mean + rn * r_mean) / (bn + gn + rn)
        self.max_signal = sat_mean - bg_mean

        self.override_star_data = override_star_data or {}
        self.stars = []
        self.q = q
        self.mb_cnt_ixy = None
        self.mb_angle = None

    @classmethod
    def process_metadata(cls, frame, meta):
        if meta.get('dec', False):
            frame.set_orientation(dec_ra_pa=(meta.get('dec'), meta.get('ra'), meta.get('pa', 0)))
        if meta.get('mb_cnt_ixy', False) is not False:
            frame.mb_cnt_ixy = meta.get('mb_cnt_ixy')
            frame.mb_angle = math.radians(meta.get('mb_angle'))

    def motion_in_px(self, ixy):
        r = np.linalg.norm(np.array(ixy) - self.mb_cnt_ixy)
        x, y = np.array(ixy) - self.mb_cnt_ixy
        line_dir = np.arctan2(-y, x) - np.pi/2

        # (2 * np.pi * r) * (self.mb_angle / 2 / np.pi) -- full circle perimeter * ratio of the whole circle
        line_len = r * self.mb_angle

        x, y = line_len * np.cos(line_dir), -line_len * np.sin(line_dir)
        return x, y

    def set_orientation(self, q=None, angleaxis=None, dec_ra_pa=None):
        if q is not None:
            self.q = q
        elif angleaxis is not None:
            self.q = tools.angleaxis_to_q(angleaxis)
        else:
            assert dec_ra_pa is not None, 'no orientation given'
            dec, ra, pa = map(math.radians, dec_ra_pa)
            self.q = tools.ypr_to_q(dec, ra, pa)

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
                                                t_eff, fe_h, log_g, m[cols['mag_v']], (stars[i]['x'], stars[i]['y'])))
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
            if st[cols['id']] in self.override_star_data:
                for f in ('mag_v', 'mag_b', 't_eff', 'log_g', 'fe_h'):
                    od = self.override_star_data[st[cols['id']]]
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

        M = (np.abs(np.repeat(np.expand_dims(img_st[:, 2:3], axis=0), len(db_st), axis=0)
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
