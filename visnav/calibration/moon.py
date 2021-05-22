import math

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from visnav.algo.image import ImageProc
from visnav.algo.model import Camera
from visnav.calibration.base import Measure, Frame, plot_bgr_qeff, RAW_IMG_MAX_VALUE
from visnav.calibration import ssi_table
from visnav.render.sun import Sun


SHOW_MEASURES = 0


def use_moon_img(img_file, get_bgr_cam, moon_gain_adj):
    debug = 1
    Frame.MISSING_BG_REMOVE_STRIPES = 0   # force to false

    bgr_cam = get_bgr_cam(thumbnail=False, estimated=0)
    f = Frame.from_file(bgr_cam, img_file, img_file[:-4]+'.lbl', debug=debug)

    measures = f.detect_moon()
    md = [m.du_count for m in measures]
    ed = [m.expected_du(post_sat_gain=moon_gain_adj, plot=debug) for m in measures]
    print('Measured DUs (B, G, R): %s => [G/B=%d%%, R/G=%d%%]\nExpected DUs (B, G, R): %s => [G/B=%d%%, R/G=%d%%]' % (
            [round(m) for m in md], md[1]/md[0]*100, md[2]/md[1]*100,
            [round(m) for m in ed], ed[1]/ed[0]*100, ed[2]/ed[1]*100,
    ))


class MoonMeasure(Measure):
    """
    Expected measures based on
    [1] "THE SPECTRAL IRRADIANCE OF THE MOON", Hugh H. Kieffer and Thomas C. Stone, The Astronomical Journal, 2005
        https://pdfs.semanticscholar.org/7bd7/e5f41e1113bd47dd616d71cde1fd2d546546.pdf
    """

    ROLO_ABD = np.array((
        (300.0, -2.67511, -1.78539, 0.50612, -0.25578, 0.03744, 0.00981, -0.00322, 0.34185, 0.01441, -0.01602),
        # first one (300nm) not real, just copied from 350 nm row
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
                    axs[0].set_xlabel('Wavelength [nm]')
                    axs[0].set_ylabel(r'Spectral Irradiance [$\mathregular{W/m^{2}/nm}$]', color='tab:blue')
                    axs[0].tick_params(axis='y', labelcolor='tab:blue')
        #            axs[0].title('Sunlight SI on 2018-12-14')

                if i == 1 or one_fig:
                    ax_qe = axs[1].twinx()
                    ax_qe.set_ylim([None, 45])
                    ax_qe.set_ylabel('Quantum efficiency [%]')
                    plot_bgr_qeff(self.frame.cam, ax=ax_qe, color=('lightblue', 'lightgreen', 'pink'),
                                  linestyle='dashed', linewidth=1, marker="")
                    axs[1].plot(lam * 1e9, spectrum_fn(lam) * 1e-9, color='tab:blue')  # [W/m2/nm]
                    axs[1].set_xlabel('Wavelength [nm]')
                    axs[1].set_ylabel(r'Spectral Irradiance [$\mathregular{W/m^{2}/nm}$]', color='tab:blue')
                    axs[1].tick_params(axis='y', labelcolor='tab:blue')
        #            axs[1].title('Moonlight SI on 2018-12-14 19:00 UTC, ROLO-model + SSI')

                plt.tight_layout()
                plt.show()

        cgain = c.gain * c.aperture_area * c.emp_coef
        fgain = f.gain * f.exposure
        queff_coefs = tuple(c.qeff_coefs if qeff_coefs is None else qeff_coefs[self.cam_i])
        electrons, _ = Camera.electron_flux_in_sensed_spectrum_fn(queff_coefs, spectrum_fn, c.lambda_min, c.lambda_max)
        du = pre_sat_gain * RAW_IMG_MAX_VALUE * post_sat_gain * fgain * cgain * electrons
        self.c_expected_du = du
        return du


class MoonFrame(Frame):
    def __init__(self, *args, **kwargs):
        super(MoonFrame, self).__init__(*args, **kwargs)

        # about moon images
        self.moon_loc = None
        self.sun_moon_dist = None
        self.cam_moon_dist = None
        self.cam_moon_lat = None
        self.cam_moon_lon = None
        self.sun_moon_lon = None
        self.phase_angle = None

    @classmethod
    def process_metadata(cls, frame, meta):
        if meta.get('moon_loc', False):
            frame.moon_loc = meta.get('moon_loc')
            frame.sun_moon_dist = meta.get('sun_moon_dist')
            frame.cam_moon_dist = meta.get('cam_moon_dist')
            frame.cam_moon_lat = math.radians(meta.get('cam_moon_lat'))
            frame.cam_moon_lon = math.radians(meta.get('cam_moon_lon'))
            frame.sun_moon_lon = math.radians(meta.get('sun_moon_lon'))
            frame.phase_angle = math.radians(meta.get('phase_angle'))

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

        if SHOW_MEASURES:
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
