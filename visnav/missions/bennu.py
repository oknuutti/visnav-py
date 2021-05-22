import math
import warnings

import numpy as np

from astropy.time import Time
from astropy import constants as const
from astropy import units

from visnav.iotools import objloader
from visnav.settings import *
from visnav.algo import tools
from visnav.algo.model import SystemModel, Asteroid, Camera


class BennuSystemModel(SystemModel):
    def __init__(self, hi_res_shape_model=False, res_mult=1.0):
        # gives some unnecessary warning about "dubious year" even when trying to ignore it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            min_time = Time('2019-03-01 00:00:00', scale='utc', format='iso')

        super(BennuSystemModel, self).__init__(
            asteroid=Bennu(hi_res_shape_model=hi_res_shape_model),

            # see https://sbnarchive.psi.edu/pds4/orex/orex.tagcams/document/tagcams_inst_desc.pdf
            # sensor datasheet: https://www.onsemi.com/pdf/datasheet/mt9p031-d.pdf
            # lens reference: http://www.msss.com/brochures/xfov.pdf
            camera=Camera(
                int(2592*res_mult),       # width in pixels
                int(1944*res_mult),       # height in pixels
                44,          # x fov in degrees
                32,          # y fov in degrees
                focal_length=7.7,  # in mm
                f_stop=3.5,
                quantum_eff=0.65,
                px_saturation_e=6456,  # SNR_MAX = 20*log10(sqrt(sat_e)), if SNR_MAX=38.1 dB, then sat_e=6456 e-
                lambda_min=400e-9, lambda_eff=580e-9, lambda_max=700e-9,   # for bandwidth calc
                dark_noise_mu=250, dark_noise_sd=60, readout_noise_sd=6.7,    # noise params (e-/s, e-)
                point_spread_fn=0.4,
                scattering_coef=5e-9,  # affects strength of haze/veil when sun shines on the lens (TODO: just a guess now)
                exclusion_angle_x=60,
                exclusion_angle_y=50,
                emp_coef=1.0,
            ),
            limits=(
                0.7,   # min_distance in km
                1.1,   # min_med_distance in km
                3.5,   # max_med_distance in km #640
                30,    # max_distance in km
                180 - 130,  # min_elong in deg (180 - phase angle)
                min_time,   # min time instant
            )
        )
        self.mission_id = 'orex'


class Bennu(Asteroid):
    # from ast frame (axis: +z, up: -x) to opengl (axis -z, up: +y)
    ast2sc_q = np.quaternion(1, 0, 1, 0).normalized()

    # Details for from book by Hapke, 2012, "Theory of Reflectance and Emittance Spectroscopy"
    # coefficients from light curve analysis https://www.sciencedirect.com/science/article/abs/pii/S0019103515000469
    HAPKE_PARAMS = [
        # J, brightness scaling
        300,

        # th_p, average surface slope (deg), effective roughness, theta hat sub p
        20,

        # w, single scattering albedo (w, omega, SSA), range 0-1
        0.031,

        # b, SPPF asymmetry parameter (sometimes g?),
        #   if single-term HG, range is [-1, 1], from backscattering to isotropic to forward scattering
        #   if two-term HG (c>0), range is [0, 1], from isotropic to scattering in a single direction
        -0.32,

        # c, second HG term for a more complex SPPF. Range [0, 1], from forward scattering to backward scattering.
        0,

        # B_SH0, or B0, amplitude of shadow-hiding opposition effect (shoe). If zero, dont use.
        3.9,

        # hs, or h or k, angular half width of shoe
        math.radians(0.11),

        # B_CB0, amplitude of coherent backscatter opposition effect (cboe). If zero, dont use.
        0,

        # hc, angular half width of cboe
        0.005,

        # extra mode selection, first bit: use K or not
        # NOTE: K generally not in use as phase angle changes so little inside one image and exposure is adjusted to
        #       increase overall brightness
        0,
    ]

    # Tried to get Lunar-Lambert coefficients from https://www.sciencedirect.com/science/article/pii/S0019103520301159
    # but the function is defined very differently there, so too difficult
    LUNAR_LAMBERT_PARAMS = [
        -7.4364e-03, 4.0259e-05, -2.2650e-06, 2.1524e-08, 0, 0,
        0.045 * 5, 0, 0, 0      # geometric albedo
    ]

    def __init__(self, hi_res_shape_model=False):
        super(Bennu, self).__init__()
        self.name = 'Bennu'

        # xtra_hires = os.path.join(DATA_DIR, 'original-shapemodels/bennu.orex.obj')
        xtra_hires = os.path.join(DATA_DIR, 'bennu+tex-98k-v3.obj')
        if os.path.exists(xtra_hires):
            self.hires_target_model_file = xtra_hires
        else:
            raise FileNotFoundError('cant find shape model file %s' % xtra_hires)

        self.image_db_path = os.path.join(DATA_DIR, 'bennu')
        self.target_model_file = xtra_hires
        self.hires_target_model_file_textures = False
        self.render_smooth_faces = False

        self.reflmod_params = {
            1: Bennu.LUNAR_LAMBERT_PARAMS,  # REFLMOD_LUNAR_LAMBERT
            2: Bennu.HAPKE_PARAMS,  # REFLMOD_HAPKE
        }

        # done using `make-const-noise-shapemodel.py data/bennu-98k.obj data/bennu-16k.obj data/bennu-16k.nsm`
        self.constant_noise_shape_model = {
            '':   os.path.join(DATA_DIR, 'bennu+tex-16k.nsm'),  # same as target_model_file but includes error estimate
            'lo': os.path.join(DATA_DIR, 'bennu+tex-4k.nsm'),   # 1/4 the vertices
            'hi': os.path.join(DATA_DIR, 'bennu+tex-1k.nsm'),   # 1/16 the vertices
        }

        sample_image = 'placeholder'
        self.sample_image_file = os.path.join(self.image_db_path, sample_image + '_P.png')
        self.sample_image_meta_file = os.path.join(self.image_db_path, sample_image + '.LBL')

        self.real_shape_model = objloader.ShapeModel(
            fname=self.hires_target_model_file if hi_res_shape_model else self.target_model_file)

        self.max_radius = 285     # in meters, maximum extent of object from asteroid frame coordinate origin
        self.mean_radius = 245

        # for cross section, assume spherical object and 2km radius
        self.mean_cross_section = math.pi * self.mean_radius ** 2

        # epoch for orbital elements, 2011-Jan-01.0 TDB
        self.oe_epoch = Time(2455562.5, format='jd')

        # orbital elements (from https://ssd.jpl.nasa.gov/sbdb.cgi)
        # reference: JPL K154/1 (heliocentric ecliptic J2000)
        self.eccentricity = .203745108478542
        self.semimajor_axis = 1.126391025934071 * const.au
        self.inclination = math.radians(6.034939533607825)
        self.longitude_of_ascending_node = math.radians(2.060867329373625)
        self.argument_of_periapsis = math.radians(66.22306846088361)
        self.mean_anomaly = math.radians(101.7039479473255)

        # other
        self.aphelion = 1.355887687702265 * const.au
        self.perihelion = .8968943641658774 * const.au
        self.orbital_period = 436.6487281348487 * 24 * 3600  # seconds
        # self.true_anomaly = math.radians(145.5260853202137 ??)

        # rotation period from https://en.wikipedia.org/wiki/101955_Bennu
        self.rot_epoch = Time('J2000')
        self.rotation_velocity = 2 * math.pi / 4.296057 / 3600

        # rotation axis dec, ra, unknown offset
        tlat, tlon, tpm = -60.17, 85.65, 0

        self.rotation_pm = math.radians(tpm)
        self.axis_latitude, self.axis_longitude = \
            tuple(map(math.radians, (tlat, tlon) if USE_ICRS else \
                tools.equatorial_to_ecliptic(tlat * units.deg, tlon * units.deg)))

        # unknown and unused for now
        self.precession_cone_radius = math.radians(0)
        self.precession_period = 1 * 24 * 3600
        self.precession_pm = math.radians(0)

        self.set_defaults()
