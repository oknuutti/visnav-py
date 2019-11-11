import math
import warnings

import numpy as np

from astropy.time import Time
from astropy import constants as const
from astropy import units

from iotools import objloader
from settings import *
from algo import tools
from algo.model import SystemModel, Asteroid, Camera


class RosettaSystemModel(SystemModel):
    def __init__(self, hi_res_shape_model=False, rosetta_batch='mtp006', focused_attenuated=True):
        # gives some unnecessary warning about "dubious year" even when trying to ignore it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            min_time = Time('2015-01-01 00:00:00', scale='utc', format='iso')

        fa = focused_attenuated  # else defocused not attenuated
        super(RosettaSystemModel, self).__init__(
            asteroid=ChuryumovGerasimenko(hi_res_shape_model=hi_res_shape_model, rosetta_batch=rosetta_batch),

            # see https://pds-smallbodies.astro.umd.edu/holdings/ro-c-navcam-2-esc4-mtp023-v1.0/document/ro-sgs-if-0001.pdf
            camera=Camera(
                1024,       # width in pixels
                1024,       # height in pixels
                5,          # x fov in degrees
                5,          # y fov in degrees
                focal_length=152.5,  # in mm
                # sensor_size=(1024*0.013, 1024*0.013),
                aperture=30 if fa else 70,    # attenuated mode (in non-attenuated mode would be 70mm)
                # f_stop=5.1,     # attenuated mode (in non-attenuated mode would be 2.2)
                quantum_eff=0.80,     # from https://www.e2v.com/resources/account/download-datasheet/1427
                px_saturation_e=1e5,  # same source as above
                # gain can be 1.0 (low) or 1.7 (high)   # from https://pds-smallbodies.astro.umd.edu/holdings/ro-c-navcam-2-esc4-mtp023-v1.0/document/ro-sgs-if-0001.pdf
                lambda_min=500e-9, lambda_eff=650e-9, lambda_max=800e-9,   # for bandwidth calc
                dark_noise_mu=250, dark_noise_sd=60, readout_noise_sd=2,    # noise params (e-/s, e-)
                point_spread_fn=0.65 if fa else 0.25,       # 0.50-0.55 for defocused, 0.65-0.70 for focused

                # 30mm aperture and attenuation filter result in an attenuation factor of ∼580 relative to 70mm aperture and no filter
                #   => if aperture is 30, need extra attenuation coef: x*(30/70)**2==1/580 => x==0.009387
                # empirical coef used to tune the synthetic images to match real brightnesses (includes attenuation filter effect)
                #  - based on star brightnesses, note that point_spread_fn affects this a lot
                emp_coef=(0.009387 * 2.3 if fa else 2.3),
            ),
            limits=(
                25,  # min_distance in km
                70,  # min_med_distance in km
                400,  # max_med_distance in km #640
                1250,  # max_distance in km
                40,  # min_elong in deg
                min_time,  # min time instant
            )
        )
        self.mission_id = 'rose'


class ChuryumovGerasimenko(Asteroid):
    # from ast frame (axis: +z, up: -x) to opengl (axis -z, up: +y)
    ast2sc_q = np.quaternion(1, 0, 1, 0).normalized()

    # Details for from book by Hapke, 2012, "Theory of Reflectance and Emittance Spectroscopy"
    # w (SSA), B_SH0 and hs set based on Table 4 (model: Hapke 2012) from Fornasier et al (2015)
    # the rest are optimized using approx-lunar-lambert.py
    HAPKE_PARAMS = [
        # J, brightness scaling
        553.38,

        # th_p, average surface slope (deg), effective roughness, theta hat sub p
        27.07,  # vs 15 in Fornasier

        # w, single scattering albedo (w, omega, SSA), range 0-1
        0.034,

        # b, SPPF asymmetry parameter (sometimes g?),
        #   if single-term HG, range is [-1, 1], from backscattering to isotropic to forward scattering
        #   if two-term HG (c>0), range is [0, 1], from isotropic to scattering in a single direction
        -0.078577,

        # c, second HG term for a more complex SPPF. Range [0, 1], from forward scattering to backward scattering.
        0,

        # B_SH0, or B0, amplitude of shadow-hiding opposition effect (shoe). If zero, dont use.
        2.25,

        # hs, or h or k, angular half width of shoe
        math.radians(0.061),

        # B_CB0, amplitude of coherent backscatter opposition effect (cboe). If zero, dont use.
        0,

        # hc, angular half width of cboe
        0.005,

        # extra mode selection, first bit: use K or not
        # NOTE: K generally not in use as phase angle changes so little inside one image and exposure is adjusted to
        #       increase overall brightness
        1,
    ]

    # Lunar-Lambert coefficients were fitted using iotools/approx-lunar-lambert.py
    LUNAR_LAMBERT_PARAMS = [
        1,
        -7.4364e-03, 4.0259e-05, -2.2650e-06, 2.1524e-08, -5.7964e-11, 7.8620e-01,
        0, 0, 0
    ]

    def __init__(self, hi_res_shape_model=False, rosetta_batch='default'):
        super(ChuryumovGerasimenko, self).__init__()
        self.name = '67P/Churyumov-Gerasimenko'

        # from http://imagearchives.esac.esa.int/index.php?/category/167/start-224
        # self._image_db_path = os.path.join(DATA_DIR, 'rosetta-mtp017')
        self.rosetta_batch = 'mtp006' if rosetta_batch == 'default' else rosetta_batch
        self.image_db_path = os.path.join(DATA_DIR, 'rosetta-'+self.rosetta_batch)
        self.target_model_file = os.path.join(DATA_DIR, '67p-17k.obj')

        xtra_hires = os.path.join(DATA_DIR, 'original-shapemodels/67P_C-G_shape_model_MALMER_2015_11_20-in-km.obj')
        if os.path.exists(xtra_hires):
            self.hires_target_model_file = xtra_hires
        else:
            self.hires_target_model_file = os.path.join(DATA_DIR, '67p-83k-b.obj')
            print(('Using lower quality shape model for synthetic navcam '
                + 'images as highest quality shape model not found: %s') % xtra_hires)

        self.hires_target_model_file_textures = False

        self.render_smooth_faces = False

        self.reflmod_params = {
            1: ChuryumovGerasimenko.LUNAR_LAMBERT_PARAMS,  # REFLMOD_LUNAR_LAMBERT
            2: ChuryumovGerasimenko.HAPKE_PARAMS,  # REFLMOD_HAPKE
        }

        # done using `make-const-noise-shapemodel.py data/67p-83k-b.obj data/67p-17k.obj data/67p-17k.nsm`
        self.constant_noise_shape_model = {
            '' : os.path.join(DATA_DIR, '67p-17k.nsm'),   # same as target_model_file but includes error estimate
            'lo': os.path.join(DATA_DIR, '67p-4k.nsm'),  # 1/4 the vertices
            'hi': os.path.join(DATA_DIR, '67p-1k.nsm'),  # 1/17 the vertices
        }

        sample_image = {
            'mtp003': 'ROS_CAM1_20140531T114923',
            'mtp006': 'ROS_CAM1_20140808T140718',
            'mtp007': 'ROS_CAM1_20140902T113852',          # ROS_CAM1_20140902T113852, ROS_CAM1_20140923T060854
            'mtp017': 'ROS_CAM1_20150630T230217',          # ROS_CAM1_20150603T094509, ROS_CAM1_20150612T230217
            'mtp024': 'ROS_CAM1_20160112T230217',          # ROS_CAM1_20151216T060218, ROS_CAM1_20160112T230217
            'mtp025': 'ROS_CAM1_20160209T231753',          # ROS_CAM1_20160113T060218, ROS_CAM1_20160209T231753
            'mtp026': 'ROS_CAM1_20160301T131104',          # ROS_CAM1_20160210T060423, ROS_CAM1_20160301T131104, ROS_CAM1_20160308T231754
        }[self.rosetta_batch]
        self.sample_image_file = os.path.join(self.image_db_path, sample_image + '_P.png')
        self.sample_image_meta_file = os.path.join(self.image_db_path, sample_image + '.LBL')

        self.real_shape_model = objloader.ShapeModel(
            fname=self.hires_target_model_file if hi_res_shape_model else self.target_model_file)

        self.max_radius = 3000     # in meters, maximum extent of object from asteroid frame coordinate origin
        self.mean_radius = 2000

        # for cross section, assume spherical object and 2km radius
        self.mean_cross_section = math.pi * self.mean_radius ** 2

        # epoch for orbital elements, 2010-Oct-22.0 TDB
        self.oe_epoch = Time(2455491.5, format='jd')

        # orbital elements (from https://ssd.jpl.nasa.gov/sbdb.cgi)
        # reference: JPL K154/1 (heliocentric ecliptic J2000)
        self.eccentricity = .6405823233437267
        self.semimajor_axis = 3.464737502510219 * const.au
        self.inclination = math.radians(7.043680712713979)
        self.longitude_of_ascending_node = math.radians(50.18004588418096)
        self.argument_of_periapsis = math.radians(12.69446409956478)
        self.mean_anomaly = math.radians(91.76808585530111)

        # other
        self.aphelion = 5.684187101644357 * const.au
        self.perihelion = 1.245287903376082 * const.au
        self.orbital_period = 2355.612944885578 * 24 * 3600  # seconds
        # self.true_anomaly = math.radians(145.5260853202137 ??)

        # rotation period
        # from http://www.aanda.org/articles/aa/full_html/2015/11/aa26349-15/aa26349-15.html
        #   - 12.4043h (2014 aug-oct)
        # from http://www.sciencedirect.com/science/article/pii/S0019103516301385?via%3Dihub
        #   - 12.4304h (19 May 2015)
        #   - 12.305h (10 Aug 2015)
        self.rot_epoch = Time('J2000')

        # self.rotation_velocity = 2*math.pi/12.4043/3600 # prograde, in rad/s
        # --- above seems incorrect based on the pics, own estimate
        # based on ROS_CAM1_20150720T165249 - ROS_CAM1_20150721T075733
        if False:
            self.rotation_velocity = 2 * math.pi / 12.4043 / 3600
        else:
            # variable rotation velocity correction in degrees per day
            correction = {
                'default': -0.4/25,   # 2014-08-01 - 2014-09-02
                'mtp003': 0.00,    # 2014-08-01 - 2014-09-02
                'mtp006': 0.006088,    # 2014-08-01 - 2014-09-02
                'mtp007': 0.011987,  # 2014-09-02 - 2014-09-23
                'mtp017': -0.652648,  # 2015-06-03 - 2015-06-30
                'mtp024': 19.170419,  # 2015-12-16 - 2016-01-12
                'mtp025': 19.623067,  # 2016-01-13 - 2016-02-09
                'mtp026': 19.857628,  # 2016-02-10 - 2016-03-08
            }[rosetta_batch]
            self.rotation_velocity = 2 * math.pi / 12.4043 / 3600 + math.radians(correction) / 24 / 3600  # 0.3754

        # for rotation phase shift, will use as equatorial longitude of
        #   asteroid zero longitude (cheops) at J2000, based on 20150720T165249
        #   papar had 114deg in it
        # for precession cone center (J2000), paper had 69.54, 64.11
        if False:
            tlat, tlon, tpm = 69.54, 64.11, 114
        else:
            # rotation phase shift in degrees for different batches
            tpm = {
                'default': -9,      # 2014-08-01 - 2014-09-02
                'mtp003': 0,       # 2014-08-01 - 2014-09-02
                'mtp006': -127.05,       # 2014-08-01 - 2014-09-02
                'mtp007': -158.68,   # 2014-09-02 - 2014-09-23
                'mtp017': -150.09,  # 2015-06-03 - 2015-06-30
                'mtp024': 83.90,  # 2015-12-16 - 2016-01-12
                'mtp025': -46.27,  # 2016-01-13 - 2016-02-09
                'mtp026': 13.40,   # 2016-02-10 - 2016-03-08
            }[rosetta_batch]
            tlat, tlon = 64.11, 69.54

        self.rotation_pm = math.radians(tpm)
        self.axis_latitude, self.axis_longitude = \
            tuple(map(math.radians, (tlat, tlon) if USE_ICRS else \
                tools.equatorial_to_ecliptic(tlat * units.deg, tlon * units.deg)))

        self.precession_cone_radius = math.radians(0.14)  # other paper 0.15+-0.03 deg
        self.precession_period = 10.7 * 24 * 3600  # other paper had 11.5+-0.5 days
        self.precession_pm = math.radians(0.288)

        self.set_defaults()
