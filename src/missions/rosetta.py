import math
import warnings

from astropy.time import Time
from astropy import constants as const
from astropy import units

from iotools import objloader
from settings import *
from algo import tools
from algo.model import SystemModel, Asteroid, Camera


class RosettaSystemModel(SystemModel):
    def __init__(self, hi_res_shape_model=False):
        # gives some unnecessary warning about "dubious year" even when trying to ignore it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            min_time = Time('2015-01-01 00:00:00', scale='utc', format='iso')

        super(RosettaSystemModel, self).__init__(
            asteroid=ChuryumovGerasimenko(hi_res_shape_model=hi_res_shape_model),
            camera=Camera(
                1024,       # width in pixels
                1024,       # height in pixels
                5,          # x fov in degrees
                5,          # y fov in degrees
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
    def __init__(self, hi_res_shape_model=False):
        super(ChuryumovGerasimenko, self).__init__()
        self.name = '67P/Churyumov-Gerasimenko'

        # from http://imagearchives.esac.esa.int/index.php?/category/167/start-224
        # self._image_db_path = os.path.join(SCRIPT_DIR, '../data/rosetta-mtp017')
        self.image_db_path = os.path.join(BASE_DIR, 'data/rosetta-mtp006')
        self.target_model_file = os.path.join(BASE_DIR, 'data/67p-17k.obj')
        self.hires_target_model_file = os.path.join(BASE_DIR, 'data/67p-83k-b.obj')
        self.render_smooth_faces = False

        # done using `make-const-noise-shapemodel.py data/67p-83k-b.obj data/67p-17k.obj data/67p-17k.nsm`
        self.constant_noise_shape_model = {
            '' : os.path.join(BASE_DIR, 'data/67p-17k.nsm'),   # same as target_model_file but includes error estimate
            'lo': os.path.join(BASE_DIR, 'data/67p-4k.nsm'),  # 1/4 the vertices
            'hi': os.path.join(BASE_DIR, 'data/67p-1k.nsm'),  # 1/17 the vertices
        }

        sample_image = 'ROS_CAM1_20140808T140718'
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
            # 2014-08-01 - 2014-09-02: 0.4/25
            self.rotation_velocity = 2 * math.pi / 12.4043 / 3600 - math.radians(0.4 / 25) / 24 / 3600  # 0.3754

        # for rotation phase shift, will use as equatorial longitude of
        #   asteroid zero longitude (cheops) at J2000, based on 20150720T165249
        #   papar had 114deg in it
        # for precession cone center (J2000), paper had 69.54, 64.11
        if False:
            tlat, tlon, tpm = 69.54, 64.11, 114
        else:
            # pm: 2014-08-01 - 2014-09-02: -9; 2015-06-13: -143
            tlat, tlon, tpm = 64.11, 69.54, -9

        self.rotation_pm = math.radians(tpm)
        self.axis_latitude, self.axis_longitude = \
            tuple(map(math.radians, (tlat, tlon) if USE_ICRS else \
                tools.equatorial_to_ecliptic(tlat * units.deg, tlon * units.deg)))

        self.precession_cone_radius = math.radians(0.14)  # other paper 0.15+-0.03 deg
        self.precession_period = 10.7 * 24 * 3600  # other paper had 11.5+-0.5 days
        self.precession_pm = math.radians(0.288)

        self.set_defaults()
