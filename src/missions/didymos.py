import os
import math
import warnings

from astropy.time import Time
from astropy import constants as const
from astropy import units

from algo import tools
from algo.model import SystemModel, Camera, Asteroid
from iotools import objloader

from settings import *


class DidymosSystemModel(SystemModel):
    def __init__(self, target_primary=True, hi_res_shape_model=False, use_narrow_cam=True):
        # gives some unnecessary warning about "dubious year" even when trying to ignore it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            min_time = Time('2023-01-01 00:00:00', scale='utc', format='iso')

        narrow_cam = Camera(
            2048,   # width in pixels
            1944,   # height in pixels
            7.7,    # x fov in degrees  (could be 6 & 5.695, 5.15 & 4.89, 7.7 & 7.309)
            7.309,  # y fov in degrees
        )
        wide_cam = Camera(
            2048,   # width in pixels
            1944,   # height in pixels
            61.5,     # x fov in degrees
            58.38,  # y fov in degrees
        )

        if target_primary:
            if use_narrow_cam:
                mission_id = 'didy1n'
                limits = (
                    6.0,    # min_distance in km
                    7.0,    # min_med_distance in km
                    10.5,   # max_med_distance in km
                    10.5,   # max_distance in km
                    45,     # min_elong in deg
                    min_time,  # min time instant
                )
            else:
                mission_id = 'didy1w'
                limits = (
                    1.1,  # min_distance in km
                    1.1,  # min_med_distance in km
                    10.5,  # max_med_distance in km
                    10.5,  # max_distance in km
                    45,  # min_elong in deg
                    min_time,  # min time instant
                )
        else:
            if use_narrow_cam:
                mission_id = 'didy2n'
                limits = (
                    1.1,    # min_distance in km
                    1.4,    # min_med_distance in km
                    5.3,   # max_med_distance in km
                    5.3,   # max_distance in km
                    45,     # min_elong in deg
                    min_time,  # min time instant
                )
            else:
                mission_id = 'didy2w'
                limits = (
                    0.15,  # min_distance in km
                    0.25,  # min_med_distance in km
                    1.3,  # max_med_distance in km
                    1.3,  # max_distance in km
                    45,  # min_elong in deg
                    min_time,  # min time instant
                )

        super(DidymosSystemModel, self).__init__(
            asteroid=DidymosPrimary(hi_res_shape_model=hi_res_shape_model) if target_primary
                else DidymosSecondary(hi_res_shape_model=hi_res_shape_model),
            camera=narrow_cam if use_narrow_cam else wide_cam,
            limits=limits,
        )
        self.mission_id = mission_id


class DidymosPrimary(Asteroid):
    def __init__(self, hi_res_shape_model=False):
        super(DidymosPrimary, self).__init__()
        self.name = 'Didymos Primary'

        self.image_db_path = None

        # use ryugu model for this, ryugu ~162m diameter, ryugu-big ~772m diameter (Didy2 & Didy1)
        self.target_model_file = os.path.join(BASE_DIR, 'data/ryugu-big-lo-res.obj')
        self.hires_target_model_file = os.path.join(BASE_DIR, 'data/ryugu-big-hi-res.obj')

        self.constant_noise_shape_model = {
            '' : os.path.join(BASE_DIR, 'data/ryugu_baseline.nsm'),   # same as target_model_file but includes error estimate
            'lo': os.path.join(BASE_DIR, 'data/ryugu_lo_noise.nsm'),  # 1/3 the vertices
            'hi': os.path.join(BASE_DIR, 'data/ryugu_hi_noise.nsm'),  # 1/10 the vertices
        }

        self.sample_image_file = None
        self.sample_image_meta_file = None

        self.real_shape_model = objloader.ShapeModel(
            fname=self.hires_target_model_file if hi_res_shape_model else self.target_model_file)
        self.render_smooth_faces = False if hi_res_shape_model else True

        # for cross section, assume spherical object
        self.max_radius = 420      # in meters, maximum extent of object from asteroid frame coordinate origin
        self.mean_radius = 775/2   # in meters
        self.mean_cross_section = math.pi * self.mean_radius ** 2

        # epoch for orbital elements, 2019-Apr-27.0 TDB
        self.oe_epoch = Time(2458600.5, format='jd')

        # orbital elements (from https://ssd.jpl.nasa.gov/sbdb.cgi?sstr=2065803#content)
        # reference: JPL 134 (heliocentric ecliptic J2000)
        self.eccentricity = .3840204901532592
        self.semimajor_axis = 1.644267950023408 * const.au
        self.inclination = math.radians(3.408560852149408)
        self.longitude_of_ascending_node = math.radians(73.20707998527304)
        self.argument_of_periapsis = math.radians(319.3188822767833)
        self.mean_anomaly = math.radians(124.6176776030496)

        # other, not used
        self.aphelion = 2.275700534134692 * const.au
        self.perihelion = 1.012835365912124 * const.au
        self.orbital_period = 770.1180709731267 * 24 * 3600  # seconds
        # self.true_anomaly = math.radians(145.5260853202137 ??)

        # rotation period 11.92h from https://ssd.jpl.nasa.gov/sbdb.cgi?sstr=2065803#content
        self.rot_epoch = Time('J2000')
        self.rotation_velocity = 2 * math.pi / (2.2593 * 3600)  # rad/s

        # asteroid rotation axis in equatorial coordinates
        ra, de, tpm = 310, -84, 0   # ra=lon, de=lat

        self.rotation_pm = math.radians(tpm)
        self.axis_latitude, self.axis_longitude = \
            tuple(map(math.radians, (de, ra) if USE_ICRS else \
                tools.equatorial_to_ecliptic(de * units.deg, ra * units.deg)))

        self.precession_cone_radius = None
        self.precession_period = None
        self.precession_pm = None

        self.set_defaults()


class DidymosSecondary(Asteroid):
    def __init__(self, hi_res_shape_model=False):
        super(DidymosSecondary, self).__init__()
        self.name = 'Didymos Secondary'

        self.image_db_path = None

        # use ryugu model for this, ryugu ~162m diameter, ryugu-big ~772m diameter (Didy2 & Didy1)
        self.target_model_file = os.path.join(BASE_DIR, 'data/ryugu-lo-res.obj')
        self.hires_target_model_file = os.path.join(BASE_DIR, 'data/ryugu-hi-res.obj')

        self.sample_image_file = None
        self.sample_image_meta_file = None

        self.real_shape_model = objloader.ShapeModel(
            fname=self.hires_target_model_file if hi_res_shape_model else self.target_model_file)
        self.render_smooth_faces = False if hi_res_shape_model else True

        # for cross section, assume spherical object
        self.max_radius = 85      # in meters, maximum extent of object from asteroid frame coordinate origin
        self.mean_radius = 163/2  # in meters
        self.mean_cross_section = math.pi * self.mean_radius ** 2

        # epoch for orbital elements, 2019-Apr-27.0 TDB
        self.oe_epoch = Time(2458600.5, format='jd')

        # orbital elements (from https://ssd.jpl.nasa.gov/sbdb.cgi?sstr=2065803#content)
        # reference: JPL 134 (heliocentric ecliptic J2000)
        self.eccentricity = .3840204901532592
        self.semimajor_axis = 1.644267950023408 * const.au
        self.inclination = math.radians(3.408560852149408)
        self.longitude_of_ascending_node = math.radians(73.20707998527304)
        self.argument_of_periapsis = math.radians(319.3188822767833)
        self.mean_anomaly = math.radians(124.6176776030496)

        # other, not used
        self.aphelion = 2.275700534134692 * const.au
        self.perihelion = 1.012835365912124 * const.au
        self.orbital_period = 770.1180709731267 * 24 * 3600  # seconds
        # self.true_anomaly = math.radians(145.5260853202137 ??)

        # rotation period 11.92h from https://ssd.jpl.nasa.gov/sbdb.cgi?sstr=2065803#content
        self.rot_epoch = Time('J2000')
        self.rotation_velocity = 2 * math.pi / (11.92 * 3600)  # rad/s

        # asteroid rotation axis in equatorial coordinates
        ra, de, tpm = 310, -84, 0   # ra=lon, de=lat

        self.rotation_pm = math.radians(tpm)
        self.axis_latitude, self.axis_longitude = \
            tuple(map(math.radians, (de, ra) if USE_ICRS else \
                tools.equatorial_to_ecliptic(de * units.deg, ra * units.deg)))

        self.precession_cone_radius = None
        self.precession_period = None
        self.precession_pm = None

        self.set_defaults()
