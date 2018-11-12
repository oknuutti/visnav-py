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
    def __init__(self, hi_res_shape_model=False, use_narrow_cam=True):
        # gives some unnecessary warning about "dubious year" even when trying to ignore it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            min_time = Time('2023-01-01 00:00:00', scale='utc', format='iso')

        narrow_cam = Camera(
            1024,   # width in pixels
            1024,   # height in pixels
            10,     # x fov in degrees
            10,     # y fov in degrees
        )
        wide_cam = Camera(
            2048,   # width in pixels
            1944,   # height in pixels
            50,     # x fov in degrees
            47.46,  # y fov in degrees
        )
        narrow_cam_limits = (
            0.150,  # min_distance in km
            1.2,    # min_med_distance in km
            5.0,    # max_med_distance in km
            20.0,   # max_distance in km
            45,     # min_elong in deg
            min_time,  # min time instant
        )
        wide_cam_limits = (
            0.150,  # min_distance in km
            0.230,  # min_med_distance in km
            2.0,    # max_med_distance in km
            2.0,    # max_distance in km
            45,     # min_elong in deg
            min_time,  # min time instant
        )

        super(DidymosSystemModel, self).__init__(
            asteroid=DidymosSecondary(hi_res_shape_model=hi_res_shape_model),
            camera=narrow_cam if use_narrow_cam else wide_cam,
            limits=narrow_cam_limits if use_narrow_cam else wide_cam_limits,
        )
        self.mission_id = 'didy' if use_narrow_cam else 'didw'


class DidymosSecondary(Asteroid):
    def __init__(self, hi_res_shape_model=False):
        super(DidymosSecondary, self).__init__()
        self.name = 'Didymos Secondary'

        self.image_db_path = None
        self.target_model_file = os.path.join(BASE_DIR, 'data/ryugu-lo-res.obj')          # use ryugu model for this
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
            (math.radians(de), math.radians(ra)) if USE_ICRS else \
                tools.equatorial_to_ecliptic(de * units.deg, ra * units.deg)

        self.precession_cone_radius = None
        self.precession_period = None
        self.precession_pm = None

        self.set_defaults()
