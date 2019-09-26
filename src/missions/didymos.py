import os
import math
import warnings

import numpy as np

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
                    3.5,    # min_distance in km
                    6.5,    # min_med_distance in km
                    10.5,   # max_med_distance in km
                    10.5,   # max_distance in km
                    45,     # min_elong in deg
                    min_time,  # min time instant
                )
            else:
                mission_id = 'didy1w'
                limits = (
                    1.0,  # min_distance in km
                    1.1,  # min_med_distance in km
                    6.0,  # max_med_distance in km
                    7.0,  # max_distance in km
                    45,  # min_elong in deg
                    min_time,  # min time instant
                )
        else:
            if use_narrow_cam:
                mission_id = 'didy2n'
                limits = (
                    1.0,    # min_distance in km
                    1.65,    # min_med_distance in km
                    5.3,   # max_med_distance in km
                    7.0,   # max_distance in km
                    45,     # min_elong in deg
                    min_time,  # min time instant
                )
            else:
                mission_id = 'didy2w'
                limits = (
                    0.135,  # min_distance in km
                    0.28,  # min_med_distance in km
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
        self.sc_model_file = os.path.join(BASE_DIR, 'data/apex-x1-2019-05-28.obj')


class DidymosPrimary(Asteroid):
    # from ast frame (axis: +y, up: +z) to spacecraft coords (axis: +x, up: +z)
    ast2sc_q = np.quaternion(1, 1, 0, 0).normalized()

    # based on itokawa, Tatsumi et al 2018
    # "Regolith Properties on the S-Type Asteroid Itokawa Estimated from Photometrical Measurements"
    HAPKE_PARAMS = [
        # J, brightness scaling
        15,

        # th_p, average surface slope (deg), effective roughness, theta hat sub p
        40,

        # w, single scattering albedo (w, omega, SSA), range 0-1
        0.57,

        # b, SPPF asymmetry parameter (sometimes g?),
        #   if single-term HG, range is [-1, 1], from backscattering to isotropic to forward scattering
        #   if two-term HG (c>0), range is [0, 1], from isotropic to scattering in a single direction
        0.35,

        # c, second HG term for a more complex SPPF. Range [0, 1], from forward scattering to backward scattering.
        0.56,

        # B_SH0, or B0, amplitude of shadow-hiding opposition effect (shoe). If zero, dont use.
        0.98,

        # hs, or h or k, angular half width of shoe (rad)
        math.radians(0.05),

        # B_CB0, amplitude of coherent backscatter opposition effect (cboe). If zero, dont use.
        0,

        # hc, angular half width of cboe
        0.005,

        # extra mode selection, first bit: use K or not
        # NOTE: K generally not in use as phase angle changes so little inside one image and exposure is adjusted to
        #       increase overall brightness
        0,
    ]

    # fitted to above hapke params using approx-lunar-lambert.py: match_ll_with_hapke
    LUNAR_LAMBERT_PARAMS = [
        1.0,
        -3.2261e-02, 8.4991e-05, 1.4809e-06, -7.7885e-09, 8.7950e-12, 7.7132e-01,
        0, 0, 0
    ]

    def __init__(self, hi_res_shape_model=False):
        super(DidymosPrimary, self).__init__()
        self.name = 'Didymos Primary'

        self.image_db_path = None

        # use ryugu model for this, ryugu ~162m diameter, ryugu-big ~772m diameter (Didy2 & Didy1)
        self.target_model_file = os.path.join(BASE_DIR, 'data/ryugu+tex-d1-16k.obj')
        self.hires_target_model_file = os.path.join(BASE_DIR, 'data/ryugu+tex-d1-400k.obj')
        self.hires_target_model_file_textures = True

        self.constant_noise_shape_model = {
            '' : os.path.join(BASE_DIR, 'data/ryugu+tex-d1-16k.nsm'),  # same as target_model_file but includes error estimate
            'lo': os.path.join(BASE_DIR, 'data/ryugu+tex-d1-4k.nsm'),  # 1/4 the vertices
            'hi': os.path.join(BASE_DIR, 'data/ryugu+tex-d1-1k.nsm'),  # 1/16 the vertices
        }

        self.sample_image_file = None
        self.sample_image_meta_file = None

        self.real_shape_model = objloader.ShapeModel(
            fname=self.hires_target_model_file if hi_res_shape_model else self.target_model_file)
        self.render_smooth_faces = False if hi_res_shape_model else True

        self.reflmod_params = {
            1: DidymosPrimary.LUNAR_LAMBERT_PARAMS, # REFLMOD_LUNAR_LAMBERT
            2: DidymosPrimary.HAPKE_PARAMS, # REFLMOD_HAPKE
        }

        # for cross section, assume spherical object
        self.max_radius = 470      # in meters, maximum extent of object from asteroid frame coordinate origin
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
    ast2sc_q = DidymosPrimary.ast2sc_q

    def __init__(self, hi_res_shape_model=False):
        super(DidymosSecondary, self).__init__()
        self.name = 'Didymos Secondary'

        self.image_db_path = None

        # use ryugu model for this, ryugu ~162m diameter, ryugu-big ~772m diameter (Didy2 & Didy1)
        self.target_model_file = os.path.join(BASE_DIR, 'data/ryugu+tex-d2-16k.obj')
        self.hires_target_model_file = os.path.join(BASE_DIR, 'data/ryugu+tex-d2-400k.obj')

        self.constant_noise_shape_model = {
            '' : os.path.join(BASE_DIR, 'data/ryugu+tex-d2-16k.nsm'),  # same as target_model_file but includes error estimate
            'lo': os.path.join(BASE_DIR, 'data/ryugu+tex-d2-4k.nsm'),  # 1/4 the vertices
            'hi': os.path.join(BASE_DIR, 'data/ryugu+tex-d2-1k.nsm'),  # 1/16 the vertices
        }

        self.sample_image_file = None
        self.sample_image_meta_file = None

        self.real_shape_model = objloader.ShapeModel(
            fname=self.hires_target_model_file if hi_res_shape_model else self.target_model_file)
        self.render_smooth_faces = False if hi_res_shape_model else True

        self.reflmod_params = {
            1: DidymosPrimary.LUNAR_LAMBERT_PARAMS, # REFLMOD_LUNAR_LAMBERT
            2: DidymosPrimary.HAPKE_PARAMS, # REFLMOD_HAPKE
        }

        # for cross section, assume spherical object
        self.max_radius = 105     # in meters, maximum extent of object from asteroid frame coordinate origin
        self.mean_radius = 163/2  # in meters, dims = [206, 158, 132]
        self.mean_cross_section = math.pi * self.mean_radius ** 2

        # Distance = 1180 (1160-1220) m
        # L1, L2 = 999.3, 1354.4 m

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
