import math
from math import radians as rads, degrees as degs
import re
from configparser import ConfigParser
from ast import literal_eval

from decimal import *

getcontext().prec = 6

import numpy as np
import quaternion
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units

from visnav.settings import *
from visnav.algo import tools


# Data main page: https://sbn.psi.edu/pds/resource/hayamica.html
#
# Two sets of (bad) metadata!
#
# Photometry based, supposedly more accurate, used for image lbl files:
#   - Info from https://sbnarchive.psi.edu/pds3/hayabusa/HAY_A_AMICA_3_HAYAMICA_V1_0/data/parameters/paramphot.lbl
#   - Data from https://sbnarchive.psi.edu/pds3/hayabusa/HAY_A_AMICA_3_HAYAMICA_V1_0/data/parameters/paramphot.tab
#
# LIDAR based, at least distances are more reasonable (phot 300m vs lidar 8km)
# However, places asteroid between s/c and sun so that phase angle almost 180 deg instead of close to 0 deg as should
#   - Info from https://sbnarchive.psi.edu/pds3/hayabusa/HAY_A_AMICA_3_HAYAMICA_V1_0/data/parameters/paramlid.lbl
#   - Data from https://sbnarchive.psi.edu/pds3/hayabusa/HAY_A_AMICA_3_HAYAMICA_V1_0/data/parameters/paramlid.tab
#
#
# >>>>
#    ======================================================================
#    Geometry Information - Coordinate System:
#    ======================================================================
#
#    The label files include the following geometric variables:
#    - SC SUN POSITION VECTOR: The vector from the spacecraft to the Sun
#      in equatorial J2000 inertial frame.
#    - SC TARGET POSITION VECTOR: The vector from the spacecraft to the
#      centre of the comet nucleus in equatorial J2000 inertial frame.
#    - SC TARGET VELOCITY VECTOR: The spacecraft to comet nucleus velocity
#      vector in in equatorial J2000 inertial frame.
#    - TARGET CENTER DISTANCE: The distance between the spacecraft and the
#      comet nucleus centre. (Note that also for checkout and stellar
#      calibration images the comet nucleus distance is given here.)
#    - SUB SPACECRAFT LATITUDE and SUB SPACECRAFT LONGITUDE: The latitude
#    and longitude of the sub-spacecraft point derived from the Flight
#    Dynamics body-fixed reference frame implicitly specified by the
#    information provided in the comet attitude file CATT.
#    - RIGHT ASCENSION and DECLINATION: Right Ascension and Declination of
#    the camera boresight direction in equatorial J2000 inertial frame.
#    - CELESTIAL NORTH CLOCK ANGLE: The direction of celestial north at the
#      center of the image - measured from the upward direction,
#      clockwise to the direction toward celestial north.
#    - SOLAR ELONGATION: The angle between the line of sight of observation
#      and the direction of the Sun.
#    - BODY_POLE_CLOCK_ANGLE: specifies the direction of the target body's
#      rotation axis in an image. It is measured from the 'upward' direction,
#      clockwise to the direction of the northern rotational pole as projected
#      into the image plane, assuming the image is displayed as defined by
#      the SAMPLE_DISPLAY_DIRECTION and LINE_DISPLAY_DIRECTION elements.
#    - HAY:BODY_POLE_SUN_ANGLE: specifies the angle between the rotation pole
#      of the target body and the direction from the target body to the sun
#    - HAY:BODY_POLE_ASPECT_ANGLE: Specifies the angle of the rotation pole
#      of the body with respect to the image plane

#
#    All geometric values are calculated for the time t = IMAGE TIME
#    (and not START TIME).


def load_image_meta(src, sm):
    # params given in equatorial J2000 coordinates, details:
    # https://pds.nasa.gov/ds-view/pds/viewProfile.jsp
    #                                   ?dsid=RO-C-NAVCAM-2-ESC3-MTP021-V1.0

    with open(src, 'r') as f:
        config_data = f.read()

    config_data = '[meta]\n' + config_data
    config_data = re.sub(r'^/\*', '#', config_data, flags=re.M)
    config_data = re.sub(r'^\^', '', config_data, flags=re.M)
    config_data = re.sub(r'^(\w+):(\w+)', r'\1__\2', config_data, flags=re.M)
    config_data = re.sub(r'^END\s*$', '', config_data, flags=re.M)
    config_data = re.sub(r'^NOTE\s*=\s*"[^"]*"', '', config_data, flags=re.M)
    config_data = re.sub(r'^OBJECT\s*=\s*.*?END_OBJECT\s*=\s*\w+', '', config_data, flags=re.M|re.S)
    config_data = re.sub(r' <(DEGREE|SECOND|KILOMETER)>', '', config_data)

    config = ConfigParser(converters={'tuple': literal_eval})
    config.read_string(config_data)

    image_time = config.get('meta', 'START_TIME')

    # spacecraft orientation, equatorial J2000
    sc_rot_ra = config.getfloat('meta', 'RIGHT_ASCENSION')
    sc_rot_dec = config.getfloat('meta', 'DECLINATION')
    sc_rot_cnca = config.getfloat('meta', 'CELESTIAL_NORTH_CLOCK_ANGLE')
    sc_igrf_q = tools.ypr_to_q(rads(sc_rot_dec), rads(sc_rot_ra), -rads(sc_rot_cnca))  # same with rosetta lbls also

    # from asteroid to spacecraft, asteroid body fixed coordinates
    # TODO: figure out why FOR SOME REASON distance is given ~30x too close
    ast_sc_dist = config.getfloat('meta', 'TARGET_CENTER_DISTANCE') * 30
    ast_sc_lat = config.getfloat('meta', 'SUB_SPACECRAFT_LATITUDE')
    ast_sc_lon = config.getfloat('meta', 'SUB_SPACECRAFT_LONGITUDE')
    ast_sc_bff_r = tools.spherical2cartesian(rads(ast_sc_lat), rads(ast_sc_lon), ast_sc_dist)

    ast_axis_img_clk_ang = config.getfloat('meta', 'BODY_POLE_CLOCK_ANGLE')
    ast_axis_img_plane_ang = config.getfloat('meta', 'HAY__BODY_POLE_ASPECT_ANGLE')  # what is the use?

    # from sun to spacecraft, equatorial J2000
    ast_sun_dist = config.getfloat('meta', 'TARGET_HELIOCENTRIC_DISTANCE')
    ast_sun_lat = config.getfloat('meta', 'SUB_SOLAR_LATITUDE')
    ast_sun_lon = config.getfloat('meta', 'SUB_SOLAR_LONGITUDE')
    sun_ast_bff_r = -tools.spherical2cartesian(rads(ast_sun_lat), rads(ast_sun_lon), ast_sun_dist)
    sun_sc_bff_r = sun_ast_bff_r + ast_sc_bff_r

    ast_axis_sun_ang = config.getfloat('meta', 'HAY__BODY_POLE_SUN_ANGLE')
    a = config.getfloat('meta', 'SUB_SOLAR_AZIMUTH')  # what is this!?

    # TODO: continue here
    ast_axis_scf_q = tools.ypr_to_q(-rads(ast_sc_lat), -rads(ast_sc_lon), 0)
    # TODO: figure out: how to get roll as some ast_axis_img_clk_ang come from ast_sc_lat?
    ast_rot_scf_q = tools.ypr_to_q(0, 0, -rads(ast_axis_img_clk_ang))
    ast_scf_q = ast_axis_scf_q  #* ast_rot_scf_q

    dec = 90 - ast_sc_lat
    ra = -ast_sc_lon
    if dec > 90:
        dec = 90 + ast_sc_lat
        ra = tools.wrap_degs(ra + 180)

    print('ra: %f, dec: %f, zlra: %f' % (ra, dec, ast_axis_img_clk_ang))

    ast_igrf_q = ast_scf_q * sc_igrf_q
    sun_ast_igrf_r = tools.q_times_v(ast_igrf_q, sun_ast_bff_r)
    ast_sc_igrf_r = tools.q_times_v(ast_igrf_q, ast_sc_bff_r)
    sun_sc_igrf_r = tools.q_times_v(ast_igrf_q, sun_sc_bff_r)

    z_axis = np.array([0, 0, 1])
    x_axis = np.array([1, 0, 0])
    ast_axis_u = tools.q_times_v(ast_igrf_q, z_axis)
    ast_zlon_u = tools.q_times_v(ast_igrf_q, x_axis)
    ast_axis_dec, ast_axis_ra, _ = tools.cartesian2spherical(*ast_axis_u)
    ast_zlon_proj = tools.vector_rejection(ast_zlon_u, z_axis)
    ast_zlon_ra = tools.angle_between_v(ast_zlon_proj, x_axis)
    ast_zlon_ra *= 1 if np.cross(x_axis, ast_zlon_proj).dot(z_axis) > 0 else -1

    # frame where ast zero lat and lon point towards the sun?
    # ast_axis_ra = -ast_sun_lon
    # ast_axis_dec = 90 - ast_sun_lat
    # ast_axis_zero_lon_ra = 0

    arr2str = lambda arr: '[%s]' % ', '.join(['%f' % v for v in arr])

    print('sun_ast_bff_r: %s' % arr2str(sun_ast_bff_r * 1e3))
    print('sun_sc_bff_r: %s' % arr2str(sun_sc_bff_r * 1e3))
    print('ast_sc_bff_r: %s' % arr2str(ast_sc_bff_r * 1e3))
    # TODO: even the light is wrong, should be right based on the sun_ast and sun_sc vectors!!

    print('sun_ast_igrf_r: %s' % arr2str(sun_ast_igrf_r * 1e3))
    print('sun_sc_igrf_r: %s' % arr2str(sun_sc_igrf_r * 1e3))
    print('ast_sc_igrf_r: %s' % arr2str(ast_sc_igrf_r * 1e3))
    print('ast_axis_ra: %f' % degs(ast_axis_ra))
    print('ast_axis_dec: %f' % degs(ast_axis_dec))
    print('ast_zlon_ra: %f' % degs(ast_zlon_ra))

    aa = quaternion.as_rotation_vector(sc_igrf_q)
    angle = np.linalg.norm(aa)
    sc_angleaxis = [angle] + list(aa/angle)
    print('sc_angleaxis [rad]: %s' % arr2str(sc_angleaxis))


def load_image_data(image_filename, table_file):
    cols = ["OBSERVATION_END_MET", "IMAGE_FILENAME", "OBSERVATION_END_TIME", "SPC_X", "SPC_Y", "SPC_Z", "AST_J2_X",
            "AST_J2_Y", "AST_J2_Z", "SPC_J2_X", "SPC_J2_Y", "SPC_J2_Z", "BODY_SURFACE_DISTANCE", "CENTER_LON",
            "CENTER_LAT", "CENTER_PIXEL_RES", "CELESTIAL_N_CLOCK_ANGLE", "BODY_POLE_CLOCK_ANGLE",
            "BODY_POLE_ASPECT_ANGLE", "SUN_DIR_CLOCK_ANGLE", "RIGHT_ASCENSION", "DECLINATION", "SUBSOLAR_LON",
            "SUBSOLAR_LAT", "INCIDENCE_ANGLE", "EMISSION_ANGLE", "PHASE_ANGLE", "SOLAR_ELONGATION", "SUB_SC_LON",
            "SUB_SC_LAT", "BODY_CENTER_DISTANCE", "PIXEL_OFFSET_X", "PIXEL_OFFSET_Y", "AST_SUN_ROT_ANGLE"]
    idx = dict(zip(cols, range(len(cols))))

    with open(table_file, 'r') as fh:
        alldata = [re.split(r'\s+', row)[1:] for row in fh]

    d = None
    for row in alldata:
        if row[idx['IMAGE_FILENAME']] == image_filename:
            d = row
            break

    assert d is not None, 'data for image %s not found' % image_filename

    # spacecraft orientation, equatorial J2000
    sc_rot_ra = float(d[idx['RIGHT_ASCENSION']])
    sc_rot_dec = float(d[idx['DECLINATION']])
    sc_rot_cnca = float(d[idx['CELESTIAL_N_CLOCK_ANGLE']])
    sc_igrf_q = tools.ypr_to_q(rads(sc_rot_dec), rads(sc_rot_ra), -rads(sc_rot_cnca))  # same with rosetta lbls also

    sun_ast_igrf_r = np.array([d[idx['AST_J2_X']], d[idx['AST_J2_Y']], d[idx['AST_J2_Z']]]).astype(np.float)
    sun_sc_igrf_r = np.array([d[idx['SPC_J2_X']], d[idx['SPC_J2_Y']], d[idx['SPC_J2_Z']]]).astype(np.float)

    arr2str = lambda arr: '[%s]' % ', '.join(['%f' % v for v in arr])
    print('sun_ast_igrf_r: %s' % arr2str(sun_ast_igrf_r * 1e3))
    print('sun_sc_igrf_r: %s' % arr2str(sun_sc_igrf_r * 1e3))
    print('ast_sc_igrf_r: %s' % arr2str((sun_sc_igrf_r - sun_ast_igrf_r) * 1e3))
#     print('ast_axis_ra: %f' % degs(ast_axis_ra))
#     print('ast_axis_dec: %f' % degs(ast_axis_dec))
#     print('ast_zlon_ra: %f' % degs(ast_zlon_ra))

    aa = quaternion.as_rotation_vector(sc_igrf_q)
    angle = np.linalg.norm(aa)
    sc_angleaxis = [angle] + list(aa/angle)
    print('sc_angleaxis [rad]: %s' % arr2str(sc_angleaxis))


if __name__ == '__main__':
    if 1:
        load_image_meta(r'C:\projects\sispo\data\targets\st_2422895458_v.lbl', None)
    else:
        load_image_data('st_2422895458_v.fit', r'C:\projects\sispo\data\targets\hayabusa_paramlid.tab')

