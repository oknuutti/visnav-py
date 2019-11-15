import math
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

from settings import *
from algo import tools

# Doc for SkyCoord:
#  http://docs.astropy.org/en/stable/coordinates/skycoord.html
#
# C-G P67 coordinate system:
#  http://www.aanda.org/articles/aa/full_html/2015/11/aa26349-15/aa26349-15.html
#
# COPIED FROM https://pds.nasa.gov/ds-view/pds/viewProfile.jsp
#                                     ?dsid=RO-C-NAVCAM-2-ESC3-MTP021-V1.0
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
    config_data = re.sub(r' <(deg|km)>','', config_data)

    config = ConfigParser(converters={'tuple':literal_eval})
    config.read_string(config_data)

    image_time = config.get('meta', 'IMAGE_TIME')

    # from sun to spacecraft, equatorial J2000
    sun_sc_eq_x, sun_sc_eq_y, sun_sc_eq_z = \
            -np.array(config.gettuple('meta', 'SC_SUN_POSITION_VECTOR'))
            
    if USE_ICRS:
        sun_sc_ec_p = np.array([sun_sc_eq_x, sun_sc_eq_y, sun_sc_eq_z])
    else:
        sc = SkyCoord(x=sun_sc_eq_x, y=sun_sc_eq_y, z=sun_sc_eq_z, unit='km',
                frame='icrs', representation_type='cartesian', obstime='J2000')\
                .transform_to('heliocentrictrueecliptic')\
                .represent_as('cartesian')
        sun_sc_ec_p = np.array([sc.x.value, sc.y.value, sc.z.value])
    sun_sc_dist = np.sqrt(np.sum(sun_sc_ec_p**2))
    
    # from spacecraft to asteroid, equatorial J2000
    sc_ast_x, sc_ast_y, sc_ast_z = \
            config.gettuple('meta', 'SC_TARGET_POSITION_VECTOR')

    # from asteroid to spacecraft, asteroid fixed body coordinates
    ast_sc_r = config.getfloat('meta', 'TARGET_CENTER_DISTANCE')
    ast_sc_lat = config.getfloat('meta', 'SUB_SPACECRAFT_LATITUDE')
    ast_sc_lon = config.getfloat('meta', 'SUB_SPACECRAFT_LONGITUDE')

    # spacecraft orientation, equatorial J2000
    sc_rot_ra = config.getfloat('meta', 'RIGHT_ASCENSION')
    sc_rot_dec = config.getfloat('meta', 'DECLINATION')
    sc_rot_cnca = config.getfloat('meta', 'CELESTIAL_NORTH_CLOCK_ANGLE')
    
    solar_elongation = config.getfloat('meta', 'SOLAR_ELONGATION')

    ## set time
    ##
    sm.asteroid.reset_to_defaults()
    half_range = sm.asteroid.rotation_period/2
    timestamp = Time(image_time, scale='utc', format='isot').unix
    sm.time.range = (timestamp - half_range, timestamp + half_range)
    sm.time.value = timestamp
    sm.time.real_value = timestamp

    ## set spacecraft orientation
    ##
    xc, yc, zc = 0, 0, 0
    #xc, yc, zc = -0.283, -0.127, 0     # ROS_CAM1_20150720T113057
    #xc, yc, zc = 0.2699, -0.09, 0  # ROS_CAM1_20150720T165249
    #xc, yc, zc = 0.09, -0.02, 0    # ROS_CAM1_20150720T064939

    if USE_ICRS:
        assert sc_rot_dec+xc<90 and sc_rot_dec+xc>-90, 'bad correction'
        sm.spacecraft_rot = (
            sc_rot_dec+xc,                   # axis lat
            (sc_rot_ra+yc+180)%360 - 180,    # axis lon
            (360-sc_rot_cnca+zc)%360 - 180,  # rotation
        )
    else:
        sc = SkyCoord(ra=sc_rot_ra*units.deg, dec=sc_rot_dec*units.deg,
                      frame='icrs', obstime='J2000')
        sc = sc.transform_to('barycentrictrueecliptic')
        assert sc.lat.value+xc<90 and sc.lat.value+xc>-90, 'bad correction'
        sm.spacecraft_rot = (
            sc.lat.value+xc,                 # axis lat
            (sc.lon.value+yc+180)%360 - 180, # axis lon
            (sc_rot_cnca+zc+180)%360 - 180,  # rotation
        )
        
    sm.real_spacecraft_rot = sm.spacecraft_rot

    ## set spacecraft position
    ##
    if USE_ICRS:
        sc_ast_ec_p = np.array([sc_ast_x, sc_ast_y, sc_ast_z])
    else:
        sc = SkyCoord(x=sc_ast_x, y=sc_ast_y, z=sc_ast_z, unit='km', frame='icrs',
                representation_type='cartesian', obstime='J2000')\
                .transform_to('barycentrictrueecliptic')\
                .represent_as('cartesian')
        sc_ast_ec_p = np.array([sc.x.value, sc.y.value, sc.z.value])

    sm.asteroid.real_position = sun_sc_ec_p + sc_ast_ec_p

    # s/c orientation
    sco = list(map(math.radians, sm.spacecraft_rot))
    scoq = tools.ypr_to_q(*sco)
    
    # project old position to new base vectors
    sc2gl_q = sm.frm_conv_q(sm.SPACECRAFT_FRAME, sm.OPENGL_FRAME)
    scub = tools.q_to_unitbase(scoq * sc2gl_q)
    scub_o = tools.q_to_unitbase(scoq)
    sc_ast_p = scub.dot(sc_ast_ec_p.transpose())

    sm.real_spacecraft_pos = sc_ast_p
    # if USE_IMG_LABEL_FOR_SC_POS:
    #    sm.spacecraft_pos = sc_ast_p
    ##
    ## done setting spacecraft position

    # use calculated asteroid axis as real axis
    sm.asteroid_rotation_from_model()
    sm.real_asteroid_axis = sm.asteroid_axis

    sm.asteroid.real_sc_ast_vertices = sm.sc_asteroid_vertices(real=True)

    if not np.isclose(float(Decimal(sm.time.value) - Decimal(sm.time.real_value)), 0):
        sm.time.real_value = sm.time.value
        if DEBUG:
            print('Strange Python problem where float memory values get corrupted a little in random places of code')

    if False:
        print((''
            + '\nsco:\n%s\n'
            + '\nscoq:\n%s\n'
            + '\nscub_o:\n%s\n'
            + '\nscub:\n%s\n'
            + '\nast_sc_ec_p:\n%s\n'
            + '\nast_sc_p:\n%s\n'
        ) % (
            sm.spacecraft_rot,
            scoq,
            scub,
            scub_o,
            sc_ast_ec_p,
            sc_ast_p,
        ))
    
    if DEBUG:
        lbl_sun_ast_v = (sun_sc_ec_p+sc_ast_ec_p)*1e3
        lbl_se, lbl_dir = tools.solar_elongation(lbl_sun_ast_v, scoq)
        
        m_elong, m_dir = sm.solar_elongation()
        mastp = sm.asteroid.position(sm.time.value)
        print((
            'solar elongation (deg), file: %.1f (%.1f), model: %.1f\n'
            + 'light direction (deg), file: %s, model: %s\n'
            + 'sun-asteroid loc (Gm), file: %s, model: %s\n'
            ) % (
            solar_elongation, math.degrees(lbl_se), math.degrees(m_elong),
            math.degrees(lbl_dir), math.degrees(m_dir),
            lbl_sun_ast_v*1e-9, (mastp)*1e-9,
        ))
        
        sm.save_state('none',printout=True)
    #quit()
    
    ## Impossible to calculate asteroid rotation axis based on given data!!
    ## TODO: Or is it? Can use some help data from model.AsteroidModel?
    ## These should be used: ast_sc_lat, ast_sc_lon
    
    
#FOR TARGET_IMAGE = 'ROS_CAM1_20150720T113057', this seems to be a perfect match:
#system state:
#        ast_x_rot = -74.81 in [-90.00, 90.00]
#        ast_y_rot = -94.82 in [-180.00, 180.00]
#        ast_z_rot = 138.96 in [-180.00, 180.00]
#        time = 1437391848.27 in [1437376452.06, 1437407263.16]
#        x_off = -0.54 in [-4.53, 3.45]
#        x_rot = -29.99 in [-90.00, 90.00]
#        y_off = 2.68 in [-1.34, 6.64]
#        y_rot = 122.54 in [-180.00, 180.00]
#        z_off = -170.19 in [-1280.00, -16.00]
#        z_rot = -103.10 in [-180.00, 180.00]
#
#solar elongation: (94.07914335833404, 87.37274850492905)
#
#asteroid rotation: 104.05
#
#[main]
#ast_x_rot = -73.584
#ast_y_rot = -92.664
#ast_z_rot = 144.216
#time = 1437391848.267516
#x_off = -0.526339523473
#x_rot = -29.628
#y_off = 2.61886006801
#y_rot = 121.824
#z_off = -170.55296469020652
#z_rot = -104.544
#
#[real]
#x_off = 0.545467755596
#y_off = 2.48761450039
#z_off = -170.626950251
