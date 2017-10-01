import math
import re
from configparser import ConfigParser
from ast import literal_eval

import numpy as np
import quaternion
from astropy.coordinates import SkyCoord
from astropy.time import Time

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
    sc_x, sc_y, sc_z = \
            -np.array(config.gettuple('meta', 'SC_SUN_POSITION_VECTOR'))

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
    half_range = sm.asteroid.rotation_period/2
    timestamp = Time(image_time, scale='utc', format='isot').unix
    sm.time.range = (timestamp - half_range, timestamp + half_range)
    sm.time.value = timestamp

    ## set spacecraft orientation
    ##
    #xc, yc, zc = 0, 0 ,0
    xc, yc, zc = 0.2699, -0.09, 0 # based on ROS_CAM1_20150720T165249
    #xc, yc, zc = 0.09, -0.02, 0 # based on ROS_CAM1_20150720T064939

    sc = SkyCoord(ra=sc_rot_ra, dec=sc_rot_dec, unit='deg',
            frame='icrs', obstime='J2000')\
            .transform_to('barycentrictrueecliptic')
    sm.x_rot.value = sc.lat.value+xc       # axis lat
    sm.y_rot.value = (sc.lon.value+yc+180)%360 - 180 # axis lon
    sm.z_rot.value = -((sc_rot_cnca-zc+180)%360 - 180) # rotation

    sco = list(map(math.radians,
            (sm.x_rot.value, sm.y_rot.value, sm.z_rot.value)))


    ## set spacecraft position
    ##
    sc = SkyCoord(x=sc_ast_x, y=sc_ast_y, z=sc_ast_z, frame='icrs',
            representation='cartesian', obstime='J2000')\
            .transform_to('barycentrictrueecliptic')\
            .represent_as('cartesian')
    sc_ast_ec_p = np.array([sc.x.value, sc.y.value, sc.z.value])

    # s/c orientation
    scoq = tools.ypr_to_q(*sco)

    # project old position to new base vectors
    scub = tools.q_to_unitbase(scoq)
    ast_sc_p = - scub.dot(sc_ast_ec_p.transpose())

    # rotate these coords to default opengl -z aligned view
    ast_sc_p = tuple(ast_sc_p[((1, 2, 0),)])

    if False:
        print((''
            + '\nsco:\n%s\n'
            + '\nscoq:\n%s\n'
            + '\nast_sc_ec_p:\n%s\n'
            + '\nast_sc_p:\n%s\n'
        ) % (
            list(map(math.degrees, sco)),
            scoq,
            sc_ast_ec_p,
            ast_sc_p,
        ))

    sm.real_spacecraft_pos = ast_sc_p
    if USE_IMG_LABEL_FOR_SC_POS:
        sm.spacecraft_pos = ast_sc_p
    ##
    ## done setting spacecraft position

    ## impossible to calculate asteroid rotation axis based on given data!!
    ## ast_sc_lat, ast_sc_lon
    