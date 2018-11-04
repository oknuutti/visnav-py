
import os

SCRIPT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.join(SCRIPT_DIR, '..')
#LOG_DIR = os.path.join(SCRIPT_DIR, '../logs/')
LOG_DIR = 'd:/projects/visnav/logs/'
#CACHE_DIR = os.path.join(SCRIPT_DIR, '../cache/')
CACHE_DIR = 'd:/projects/visnav/cache/'

CAMERA_X_FOV = 5            # in deg
CAMERA_Y_FOV = 5            # in deg
CAMERA_WIDTH = 1024         # in pixels
CAMERA_HEIGHT = 1024        # in pixels
VIEW_WIDTH  = 512  #256 #384 #512 #768
VIEW_HEIGHT = 512  #256 #384 #512 #768

#MAX_TEST_X_RES = 873        #
#MAX_TEST_Y_RES = 873        # visit fails to generate higher images
MAX_TEST_X_RES = 1024        #
MAX_TEST_Y_RES = 1024        # visit fails to generate higher images

MIN_DISTANCE = 16           # in km
MIN_MED_DISTANCE = 64       # in km
MAX_MED_DISTANCE = 480      # in km #640
MAX_DISTANCE = 1280         # in km

# from http://imagearchives.esac.esa.int/index.php?/category/167/start-224
#IMAGE_DB_PATH = os.path.join(SCRIPT_DIR, '../data/rosetta-mtp017')
IMAGE_DB_PATH = os.path.join(SCRIPT_DIR, '../data/rosetta-mtp006')

#TARGET_IMAGE = 'ROS_CAM1_20140801T110718'  # 1
#TARGET_IMAGE = 'ROS_CAM1_20140804T220718'   # 80
#TARGET_IMAGE = 'ROS_CAM1_20140805T110718'   # 90
#TARGET_IMAGE = 'ROS_CAM1_20140805T170718'   
#TARGET_IMAGE = 'ROS_CAM1_20140805T210719'
#TARGET_IMAGE = 'ROS_CAM1_20140810T150718'   # 220
#TARGET_IMAGE = 'ROS_CAM1_20140810T170718'  # 222
#TARGET_IMAGE = 'ROS_CAM1_20140805T190719'  # 222
#TARGET_IMAGE = 'ROS_CAM1_20140829T140853'  #
#TARGET_IMAGE = 'ROS_CAM1_20140830T100433'
#TARGET_IMAGE = 'ROS_CAM1_20140830T101833'
#TARGET_IMAGE = 'ROS_CAM1_20140830T142253' 
TARGET_IMAGE = 'ROS_CAM1_20140824T060433'
#TARGET_IMAGE = 'ROS_CAM1_20140901T221833'
#TARGET_IMAGE = 'ROS_CAM1_20140902T062253'

#TARGET_IMAGE = 'ROS_CAM1_20150613T143701'   # rosetta-mtp017

#TARGET_IMAGE = 'ROS_CAM1_20150720T064939'  # 196.99
#TARGET_IMAGE = 'ROS_CAM1_20150720T113057'  # 141.70
#TARGET_IMAGE = 'ROS_CAM1_20150720T165249'  # 341.43
#TARGET_IMAGE = 'ROS_CAM1_20150720T215423'  # 197.71
#TARGET_IMAGE = 'ROS_CAM1_20150721T025558'  # 43.20
#TARGET_IMAGE = 'ROS_CAM1_20150721T075733'  # 255.96

SHOW_TARGET_IMAGE = True
TARGET_IMAGE_FILE = os.path.join(IMAGE_DB_PATH, TARGET_IMAGE+'_P.png')
TARGET_IMAGE_META_FILE = os.path.join(IMAGE_DB_PATH, TARGET_IMAGE+'.LBL')
TARGET_MODEL_FILE = os.path.join(SCRIPT_DIR, '../data/CSHP_DV_130_01_XLRESb_00200.obj') # _XLRES_, _LORES_
HIRES_TARGET_MODEL_FILE = os.path.join(SCRIPT_DIR, '../data/CSHP_DV_130_01_LORES_00200.obj') # _XLRES_, _LORES_
#HIRES_TARGET_MODEL_FILE = os.path.join(SCRIPT_DIR, '../data/CSHP_DV_130_01_HIRESb_00200.obj') # _XLRES_, _LORES_
#HIRES_TARGET_MODEL_FILE = os.path.join(SCRIPT_DIR, '../data/CSHP_DV_130_01_XLRESb_00200.obj') # _XLRES_, _LORES_
#TARGET_MODEL_FILE = os.path.join(SCRIPT_DIR, '../data/test-ball-hires.obj')
#TARGET_MODEL_FILE = os.path.join(SCRIPT_DIR, '../data/test-ball.obj')

ADD_SHAPE_MODEL_NOISE = False
#SHAPE_MODEL_NOISE_SUPPORT = os.path.join(SCRIPT_DIR, '../data/CSHP_DV_130_01_X3LRES_00200.obj')
#SHAPE_MODEL_NOISE_SUPPORT = os.path.join(SCRIPT_DIR, '../data/test-ball.obj')
SHAPE_MODEL_NOISE_LEN_SC = 1
SHAPE_MODEL_NOISE_LV = 0.01    # low
#SHAPE_MODEL_NOISE_LV = 0.03     # high

## profile algorithm by setting PROFILE to True
##  - profile testing setup by "D:\Program Files\Anaconda3\python" -m cProfile -o profile.out src\batch1.py keypoint 10
## then snakeviz profile.out
PROFILE = False
PROFILE_OUT_FILE = os.path.join(SCRIPT_DIR, '../profile.out')

START_IN_THREAD = False
USE_IMG_LABEL_FOR_SC_POS = True
BATCH_MODE = False
DEBUG = False

USE_ICRS = True # if true, use barycentric equatorial system, else heliocentric ecliptic