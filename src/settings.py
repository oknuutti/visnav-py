
import os

SCRIPT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.join(SCRIPT_DIR, '..')

CAMERA_X_FOV = 5            # in deg
CAMERA_Y_FOV = 5            # in deg
CAMERA_WIDTH = 1024         # in pixels
CAMERA_HEIGHT = 1024        # in pixels
VIEW_WIDTH  = 256  #256 #384 #512 #768
VIEW_HEIGHT = 256  #256 #384 #512 #768

MAX_TEST_X_RES = 873        # 
MAX_TEST_Y_RES = 873        # visit fails to generate higher images

MIN_DISTANCE = 16           # in km
MIN_MED_DISTANCE = 64       # in km
MAX_MED_DISTANCE = 480      # in km #640
MAX_DISTANCE = 1280         # in km

# from http://imagearchives.esac.esa.int/index.php?/category/167/start-224
#TARGET_IMAGE = 'ROS_CAM1_20150720T064939'  # 196.99
TARGET_IMAGE = 'ROS_CAM1_20150720T113057'  # 141.70
#TARGET_IMAGE = 'ROS_CAM1_20150720T165249'  # 341.43
#TARGET_IMAGE = 'ROS_CAM1_20150720T215423'  # 197.71
#TARGET_IMAGE = 'ROS_CAM1_20150721T025558'  # 43.20
#TARGET_IMAGE = 'ROS_CAM1_20150721T075733'  # 255.96
TARGET_IMAGE_FILE = os.path.join(SCRIPT_DIR, '../data/targetimgs/'+TARGET_IMAGE+'_P.png')
TARGET_IMAGE_META_FILE = os.path.join(SCRIPT_DIR, '../data/targetimgs/'+TARGET_IMAGE+'.LBL')
TARGET_MODEL_FILE = os.path.join(SCRIPT_DIR, '../data/CSHP_DV_130_01_XLRES_00200.obj') # _XLRES_, _LORES_
#TARGET_MODEL_FILE = os.path.join(SCRIPT_DIR, '../data/test-ball-hires.obj')

ADD_SHAPE_MODEL_NOISE = True
SHAPE_MODEL_NOISE_SUPPORT = os.path.join(SCRIPT_DIR, '../data/CSHP_DV_130_01_X3LRES_00200.obj')
#SHAPE_MODEL_NOISE_SUPPORT = os.path.join(SCRIPT_DIR, '../data/test-ball.obj')
SHAPE_MODEL_NOISE_LEN_SC = 1
SHAPE_MODEL_NOISE_LV = 0.003

LOG_DIR = os.path.join(SCRIPT_DIR, '../logs/')
VISIT_SCRIPT_PY_FILE = os.path.join(SCRIPT_DIR, 'visit-py-script.py')
VISIT_PORT = 8787

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