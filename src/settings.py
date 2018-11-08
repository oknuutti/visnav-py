
import os

SCRIPT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.join(SCRIPT_DIR, '..')
#LOG_DIR = os.path.join(SCRIPT_DIR, '../logs/')
LOG_DIR = 'd:/projects/visnav/logs/'
#CACHE_DIR = os.path.join(SCRIPT_DIR, '../cache/')
CACHE_DIR = 'd:/projects/visnav/cache/'

# render this size synthetic images on the s/c
VIEW_WIDTH  = 512  #256 #384 #512 #768
VIEW_HEIGHT = 512  #256 #384 #512 #768

# if your machine cant for some reason generate images with the camera resolution
MAX_TEST_X_RES = 1024
MAX_TEST_Y_RES = 1024

# DEPRACATED (used only by e.g. visnav.py) >>
# SHOW_TARGET_IMAGE = True
# TARGET_IMAGE_FILE = os.path.join(IMAGE_DB_PATH, TARGET_IMAGE+'_P.png')
# TARGET_IMAGE_META_FILE = os.path.join(IMAGE_DB_PATH, TARGET_IMAGE+'.LBL')
# SHAPE_MODEL_NOISE_SUPPORT = os.path.join(SCRIPT_DIR, '../data/CSHP_DV_130_01_X3LRES_00200.obj')
# SHAPE_MODEL_NOISE_SUPPORT = os.path.join(SCRIPT_DIR, '../data/test-ball.obj')
# USE_IMG_LABEL_FOR_SC_POS = True

ADD_SHAPE_MODEL_NOISE = False
SHAPE_MODEL_NOISE_LEN_SC = 1
SHAPE_MODEL_NOISE_LV = 0.01    # low
#SHAPE_MODEL_NOISE_LV = 0.03     # high

## profile algorithm by setting PROFILE to True
##  - profile testing setup by "D:\Program Files\Anaconda3\python" -m cProfile -o profile.out src\batch1.py keypoint 10
## then snakeviz profile.out
PROFILE = False
PROFILE_OUT_FILE = os.path.join(SCRIPT_DIR, '../profile.out')

START_IN_THREAD = False
BATCH_MODE = False          # used only in relation to visnav.py
DEBUG = False

USE_ICRS = True  # if true, use barycentric equatorial system, else heliocentric ecliptic