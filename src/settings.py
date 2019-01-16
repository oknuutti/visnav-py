
import os

SCRIPT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.join(SCRIPT_DIR, '..')
#LOG_DIR = os.path.join(SCRIPT_DIR, '../logs/')
LOG_DIR = 'd:/projects/visnav/logs/'
#CACHE_DIR = os.path.join(SCRIPT_DIR, '../cache/')
CACHE_DIR = 'd:/projects/visnav/cache/'

USE_ICRS = True  # if true, use barycentric equatorial system, else heliocentric ecliptic

# if your machine cant for some reason generate images with the camera resolution (not in use)
MAX_TEST_X_RES = 1024
MAX_TEST_Y_RES = 1024

# render this size synthetic images on the s/c, height is proportional to cam height/width
VIEW_WIDTH = 512

SHAPE_MODEL_NOISE_LEN_SC = 0.33
SHAPE_MODEL_NOISE_LV = {
    'lo': 0.0003,   # was 0.005, resulted in very spiky 3d models, ~5% err
    'hi': 0.0012,    # was 0.020, resulted in very spiky 3d models. ~10% err
}

## profile algorithm by setting PROFILE to True
##  - profile testing setup by "D:\Program Files\Anaconda3\python" -m cProfile -o profile.out src\batch1.py keypoint 10
## then snakeviz profile.out
PROFILE = False
PROFILE_OUT_FILE = os.path.join(SCRIPT_DIR, '../profile.out')

START_IN_THREAD = False
BATCH_MODE = True          # used only in relation to visnav.py
DEBUG = False

ONLY_POPULATE_CACHE = True  # work-around for an unfixed bug
