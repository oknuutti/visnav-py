import math

import numpy as np
import cv2
import sys

try:
    from visnav.render.render import RenderEngine
except:
    from visnav.render import RenderEngine
from visnav.algo import tools
from visnav.missions.didymos import DidymosSystemModel
from visnav.settings import *


if __name__ == '__main__':
    sm = DidymosSystemModel(use_narrow_cam=False, target_primary=False, hi_res_shape_model=False)

    re = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=0)
    re.set_frustum(sm.cam.x_fov, sm.cam.y_fov, 0.05, 2)
    obj_idx = re.load_object(sm.asteroid.real_shape_model)

    pos = np.array([0, 0, -sm.min_med_distance * 1])
    q = tools.angleaxis_to_q((math.radians(20), 0, 1, 0))
    light_v = np.array([1, 0, 0]) / math.sqrt(1)

    img = re.render(obj_idx, pos, q, light_v, gamma=1.8, get_depth=False, shadows=True, textures=True,
                    reflection=RenderEngine.REFLMOD_HAPKE)
    cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else 'test.jpg', img)
