import math

import numpy as np
import quaternion
import cv2

from visnav.algo.base import AlgorithmBase
from visnav.settings import *
from visnav.algo import tools
#from visnav.algo.centroid import CentroidAlgo
#from visnav.algo.keypoint import KeypointAlgo
from visnav.algo.tools import PositioningException


class MixedAlgo(AlgorithmBase):
    EXPECTED_LATERAL_ERR_ANGLE_SD = 0.30      # in deg
    EXPECTED_RELATIVE_DISTANCE_ERR_SD = 0.10  # 0.1 == 10% err

    def __init__(self, centroid, keypoint, **kwargs):
        super(MixedAlgo, self).__init__(centroid.system_model, centroid.render_engine, centroid.obj_idx)
        self._centroid = centroid
        self._keypoint = keypoint

    def run(self, sce_img, outfile, **kwargs):
        centroid_result = None
        keypoint_ok = False
        try:
            if True:
                self._centroid.adjust_iteratively(sce_img, None, **kwargs)
                sc_r1 = self.system_model.spacecraft_rot
                ast_r1 = self.system_model.asteroid_axis
                rel_q1 = self.system_model.sc_asteroid_rel_q()
                centroid_result = self.system_model.spacecraft_pos
            else:
                centroid_result = self.system_model.real_spacecraft_pos
            #kwargs['init_z'] = centroid_result[2]

            x_off, y_off = self._cam.calc_img_xy(*centroid_result)
            uncertainty_radius = math.tan(math.radians(MixedAlgo.EXPECTED_LATERAL_ERR_ANGLE_SD) * 2) \
                                 * abs(centroid_result[2]) * (1 + MixedAlgo.EXPECTED_RELATIVE_DISTANCE_ERR_SD * 2)
            kwargs['match_mask_params'] = (
                x_off-self._cam.width/2,
                y_off-self._cam.height/2,
                centroid_result[2],
                uncertainty_radius,
            )
            d1 = np.linalg.norm(centroid_result)
        except PositioningException as e:
            if str(e) == 'No asteroid found':
                raise e
            elif not 'Asteroid too close' in str(e) and DEBUG:
                print('Centroid algo failed with: %s'%(e,))

        centroid_ok = centroid_result is not None
        fallback = centroid_ok and kwargs.get('centroid_fallback', False) \
                   and self.system_model.max_distance > d1 > self.system_model.min_med_distance
        try:
            self._keypoint.solve_pnp(sce_img, outfile, **kwargs)

            d2 = np.linalg.norm(self.system_model.spacecraft_pos)
            rel_q2 = self.system_model.sc_asteroid_rel_q()
            d_ang = abs(tools.wrap_rads(tools.angle_between_q(rel_q1, rel_q2))) if fallback else None
            if fallback and (d2 > d1 * 1.2 or d_ang > math.radians(20)):
                # if keypoint res distance significantly larger than from centroid method, override
                # also, if orientation significantly different from initial, override
                self.system_model.spacecraft_pos = centroid_result
                self.system_model.spacecraft_rot = sc_r1
                self.system_model.asteroid_axis = ast_r1
            elif fallback and d2 > self.system_model.max_med_distance:
                # if distance more than max medum distance, assume orientation result is wrong
                self.system_model.spacecraft_rot = sc_r1
                self.system_model.asteroid_axis = ast_r1
            else:
                keypoint_ok = True
        except PositioningException as e:
            if fallback:
                self.system_model.spacecraft_pos = centroid_result
                self.system_model.spacecraft_rot = sc_r1
                self.system_model.asteroid_axis = ast_r1
                if DEBUG:
                    print('Using centroid result as keypoint algo failed: %s'%(e,))
            else:
                raise e

        return (centroid_ok or keypoint_ok, keypoint_ok)