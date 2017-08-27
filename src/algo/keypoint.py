
import math

import numpy as np
import quaternion
import cv2

from settings import *
from algo import tools
from algo.tools import PositioningException

# TODO:
#  - call solve_pnp from visnav.py (& testloop.py)
# 

class KeypointAlgo():
    def __init__(self, system_model, glWidget):
        self.system_model = system_model
        self.glWidget = glWidget
        self.min_options = None
        self.errval0 = None
        self.errval1 = None
        self.errval  = None
        self.iter_count = 0
        self.start_time = None
        self.optfun_values = []
        
        self.debug_filebase = None
        self._target_image = None
        self._hannw = None
        self._render_result = None


    def solve_pnp(self, scenefile, **kwargs):
        # load scene image
        self.glWidget.loadTargetImage(scenefile, remove_bg=False)
        
        # orb features from scene image
        orb = cv2.ORB_create()
        sc_kp, sc_desc = orb.detectAndCompute(self.glWidget.full_image, None)

        # render model image
        #zoff = self.system_model.z_off.value
        self.system_model.z_off.value = -MIN_DISTANCE*1.6
        render_result, depth_result = self.glWidget.render(depth=True)
        #self.system_model.z_off.value = zoff
        
        # orb features from model image
        ref_kp, ref_desc = orb.detectAndCompute(render_result, None)
        
        # match features
        matches = self._flann_matcher(sc_desc, ref_desc)

#        cv2.imshow('scene', self.glWidget.full_image)
#        cv2.imshow('model', render_result)
#        cv2.waitKey()

        # debug by drawing matches
#        self._draw_matches(self.glWidget.full_image, sc_kp,
#                           render_result, ref_kp, matches)
        
        # get feature 3d points using 3d model
        ref_kp_3d = self._inverse_project([ref_kp[m.trainIdx].pt for m in matches], depth_result)
        # TODO: scale coordinates from scene resolution to render resolution?
        sc_kp_2d = np.array([sc_kp[m.queryIdx].pt for m in matches], dtype='float')
        
#        print('keypoints: %s, %s'%(ref_kp_3d, sc_kp_2d), flush=True)
#        print('3d range: %s'%(ref_kp_3d.max(axis=(0,2))-ref_kp_3d.min(axis=(0,2)),), flush=True)
        
        # solve pnp with ransac
        rvec, tvec, inliers = self._solve_pnp_ransac(sc_kp_2d, ref_kp_3d)
        
#        print('results: %s, %s, %s'%(rvec, tvec, len(inliers)), flush=True)
        
        # debug by drawing inlier matches
        self._draw_matches(self.glWidget.full_image, sc_kp,
                           render_result, ref_kp,
                           [matches[i[0]] for i in inliers], pause=False)
        
        # set model params to solved pose & pos
        self._set_sc_from_ast_rot_and_trans(rvec, tvec)
        
        # TODO: other stuff? e.g. save some image for debugging perf?
    
    
    def _flann_matcher(self, desc1, desc2):
        MIN_FEATURES = 4
        
        if desc1 is None or desc2 is None or len(desc1)<MIN_FEATURES or len(desc2)<MIN_FEATURES:
            raise PositioningException('Not enough features found')
        
        FLANN_INDEX_LSH = 6
#        FLANN_INDEX_KDTREE = 0
        index_params = {
#            'algorithm':FLANN_INDEX_KDTREE,
#            'trees':5,
            'algorithm':FLANN_INDEX_LSH,
            'table_number':6,      # 12
            'key_size':12,         # 20
            'multi_probe_level':1, #2
        }
        search_params = {
            'checks':100,
        }
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)

        if len(matches)<MIN_FEATURES:
            raise PositioningException('Not enough features matched')

        # ratio test as per Lowe's paper
        matches = list(m[0] for m in matches if len(m)>1 and m[0].distance < 0.7*m[1].distance)
        
        if len(matches)<MIN_FEATURES:
            raise PositioningException('Too many features discarded')
        
        return matches
    
    
    def _draw_matches(self, img1, kp1, img2, kp2, matches, pause=True):
        matches = list([m] for m in matches)
        draw_params = {
#            matchColor: (88, 88, 88),
#            singlePointColor: (155, 155, 155),
            'flags': 2,
        }
        
        # scale keypoint positions
        for kp in kp1:
            kp.pt = (kp.pt[0]*VIEW_WIDTH/CAMERA_WIDTH, kp.pt[1]*VIEW_HEIGHT/CAMERA_HEIGHT)
        
        # scale image
        img1sc = cv2.cvtColor(cv2.resize(img1, None,
                            fx=VIEW_WIDTH/CAMERA_WIDTH,
                            fy=VIEW_HEIGHT/CAMERA_HEIGHT,
                            interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
        img2c = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        img3 = cv2.drawMatchesKnn(img1sc, kp1, img2c, kp2, matches, None, **draw_params)

        # restore original keypoint positions
        for kp in kp1:
            kp.pt = (kp.pt[0]*CAMERA_WIDTH/VIEW_WIDTH, kp.pt[1]*CAMERA_HEIGHT/VIEW_HEIGHT)

        cv2.imshow('matches', img3)
        if pause:
            cv2.waitKey()
    
    
    def _solve_pnp_ransac(self, sc_kp_2d, ref_kp_3d):
        x = CAMERA_WIDTH/2
        y = CAMERA_HEIGHT/2
        fl_x = x / math.tan( math.radians(CAMERA_X_FOV)/2 )
        fl_y = y / math.tan( math.radians(CAMERA_Y_FOV)/2 )
        cam_mx = np.array([[fl_x, 0, x],
                           [0, fl_y, y],
                           [0, 0, 1]], dtype = "float")
        
        # assuming no lens distortion
        dist_coeffs = np.zeros((4,1), dtype="float")
        ref_kp_3d = np.reshape(ref_kp_3d, (len(ref_kp_3d),1,3))
        sc_kp_2d = np.reshape(sc_kp_2d, (len(sc_kp_2d),1,2))
        
        try:
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                    ref_kp_3d, sc_kp_2d, cam_mx, dist_coeffs,
                    iterationsCount = 100,
                    reprojectionError = 8.0)
            if not retval:
                raise PositioningException('RANSAC algorithm returned False')
        except Exception as e:
            raise PositioningException('RANSAC algorithm ran into problems') from e

        return rvec, tvec, inliers

    
    def _inverse_project(self, points_2d, depths):
        z0 = self.system_model.z_off.value # translate to object origin
        def invproj(xi, yi):
            d = depths[int(yi)][int(xi)]
            x, y = tools.calc_xy(xi-VIEW_WIDTH/2, yi-VIEW_HEIGHT/2, d,
                                 width=VIEW_WIDTH, height=VIEW_HEIGHT)
            return x, y, -d-z0
        
        points_3d = np.array([invproj(pt[0], pt[1]) for pt in points_2d])
        return points_3d
    
    
    def _set_sc_from_ast_rot_and_trans(self, rvec, tvec):
        self.system_model.set_spacecraft_pos([tvec[0], -tvec[1], -tvec[2]])
        self.system_model.rotate_spacecraft(
                quaternion.from_rotation_vector(rvec).conj())
        