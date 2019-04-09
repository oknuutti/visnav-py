import math

import numpy as np
import cv2

from algo.image import ImageProc
from algo.model import SystemModel
from iotools import lblloader
from missions.rosetta import RosettaSystemModel
from settings import *
from render.render import RenderEngine
from algo import tools


class AlgorithmBase:
    def __init__(self, system_model, render_engine, obj_idx):
        self.system_model = system_model
        self.render_engine = render_engine
        self.obj_idx = obj_idx
        self.debug_filebase = None
        self.timer = None
        self._cam = self.system_model.cam

        self.latest_discretization_err_q = None
        self.latest_discretization_light_err_angle = None

        # initialize by the call to set_image_zoom_and_resolution
        self.im_xoff = None
        self.im_yoff = None
        self.im_width = None
        self.im_height = None
        self.im_def_scale = None
        self.im_scale = None

        self.set_image_zoom_and_resolution()

    def set_image_zoom_and_resolution(self, im_xoff=0, im_yoff=0, im_width=None, im_height=None):
        self.im_xoff = im_xoff
        self.im_yoff = im_yoff
        self.im_width = im_width or self._cam.width
        self.im_height = im_height or self._cam.height
        self.im_def_scale = self.system_model.view_width/self.im_width
        self.im_scale = self.im_def_scale

        # calculate frustum based on fov, aspect & near
        # NOTE: with wide angle camera, would need to take into account
        #       im_xoff, im_yoff, im_width and im_height
        x_fov = self._cam.x_fov * self.im_def_scale / self.im_scale
        y_fov = self._cam.y_fov * self.im_def_scale / self.im_scale
        self.render_engine.set_frustum(x_fov, y_fov, self.system_model.min_altitude*.1, self.system_model.max_distance)

    def load_target_image(self, src):
        tmp = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if tmp is None:
            raise Exception('Cant load image from file %s' % (src,))

        if tmp.shape != (self._cam.height, self._cam.width):
            # visit fails to generate 1024 high images
            tmp = cv2.resize(tmp, None,
                             fx=self._cam.width / tmp.shape[1],
                             fy=self._cam.height / tmp.shape[0],
                             interpolation=cv2.INTER_CUBIC)
        return tmp

    def remove_background(self, img):
        res_img, h, th = ImageProc.process_target_image(img)
        return res_img, th

    def load_obj(self, obj_file, obj_idx=None):
        self.obj_idx = self.render_engine.load_object(obj_file, obj_idx)

    def render(self, center=False, depth=False, discretize_tol=False, shadows=False, gamma=1.8,
               reflection=RenderEngine.REFLMOD_LUNAR_LAMBERT):
        assert not discretize_tol, 'discretize_tol deprecated at render function'

        rel_pos_v, rel_rot_q, light_v = self._render_params(discretize_tol, center)
        res = self.render_engine.render(self.obj_idx, rel_pos_v, rel_rot_q, light_v, get_depth=depth, shadows=shadows,
                                        gamma=gamma, reflection=reflection)
        if depth:
            img, dth = res
        else:
            img = res

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return (img, dth) if depth else img


    def _render_params(self, discretize_tol=False, center_model=False):
        assert not discretize_tol, 'discretize_tol deprecated at _render_params function'

        m = self.system_model

        # NOTE: with wide angle camera, would need to take into account
        #       im_xoff, im_yoff, im_width and im_height
        xc_off = (self.im_xoff + self.im_width / 2 - self._cam.width / 2)
        xc_angle = xc_off / self._cam.width * math.radians(self._cam.x_fov)

        yc_off = (self.im_yoff + self.im_height / 2 - self._cam.height / 2)
        yc_angle = yc_off / self._cam.height * math.radians(self._cam.y_fov)

        # first rotate around x-axis, then y-axis,
        # note that diff angle in image y direction corresponds to rotation
        # around x-axis and vise versa
        q_crop = (
                np.quaternion(math.cos(-yc_angle / 2), math.sin(-yc_angle / 2), 0, 0)
                * np.quaternion(math.cos(-xc_angle / 2), 0, math.sin(-xc_angle / 2), 0)
        )

        x = m.x_off.value
        y = m.y_off.value
        z = m.z_off.value

        # rotate offsets using q_crop
        x, y, z = tools.q_times_v(q_crop.conj(), np.array([x, y, z]))

        # maybe put object in center of view
        if center_model:
            x, y = 0, 0

        # get object rotation and turn it a bit based on cropping effect
        q, err_q = m.gl_sc_asteroid_rel_q(discretize_tol)
        sc2gl_q = m.frm_conv_q(m.SPACECRAFT_FRAME, m.OPENGL_FRAME)
        self.latest_discretization_err_q = sc2gl_q * err_q * sc2gl_q.conj() if discretize_tol else False

        qfin = (q * q_crop.conj())

        # light direction
        light, err_angle = m.gl_light_rel_dir(err_q, discretize_tol)
        self.latest_discretization_light_err_angle = err_angle if discretize_tol else False

        return (x, y, z), qfin, light


if __name__ == '__main__':
    sm = RosettaSystemModel(rosetta_batch='mtp006')
    #img = 'ROS_CAM1_20140823T021833'
    img = 'ROS_CAM1_20140822T020718'
    #img = 'ROS_CAM1_20150606T213002'  # 017

    lblloader.load_image_meta(os.path.join(sm.asteroid.image_db_path, img + '.LBL'), sm)
    sm.swap_values_with_real_vals()

    if False:
        re = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=16)
        obj_idx = re.load_object(sm.asteroid.hires_target_model_file, smooth=False)
    else:
        re = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=0)
        obj_idx = re.load_object(sm.asteroid.target_model_file, smooth=False)

    ab = AlgorithmBase(sm, re, obj_idx)

    hapke = True
    model = RenderEngine.REFLMOD_HAPKE if hapke else RenderEngine.REFLMOD_LUNAR_LAMBERT
    size = (1024, 1024)  # (256, 256)

    real = cv2.imread(os.path.join(sm.asteroid.image_db_path, img + '_P.png'), cv2.IMREAD_GRAYSCALE)

    synth = []
#    for i, p in enumerate(np.linspace(0, 30, 11)):
#        if hapke:
            # L, th, w, b (scattering anisotropy), c (scattering direction from forward to back)
#            RenderEngine.REFLMOD_PARAMS[model][0:5] = [600, p, 0.052, -0.42, 0]
    synth.append(cv2.resize(ab.render(shadows=True, reflection=model), size))
#        if not hapke:
#            break
    #synth[0] = ImageProc.equalize_brightness(synth[0], real, percentile=99.999, image_gamma=1.8)
    #synth[0] = ImageProc.adjust_gamma(synth[0], 1/4)
    #real = ImageProc.adjust_gamma(real, 1/4)

    real = cv2.resize(real, size)
    cv2.imshow('real vs synthetic', np.concatenate([real, 255*np.ones((real.shape[0], 1), dtype='uint8')] + synth, axis=1))
    cv2.waitKey()
