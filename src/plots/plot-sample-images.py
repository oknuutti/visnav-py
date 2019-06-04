import os
import pickle

import numpy as np
import cv2

from algo.base import AlgorithmBase
from iotools import lblloader, objloader
from missions.rosetta import RosettaSystemModel
from render.render import RenderEngine

from settings import *

if __name__ == '__main__':
    size = (1024, 1024)  # (256, 256)
    imgs = [None]*6

    sm = RosettaSystemModel()
    #img = 'ROS_CAM1_20140823T021833'
    img = 'ROS_CAM1_20140822T020718'

    real = cv2.imread(os.path.join(sm.asteroid.image_db_path, img + '_P.png'), cv2.IMREAD_GRAYSCALE)
    imgs[0] = cv2.resize(real, size)

    lblloader.load_image_meta(os.path.join(sm.asteroid.image_db_path, img + '.LBL'), sm)
    sm.swap_values_with_real_vals()

    render_engine = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=16)
    obj_idx = render_engine.load_object(sm.asteroid.hires_target_model_file, smooth=sm.asteroid.render_smooth_faces)
    ab = AlgorithmBase(sm, render_engine, obj_idx)
    imgs[1] = cv2.resize(ab.render(shadows=True, reflection=RenderEngine.REFLMOD_HAPKE), size)
    imgs[2] = np.zeros_like(imgs[1])

    render_engine = RenderEngine(sm.view_width, sm.view_height, antialias_samples=0)
    obj_idx = render_engine.load_object(sm.asteroid.target_model_file, smooth=sm.asteroid.render_smooth_faces)
    ab = AlgorithmBase(sm, render_engine, obj_idx)

    def render(fname):
        with open(fname, 'rb') as fh:
            noisy_model, _loaded_sm_noise = pickle.load(fh)
        render_engine.load_object(objloader.ShapeModel(data=noisy_model), obj_idx, smooth=sm.asteroid.render_smooth_faces)
        return cv2.resize(ab.render(shadows=True, reflection=RenderEngine.REFLMOD_LUNAR_LAMBERT), size)

    imgs[3:] = [render(sm.asteroid.constant_noise_shape_model[key]) for key in ('', 'lo', 'hi')]

    cv2.imshow('image types', np.concatenate([np.concatenate(imgs[:3], axis=1), np.concatenate(imgs[3:], axis=1)], axis=0))
    cv2.waitKey()
