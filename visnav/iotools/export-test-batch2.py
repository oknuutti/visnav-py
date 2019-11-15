import math
import sys
import re

import cv2
import numpy as np
import quaternion

from algo import tools
from algo.base import AlgorithmBase
from algo.image import ImageProc
from algo.model import SystemModel
from batch1 import get_system_model
from render.render import RenderEngine
from settings import *


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('USAGE: python %s <mission> <visnav-folder> <dst-folder>' % (sys.argv[0],))
        quit()

    mission = sys.argv[1]
    src_path = sys.argv[2]
    dst_path = sys.argv[3]
    sm = get_system_model(mission)
    trg_h, trg_w = 224, 224

    if DEBUG:
        renderer = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=0)
        obj_idx = renderer.load_object(sm.asteroid.target_model_file,
                                       smooth=sm.asteroid.render_smooth_faces)
        algo = AlgorithmBase(sm, renderer, obj_idx)

    with open(os.path.join(dst_path, 'dataset_all.txt'), 'w') as f:
        f.write('\n'.join(['Ryugu Synthetic Image Set V1, camera centric coordinate frame used',
                           'ImageFile, Target Pose [X Y Z W P Q R], Sun Vector [X Y Z]', '', '']))

    files = list(os.listdir(src_path))
    for i, fn in enumerate(files):
        if re.search(r'(?<!far_)\d{4}\.png$', fn):
            tools.show_progress(len(files)//2, i//2)

            # read system state, write out as relative to s/c
            base_path = os.path.join(src_path, fn[:-4])
            sm.load_state(base_path + '.lbl')
            sm.swap_values_with_real_vals()

            # read image, detect box, resize, adjust relative pose
            img = cv2.imread(base_path + '.png', cv2.IMREAD_GRAYSCALE)

            # detect target, get bounds
            x, y, w, h = ImageProc.single_object_bounds(img, threshold=AbsoluteNavigationNN.DEF_LUMINOSITY_THRESHOLD,
                                                        crop_margin=AbsoluteNavigationNN.DEF_CROP_MARGIN,
                                                        min_px=AbsoluteNavigationNN.DEF_MIN_PIXELS, debug=DEBUG)
            if x is None:
                continue

            # write image metadata
            sc_ast_lf_r, sc_ast_lf_q, ast_sun_lf_u = sm.get_cropped_system_scf(x, y, w, h)
            with open(os.path.join(dst_path, 'dataset_all.txt'), 'a') as f:
                f.write(fn + ' ' + (' '.join('%f' % f for f in (
                    tuple(sc_ast_lf_r)
                    + tuple(sc_ast_lf_q.components)
                    + tuple(ast_sun_lf_u)))) + '\n')

            # crop & resize image, write it
            imgd = ImageProc.crop_and_zoom_image(img, x, y, w, h, None, (trg_w, trg_h))
            cv2.imwrite(os.path.join(dst_path, fn), imgd, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            if DEBUG:
                sc, dq = sm.cropped_system_tf(x, y, w, h)

                sm.spacecraft_pos = tools.q_times_v(SystemModel.sc2gl_q.conj(), sc_ast_lf_r)
                sm.rotate_spacecraft(dq)
                #sm.set_cropped_system_scf(x, y, w, h, sc_ast_lf_r, sc_ast_lf_q)

                if False:
                    sm.load_state(base_path + '.lbl')
                    sm.swap_values_with_real_vals()
                    imgd = cv2.resize(img, (trg_h, trg_w))

                imge = algo.render(center=False, depth=False, shadows=True)
                h, w = imge.shape
                imge = cv2.resize(imge[:, (w - h)//2:(w - h)//2+h], imgd.shape)
                cv2.imshow('equal?', np.hstack((
                    imgd,
                    np.ones((imgd.shape[0], 1), dtype=imgd.dtype) * 255,
                    imge,
                )))
                cv2.waitKey()

                if i > 60:
                    quit()
