import math
import sys
import re

import cv2
import numpy as np
import quaternion

from visnav.algo import tools
from visnav.algo.absnet import AbsoluteNavigationNN
from visnav.algo.base import AlgorithmBase
from visnav.algo.image import ImageProc
from visnav.algo.model import SystemModel
from visnav.batch1 import get_system_model
from visnav.render.render import RenderEngine
from visnav.settings import *


def export(sm, dst_path, src_path=None, src_imgs=None, trg_shape=(224, 224), debug=False, img_postfix="", title=""):
    trg_w, trg_h = trg_shape
    assert (src_path is not None) + (src_imgs is not None) == 1, 'give either src_path or src_imgs, not both'

    if debug:
        renderer = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=0)
        obj_idx = renderer.load_object(sm.asteroid.target_model_file,
                                       smooth=sm.asteroid.render_smooth_faces)
        algo = AlgorithmBase(sm, renderer, obj_idx)

    metadatafile = os.path.join(dst_path, 'dataset_all.txt')
    if not os.path.exists(metadatafile):
        with open(metadatafile, 'w') as f:
            f.write('\n'.join(['%s, camera centric coordinate frame used' % title,
                               'ImageFile, Target Pose [X Y Z W P Q R], Sun Vector [X Y Z]', '', '']))

    files = list(os.listdir(src_path)) if src_imgs is None else src_imgs
    for i, fn in enumerate(files):
        if src_imgs is not None or re.search(r'(?<!far_)\d{4}\.png$', fn):
            c = 2 if src_imgs is None else 1
            tools.show_progress(len(files)//c, i//c)

            # read system state, write out as relative to s/c
            fname = os.path.basename(fn)
            base_path = fn[:-4-len(img_postfix)]
            if src_imgs is None:
                base_path = os.path.join(src_path, base_path)
            sm.load_state(base_path + '.lbl')
            sm.swap_values_with_real_vals()

            # read image, detect box, resize, adjust relative pose
            img = cv2.imread(base_path + img_postfix + '.png', cv2.IMREAD_GRAYSCALE)
            assert img is not None, 'image file %s not found' % (base_path + img_postfix + '.png')

            # detect target, get bounds
            x, y, w, h = ImageProc.single_object_bounds(img, threshold=AbsoluteNavigationNN.DEF_LUMINOSITY_THRESHOLD,
                                                        crop_marg=AbsoluteNavigationNN.DEF_CROP_MARGIN,
                                                        min_px=AbsoluteNavigationNN.DEF_MIN_PIXELS, debug=debug)
            if x is None:
                continue

            # write image metadata
            sc_ast_lf_r, sc_ast_lf_q, ast_sun_lf_u = sm.get_cropped_system_scf(x, y, w, h)
            with open(metadatafile, 'a') as f:
                f.write(fname + ' ' + (' '.join('%f' % f for f in (
                    tuple(sc_ast_lf_r)
                    + tuple(sc_ast_lf_q.components)
                    + tuple(ast_sun_lf_u)))) + '\n')

            depth = None
            if os.path.exists(base_path + img_postfix + '_d.exr'):
                depth = cv2.imread(base_path + img_postfix + '_d.exr', cv2.IMREAD_UNCHANGED)

            # crop & resize image, write it
            cropped = ImageProc.crop_and_zoom_image(img, x, y, w, h, None, (trg_w, trg_h), depth=depth)
            cropped = [cropped] if depth is None else cropped

            cv2.imwrite(os.path.join(dst_path, fname), cropped[0], [cv2.IMWRITE_PNG_COMPRESSION, 9])
            if depth is not None:
                cv2.imwrite(os.path.join(dst_path, fname[:-4] + '_d.exr'), cropped[1],
                            (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT))

            if debug:
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
                imge = cv2.resize(imge[:, (w - h)//2:(w - h)//2+h], cropped[0].shape)
                cv2.imshow('equal?', np.hstack((
                    cropped[0],
                    np.ones((cropped[0].shape[0], 1), dtype=cropped[0].dtype) * 255,
                    imge,
                )))
                cv2.waitKey()

                if i > 60:
                    quit()


def get_files_with_metadata(dst_path):
    files = set()
    metadatafile = os.path.join(dst_path, 'dataset_all.txt')
    if os.path.exists(metadatafile):
        with open(metadatafile, 'r') as f:
            for line in f:
                files.add(line.split(' ')[0].strip())

    return files


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('USAGE: python %s <mission> <visnav-folder> <dst-folder>' % (sys.argv[0],))
        quit()

    mission = sys.argv[1]
    src_path = sys.argv[2]
    dst_path = sys.argv[3]
    sm = get_system_model(mission)
    trg_h, trg_w = 224, 224

    export(sm, dst_path, src_path=src_path, trg_shape=(trg_w, trg_h), debug=0)
