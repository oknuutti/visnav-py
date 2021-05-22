import math
import sys
import re
import shutil

import cv2
import numpy as np
import quaternion
from tqdm import tqdm

from visnav.algo import tools
from visnav.algo.base import AlgorithmBase
from visnav.algo.image import ImageProc
from visnav.algo.model import SystemModel
from visnav.batch1 import get_system_model
from visnav.render.render import RenderEngine
from visnav.settings import *


def _write_metadata(metadatafile, id, fname, system_scf):
    with open(metadatafile, 'a') as f:
        f.write(str(id) + ' ' + fname + ' ' + (' '.join('%f' % f for f in (
                tuple(system_scf[0])
                + tuple(system_scf[1].components)
                + tuple(system_scf[2])))) + '\n')


def export(sm, dst_path, src_path=None, src_imgs=None, trg_shape=(224, 224), crop=False, debug=False,
           img_prefix="", title=""):

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
                               'Image ID, ImageFile, Target Pose [X Y Z W P Q R], Sun Vector [X Y Z]', '', '']))

    files = list(os.listdir(src_path)) if src_imgs is None else src_imgs
    files = sorted(files)

    id = 0
    for i, fn in enumerate(files):
        if src_imgs is not None or re.search(r'(?<!far_)\d{4}\.png$', fn):
            c = 2 if src_imgs is None else 1
            tools.show_progress(len(files)//c, i//c)
            id += 1

            # read system state, write out as relative to s/c
            fname = os.path.basename(fn)
            if src_imgs is None:
                fn = os.path.join(src_path, fn)
            lbl_fn = re.sub(r'_%s(\d{4})' % img_prefix, r'_\1', fn[:-4]) + '.lbl'

            sm.load_state(lbl_fn)
            sm.swap_values_with_real_vals()

            if not crop:
                shutil.copy2(fn, os.path.join(dst_path, fname))
                if os.path.exists(fn[:-4] + '.d.exr'):
                    shutil.copy2(fn[:-4] + '.d.exr', os.path.join(dst_path, fname[:-4] + '.d.exr'))
                if os.path.exists(fn[:-4] + '.xyz.exr'):
                    shutil.copy2(fn[:-4] + '.xyz.exr', os.path.join(dst_path, fname[:-4] + '.xyz.exr'))
                if os.path.exists(fn[:-4] + '.s.exr'):
                    shutil.copy2(fn[:-4] + '.s.exr', os.path.join(dst_path, fname[:-4] + '.s.exr'))
                _write_metadata(metadatafile, id, fname, sm.get_system_scf())
                continue

            from visnav.algo.absnet import AbsoluteNavigationNN

            # read image, detect box, resize, adjust relative pose
            img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
            assert img is not None, 'image file %s not found' % fn

            # detect target, get bounds
            x, y, w, h = ImageProc.single_object_bounds(img, threshold=AbsoluteNavigationNN.DEF_LUMINOSITY_THRESHOLD,
                                                        crop_marg=AbsoluteNavigationNN.DEF_CROP_MARGIN,
                                                        min_px=AbsoluteNavigationNN.DEF_MIN_PIXELS, debug=debug)
            if x is None:
                continue

            # write image metadata
            system_scf = sm.get_cropped_system_scf(x, y, w, h)
            _write_metadata(metadatafile, id, fname, system_scf)

            others, (depth, coords, px_size), k = [], [False] * 3, 1
            if os.path.exists(fn[:-4] + '.d.exr'):
                depth = True
                others.append(cv2.imread(fn[:-4] + '.d.exr', cv2.IMREAD_UNCHANGED))
            if os.path.exists(fn[:-4] + '.xyz.exr'):
                coords = True
                others.append(cv2.imread(fn[:-4] + '.xyz.exr', cv2.IMREAD_UNCHANGED))
            if os.path.exists(fn[:-4] + '.s.exr'):
                px_size = True
                others.append(cv2.imread(fn[:-4] + '.s.exr', cv2.IMREAD_UNCHANGED))

            # crop & resize image, write it
            cropped = ImageProc.crop_and_zoom_image(img, x, y, w, h, None, (trg_w, trg_h), others=others)

            cv2.imwrite(os.path.join(dst_path, fname), cropped[0], [cv2.IMWRITE_PNG_COMPRESSION, 9])
            if depth:
                cv2.imwrite(os.path.join(dst_path, fname[:-4] + '.d.exr'), cropped[k],
                            (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT))
                k += 1
            if coords:
                cv2.imwrite(os.path.join(dst_path, fname[:-4] + '.xyz.exr'), cropped[k],
                            (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT))
                k += 1
            if px_size:
                cv2.imwrite(os.path.join(dst_path, fname[:-4] + '.s.exr'), cropped[k],
                            (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT))

            if debug:
                sc, dq = sm.cropped_system_tf(x, y, w, h)

                sm.spacecraft_pos = tools.q_times_v(SystemModel.sc2gl_q.conj(), sc_ast_lf_r)
                sm.rotate_spacecraft(dq)
                #sm.set_cropped_system_scf(x, y, w, h, sc_ast_lf_r, sc_ast_lf_q)

                if False:
                    sm.load_state(lbl_fn)
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


def export_relative(sm, dst_path, src_path=None, src_imgs=None, debug=False, img_prefix="", title=""):
    assert (src_path is not None) + (src_imgs is not None) == 1, 'give either src_path or src_imgs, not both'

    metadatafile = os.path.join(dst_path, 'dataset_all.txt')
    if not os.path.exists(metadatafile):
        with open(metadatafile, 'w') as f:
            f.write('\n'.join(['%s, ICRS J2000' % title,
                               'Image ID, Trajectory ID, Frame Number, Image File, S/C-Target Vector [X Y Z], '
                               'S/C Orientation [W X Y Z], Target Orientation [W X Y Z], S/C-Sun Vector [X Y Z]', '', '']))

    files = list(os.listdir(src_path)) if src_imgs is None else src_imgs
    data = []
    for i, fn in enumerate(files):
        m = re.search(r'(?<!far_)%s(\d{4})_(\d+)\.(png|jpg)$' % img_prefix, fn)
        if m:
            # read system state, write out
            fname = os.path.basename(fn)
            if src_imgs is None:
                fn = os.path.join(src_path, fn)
            lbl_fn = re.sub(r'_%s(\d{4})' % img_prefix, r'_\1', fn[:-4]) + '.lbl'

            sm.load_state(lbl_fn)
            sm.swap_values_with_real_vals()
            sc_gf_q, ast_gf_q, sc_ast_gf_r, sc_sun_gf_r = sm.get_system_gf()
            data.append((fn, int(m[1]), int(m[2]), fname, sc_ast_gf_r, sc_gf_q.components,
                                                          ast_gf_q.components, sc_sun_gf_r))

    data = sorted(data, key=lambda r: (r[1], r[2]))
    for i, row in enumerate(tqdm(data)):
        # img_id, traj_id, frame_nr, file, sc_trg_x, sc_trg_y, sc_trg_z, sc_qw, sc_qx, sc_qy, sc_qz, \
        #                                  trg_qw, trg_qx, trg_qy, trg_qz, sc_sun_x, sc_sun_y, sc_sun_z
        with open(metadatafile, 'a') as f:
            f.write(' '.join(['%d' % c for c in ((i,) + row[1:3])]) + (' %s ' % row[3]) + (' '.join('%f' % f for f in (
                tuple(row[4]) + tuple(row[5]) + tuple(row[6]) + tuple(row[7])))) + '\n')

        src_file = row[0]
        dst_file = os.path.join(dst_path, row[3])

        # copy pixel size file
        shutil.copy2(src_file[:-4] + '.s.exr', dst_file[:-4] + '.s.exr')

        # copy model coordinates file
        shutil.copy2(src_file[:-4] + '.xyz.exr', dst_file[:-4] + '.xyz.exr')

        # copy depth file
        shutil.copy2(src_file[:-4] + '.d.exr', dst_file[:-4] + '.d.exr')

        # copy label file
        shutil.copy2(src_file[:-4].replace('_cm', '') + '.lbl', dst_file[:-4] + '.lbl')

        # copy img file
        shutil.copy2(src_file, dst_file)


def get_files_with_metadata(dst_path, traj_len=1):
    files = set()
    metadatafile = os.path.join(dst_path, 'dataset_all.txt')
    if os.path.exists(metadatafile):
        with open(metadatafile, 'r') as f:
            for line in f:
                cells = line.split(' ')
                if len(cells) < 5:
                    pass
                elif traj_len == 1:
                    files.add(cells[1].strip())
                else:
                    files.add(cells[3].strip())

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
