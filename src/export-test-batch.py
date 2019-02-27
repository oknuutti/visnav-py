import pickle
import sys
import re
from zipfile import ZipFile, ZIP_BZIP2, ZIP_DEFLATED, ZIP_LZMA

import cv2
import numpy as np

from algo.base import AlgorithmBase
from batch1 import get_system_model
from iotools import objloader
from render.render import RenderEngine
from settings import *


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[2] not in ('hi', 'med', 'lo'):
        print('USAGE: python %s <mission> <sm-quality: hi|med|lo>' % (sys.argv[0],))
        quit()

    mission = sys.argv[1]
    sm_quality = sys.argv[2]
    noise = {'hi':'', 'med':'lo', 'lo':'hi'}[sm_quality]

    sm = get_system_model(mission)
    with open(sm.asteroid.constant_noise_shape_model[noise], 'rb') as fh:
        noisy_model, sm_noise = pickle.load(fh)

    renderer = RenderEngine(sm.view_width, sm.view_height, antialias_samples=0)
    obj_idx = renderer.load_object(objloader.ShapeModel(data=noisy_model), smooth=sm.asteroid.render_smooth_faces)
    algo = AlgorithmBase(sm, renderer, obj_idx)
    cache_path = os.path.join(CACHE_DIR, mission)

    # i = 0
    for fn in os.listdir(cache_path):
        m = re.match('^(' + mission + r'_(\d+))\.lbl$', fn)
        if m and float(m[2]) < 2000:
            base_path = os.path.join(cache_path, m[1])
            sm.load_state(base_path + '.lbl')

            # save state in a more user friendly way
            sm.export_state(base_path + '_meta.csv')

            # render
            sm.z_off.value = -sm.min_med_distance
            image, depth = algo.render(center=True, depth=True, shadows=True)

            # write rendered image
            cv2.imwrite(base_path + '_'+sm_quality+'.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            if False:
                # write depth as zipped csv file
                depth_fn = base_path + '_' + sm_quality + '.csv'
                np.savetxt(depth_fn, depth, '%f', delimiter='\t', encoding='utf8')
                with ZipFile(depth_fn[:-4] + '.zip', 'w', ZIP_LZMA) as myzip:
                    myzip.write(depth_fn, m[1] + '_' + sm_quality + '.csv')
                os.unlink(depth_fn)
            else:
                # write as zipped binary file
                #  - extract all .zip files first
                #  - .bin file can be read in e.g. matlab using:  fid = fopen(fname); D = fread(fid, [512 512], 'float');
                depth_fn = base_path + '_' + sm_quality + '.bin'
                with open(depth_fn, 'wb') as file:
                    file.write(depth.astype('f4').tobytes())
                with ZipFile(depth_fn[:-4] + '.zip', 'w', ZIP_LZMA) as myzip:
                    myzip.write(depth_fn, m[1] + '_' + sm_quality + '.bin')
                os.unlink(depth_fn)

            # if i>9:
            #     quit()
            # i += 1
