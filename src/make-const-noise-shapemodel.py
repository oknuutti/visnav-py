import math
import pickle
import sys

import numpy as np
import numba as nb

from algo import tools
from iotools import objloader
from settings import *

# data/CSHP_DV_130_01_XLRES_00200.obj data/CSHP_DV_130_01_XXLRES_00200.obj data/67P_test.nsm
# data/bennu-hi-res.obj data/bennu-lo-res.obj data/bennu_baseline.nsm

# data/CSHP_DV_130_01_LORES_00200.obj data/CSHP_DV_130_01_XLRESb_00200.obj data/67P_baseline.nsm
# data/CSHP_DV_130_01_LORES_00200.obj data/CSHP_DV_130_01_XXLRES_00200.obj data/67P_xl.nsm
# data/CSHP_DV_130_01_LORES_00200.obj data/CSHP_DV_130_01_X3RES_00200.obj data/67P_xxl.nsm

# data/ryugu-hi-res.obj data/ryugu-lo-res.obj data/ryugu_baseline.nsm
# data/ryugu-hi-res.obj data/ryugu-xl-res.obj data/ryugu_xl.nsm
# data/ryugu-hi-res.obj data/ryugu-xxl-res.obj data/ryugu_xxl.nsm

@nb.jit(nb.f8[:](nb.f8[:,:], nb.f8[:,:], nb.i4[:,:]), nogil=True, parallel=False)
def get_model_errors(full_vertices, vertices, faces):
    count = len(full_vertices)
    digits = int(math.ceil(math.log10(count//10+1)))
    print('%s/%d' % ('0' * digits, count//10), end='', flush=True)

    devs = np.empty(full_vertices.shape[0])
    for i in nb.prange(count):
        vx = full_vertices[i, :]
        err = tools.intersections(faces, vertices, np.array(((0, 0, 0), vx)))
        if math.isinf(err):  # len(pts) == 0:
            print('no intersections!')
            continue

        if False:
            idx = np.argmin([np.linalg.norm(pt-vx) for pt in pts])
            err = np.linalg.norm(pts[idx]) - np.linalg.norm(vx)

        devs[i] = err
        print(('%s%0' + str(digits) + 'd/%d') % ('\b' * (digits * 2 + 1), (i+1)//10, count//10), end='', flush=True)

    return devs


if __name__ == '__main__':
    if False:
        res = tools.poly_line_intersect(((0, 0, 1), (0, 1, 1), (1, 0, 1)), ((0, 0, 0), (.3, .7, 1)))
        print('%s' % res)
        quit()

    assert len(sys.argv) == 4, 'USAGE: %s [full-res-model] [target-model] [output]' % sys.argv[0]

    full_res_model = os.path.join(BASE_DIR, sys.argv[1])
    infile = os.path.join(BASE_DIR, sys.argv[2])
    outfile = os.path.join(BASE_DIR, sys.argv[3])
    sc = 1000  # bennu in meters, ryugu & 67P in km

    # load shape models
    obj_fr = objloader.ShapeModel(fname=full_res_model)
    obj = objloader.ShapeModel(fname=infile)

    faces = np.array([f[0] for f in obj.faces], dtype='uint')
    vertices = np.array(obj.vertices)
    full_vertices = np.array(obj_fr.vertices)

    timer = tools.Stopwatch()
    timer.start()
    devs = get_model_errors(full_vertices, vertices, faces)
    timer.stop()
    # doesnt work: tools.intersections.parallel_diagnostics(level=4)

    p50 = np.median(devs)
    p68, p95, p99 = np.percentile(np.abs(devs-p50), (68, 95, 99.7))

    idxs = np.abs(devs-p50) < p95
    clean_devs = devs[idxs]
    dev_mean = np.mean(clean_devs)
    dev_std = np.std(clean_devs)
    print('\n\n(%.2fms/vx) dev mean %.6fm/%.6fm, std %.6fm/%.6fm, 2s %.6fm/%.6fm, 3s %.6fm/%.6fm' % tuple(
        map(lambda x: sc*x, (
            timer.elapsed/(full_vertices.shape[0]),
            dev_mean, p50,
            dev_std*1, p68,
            dev_std*2, p95,
            dev_std*3, p99,
        ))
    ))

    with open(outfile, 'wb') as fh:
        pickle.dump((obj.as_dict(), dev_mean), fh, -1)
