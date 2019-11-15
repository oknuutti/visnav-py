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

# data/ryugu+tex-d1-80k.obj data/ryugu+tex-d1-16k.obj data/ryugu+tex-d1-16k.nsm
# data/ryugu+tex-d1-80k.obj data/ryugu+tex-d1-4k.obj data/ryugu+tex-d1-4k.nsm
# data/ryugu+tex-d1-80k.obj data/ryugu+tex-d1-1k.obj data/ryugu+tex-d1-1k.nsm
# data/ryugu+tex-d2-80k.obj data/ryugu+tex-d2-16k.obj data/ryugu+tex-d2-16k.nsm
# data/ryugu+tex-d2-80k.obj data/ryugu+tex-d2-4k.obj data/ryugu+tex-d2-4k.nsm
# data/ryugu+tex-d2-80k.obj data/ryugu+tex-d2-1k.obj data/ryugu+tex-d2-1k.nsm


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

    timer = tools.Stopwatch()
    timer.start()
    devs = tools.point_cloud_vs_model_err(np.array(obj_fr.vertices), obj)
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
            timer.elapsed/len(obj_fr.vertices),
            dev_mean, p50,
            dev_std*1, p68,
            dev_std*2, p95,
            dev_std*3, p99,
        ))
    ))

    with open(outfile, 'wb') as fh:
        pickle.dump((obj.as_dict(), dev_mean), fh, -1)
