import math
import sys
from dateutil import parser

import numpy as np
import quaternion

from astropy.time import Time
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from algo import tools
from missions.rosetta import RosettaSystemModel

from settings import *

def manual_approach():
    ## earlier, manual approach:
    ##
    data = {
        'mtp007': {'phi1': -99.072, 'd1': '2014-09-02 11:38:52',
                   'phi2': -95.39, 'd2': '2014-09-23 06:08:54'},
        'mtp017': {'phi1': 136.9, 'd1': '2015-06-03 09:45:00',
                   'phi2': 127.3, 'd2': '2015-06-12 23:02:00'},
        'mtp024': {'phi1': -154.008, 'd1': '2015-12-16 06:02:18',
                   'phi2': 16.704 - 360, 'd2': '2016-01-12 23:02:17'},
        'mtp025': {'phi1': 22.608, 'd1': '2016-01-13 06:02:18',
                   'phi2': -154.728, 'd2': '2016-02-09 23:17:53'},
        'mtp026': {'phi1': -150.33, 'd1': '20160210T060423',
                   'phi2': 110.232 - 360 * 1, 'd2': '20160223T080929'},
        # 'phi2': -107.28, 'd2': '20160301T131104'},
        # 'phi2': 48.6-360, 'd2': '20160308T231754'},
    }

    for batch, r in data.items():
        t1 = (Time(parser.parse(r['d1'])) - Time('J2000')).sec / 3600 / 24  # in days
        t2 = (Time(parser.parse(r['d2'])) - Time('J2000')).sec / 3600 / 24  # in days
        delta_w = (r['phi2'] - r['phi1'] + 360*1) / (t2 - t1)  # deg/day
        phi = tools.wrap_degs((r['phi1'] - delta_w * t1 + r['phi2'] - delta_w * t2) / 2)  # in degs
        print('%s: %.6f deg/day, %.2f deg' % (batch, delta_w, phi))
    # as phi;


if __name__ == '__main__':
    if False:
        manual_approach()
        quit()

    fname = sys.argv[1] if len(sys.argv)>1 else 'rose-akaze+real-20190227-001506-fvals.log'
    with open(os.path.join(LOG_DIR, fname)) as fh:
        data = np.array([line.split('\t') for line in fh.readlines()]).astype('float')

    sm = RosettaSystemModel()
    Y = quaternion.from_float_array(data[:, 1:5])
    T = data[:, 5]

    # n=195
    I = data[:, 6] < 30
    T = T[I]
    Y = Y[I]

    def costfun(x, sm, T, Y, verbose=0):
        sm.asteroid.rotation_pm = tools.wrap_rads(x[0])
        if len(x) > 1:
            sm.asteroid.rotation_velocity = x[1]
        if len(x) > 2:
            sm.asteroid.axis_latitude = x[2]
            sm.asteroid.axis_longitude = x[3]

        errs = []
        for i, y in enumerate(Y):
            sm.time.value = T[i]
            ast_q = sm.asteroid_q()
            errs.append(tools.wrap_degs(math.degrees(tools.angle_between_q(ast_q, y))))
        errs = np.array(errs)
        max_err = np.percentile(np.abs(errs), 99)+100
        err = np.mean(errs[np.abs(errs) < max_err]**2)

        if verbose > 1:
            I = np.abs(errs) < max_err
            #plt.plot((T[I]-np.min(T))/np.ptp(T), errs[I])
            plt.plot(errs[I])
            plt.show()
        if verbose > 0:
            base_w = 2 * math.pi / 12.4043 * 24
            print(('%.2f' + (', %.6f' if len(x) > 1 else '') + (', %.2f, %.2f' if len(x) > 2 else '') + ' => %.3f') % (
                (math.degrees(tools.wrap_rads(x[0])),)
                + ((math.degrees(x[1]*3600*24 - base_w),) if len(x) > 1 else tuple())
                + ((math.degrees(x[2]), math.degrees(x[3]),) if len(x) > 2 else tuple())
                + (err,)))
        return err

    ast = sm.asteroid
    inival = [ast.rotation_pm]
    if True:
        inival += [ast.rotation_velocity]
    if False:
        inival += [ast.axis_latitude, ast.axis_longitude]

    res = minimize(costfun, inival, args=(sm, T, Y, 1),
                   #method="BFGS", options={'maxiter': 10, 'eps': 1e-3, 'gtol': 1e-3})
                   method="Nelder-Mead", options={'maxiter': 120, 'xtol': 1e-4, 'ftol': 1e-4})
                   #method="COBYLA", options={'rhobeg': 1.0, 'maxiter': 200, 'disp': False, 'catol': 0.0002})

    print('%s' % res)
    costfun(res.x, sm, T, Y, 2)