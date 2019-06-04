import sys
import csv
import math

import numpy as np
import matplotlib.pyplot as plt

from batch1 import get_system_model
from missions.didymos import DidymosSystemModel
from missions.rosetta import RosettaSystemModel

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
except:
    print('Requires scikit-learn, install using "conda install scikit-learn"')
    sys.exit()

from settings import *
from algo import tools

EASY_LIMITS = ((80, 180), (0, 12), (50, 270), (0.8, 1))
FAIL_ERRS = {
    'rel shift error (m/km)': 200,
    'altitude error': 2000,
    'dist error (m/km)': 200,
    'lat error (m/km)': 200,
    'rot error': 25,
}
MAX_ROTATION_ERR = 7

# read logfiles
def read_data(sm, logfile, predictors, target):
    X, y, rot_err, labels = [], [], [], []
    
    with open(logfile, newline='') as csvfile:
        rad = sm.asteroid.mean_radius * 0.001
        data = csv.reader(csvfile, delimiter='\t')
        first = True
        for row in data:
            if len(row)>10:
                if first:
                    first = False
                    prd_i = [row.index(p) for p in predictors if p not in ('distance', 'visible')]
                    trg_i = row.index(target)
                    rot_i = row.index('rot error')
                    pos_i = [row.index(p+' sc pos') for p in ('x','y','z')]
                    lbl_i = row.index('iter')
                else:
                    row = np.array(row)
                    try:
                        pos = row[pos_i].astype(np.float)
                    except ValueError as e:
                        print('Can\'t convert cols %s to float on row %s' % (pos_i, row[0]))
                        raise e
                    distance = np.sqrt(np.sum(pos**2))
                    xt = abs(pos[2])*math.tan(math.radians(sm.cam.x_fov)/2)
                    yt = abs(pos[2])*math.tan(math.radians(sm.cam.y_fov)/2)

                    #xm = np.clip((xt - (abs(pos[0])-rad))/rad/2, 0, 1)
                    #ym = np.clip((yt - (abs(pos[1])-rad))/rad/2, 0, 1)
                    xm = 1 - (max(0, pos[0]+rad - xt) + max(0, rad-pos[0] - xt))/rad/2
                    ym = 1 - (max(0, pos[1]+rad - yt) + max(0, rad-pos[1] - yt))/rad/2

                    X.append(np.concatenate((
                        row[prd_i].astype(np.float),
                        [distance],
                        [xm*ym],
                    )))
                    
                    # err m/km
                    tmp = row[trg_i].astype(np.float) if len(row)>trg_i else float('nan')
                    y.append(tmp)
                    rot_err.append(row[rot_i].astype(np.float))
                    labels.append(row[lbl_i])
    
    X = np.array(X)
    
    # for classification of fails
    yc = np.isnan(y)
    rot_err = np.array(rot_err)
    if True:
        yc = np.logical_or(yc, np.isnan(rot_err))
    if MAX_ROTATION_ERR > 0:
        I = np.logical_not(yc)
        rot_err[I] = np.abs(tools.wrap_degs(rot_err[I]))
        yc[I] = np.logical_or(yc[I], rot_err[I] > MAX_ROTATION_ERR)

    # for regression
    yr = np.array(y)
    #yr[np.isnan(yr)] = FAIL_ERRS[target]   # np.nanmax(yr)

    if target == 'rot error':
        yr = np.abs(tools.wrap_degs(yr))

    return X, yc, yr, labels


if __name__ == '__main__':
    if len(sys.argv)<2:
        print('USAGE: python analyze-log.py <path to log file> [gpr|1d|easy] [shift|alt|dist|lat|orient]')
        sys.exit()
        
    mode = sys.argv[2]
    if len(sys.argv) > 3:
        sc = 1
        if sys.argv[3] == 'shift':
            target = 'rel shift error (m/km)'
        elif sys.argv[3] == 'alt':
            target = 'altitude error'
            sc = 1000
        elif sys.argv[3] == 'dist':
            target = 'dist error (m/km)'
        elif sys.argv[3] == 'lat':
            target = 'lat error (m/km)'
        elif sys.argv[3] == 'orient':
            target = 'rot error'
        else:
            assert False, 'unknown target: %s' % sys.argv[3]

    predictors = (
        'sol elong',    # solar elongation
        'total dev angle',  # total angle between initial estimate and actual relative orientation
        'distance',    # distance of object
        'visible',      # esimate of % visible because of camera view edge
    )
    predictor_labels = (
        'Solar Elongation (deg)',
        'Initial orientation error (deg)',
        'Distance (km)',
        'In camera view (%)',
    )
    target = target or 'rel shift error (m/km)'  #'shift error km' #if not one_d_only else 'dist error'

    data = []
    for logfile in sys.argv[1].split(" "):
        mission = logfile.split('-')[0]
        sm = get_system_model(mission)

        # read data
        X, yc, yr, labels = read_data(sm, os.path.join(LOG_DIR, logfile), predictors, target)
        X[:, 1] = np.abs(tools.wrap_degs(X[:, 1]))
        data.append((logfile, X, yc, yr*sc, labels))


    if mode in ('1d', 'easy'):
        n_groups = 6
        #yr = yr/1000
        #idxs = (0, 1, 2, 3)
        idxs = (2,)
        for idx in idxs:
            fig, axs = plt.subplots(len(data), 1, figsize=(20, 18), sharex=True)
            for i, (logfile, X, yc, yr, labels) in enumerate(data):
                if mode == 'easy':
                    q997 = np.percentile(np.abs(yr), 99.7)
                    tmp = tuple((X[:, k] >= EASY_LIMITS[k][0], X[:, k] <= EASY_LIMITS[k][1]) for k in idxs if k!=idx)

                    # concatenate above & take logical and, also remove worst 0.3%
                    I = np.logical_and.reduce(sum(tmp, ()) + (np.logical_or(np.abs(yr) < q997, yr == FAIL_ERRS[target]),))
                else:
                    I = np.ones((X.shape[0],), dtype='bool')

                xmin, xmax = np.min(X[I, idx]), np.max(X[I, idx])
                ax = axs[i] if len(data) > 1 else axs
                line, = ax.plot(X[I, idx], yr[I], 'x')

                if n_groups:
                    # calc means and stds in bins

                    #x = [1/v for v in np.linspace(1/xmin, 1/xmax, n_groups+1)]
                    x = np.linspace(xmin, xmax, n_groups + 1)
                    y_grouped = [yr[np.logical_and.reduce((
                        I,
                        np.logical_not(yc),
                        X[:, idx] > x[i],
                        X[:, idx] < x[i+1],
                    ))] for i in range(n_groups)]
                    #means = [np.percentile(yg, 50) for yg in y_grouped]
                    means = np.array([np.mean(yg) for yg in y_grouped])
                    #stds = np.subtract([np.percentile(yg, 68) for yg in y_grouped], means)
                    stds = np.array([np.std(yg) for yg in y_grouped])
                    x = x.reshape((-1, 1))
                    stds = stds.reshape((-1, 1))
                    means = means.reshape((-1, 1))
                    xstep = np.concatenate((x, x), axis=1).flatten()[1:-1]
                    sstep = np.concatenate((stds, stds), axis=1).flatten()
                    mstep = np.concatenate((means, means), axis=1).flatten()
                    ax.plot(xstep, sstep, '-')
                    ax.plot(xstep, mstep, '-')
                    # bar_width = (xmax - xmin)/n_groups * 0.2
                    # rects1 = ax.bar((x[1:] + x[:-1]) * 0.5, stds, width=bar_width, bottom=means-stds/2,
                    #                 alpha=0.4, color='b', yerr=stds, error_kw={'ecolor': '0.3'}, label='error')

                else:
                    # filtered means, stds
                    xt = np.linspace(xmin, xmax, 100)

                    if False:
                        # exponential weight
                        weight_fun = lambda d: 0.01**abs(d/(xmax-xmin))
                    else:
                        # gaussian weight
                        from scipy.stats import norm
                        from scipy.interpolate import interp1d
                        interp = interp1d(xt-xmin, norm.pdf(xt-xmin, 0, (xmax-xmin)/10))
                        weight_fun = lambda d: interp(abs(d))

                    if False:
                        # use smoothed mean for std calc
                        yma = tools.smooth1d(X[I, idx], X[I, idx], yr[I], weight_fun)
                    else:
                        # use global mean for std calc (fast)
                        yma = np.mean(yr[I])

                    ym = tools.smooth1d(xt, X[I, idx], yr[I], weight_fun)
                    ystd = tools.smooth1d(xt, X[I, idx], (yr[I] - yma)**2, weight_fun) ** (1/2)

                    ax.plot(xt, ym, '-')
                    ax.plot(xt, ystd, '-')

                ax.set_title('%s: %s by %s' % (logfile, target, predictor_labels[idx]))
                ax.set_xlabel(predictor_labels[idx])
                ax.set_ylabel(target)
                ax.set_yticks(range(-200, 201, 50))
                ax.hlines(range(-200, 201, 10), xmin, xmax, '0.95', '--')
                ax.hlines(range(-200, 201, 50), xmin, xmax, '0.7', '-')
                plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=14)
                plt.setp(ax.get_yticklabels(), fontsize=14)
                tools.hover_annotate(fig, ax, line, np.array(labels)[I])

                #ax.set_xticks((x[1:] + x[:-1]) * 0.5)
                #ax.set_xticklabels(['%.2f-%.2f' % (x[i], x[i+1]) for i in range(n_groups)])
                #ax.legend()

                # operation zones for didymos mission
                if mission[:4]=='didy' and idx==2:
                    ax.set_xticks(np.arange(0.1, 10.5, 0.2))
                    if i==0:
                        ax.axvspan(1.1, 1.3, facecolor='cyan', alpha=0.3)
                        ax.axvspan(3.8, 4.2, facecolor='orange', alpha=0.3)
                    elif i==1:
                        ax.axvspan(0.15, 0.3, facecolor='pink', alpha=0.5)
                        ax.axvspan(1.1, 1.3, facecolor='cyan', alpha=0.3)
                    elif i == 2:
                        ax.axvspan(3.8, 4.2, facecolor='orange', alpha=0.3)
                    elif i==3:
                        ax.axvspan(1.1, 1.3, facecolor='cyan', alpha=0.3)
                        ax.axvspan(2.8, 5.2, facecolor='orange', alpha=0.3)

            plt.tight_layout()
            while(plt.waitforbuttonpress() == False):
                pass

    elif mode == '2d':

        # 0: solar elong
        # 1: initial deviation angle
        # 2: distance
        # 3: ratio in view
        idxs = tuple(range(4))
        pairs = (
        #    (2, 0),
        #    (1, 3),
            (2, 3),
            (0, 1),
        )

        titles = ['ORB', 'AKAZE', 'SURF', 'SIFT']
        nd = len(data)
        r, c = {
            1: (1, 1),
            2: (1, 2),
            3: (3, 1),
            4: (2, 2),
        }[nd]
        fig, axs = plt.subplots(r, c*len(pairs), figsize=(32, 18))

        for j, (logfile, X, yc, yr, labels) in enumerate(data):
            for i, (i0, i1) in enumerate(pairs):
                ax = axs.flatten()[j*len(pairs) + i]

                # filter out difficult regions of axis that are not shown
                tmp = tuple((X[:, k] >= EASY_LIMITS[k][0], X[:, k] <= EASY_LIMITS[k][1]) for k in idxs if k not in (i0, i1))
                I = np.logical_and.reduce(sum(tmp, ()))

                # add some offset if ratio in view is one so that they dont all stack in same place
                offsets = (X[I, 3] == 1) * np.random.uniform(0, 0.2, (np.sum(I),))
                off0 = 0 if i0 != 3 else offsets
                off1 = 0 if i1 != 3 else offsets

                line = ax.scatter(X[I, i0] + off0, X[I, i1] + off1, s=60, c=yc[I], cmap=plt.cm.Paired, alpha=0.5)  #edgecolors=(0, 0, 0))
                ax.tick_params(labelsize=18)
                ax.set_xlabel(predictors[i0], fontsize=22)
                ax.set_ylabel(predictors[i1], fontsize=22)
                tools.hover_annotate(fig, ax, line, np.array(labels)[I])

                if i==0:
                    col, row = j%c, j//c
                    fig.text(0.26+col*0.5, 0.96-row*0.5, titles[j], fontsize=30, horizontalalignment='center')
                # ax.set_xbound(xmin, xmax)
                # ax.set_ybound(ymin, ymax)

        plt.tight_layout()
        plt.subplots_adjust(top=0.94, hspace=0.3, wspace=0.25)
        plt.show()

    elif mode == 'gpr':
        logfile, X, yc, yr, labels = data[0]

        pairs = (
            (0, 1),
            (0, 2),
            (1, 2),
            #        (0,3),(1,3),(2,3),
        )
        for pair in pairs:
            xmin, xmax = np.min(X[:, pair[0]]), np.max(X[:, pair[0]])
            ymin, ymax = np.min(X[:, pair[1]]), np.max(X[:, pair[1]])
            xx, yy = np.meshgrid(np.linspace(xmin, xmax, 50), np.linspace(ymin, ymax, 50))

            kernel = 0.01 * RBF(length_scale=((xmax - xmin) * 2, (ymax - ymin) * 2))
            if False:
                y = yc
                # fit hyper parameters
                kernel += 0.1 * WhiteKernel(noise_level=0.001)
                gpc = GaussianProcessClassifier(kernel=kernel, warm_start=True).fit(X[:, pair], yc)
                # hyper parameter results
                res = gpc.kernel_, gpc.log_marginal_likelihood(gpc.kernel_.theta)
                # classify on each grid point
                P = gpc.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
            else:
                y = yr
                # fit hyper parameters
                kernel += 4.0 * WhiteKernel(noise_level=4.0)
                gpr = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True).fit(X[:, pair], yr)
                # hyper parameter results
                res = gpr.kernel_, gpr.log_marginal_likelihood(gpr.kernel_.theta)
                # regress on each grid point
                P = gpr.predict(np.vstack((xx.ravel(), yy.ravel())).T)

            P = P.reshape(xx.shape)

            # plot classifier output
            fig = plt.figure(figsize=(8, 8))
            if True:
                print('%s' % ((np.min(P), np.max(P), np.min(y), np.max(y)),))
                image = plt.imshow(P, interpolation='nearest', extent=(xmin, xmax, ymin, ymax),
                                   aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
                plt.scatter(X[:, pair[0]], X[:, pair[1]], s=30, c=y, cmap=plt.cm.Paired, edgecolors=(0, 0, 0))
                cb = plt.colorbar(image)
                ax = fig.gca()
            else:
                from mpl_toolkits.mplot3d import Axes3D
                from matplotlib.colors import Normalize

                ax = fig.gca(projection='3d')
                scalarMap = plt.cm.ScalarMappable(norm=Normalize(vmin=np.min(P), vmax=np.max(P)),
                                                  cmap=plt.cm.PuOr_r)
                ax.plot_surface(xx, yy, P, rstride=1, cstride=1, facecolors=scalarMap.to_rgba(P), antialiased=True)

            cb.ax.tick_params(labelsize=18)
            ax.tick_params(labelsize=18)
            plt.xlabel(predictors[pair[0]], fontsize=22)
            plt.ylabel(predictors[pair[1]], fontsize=22)
            plt.axis([xmin, xmax, ymin, ymax])
            # plt.title("%s\n Log-Marginal-Likelihood:%.3f" % res, fontsize=12)
            plt.tight_layout()
            plt.show()

    elif mode == '3d':
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

        logfile, X, yc, yr, labels = data[0]
        xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
        ymin, ymax = np.min(X[:, 1]), np.max(X[:, 1])
        zmin, zmax = np.min(X[:, 2]), np.max(X[:, 2])

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=yr, cmap=plt.cm.Paired, edgecolors=(0, 0, 0))

        #        cb = plt.colorbar(image)
        #        cb.ax.tick_params(labelsize=18)
        ax.tick_params(labelsize=18)
        ax.set_xlabel(predictors[0], fontsize=22)
        ax.set_ylabel(predictors[1], fontsize=22)
        ax.set_zlabel(predictors[2], fontsize=22)
        ax.set_xbound(xmin, xmax)
        ax.set_ybound(ymin, ymax)
        ax.set_zbound(zmin, zmax)

        plt.tight_layout()
        plt.show()
    else:
        assert False, 'wrong mode'

    #plt.waitforbuttonpress()
