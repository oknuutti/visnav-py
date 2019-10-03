import sys
import csv
import math
from itertools import product

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.linalg import inv

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

from algo import tools
from iotools.readlog import read_data
from settings import *


# phase angle (180-elong), ini err, distance, visibility
EASY_LIMITS = ((20, 100), (0, 10), (3.5, 4.5), (0.8, 1))
FIG_SIZE = (8, 6)
FONT_SIZE = 5
MARKER_SIZE = 2
LINE_WIDTH = 0.5


def main():
#    mpl.style.use('classic')
    mpl.rcParams['font.size'] = FONT_SIZE
    mpl.rcParams['lines.markersize'] = MARKER_SIZE
    mpl.rcParams['lines.linewidth'] = LINE_WIDTH

    if len(sys.argv) < 2:
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
    else:
        target = 'rel shift error (m/km)'  # 'shift error km' #if not one_d_only else 'dist error'

    predictors = (
        'sol elong',  # solar elongation
        'total dev angle',  # total angle between initial estimate and actual relative orientation
        'distance',  # distance of object
        'visible',  # esimate of % visible because of camera view edge
    )
    predictor_labels = (
        'Phase Angle (deg)',
        'Initial orientation error (deg)',
        'Distance (km)',
        'In camera view (%)',
    )

    data = []
    for logfile in sys.argv[1].split(" "):
        mission = logfile.split('-')[0]
        sm = get_system_model(mission)

        # read data
        X, Y, yc, labels = read_data(sm, os.path.join(LOG_DIR, logfile), predictors, [target])
        #X[:, 1] = np.abs(tools.wrap_degs(X[:, 1]))
        data.append((logfile, X, yc, Y.flatten(), labels))

    title_map = {
        'didy2w': 'D2, WAC',
        'didy2n': 'D2, NAC',
        'didy1w': 'D1, WAC',
        'didy1n': 'D1, NAC',
    }

    if mode in ('1d', 'easy'):
        n_groups = 6
        # yr = yr/1000
        # idxs = (0, 1, 2, 3)
        idxs = (2,)
        for idx in idxs:
            fig, axs = plt.subplots(len(data), 1, figsize=FIG_SIZE, sharex=True)
            for i, (logfile, X, yc, yr, labels) in enumerate(data):
                if mode == 'easy':
                    q997 = np.percentile(np.abs(yr), 99.7)
                    tmp = tuple((X[:, k] >= EASY_LIMITS[k][0], X[:, k] <= EASY_LIMITS[k][1]) for k in idxs if k != idx)

                    # concatenate above & take logical and, also remove worst 0.3%
                    I = np.logical_and.reduce(
                        sum(tmp, ()) + (np.logical_or(np.abs(yr) < q997, yr == FAIL_ERRS[target]),))
                else:
                    I = np.ones((X.shape[0],), dtype='bool')

                xmin, xmax = np.min(X[I, idx]), np.max(X[I, idx])
                ax = axs[i] if len(data) > 1 else axs
                line, = ax.plot(X[I, idx], yr[I], 'x')

                if n_groups:
                    # calc means and stds in bins

                    # x = [1/v for v in np.linspace(1/xmin, 1/xmax, n_groups+1)]
                    x = np.linspace(xmin, xmax, n_groups + 1)
                    y_grouped = [yr[np.logical_and.reduce((
                        I,
                        np.logical_not(yc),
                        X[:, idx] > x[i],
                        X[:, idx] < x[i + 1],
                    ))] for i in range(n_groups)]
                    # means = [np.percentile(yg, 50) for yg in y_grouped]
                    means = np.array([np.mean(yg) for yg in y_grouped])
                    # stds = np.subtract([np.percentile(yg, 68) for yg in y_grouped], means)
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
                        weight_fun = lambda d: 0.01 ** abs(d / (xmax - xmin))
                    else:
                        # gaussian weight
                        from scipy.stats import norm
                        from scipy.interpolate import interp1d
                        interp = interp1d(xt - xmin, norm.pdf(xt - xmin, 0, (xmax - xmin) / 10))
                        weight_fun = lambda d: interp(abs(d))

                    if False:
                        # use smoothed mean for std calc
                        yma = tools.smooth1d(X[I, idx], X[I, idx], yr[I], weight_fun)
                    else:
                        # use global mean for std calc (fast)
                        yma = np.mean(yr[I])

                    ym = tools.smooth1d(xt, X[I, idx], yr[I], weight_fun)
                    ystd = tools.smooth1d(xt, X[I, idx], (yr[I] - yma) ** 2, weight_fun) ** (1 / 2)

                    ax.plot(xt, ym, '-')
                    ax.plot(xt, ystd, '-')

                # ax.set_title('%s: %s by %s' % (logfile, target, predictor_labels[idx]))
                ax.set_title('%s' % title_map[logfile.split('-')[0]])
                if i == len(data)-1:
                    ax.set_xlabel(predictor_labels[idx])
                ax.set_ylabel(target)
                ax.set_yticks(range(-200, 201, 50))
                ax.hlines(range(-200, 201, 10), xmin, xmax, '0.95', '--')
                ax.hlines(range(-200, 201, 50), xmin, xmax, '0.7', '-')
                plt.setp(ax.get_xticklabels(), rotation=45)
#                plt.setp(ax.get_yticklabels())
                tools.hover_annotate(fig, ax, line, np.array(labels)[I])

                # ax.set_xticks((x[1:] + x[:-1]) * 0.5)
                # ax.set_xticklabels(['%.2f-%.2f' % (x[i], x[i+1]) for i in range(n_groups)])
                # ax.legend()

                # operation zones for didymos mission
                if mission[:4] == 'didy' and idx == 2:
                    ax.set_xticks(np.arange(0.1, 10.5, 0.2))
                    if i == 0:
                        ax.axvspan(1.1, 1.3, facecolor='cyan', alpha=0.3)
                        ax.axvspan(3.8, 4.2, facecolor='orange', alpha=0.3)
                    elif i == 1:
                        ax.axvspan(0.15, 0.3, facecolor='pink', alpha=0.5)
                        ax.axvspan(1.1, 1.3, facecolor='cyan', alpha=0.3)
                    elif i == 2:
                        ax.axvspan(3.8, 4.2, facecolor='orange', alpha=0.3)
                    elif i == 3:
                        ax.axvspan(1.1, 1.3, facecolor='cyan', alpha=0.3)
                        ax.axvspan(2.8, 5.2, facecolor='orange', alpha=0.3)

            plt.tight_layout()
            plt.subplots_adjust(top=0.94, hspace=0.35)
            while (plt.waitforbuttonpress() == False):
                pass

    elif mode == '2d':

        scatter = False

        # 0: solar elong
        # 1: initial deviation angle
        # 2: distance
        # 3: ratio in view
        idxs = tuple(range(4))
        pairs = (
            (2, 0),
            #    (1, 3),
            #    (2, 3),
            #    (0, 1),
        )

        # titles = ['ORB', 'AKAZE', 'SURF', 'SIFT']
        # titles = [d[0][:-4] for d in data]
        titles = [title_map[d[0].split('-')[0]] for d in data]

        nd = len(data)
        r, c = {
            1: (1, 1),
            2: (1, 2),
            3: (3, 1),
            4: (2, 2),
        }[nd]
        if scatter:
            fig, axs = plt.subplots(r, c * len(pairs), figsize=FIG_SIZE)
        else:
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.colors import Normalize
            fig = plt.figure(figsize=FIG_SIZE)
            fig2 = plt.figure(figsize=FIG_SIZE)

        for j, (logfile, X, yc, yr, labels) in enumerate(data):
            for i, (i0, i1) in enumerate(pairs):
                if scatter:
                    # filter out difficult regions of axis that are not shown
                    tmp = tuple(
                        (X[:, k] >= EASY_LIMITS[k][0], X[:, k] <= EASY_LIMITS[k][1]) for k in idxs if k not in (i0, i1))
                    I = np.logical_and.reduce(sum(tmp, ()))
                else:
                    I = np.ones((X.shape[0],), dtype='bool')

                # add some offset if ratio in view is one so that they dont all stack in same place
                offsets = (X[I, 3] == 1) * np.random.uniform(0, 0.2, (np.sum(I),))
                off0 = 0 if i0 != 3 else offsets
                off1 = 0 if i1 != 3 else offsets

                if scatter:
                    ax = axs.flatten()[j * len(pairs) + i]
                    line = ax.scatter(X[I, i0] + off0, X[I, i1] + off1, s=60, c=yc[I], cmap=plt.cm.Paired,
                                      alpha=0.5)  # edgecolors=(0, 0, 0))
                    tools.hover_annotate(fig, ax, line, np.array(labels)[I])
                else:
                    ax = fig.add_subplot(r, c * len(pairs), j * len(pairs) + i + 1, projection='3d')
                    ax2 = fig2.add_subplot(r, c * len(pairs), j * len(pairs) + i + 1, projection='3d')

                    xmin, xmax = np.min(X[I, i0]), np.max(X[I, i0])
                    ymin, ymax = np.min(X[I, i1]), np.max(X[I, i1])
                    x = np.linspace(xmin, xmax, 7)
                    y = np.linspace(ymin, ymax, 7) if i1 != 0 else np.array([0, 22.5, 45, 67.5, 90, 112.5, 135])
                    xx, yy = np.meshgrid(x, y)

                    y_grouped = [[yr[np.logical_and.reduce((
                        I,
                        np.logical_not(yc),
                        X[:, i0] > x[i],
                        X[:, i0] < x[i + 1],
                        X[:, i1] > y[j],
                        X[:, i1] < y[j + 1],
                    ))] for i in range(len(x) - 1)] for j in range(len(y) - 1)]

                    # std when we assume zero mean, remove lowest 0.1% and highest 0.1% before calculating stats
                    stds = np.array(
                        [[np.sqrt(tools.robust_mean(y_grouped[j][i]**2, 0.1)) for i in range(len(x) - 1)] for j in range(len(y) - 1)])
                    samples = np.array([[len(y_grouped[j][i]) for i in range(len(x) - 1)] for j in range(len(y) - 1)])

                    d_coefs, pa_coefs, mod_stds = model_stds2(stds, samples)

                    print_model(titles[j], x, d_coefs, y, pa_coefs)

                    xstep = np.kron(xx, np.ones((2, 2)))[1:-1, 1:-1]
                    ystep = np.kron(yy, np.ones((2, 2)))[1:-1, 1:-1]
                    # mstep = np.kron(means, np.ones((2, 2)))
                    sstep = np.kron(stds, np.ones((2, 2)))
                    msstep = np.kron(mod_stds, np.ones((2, 2)))

                    scalarMap = plt.cm.ScalarMappable(norm=Normalize(vmin=np.min(sstep), vmax=np.max(sstep)), cmap=plt.cm.PuOr_r)
                    ax.plot_surface(xstep, ystep, sstep, facecolors=scalarMap.to_rgba(sstep), antialiased=True)
                    ax.view_init(30, -60)

                    ax2.plot_surface(xstep, ystep, msstep, facecolors=scalarMap.to_rgba(msstep), antialiased=True)
                    ax2.view_init(30, -60)

                # ax.tick_params(labelsize=18)
                ax.set_xlabel(predictor_labels[i0])
                ax.set_ylabel(predictor_labels[i1])
                ax2.set_xlabel(predictor_labels[i0])
                ax2.set_ylabel(predictor_labels[i1])

                if i == 0:
                    col, row = j % c, j // c
                    fig.text(0.36 + col * 0.5, 0.96 - row * 0.5, titles[j], horizontalalignment='center')
                    fig2.text(0.36 + col * 0.5, 0.96 - row * 0.5, titles[j], horizontalalignment='center')
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
            fig = plt.figure(figsize=FIG_SIZE)
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

            # cb.ax.tick_params(labelsize=18)
            # ax.tick_params(labelsize=18)
            plt.xlabel(predictors[pair[0]])
            plt.ylabel(predictors[pair[1]])
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

        fig = plt.figure(figsize=FIG_SIZE)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=yr, cmap=plt.cm.Paired, edgecolors=(0, 0, 0))

        #        cb = plt.colorbar(image)
        #        cb.ax.tick_params(labelsize=18)
        # ax.tick_params(labelsize=18)
        ax.set_xlabel(predictors[0])
        ax.set_ylabel(predictors[1])
        ax.set_zlabel(predictors[2])
        ax.set_xbound(xmin, xmax)
        ax.set_ybound(ymin, ymax)
        ax.set_zbound(zmin, zmax)

        plt.tight_layout()
        plt.show()
    else:
        assert False, 'wrong mode'

    # plt.waitforbuttonpress()


def model_stds(stds):
    """
    solve pa_coefs.dot(d_coefs.T) == stds by taking log on both sides,
    arrange equations so that can use pseudo inverse.
    E.g. for stds of size 2x2:
    A = [[1 0 1 0],
         [1 0 0 1],
         [0 1 1 0],
         [0 1 0 1]]
    y = np.log([stds[0,0],
                stds[0,1],
                stds[1,0]
                stds[1,1]])
    Aw == y
    => w = inv(A'A)A'y
    pa_coefs = np.exp(w[:2])
    d_coefs = np.exp(w[2:])

    ==> regrettably, turns out there's some bug, also cant figure out how to weight by sample count
    """

    n = len(stds)       # phase angle
    m = len(stds[0])    # distance
    A = np.zeros((n*m, n+m))
    y = np.zeros((n*m, 1))
    for k, (j, i) in enumerate(product(range(n), range(m))):
        A[k, j] = 1
        A[k, n+i] = 1
        y[k] = math.log(stds[j, i])

    # pseudo inverse
    w = inv(A.T.dot(A)).dot(A.T).dot(y)

    pa_coefs = np.exp(w[:n]).flatten()
    d_coefs = np.exp(w[n:]).flatten()
    mod_stds = np.array([[p*d for d in d_coefs] for p in pa_coefs])
    return d_coefs, pa_coefs, mod_stds


def model_stds2(stds, samples):
    n = len(stds)       # phase angle
    m = len(stds[0])    # distance

    def weighted_mse(w, n, Y, S):
        return np.average((np.array([[p * d for p in w[n:]] for d in w[:n]]) - Y)**2, weights=S)

    from scipy.optimize import minimize
    res = minimize(weighted_mse, np.ones(n+m), args=(n, stds, samples), method="BFGS",
                   options={'maxiter': 10, 'eps': 1e-06, 'gtol': 1e-4})

    #print('cost: %s' % res.fun)
    pa_coefs = res.x[:n]
    d_coefs = res.x[n:]
    mod_stds = np.array([[p * d for d in d_coefs] for p in pa_coefs])
    return d_coefs, pa_coefs, mod_stds


def print_model(name, x, d_coefs, y, pa_coefs):
    print('%s:\nd_lims = %s;\nd_coefs = %s;\npa_lims = %s;\npa_coefs = %s;\n' % (
        name, x[1:-1], d_coefs, y[1:-1], pa_coefs
    ))


if __name__ == '__main__':
    main()

