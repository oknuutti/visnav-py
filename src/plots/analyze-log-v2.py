import itertools
import sys
import re

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import MultipleLocator

from scipy.stats import norm
from scipy.interpolate import interp1d

from algo import tools
from iotools.readlog import read_data, FAIL_ERRS
from batch1 import get_system_model
from missions.didymos import DidymosSystemModel
from missions.rosetta import RosettaSystemModel

from settings import *


# distance, phase angle (180-elong), ini err, visibility
EASY_LIMITS = {
    'synth': [(50, 250), (20, 100), (0, 10), (80, 100)],
    'real': [(50, 250), (20, 100), (0, 10), (80, 100)],
}
PLOT_LIMITS = {
    'errs' : {
        'synth': [(35, 400), (0, 140), (0, 15), (20, 120)],
        'real':  [(35, 300), (20, 140), (0, 15), (20, 120)],
    },
    'fails': {
        'synth': [(0, 400), (0, 180), (0, 15), (0, 120)],
        'real': [(0, 300), (0, 180), (0, 15), (0, 120)],
    },
}

FONT_SIZE = 6
MARKER_SIZE = 4
LINE_WIDTH = 0.5
default_cycler = (cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']) +    # default colors in matplotlib v2.0
                  cycler(linestyle=['--', '-', ':', '-.']))

def windowed_percentile(xp, yr, nx, window=0.05, p_lim=68, plot_xlim=None):
    xmin, xmax = np.min(xp), np.max(xp)
    if plot_xlim:
        xmin = max(xmin, plot_xlim[0])
        xmax = min(xmax, plot_xlim[1])
    xt = np.linspace(xmin, xmax, nx)

    w = window * (xmax - xmin)
    # n = [np.sum(np.logical_and(xp > x-w/2, xp < x+w/2)) for x in xt]
    yt = np.array([np.percentile(yr[np.logical_and(xp > x-w/2, xp < x+w/2)], p_lim) for x in xt])
    return xt, yt


def smooth(xp, yr, nx, weight='mavg', wfun_coef=None, mode='mean+std', p_bins=1, p_lim=99, plot_xlim = None, high_acc_std=False):
    I = np.ones((xp.shape[0],), dtype='bool')
    if yr.dtype != np.bool:
        if p_bins > 1:
            # remove worst 1%, binned
            xmin, xmax = np.min(xp), np.max(xp)
            x = np.linspace(xmin, xmax, p_bins + 1)
            Is = [np.logical_and(xp > x[j], xp < x[j + 1]) for j in range(p_bins)]
            ylim = [np.percentile(yr[Is[j]], p_lim) for j in range(p_bins)]
            for j in range(p_bins):
                I[Is[j]] = yr[Is[j]] < ylim[j]
        elif p_bins == 1:
            # remove worst 1%, global
            ylim = np.percentile(yr, p_lim)
            I = yr < ylim

    xp, yr = xp[I], yr[I]
    xmin, xmax = xmin_t, xmax_t = np.min(xp), np.max(xp)
    if plot_xlim:
        xmin = max(xmin, plot_xlim[0])
        xmax = min(xmax, plot_xlim[1])

    # calculate filtered means, stds
    xt = np.linspace(xmin, xmax, nx)

    if weight == 'exp':
        # exponential weight (def wfun_coef=0.0001)
        wfun_coef = wfun_coef or 0.0001
        weight_fun = lambda d: wfun_coef ** abs(d / (xmax - xmin))
    elif weight == 'mavg':
        # moving average, i.e. sliding window (def wfun_coef=0.1)
        wfun_coef = wfun_coef or 0.1
        weight_fun = lambda d: 1 if abs(d) < wfun_coef * (xmax - xmin) else 0
    elif weight == 'gaussian':
        # gaussian weight (def wfun_coef=0.03)
        wfun_coef = wfun_coef or 0.03
        xi = np.linspace(xmin_t, xmax_t, nx*2)
        interp = interp1d(xi - xmin_t, norm.pdf(xi - xmin_t, 0, wfun_coef * (xmax - xmin)))
        weight_fun = lambda d: interp(abs(d))
    else:
        raise ValueError('wrong weight: %s'%weight)

    yt = 0
    if 'mean' in mode:
        try:
            ym = tools.smooth1d(xt, xp, yr, weight_fun)
        except ValueError:
            assert False, 'distance range %s doesnt cover that of %s' % (
                (np.min(xi - xmin_t), np.max(xi - xmin_t)),
                (0, np.max(xt) - np.min(xp)),
            )
        yt = ym

    if 'std' in mode:
        if high_acc_std:
            # use smoothed mean for std calc
            yma = tools.smooth1d(xp, xp, yr, weight_fun)
        else:
            # use global mean for std calc (fast)
            yma = np.mean(yr)

        try:
            ystd = tools.smooth1d(xt, xp, (yr - yma) ** 2, weight_fun) ** (1 / 2)
        except ValueError:
            assert False, 'xp range doesnt cover that of xt'

        if 'std2' in mode:
            yt += 2*ystd
        else:
            yt += ystd

    return xt, yt


if __name__ == '__main__':
    algos = ('orb', 'akaze', 'surf', 'sift')
    try:
        mission = re.match(r'^\w+$', sys.argv[1])[0]
        postfix = re.match(r'^\w+$', sys.argv[2])[0]
        mode = re.match(r'^fails|errs$', sys.argv[3])[0]
        image_type = re.match(r'^synth|real|both$', sys.argv[4])[0]
        sm_quality = re.match(r'^1k|4k|16k|17k$', sys.argv[5])[0]
        single_algo = False if len(sys.argv) < 7 else re.match('^'+'|'.join(algos)+'$', sys.argv[6])[0]
        assert not (mode == 'fails' and image_type == 'both'), 'Can only show both real and synth results when ' \
                                                             + 'plotting error curves, not for the failure scatter plot'
    except:
        print(('USAGE: python %s <mission> <logfile-postfix> <fails|errs> <synth|real|both> <1k|4k|16k|17k> ['+'|'.join(algos)+']') % (sys.argv[0],))
        quit()

    mpl.rcParams['font.size'] = FONT_SIZE
    mpl.rcParams['lines.markersize'] = MARKER_SIZE
    mpl.rcParams['lines.linewidth'] = LINE_WIDTH
    plt.rc('axes', prop_cycle=default_cycler)
    #mpl.rcParams['legend.fontsize'] = FONT_SIZE

    predictors = (
        'distance',     # distance of object
        'sol elong',    # solar elongation
        'total dev angle',  # total angle between initial estimate and actual relative orientation
        'visible',      # esimate of % visible because of camera view edge
    )
    predictor_labels = (
        'Distance (km)',
        'Phase Angle (deg)',
        'Initial orientation error (deg)',
        'In camera view (%)',
    )
    targets = (
        'dist error (m/km)',
        'lat error (m/km)',
        'rot error',
    )
    image_types = ('real', 'synth') if image_type == 'both' else (image_type,)
    algos = (single_algo,) if single_algo else algos

    sm = get_system_model(mission)
    if isinstance(sm, DidymosSystemModel):
        EASY_LIMITS['synth'][0] = (sm.min_med_distance, sm.max_med_distance)
        PLOT_LIMITS['errs']['synth'][0] = (sm.min_distance, sm.max_distance)
        PLOT_LIMITS['fails']['synth'][0] = (0, sm.max_distance)

    data = {'real': {a: None for a in algos}, 'synth': {a: None for a in algos}}
    for fname in os.listdir(LOG_DIR):
        m = re.match(mission + r"-([^-]+)-" + postfix + r"\.log", fname)
        if m:
            # further filter log-files
            specs = m[1].split('+')
            algo = re.match('|'.join(algos), m[1])
            itype = 'real' if 'real' in specs else 'synth'

            ok = algo and (
                sm_quality == '1k' and 'smn_' in specs
                or sm_quality == '4k' and 'smn' in specs
                or sm_quality in ('17k','16k') and ('smn' not in specs and 'smn_' not in specs)
            ) and (
                image_type == 'both'
                or image_type == 'real' and 'real' in specs
                or image_type == 'synth' and 'real' not in specs
            )

            if ok:
                # read data
                X, Y, yc, labels = read_data(sm, os.path.join(LOG_DIR, fname), predictors, targets)
                data[itype][algo[0]] = (X, Y, yc, labels)


    if mode == 'errs':
        predictor_idxs = (0, 1, 2)
        plot_ymax = {
            'synth' : (0.6, 15, 1.5, 3.5),
            'real'  : (0.6, 15, 1.5, 3.5),
        }
        plot_ymax['both'] = plot_ymax['real']
        fig, axs = plt.subplots(len(targets) + 1, len(predictor_idxs), figsize=(8, 6))
        for r, axr in enumerate(axs):
            axr[0].get_shared_y_axes().join(*axs[r])
        for c, axc in enumerate(axs[0]):
            axc.get_shared_x_axes().join(*[axr[c] for axr in axs])

        count = len(predictor_idxs) * (len(targets)+1) * len(image_types) * len(algos)
        i = 0

        old_itype = None
        for (pj, pi), (ti, target), itype, algo in itertools.product(
                enumerate(predictor_idxs), enumerate(('fails',) + targets), image_types, algos):
            tools.show_progress(count, i)
            ti -= 1
            i += 1
            try:
                X, Y, yc, labels = data[itype][algo]
            except TypeError:
                assert False, 'No log file found for: %s %s %s %s %s' % (mission, postfix, itype, algo, sm_quality)

            # remove difficult samples on other dimensions so that effect of plotted dimension more clear
            I0 = tuple((X[:, k] >= EASY_LIMITS[itype][k][0], X[:, k] <= EASY_LIMITS[itype][k][1])
                       for k in predictor_idxs if k != pi)
            I = np.logical_and.reduce(sum(I0, ()))
            if ti >= 0:
                if False:
                    # remove all failures and effect of FAIL_ERRS, and over MAX_ROTATION_ERR iterations
                    I = np.logical_and(I, np.logical_not(yc))
                else:
                    # remove only hard failures and only effect of FAIL_ERRS
                    I = np.logical_and(I, Y[:, targets.index('rot error')] != FAIL_ERRS['rot error'])
                yr = Y[I, ti]
            else:
                yr = yc[I]

            ax = axs[ti+1][pi]
            if single_algo:
                line, = ax.plot(X[I, pi], yr, 'x')

            xlim = PLOT_LIMITS[mode][itype][pi]
            if ti == -1:
                xt, yt = smooth(X[I, pi], yr, 100, mode='mean', plot_xlim=xlim, p_bins=0,
                                #weight='mavg', wfun_coef=0.1)
                                weight='gaussian', wfun_coef=0.05)
                lbl = ' μ'
            elif False:
                xt, yt = smooth(X[I, pi], yr, 100, mode='mean+std', plot_xlim=xlim,
                                p_bins=5, p_lim=95, weight='mavg', wfun_coef=0.1, high_acc_std=0)
                lbl = ' μ+1σ'
            else:
                xt, yt = windowed_percentile(X[I, pi], yr, 100, window=0.2, p_lim=50, plot_xlim=xlim)
                lbl = ' p50'

            # reset color cycling
            if old_itype is not None and old_itype != itype:
                ax.set_prop_cycle(None)
            old_itype = itype

            if single_algo:
                ax.plot(xt, yt, label=', '.join((itype, algo)) + lbl)
                if ti != -1:
                    xt2, yt2 = windowed_percentile(X[I, pi], yr, 100, window=0.2, p_lim=95, plot_xlim=xlim)
                    ax.plot(xt2, yt2, color='C1', label=', '.join((itype, algo)) + ' p95')
            else:
                # '--' if image_type == 'both' and itype == 'real' else '-'
                ax.plot(xt, yt, label=', '.join((itype, algo))+lbl)

            ax.set_xlim(*PLOT_LIMITS[mode][itype][pi])
            if not single_algo:
                ax.set_ylim(0, plot_ymax[image_type][ti+1])

            # ax.set_title('%s: %s by %s' % (logfile, target, predictor_labels[pi]))
            # ax.set_xticks((x[1:] + x[:-1]) * 0.5)
            # ax.set_yticks(range(-200, 201, 50))
            # ax.hlines(range(-200, 201, 10), xmin, xmax, '0.95', '--')
            # ax.hlines(range(-200, 201, 50), xmin, xmax, '0.7', '-')
#            plt.setp(ax.get_xticklabels())#, rotation=45)
#            plt.setp(ax.get_yticklabels())

            if single_algo:
                tools.hover_annotate(fig, ax, line, np.array(labels)[I])

            ax.legend()

            # operation zones for didymos mission
            if mission[:4] == 'didy' and pi == 0:
                min_dist, max_dist = np.min(X[I, pi]), np.max(X[I, pi])
                interval = 0.2
                ax.set_xticks(np.arange(min_dist//interval*interval, (max_dist//interval + 1)*interval, 0.2))
                if mission[:5] == 'didy1' and min_dist <= 3.8 and max_dist >= 4.2:
                    ax.axvspan(3.8, 4.2, facecolor='orange', alpha=0.3)
                if mission[:5] == 'didy2' and min_dist <= 2.8 and max_dist >= 5.2:
                    ax.axvspan(2.8, 5.2, facecolor='orange', alpha=0.3)
                if min_dist <= 1.1 and max_dist >= 1.3:
                    ax.axvspan(1.1, 1.3, facecolor='cyan', alpha=0.3)
                if min_dist <= 0.15 and max_dist >= 0.3:
                    ax.axvspan(0.15, 0.3, facecolor='pink', alpha=0.5)


            # zones where there's samples missing for real rosetta images
            if mission[:4] == 'rose' and image_type == 'real' and pi == 0:
                ax.axvspan(100, 170, facecolor='pink', alpha=0.5)
                ax.axvspan(220, PLOT_LIMITS[mode][image_type][0][1], facecolor='pink', alpha=0.5)

            if pj == 0:
                ax.set_ylabel(target)
            if ti+1 == len(targets):
                ax.set_xlabel(predictor_labels[pi])

        plt.tight_layout()
        plt.show()


    elif mode == 'fails':

        # 0: distance
        # 1: solar elong
        # 2: initial deviation angle
        # 3: ratio in view
        idxs = tuple(range(4))
        pairs = (
            (0, 3),
            (0, 1),
            (0, 2),
        )

        titles = ['ORB', 'AKAZE', 'SURF', 'SIFT']
        fig, axs = plt.subplots(len(pairs), len(algos), figsize=(8, 6))
        axs = np.atleast_2d(axs)   # need .T if only one algo?
        for r, axr in enumerate(axs):
            if len(axr) > 1:
                axr[0].get_shared_y_axes().join(*axs[r])
        axs.flatten()[0].get_shared_x_axes().join(*axs.flatten())

        count = len(algos) * len(pairs)
        j = 0

        for (ip, (i0, i1)), (ia, algo) in itertools.product(enumerate(pairs), enumerate(algos)):
            tools.show_progress(count, j)
            j += 1
            try:
                X, Y, yc, labels = data[image_type][algo]
            except TypeError as e:
                assert False, 'No log file found for: %s %s %s %s %s' % (mission, postfix, itype, algo, sm_quality)

            ax = axs[ip][ia]

            # filter out difficult regions of axis that are not shown
            tmp0 = tuple((X[:, k] >= EASY_LIMITS[image_type][k][0], X[:, k] <= EASY_LIMITS[image_type][k][1]) for k in idxs if k not in (i0, i1))
            tmp1 = tuple((X[:, k] >= PLOT_LIMITS[mode][image_type][k][0], X[:, k] <= PLOT_LIMITS[mode][image_type][k][1]) for k in idxs)
            I = np.logical_and.reduce(sum(tmp0, ())+sum(tmp1, ()))

            # add some offset if ratio in view is one so that they dont all stack in same place
            offsets = (X[I, 3] == 100) * np.random.uniform(0, 20, (np.sum(I),))
            off0 = 0 if i0 != 3 else offsets
            off1 = 0 if i1 != 3 else offsets

            line = ax.scatter(X[I, i0] + off0, X[I, i1] + off1, s=MARKER_SIZE, linewidth=LINE_WIDTH,
                              c=yc[I], cmap=plt.cm.Paired, alpha=0.5)  #edgecolors=(0, 0, 0))
            #ax.tick_params(labelsize=18)
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            tools.hover_annotate(fig, ax, line, np.array(labels)[I])
#            ax.set_xlim(*PLOT_LIMITS[mode][image_type][i0])
#            ax.set_ylim(*PLOT_LIMITS[mode][image_type][i1])
            # ax.set_xbound(xmin, xmax)
            # ax.set_ybound(ymin, ymax)

            # operation zones for didymos mission
            if mission[:4] == 'didy' and i0 == 0:
                min_dist, max_dist = np.min(X[I, i0]), np.max(X[I, i0])
                interval = 0.2
                ax.set_xticks(np.arange(min_dist//interval*interval, (max_dist//interval + 1)*interval, 0.2))
                if mission[:5] == 'didy1' and min_dist <= 3.8 and max_dist >= 4.2:
                    ax.axvspan(3.8, 4.2, facecolor='orange', alpha=0.3)
                if mission[:5] == 'didy2' and min_dist <= 2.8 and max_dist >= 5.2:
                    ax.axvspan(2.8, 5.2, facecolor='orange', alpha=0.3)
                if min_dist <= 1.1 and max_dist >= 1.3:
                    ax.axvspan(1.1, 1.3, facecolor='cyan', alpha=0.3)
                if min_dist <= 0.15 and max_dist >= 0.3:
                    ax.axvspan(0.15, 0.3, facecolor='pink', alpha=0.5)

            if ip == 0:
                ax.set_title(algo.upper() if single_algo else titles[ia])
                # col, row = j%c, j//c
                # fig.text(0.26+col*0.5, 0.96-row*0.5, titles[j], fontsize=30, horizontalalignment='center')
            if ip == len(pairs) - 1:
                ax.set_xlabel(predictor_labels[i0])
            if ia == 0:
                ax.set_ylabel(predictor_labels[i1])

        plt.tight_layout()
        #plt.subplots_adjust(top=0.94, hspace=0.3, wspace=0.25)
        plt.subplots_adjust(wspace=0.3)
        plt.show()

    else:
        assert False, 'wrong mode'
