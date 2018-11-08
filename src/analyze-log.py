import sys
import csv
import math

import numpy as np
import matplotlib.pyplot as plt

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


# read logfiles
def read_data(sm, logfile, predictors, target):
    X, y = [], []
    
    with open(logfile, newline='') as csvfile:
        diam = 2 # avg diam 2km for asteroid
        data = csv.reader(csvfile, delimiter='\t')
        first = True
        for row in data:
            if len(row)>10:
                if first:
                    first = False
                    prd_i = [row.index(p) for p in predictors if p not in ('distance', 'visible')]
                    trg_i = row.index(target)
                    pos_i = [row.index(p+' sc pos') for p in ('x','y','z')]
                else:
                    row = np.array(row)
                    pos = row[pos_i].astype(np.float)
                    distance = np.sqrt(np.sum(pos**2))
                    xt = abs(pos[2])*math.tan(math.radians(sm.cam.x_fov)/2)
                    yt = abs(pos[2])*math.tan(math.radians(sm.cam.y_fov)/2)
                    xm = np.clip((xt - (abs(pos[0])-diam/2))/diam, 0, 1)
                    ym = np.clip((yt - (abs(pos[1])-diam/2))/diam, 0, 1)
                    
                    X.append(np.concatenate((
                        row[prd_i].astype(np.float),
                        [distance],
                        [xm*ym],
                    )))
                    
                    # err m/km
                    tmp = row[trg_i].astype(np.float)*1000 if len(row)>trg_i else float('nan')
                    y.append(tmp) #/ distance
    
    X = np.array(X)
    
    # for classification of fails
    yc = np.isnan(y)
    
    # for regression
    yr = np.array(y)
    yr[np.isnan(yr)] = np.nanmax(yr)
    
    return X, yc, yr


if __name__ == '__main__':
    if len(sys.argv)<2:
        print('USAGE: python analyze-log.py <path to log file>')
        sys.exit()
        
    logfile = sys.argv[1]
    one_d_only = len(sys.argv) > 2 and sys.argv[2] == '1d'

    if logfile[:4] == 'rose':
        sm = RosettaSystemModel()
    elif logfile[:4] == 'didy':
        sm = DidymosSystemModel()
    else:
        print('defaulting to rosetta mission')
        sm = RosettaSystemModel()

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
    target = 'rel shift error' #'shift error km' #if not one_d_only else 'dist error'
    
    # read data
    X, yc, yr = read_data(sm, logfile, predictors, target)
    X[:,1] = np.abs(tools.wrap_degs(X[:,1]))

    if one_d_only:
        n_groups = 10
        #yr = yr/1000
        for idx in (0,1,2,):
            xmin, xmax = np.min(X[:,idx]), np.max(X[:,idx])
            #x = [1/v for v in np.linspace(1/xmin, 1/xmax, n_groups+1)]
            x = np.linspace(xmin, xmax, n_groups + 1)
            y_grouped = [yr[np.logical_and(np.logical_not(yc), np.logical_and(X[:,idx]>x[i], X[:,idx]<x[i+1]))] for i in range(n_groups)]
            means = [np.mean(yg) for yg in y_grouped]
            stds = [np.std(yg) for yg in y_grouped]
            #means = [np.percentile(yg, 50) for yg in y_grouped]
            #stds = np.subtract([np.percentile(yg, 68) for yg in y_grouped], means)
            fig, ax = plt.subplots(figsize=(10, 4))
            index = np.arange(n_groups)
            bar_width = 0.35
            opacity = 0.4
            error_config = {'ecolor': '0.3'}
            rects1 = ax.bar(index, means, bar_width,
                            alpha=opacity, color='b',
                            yerr=stds, error_kw=error_config,
                            label='error')

            ax.set_xlabel(predictor_labels[idx])
            ax.set_ylabel('Error (m/km)')
            ax.set_title('Error (m/km) by %s'%predictor_labels[idx])
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(['%d-%d'%(x[i],x[i+1]) for i in range(n_groups)])
            #ax.legend()

            fig.tight_layout()
            plt.show()

    else:
        pairs = (
            (0,1),
            (0,2),
            (1,2),
    #        (0,3),(1,3),(2,3),
        )
        for pair in pairs:
            xmin, xmax = np.min(X[:,pair[0]]), np.max(X[:,pair[0]])
            ymin, ymax = np.min(X[:,pair[1]]), np.max(X[:,pair[1]])
            xx, yy = np.meshgrid(np.linspace(xmin, xmax, 50), np.linspace(ymin, ymax, 50))

            kernel = 0.01*RBF(length_scale=((xmax-xmin)*2, (ymax-ymin)*2))
            if False:
                y=yc
                # fit hyper parameters
                kernel += 0.1*WhiteKernel(noise_level=0.001)
                gpc = GaussianProcessClassifier(kernel=kernel, warm_start=True).fit(X[:,pair], yc)
                # hyper parameter results
                res = gpc.kernel_, gpc.log_marginal_likelihood(gpc.kernel_.theta)
                # classify on each grid point
                P = gpc.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
            else:
                y = yr
                # fit hyper parameters
                kernel += 4.0*WhiteKernel(noise_level=4.0)
                gpr = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True).fit(X[:,pair], yr)
                # hyper parameter results
                res = gpr.kernel_, gpr.log_marginal_likelihood(gpr.kernel_.theta)
                # regress on each grid point
                P = gpr.predict(np.vstack((xx.ravel(), yy.ravel())).T)

            P = P.reshape(xx.shape)

            # plot classifier output
            fig = plt.figure(figsize=(8, 8))
            if True:
                print('%s'%((np.min(P),np.max(P),np.min(y),np.max(y)),))
                image = plt.imshow(P, interpolation='nearest', extent=(xmin, xmax, ymin, ymax),
                                   aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
                plt.scatter(X[:,pair[0]], X[:,pair[1]], s=30, c=y, cmap=plt.cm.Paired, edgecolors=(0, 0, 0))
                cb = plt.colorbar(image)
                ax = fig.gca()
            else:
                from mpl_toolkits.mplot3d import Axes3D
                from matplotlib.colors import Normalize
                ax = fig.gca(projection='3d')
                scalarMap = plt.cm.ScalarMappable(norm=Normalize(vmin=np.min(P), vmax=np.max(P)), cmap=plt.cm.PuOr_r)
                ax.plot_surface(xx, yy, P, rstride=1, cstride=1, facecolors=scalarMap.to_rgba(P), antialiased=True)

            cb.ax.tick_params(labelsize=18)
            ax.tick_params(labelsize=18)
            plt.xlabel(predictors[pair[0]], fontsize=22)
            plt.ylabel(predictors[pair[1]], fontsize=22)
            plt.axis([xmin, xmax, ymin, ymax])
            #plt.title("%s\n Log-Marginal-Likelihood:%.3f" % res, fontsize=12)
            plt.tight_layout()
            plt.show()

