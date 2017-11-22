import sys
import csv

import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
except:
    print('Requires scikit-learn, install using "conda install scikit-learn"')
    sys.exit()

from algo import tools


# read logfiles
def read_data(logfile, predictors, target):
    X, y = [], []
    with open(logfile, newline='') as csvfile:
        data = csv.reader(csvfile, delimiter='\t')
        first = True
        for row in data:
            if len(row)>10:
                if first:
                    first = False
                    prd_i = [row.index(p) for p in predictors if p != 'distance']
                    trg_i = row.index(target)
                    d = [row.index(p+' sc pos') for p in ('x','y','z')]
                else:
                    row = np.array(row)
                    X.append(np.concatenate((row[prd_i].astype(np.float), [np.sqrt(np.sum(row[d].astype(np.float)**2))])))
                    y.append(row[trg_i].astype(np.float))
    
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
    predictors = (
        'sol elong',    # solar elongation
        'total dev angle',  # total angle between initial estimate and actual relative orientation
        'distance',    # distance of object
#        'margin',      # esimate of % visible because of camera view edge
    )
    target = 'shift error km'
    
    # read data
    X, yc, yr = read_data(logfile, predictors, target)
    X[:,1] = np.abs(tools.wrap_degs(X[:,1]))
    
    pairs = ((0,1),(0,2),(1,2))
    for pair in pairs:
        xmin, xmax = np.min(X[:,pair[0]]), np.max(X[:,pair[0]])
        ymin, ymax = np.min(X[:,pair[1]]), np.max(X[:,pair[1]])
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 50), np.linspace(ymin, ymax, 50))
        
        kernel = 1.0*RBF(length_scale=(xmax-xmin, ymax-ymin)) + 1.0*WhiteKernel(noise_level=0.1)
        if True:
            y=yc
            # fit hyper parameters
            gpc = GaussianProcessClassifier(kernel=kernel, warm_start=True).fit(X[:,pair], yc)
            # hyper parameter results
            res = gpc.kernel_, gpc.log_marginal_likelihood(gpc.kernel_.theta)
            # classify on each grid point
            P = gpc.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
        else:
            y = yr
            # fit hyper parameters
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
            plt.colorbar(image)
        else:
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.colors import Normalize
            ax = fig.gca(projection='3d')
            scalarMap = plt.cm.ScalarMappable(norm=Normalize(vmin=np.min(P), vmax=np.max(P)), cmap=plt.cm.PuOr_r)
            ax.plot_surface(xx, yy, P, rstride=1, cstride=1, facecolors=scalarMap.to_rgba(P), antialiased=True)
        
        plt.xlabel(predictors[pair[0]])
        plt.ylabel(predictors[pair[1]])
        plt.axis([xmin, xmax, ymin, ymax])
        plt.title("%s\n Log-Marginal-Likelihood:%.3f" % res, fontsize=12)
        plt.tight_layout()
        plt.show()

