import csv
import math

import numpy as np

from algo import tools


MAX_ROTATION_ERR = 7
FAIL_ERRS = {
    'rel shift error (m/km)': 100,
    'altitude error': 1000,
    'dist error (m/km)': 100,
    'lat error (m/km)': 5,
    'rot error': 15,
}


# read logfiles
def read_data(sm, logfile, predictors, targets):
    X, Y, rot_err, labels = [], [], [], []

    with open(logfile, newline='') as csvfile:
        data = csv.reader(csvfile, delimiter='\t')
        first = True
        for row in data:
            if len(row) > 10:
                if first:
                    first = False
                    prd_i = [row.index(p) for p in predictors if p not in ('distance', 'visible')]
                    trg_i = [row.index(t) for t in targets]
                    rot_i = row.index('rot error')
                    pos_i = [row.index(p + ' sc pos') for p in ('x', 'y', 'z')]
                    lbl_i = row.index('iter')
                else:
                    row = np.array(row)
                    try:
                        pos = row[pos_i].astype(np.float)
                    except ValueError as e:
                        print('Can\'t convert cols %s to float on row %s' % (pos_i, row[0]))
                        raise e
                    distance = np.sqrt(np.sum(pos ** 2))
                    visib = sm.calc_visibility(pos)

                    j = 0
                    x = [None] * len(predictors)
                    for i, p in enumerate(predictors):
                        if p == 'distance':
                            x[i] = distance
                        elif p == 'visible':
                            x[i] = visib
                        elif p == 'total dev angle':
                            x[i] = abs(tools.wrap_degs(row[prd_i[j]].astype(np.float)))
                            j += 1
                        elif p == 'sol elong':
                            x[i] = 180 - row[prd_i[j]].astype(np.float)
                            j += 1
                        else:
                            x[i] = row[prd_i[j]].astype(np.float)
                            j += 1
                    X.append(x)
                    Y.append([row[t].astype(np.float) if len(row) > t else float('nan') for t in trg_i])
                    rot_err.append(row[rot_i].astype(np.float))
                    labels.append(row[lbl_i])

    X = np.array(X)
    Y = np.array(Y)
    rot_err = np.array(rot_err)

    # for classification of fails
    yc = np.any(np.isnan(Y), axis=1)
    if MAX_ROTATION_ERR > 0:
        I = np.logical_not(yc)
        yc[I] = np.abs(tools.wrap_degs(rot_err[I])) > MAX_ROTATION_ERR

    # for regression, set failed to max err
    for i, tn in enumerate(targets):
        Y[np.isnan(Y[:, i]), i] = FAIL_ERRS[tn]
        if tn == 'rot error':
            Y[:, i] = np.abs(tools.wrap_degs(Y[:, i]))
        elif tn == 'dist error (m/km)':
            Y[:, i] = np.abs(Y[:, i])

    return X, Y, yc, labels
