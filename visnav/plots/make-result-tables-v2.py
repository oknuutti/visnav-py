import datetime
import os
import sys
import re

import numpy as np

from algo import tools
from plots.mergelogs import read_logfile
from settings import LOG_DIR
from testloop import TestLoop


def main():
    try:
        mission = sys.argv[1]
        postfix = sys.argv[2]
        transpose = 'transpose' in sys.argv[3:]
        real_ = 'real_' in sys.argv[3:]
    except:
        print('USAGE: python %s <mission-id> <log-identifier-postfix>\n\tuses logs with given postfix' % (sys.argv[0],))
        quit()

    noise_rows = ['17k', '4k', '1k']
    setups = [
        "orb",
        "orb+smn",
        "orb+smn_",
        "orb+real",
        "orb+smn+real",
        "orb+smn_+real",

        "akaze",
        "akaze+smn",
        "akaze+smn_",
        "akaze+real",
        "akaze+smn+real",
        "akaze+smn_+real",

        "surf",
        "surf+smn",
        "surf+smn_",
        "surf+real",
        "surf+smn+real",
        "surf+smn_+real",

        "sift",
        "sift+smn",
        "sift+smn_",
        "sift+real",
        "sift+smn+real",
        "sift+smn_+real",

        "centroid",
        "centroid+smn",
        "centroid+smn_",
        "centroid+real",
        "centroid+smn+real",
        "centroid+smn_+real",
    ]

    row_header = (
        ("", "Samples"),
#        ("", "T [ms]"),
#        ("", "Fail [%]"),
        ("", "Fails"),
        ("Radial err [m/km]", "σ"),     # μ
#        ("", "σ"),
#        ("", "p0.13"),
#        ("", "p15.9"),
        ("", "p68.27"),
        ("", "p95.45"),
        ("", "p99.73"),     # p99.87
        ("Lateral err [m/km]", "σ"),     # μ
#        ("", "σ"),
        ("", "p68.27"),
        ("", "p95.45"),
        ("", "p99.73"),
        # ("Ori. err [m/km]", "σ"),     # μ
        # ("", "σ"),
        # ("", "p68.27"),
        # ("", "p95.45"),
        # ("", "p99.73"),
    )

    if real_:
        setups = [s.replace('real', 'real_') for s in setups]

    # initialize
    methods = []
    for s in setups:
        m = re.sub(r"\+(smn_?|real_?)", "", s)
        if m not in methods:
            methods.append(m)

    c_idxs, i = {}, 0
    for n in noise_rows:
        for m in methods:
            c_idxs['%s-%s' % (m, n)] = i
            c_idxs['%s-%s' % (m, n)] = i
            i += 1

    n_row = len(row_header)
    n_col = len(noise_rows) * len(methods)
    synth_table = np.zeros((n_row+2, n_col+2), dtype='<U20')
    real_table = np.zeros((n_row+2, n_col+2), dtype='<U20')

    for i, n in enumerate(noise_rows):
        synth_table[0, i * len(methods) + 2] = n
        real_table[0, i * len(methods) + 2] = n
        for j, m in enumerate(methods):
            synth_table[1, i * len(methods) + j + 2] = m
            real_table[1, i * len(methods) + j + 2] = m

    for i in range(n_row):
        synth_table[2 + i, 0] = row_header[i][0]
        synth_table[2 + i, 1] = row_header[i][1]
        real_table[2 + i, 0] = row_header[i][0]
        real_table[2 + i, 1] = row_header[i][1]

    # find most recent log files
    logfiles = {}
    for fname in os.listdir(LOG_DIR):
        mstr1 = r'^rose(\d{3})?' if mission == 'rose' else ('^(' + mission+')')
        match = re.match(mstr1 + r"-([^-]+)-" + postfix + r"\.log", fname)
        if match is not None:
            s = match[2]
            logfiles[s] = fname

    # read data from logfiles
    cols = {}
    columns = TestLoop.log_columns()
    for s, fname in logfiles.items():
        if s not in setups:
            continue

        m = re.sub(r"\+(smn_?|real_?)", "", s)
        n = noise_rows[2 if "smn_" in s else 1 if "smn" in s else 0]
        key = '%s-%s' % (m, n)

        data = read_logfile(fname)
        if len(data) == 0:
            continue

        data[:, columns.index('dist error (m/km)')] = np.abs(data[:, columns.index('dist error (m/km)')]\
                                                             .astype('float')).astype('<20U')
        a = Statistics(data, columns)
        col = [
            a.samples(),
#            a.mean('execution time'),
            a.fail_ratio()*a.samples()/100,
#            a.mean('dist error (m/km)'),
            a.std('dist error (m/km)'),
#            a.prct('dist error (m/km)', 0.27),
#            a.prct('dist error (m/km)', 15.9),
#            a.prct('dist error (m/km)', 50),
            a.prct('dist error (m/km)', 68.27),
            a.prct('dist error (m/km)', 95.45),
            a.prct('dist error (m/km)', 99.73),  # 99.87
#            a.mean('lat error (m/km)'),
            a.std('lat error (m/km)'),
#            a.prct('lat error (m/km)', 50),
            a.prct('lat error (m/km)', 68.27),
            a.prct('lat error (m/km)', 95.45),
            a.prct('lat error (m/km)', 99.73),
##            a.mean('rot error'),
#            a.std('rot error'),
##            a.prct('rot error', 50),
#            a.prct('rot error (m/km)', 68.27),
#            a.prct('rot error (m/km)', 95.45),
#            a.prct('rot error', 99.73),
        ]
        if "real" in s:
            real_table[2:, c_idxs[key] + 2] = col
        else:
            synth_table[2:, c_idxs[key] + 2] = col

    if transpose:
        synth_table = synth_table.T
        real_table = real_table.T

    # write tables to file
    fname = os.path.join(LOG_DIR, "result-tables-%s-%s.csv" % (mission, postfix))
    with open(fname, 'w', encoding='utf8') as fh:
        fh.write("Synthetic\n")
        for r in synth_table:
            fh.write("\t".join(r) + "\n")
        fh.write("\n\nReal\n")
        for r in real_table:
            fh.write("\t".join(r) + "\n")


def read_logfile(fname):
    data = []
    with open(os.path.join(LOG_DIR, fname)) as fh:
        for line in fh.readlines():
            match = re.match(r"^(\d{3}-)?(\d+)\t", line)
            if match:
                data.append(line.split('\t'))
    return np.array(data)


class Statistics:
    def __init__(self, data, columns):
        self.data = data
        self.columns = columns

    def samples(self):
        return len(self.data)

    def failures(self):
        return np.sum(np.isnan(self.data[:, self.columns.index('dist error (m/km)')].astype('float')))

    def fail_ratio(self):
        return 100 * self.failures() / self.samples()

    def mean(self, col):
        return tools.robust_mean(self.data[:, self.columns.index(col)].astype('float'), 100 - 99.73)

    def std(self, col):
        return tools.robust_std(self.data[:, self.columns.index(col)].astype('float'), 100 - 99.73, mean=0)

    def prct(self, col, prct):
        return np.nanpercentile(self.data[:, self.columns.index(col)].astype('float'), prct)

if __name__ == '__main__':
    main()