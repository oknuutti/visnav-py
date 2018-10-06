import datetime
import os
import sys
import re

import settings
from settings import LOG_DIR

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('USAGE: python %s <yyyy-mm-dd> <HH:MM:SS>\n\tuses logs dated after given datetime'%(sys.argv[0],))
        quit()

    start_date = datetime.datetime.strptime(sys.argv[1]+' '+sys.argv[2], '%Y-%m-%d %H:%M:%S')
    adjust_expiry = datetime.datetime.strptime('2018-09-21 14:27:00', '%Y-%m-%d %H:%M:%S')
    n_noise = 3
    n_col = 4

    setups = [
        "orb",
        "orb+smn",
        "orb+smn_",
        "orb+fdb",
        "orb+fdb+smn",
        "orb+fdb+smn_",
        "orb+real",
        "orb+smn+real",
        "orb+smn_+real",
        "orb+fdb+real",
        "orb+fdb+smn+real",
        "orb+fdb+smn_+real",

        "akaze",
        "akaze+smn",
        "akaze+smn_",
        "akaze+fdb",
        "akaze+fdb+smn",
        "akaze+fdb+smn_",
        "akaze+real",
        "akaze+smn+real",
        "akaze+smn_+real",
        "akaze+fdb+real",
        "akaze+fdb+smn+real",
        "akaze+fdb+smn_+real",

        "surf",
        "surf+smn",
        "surf+smn_",
        "surf+fdb",
        "surf+fdb+smn",
        "surf+fdb+smn_",
        "surf+real",
        "surf+smn+real",
        "surf+smn_+real",
        "surf+fdb+real",
        "surf+fdb+smn+real",
        "surf+fdb+smn_+real",

        "sift",
        "sift+smn",
        "sift+smn_",
        "sift+fdb",
        "sift+fdb+smn",
        "sift+fdb+smn_",
        "sift+real",
        "sift+smn+real",
        "sift+smn_+real",
        "sift+fdb+real",
        "sift+fdb+smn+real",
        "sift+fdb+smn_+real",
    ]

    table_header = ['method'] + [
        "time ms",
        "fail %",
        "err-p68  m/km",
        "err-p95  m/km",
    ] * n_noise

    # initialize
    methods = []
    synth_table = {}
    real_table = {}
    for s in setups:
        m = re.sub("\+(smn_?|real)", "", s)
        if m not in methods:
            methods.append(m)
            synth_table[m] = [""]*(n_noise*n_col)
            real_table[m] = [""]*(n_noise*n_col)

    # find most recent log files
    logfiles = {}
    dates = {}
    for fname in os.listdir(LOG_DIR):
        match = re.match("([^-]+)-(\d+-\d+)\.log", fname)
        if match is not None:
            s = match[1]
            t = datetime.datetime.strptime(match[2], "%Y%m%d-%H%M%S")
            if t > start_date and (s not in dates or dates[s] < t):
                dates[s] = t
                logfiles[s] = fname

    # read data from logfiles
    data = {}
    for s, fname in logfiles.items():
        if s not in setups:
            continue

        with open(os.path.join(LOG_DIR, fname)) as fh:
            line = fh.readline()

        adj = 0.001 if adjust_expiry > dates[s] else 1

        match = re.match("[^(]+\((\d+)([^(]+\(){3}[\d.]+,\s([\d.]+),\s([\d.]+).*?fail:\s([\d.]+)", line)
        if match is not None:
            data[s] = [
                match[1],
                match[5],
                '%.3f'%(float(match[3])*adj),
                '%.3f'%(float(match[4])*adj),
            ]
        else:
            data[s] = [""]*4

        assert len(data[s]) == n_col

    # arrange data to suitable rows
    for s, d in data.items():
        m = re.sub("\+(smn_?|real)", "", s)
        i = 2 if "smn_" in s else 1 if "smn" in s else 0
        assert 0 <= i < n_noise

        if "real" in s:
            real_table[m][(i * n_col):((i + 1) * n_col)] = d
        else:
            synth_table[m][(i * n_col):((i + 1) * n_col)] = d

    # write tables to file
    fname = os.path.join(LOG_DIR, "result-tables.csv")
    with open(fname, 'w') as fh:
        fh.write("Synthetic\n")
        fh.write("\t".join(table_header) + "\n")
        for m in methods:
            fh.write(m + "\t" + "\t".join(synth_table[m]) + "\n")

        fh.write("\n\nReal\n")
        fh.write("\t".join(table_header) + "\n")
        for m in methods:
            fh.write(m + "\t" + "\t".join(real_table[m]) + "\n")
