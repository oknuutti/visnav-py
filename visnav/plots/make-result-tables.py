import datetime
import os
import sys
import re

import settings
from settings import LOG_DIR

SIMPLE = False

if __name__ == '__main__':
    postfix = None
    try:
        if len(sys.argv) > 2:
            start_date = datetime.datetime.strptime(sys.argv[1]+' '+sys.argv[2], '%Y-%m-%d %H:%M:%S')
            adjust_expiry = datetime.datetime.strptime('2018-09-21 14:27:00', '%Y-%m-%d %H:%M:%S')
        else:
            postfix = sys.argv[1]
    except:
        print('USAGE: python %s <yyyy-mm-dd> <HH:MM:SS>\n\tuses logs dated after given datetime'%(sys.argv[0],))
        quit()

    noise_rows = ['17k', '4k', '1k']
    n_noise = len(noise_rows)
    n_col = 8
    nofdb = True

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

    if nofdb:
        setups = [s for s in setups if 'fdb' not in s]

    table_header = ['method', 'vertices'] + ([
        "time ms",
        "fail %",
        "distance err (m/km) p50",
        "distance err (m/km) p84.1",
        "lat err (m/km) p50",
        "lat err (m/km) p84.1",
        "orientation err (deg) p50",
        "orientation err (deg) p84.1",
    ] if SIMPLE else [
        "time ms",
        "fail %",
        "distance err (m/km) μ/μ+σ",
        "distance err (m/km) p50/p84.1",
        "lat err (m/km) μ/μ+σ",
        "lat err (m/km) p50/p84.1",
        "orientation err (deg) μ/μ+σ",
        "orientation err (deg) p50/p84.1",
    ])

    # initialize
    methods = []
    synth_table = {}
    real_table = {}
    for s in setups:
        m = re.sub(r"\+(smn_?|real)", "", s)
        if m not in methods:
            methods.append(m)
            for n in noise_rows:
                synth_table['%s-%s' % (m, n)] = [""] * n_col
                real_table['%s-%s' % (m, n)] = [""] * n_col

    # find most recent log files
    logfiles = {}
    dates = {}
    for fname in os.listdir(LOG_DIR):
        mstr3 = postfix if postfix else r'(\d+-\d+)'
        match = re.match(r"([^-]+)-([^-]+)-" + mstr3 + r"\.log", fname)
        if match is not None:
            mission = match[1]
            s = match[2]
            if postfix:
                logfiles[s] = fname
            else:
                t = datetime.datetime.strptime(match[3], "%Y%m%d-%H%M%S")
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

        match = re.match(
              r"[^(]+\((\d+)"               # start=>time, m[1]:  time
            + r"[^(]+\("+r"([\d./]+)[,)\s]+"*3   # =>Le, m[2:5]:  Le p50 p68 p95
            + r"[^(]+\("+r"([\d./]+)[,)\s]+"*3   # =>De, m[5:8]:  De p50 p68 p95
            + r"[^(]+\("+r"([\d./]+)[,)\s]+"*3   # =>De, m[8:11]:  Se p50 p68 p95
            + r"[^(]+\("+r"([\d./]+)[,)\s]+"*3   # =>Re, m[11:14]: Re p50 p68 p95
            + r"[^f]*fail:\s([\d.]+).*"         # =>fail, m[14]:  fail %, => end
            , line)
        if match is not None:
            data[s] = [
                match[1],
                match[14],
                match[5].split('/')[1],  # dist p50
                match[6].split('/')[1],  # dist p84
                match[2].split('/')[1],  # lateral p50
                match[3].split('/')[1],  # lateral p84
                match[11].split('/')[1],  # orient p50
                match[12].split('/')[1],  # orient p84
            ] if SIMPLE else [
                match[1],
                match[14],
                match[5].split('/')[0] + ' / ' + match[6].split('/')[0],  # dist mean/sd
                match[5].split('/')[1] + ' / ' + match[6].split('/')[1],  # dist p50/p84
                match[2].split('/')[0] + ' / ' + match[3].split('/')[0],  # lateral mean/sd
                match[2].split('/')[1] + ' / ' + match[3].split('/')[1],  # lateral p50/p84
                match[11].split('/')[0] + ' / ' + match[12].split('/')[0],  # orient mean/sd
                match[11].split('/')[1] + ' / ' + match[12].split('/')[1],  # orient p50/p84
            ]
        else:
            data[s] = [""]*n_col

        assert len(data[s]) == n_col

    # arrange data to suitable rows
    for s, d in data.items():
        m = re.sub(r"\+(smn_?|real)", "", s)
        n = noise_rows[2 if "smn_" in s else 1 if "smn" in s else 0]

        key = '%s-%s' % (m, n)
        if "real" in s:
            real_table[key][:] = d
        else:
            synth_table[key][:] = d

    # write tables to file
    fname = os.path.join(LOG_DIR, "result-tables.csv")
    with open(fname, 'w', encoding='utf8') as fh:
        fh.write("Synthetic\n")
        fh.write("\t".join(table_header) + "\n")
        for n in noise_rows:
            for m in methods:
                key = '%s-%s' % (m, n)
                fh.write(m + "\t" + n + "\t" + "\t".join(synth_table[key]) + "\n")

        fh.write("\n\nReal\n")
        fh.write("\t".join(table_header) + "\n")
        for n in noise_rows:
           for m in methods:
                key = '%s-%s' % (m, n)
                fh.write(m + "\t" + n + "\t" + "\t".join(real_table[key]) + "\n")
