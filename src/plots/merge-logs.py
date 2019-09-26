import datetime
import os
import sys
import re

import numpy as np

import settings
from algo import tools
from batch1 import get_system_model
from settings import LOG_DIR
from testloop import TestLoop

# distance, phase angle (180-elong), ini err, visibility
OPZONE_LIMITS = ((0, 250), (0, 140), (0, 10), (50, 120))

if __name__ == '__main__':
    try:
        mission = sys.argv[1]
        sm = get_system_model(mission)
        start_date = datetime.datetime.strptime(sys.argv[2] + ' ' + sys.argv[3], '%Y-%m-%d %H:%M:%S')
        postfix = sys.argv[4]
        opzone = 'opzone' in sys.argv[5:]
        only_least_noise = 'noiseless' in sys.argv[5:]
        only_real = 'real' in sys.argv[5:]
        only_synth = 'synth' in sys.argv[5:]
        method = next(iter(set(sys.argv[5:]) - {'opzone', 'noiseless', 'real', 'synth'}), False)
    except:
        print('USAGE: python %s <mission> <yyyy-mm-dd> <HH:MM:SS> <postfix> [opzone]\n\tmerges logs dated after given datetime'%(sys.argv[0],))
        quit()

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

        "centroid",
        "centroid+smn",
        "centroid+smn_",
        "centroid+real",
        "centroid+smn+real",
        "centroid+smn_+real",
    ]

    if nofdb:
        setups = [s for s in setups if 'fdb' not in s]
    if only_synth:
        setups = [s for s in setups if 'real' not in s]
    if only_real:
        setups = [s for s in setups if 'real' in s]
    if method:
        setups = [s for s in setups if method in s]
    if only_least_noise:
        setups = [s for s in setups if 'smn' not in s]

    setups = {s: ([], []) for s in setups}

    # find most recent log files
    for fname in os.listdir(LOG_DIR):
        match = re.match(r"([^-]+)-([^-]+)-(\d+-\d+)\.log", fname)
        if match is not None:
            m = match[1]
            s = match[2]
            t = datetime.datetime.strptime(match[3], "%Y%m%d-%H%M%S")
            if mission == 'rose' and re.match(r'^rose(\d{3})?$', m):
               m = 'rose'

            if m == mission and s in setups and t > start_date:
                setups[s][0].append((t, fname))

    # sort files in date order
    setups = {setup: (sorted(files, key=lambda f: f[0]), data) for setup, (files, data) in setups.items()}

    # read data from logfiles
    for s, (files, data) in setups.items():
        for t, fname in files:
            with open(os.path.join(LOG_DIR, fname)) as fh:
                offset = 0
                prefix = ''
                if mission == 'rose':
                    m = re.match(r'^rose(\d{3})?-.*?$', fname)
                    batch = m[1] if m[1] else '006'
                    offsets = np.cumsum([0, 718, 517, 391, 407, 487])
                    offset = offsets[['006', '007', '017', '024', '025', '026'].index(batch)]
                    prefix = batch + '-'
                for line in fh.readlines():
                    match = re.match(r"^(\d+)\t", line)
                    if match:
                        i = int(match[1]) + offset
                        if i >= len(data):
                            data.extend(['%d\t' % j for j in range(len(data), i+1)])
                        data[i] = prefix + line

    # write output
    columns = TestLoop.log_columns()
    for s, (files, rdata) in setups.items():
        fname = mission+"-"+s+"-"+postfix+".log"
        data = np.array([r.split('\t') for r in rdata])
        assert len(rdata)>0, 'No data for %s'%s
        assert len(data.shape) == 2, '%s: missing values at %s' % (s, np.where(np.array([len(d) for d in data]) == 2),)

        if len(data) > 0:
            if opzone:
                # distance, phase angle (180-elong), ini err, visibility
                phase_angle = 180 - data[:, columns.index('sol elong')].astype('float')
                inierr = np.abs(tools.wrap_degs(data[:, columns.index('total dev angle')].astype('float')))
                pos_start = columns.index('x sc pos')
                pos = data[:, pos_start:pos_start+3].astype('float')
                dist = np.linalg.norm(pos, axis=1)
                visib = sm.calc_visibility(pos)

                assert np.all(np.logical_and(phase_angle>=0, phase_angle<=180)), 'phase angle out of range'
                assert np.all(inierr > 0), 'ini err out of range'
                assert np.all(np.logical_and(visib>=0, visib<=100)), 'visibility out of range'

                I = np.logical_and.reduce((
                    dist > OPZONE_LIMITS[0][0],
                    dist < OPZONE_LIMITS[0][1],

                    phase_angle > OPZONE_LIMITS[1][0],
                    phase_angle < OPZONE_LIMITS[1][1],

                    inierr > OPZONE_LIMITS[2][0],
                    inierr < OPZONE_LIMITS[2][1],

                    visib > OPZONE_LIMITS[3][0],
                    visib < OPZONE_LIMITS[3][1],
                ))
                data = data[I, :]
                rdata = np.array(rdata)[I]

            runtimes = data[:, columns.index('execution time')].astype('float')
            laterrs = data[:, columns.index('lat error (m/km)')].astype('float')
            disterrs = data[:, columns.index('dist error (m/km)')].astype('float')
            shifterrs = data[:, columns.index('rel shift error (m/km)')].astype('float')
            roterrs = data[:, columns.index('rot error')].astype('float')

            summary = TestLoop.calc_err_summary(0, len(data), np.sum(np.isnan(roterrs)), runtimes,
                                                laterrs, disterrs, shifterrs, roterrs)

            head = summary + ' '.join(('src\\batch1.py', mission, s, str(len(data)))) + '\n' + '\t'.join(columns) + '\n'
            with open(os.path.join(LOG_DIR, fname), 'w') as fh:
                fh.write(head)
                fh.write("".join(rdata))
