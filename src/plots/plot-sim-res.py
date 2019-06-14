import csv
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from algo import tools


def main(mission_sc=False):
    try:
        filename = sys.argv[1]
        span = tuple(map(int, sys.argv[2].split(':'))) if len(sys.argv) > 2 else None
        is_4km = '4km' in filename
    except:
        print('USAGE: %s <logfile>' % sys.argv[0])
        quit()

    raw = []
    with open(filename, newline='') as fh:
        reader = csv.reader(fh, delimiter='\t', quotechar='"')
        for i, row in enumerate(reader):
            if i == 0 and mission_sc:
                continue
            raw.append(row)
    data = np.array(raw).astype('double')
    if span is not None:
        data = data[span[0]:span[1], :]

    if mission_sc:
        plot_mission_sc(data)
        quit()

    # time (1)
    time = data[:, 0]/3600

    # real D1 pose (7)
    d1_loc = data[:, 1:4]
    d1_q = data[:, 4:8]

    # real D2 pose (7)          15
    d2_loc = data[:, 8:11]
    d2_q = data[:, 11:15]

    # real s/c pose (7)
    sc_loc = data[:, 15:18]
    sc_q = data[:, 18:22]

    # init s/c pose (7)         29
    isc_loc = data[:, 22:25]
    isc_q = data[:, 25:29]

    # landmark algo loc (3)
    lma_loc = data[:, 29:32]

    # landmark algo loc ok (1)  33
    lma_loc_ok = data[:, 32:33]
    lma_loc[np.logical_not(lma_loc_ok).flatten(), :] = np.nan

    # landmark algo ori (4)
    lma_q = data[:, 33:37]

    # landmark algo ori ok (1)  38
    lma_q_ok = data[:, 37:38]
    lma_q[np.logical_not(lma_q_ok).flatten(), :] = np.nan

    # laser algo loc (3)
    lsr_loc = data[:, 38:41]

    # laser algo loc ok (1)     42
    lsr_loc_ok = data[:, 41:42]
    lsr_loc[np.logical_not(lsr_loc_ok).flatten(), :] = np.nan

    # nav filter loc (3)
    flt_loc = data[:, 42:45]

    # measurement log likelihood (1)
    meas_ll = data[:, 45:46]

    # delta-v spent (1)
    cum_delta_v = data[:, 46:47]

    has_lma = not np.all(np.isnan(lma_loc))
    has_lsr = not np.all(np.isnan(lsr_loc))

    # calculate transformation to synodic frame, apply
    if not is_4km:
        transf = calc_transf(d1_loc, d2_loc)
    else:
        # for error calculation only
        use_d2 = '-d2-' in filename.lower()
        transf = calc_transf(sc_loc, d2_loc if use_d2 else d1_loc)
    c_lab = ('distance', 'along orbit', 'above orbit')

    d1_loc_sf = apply_transf(transf, d1_loc)
    d2_loc_sf = apply_transf(transf, d2_loc)
    sc_loc_sf = apply_transf(transf, sc_loc)
    isc_loc_sf = apply_transf(transf, isc_loc)
    lma_loc_sf = apply_transf(transf, lma_loc)
    lsr_loc_sf = apply_transf(transf, lsr_loc)
    flt_loc_sf = apply_transf(transf, flt_loc)

    flt_err_mean = np.mean(flt_loc_sf - sc_loc_sf, axis=0)
    flt_err_std = np.std(flt_loc_sf - sc_loc_sf, axis=0)
    if has_lma:
        lma_err_mean = np.nanmean(lma_loc_sf - sc_loc_sf, axis=0)
        lma_err_std = np.nanstd(lma_loc_sf - sc_loc_sf, axis=0)
    if has_lsr:
        lsr_err_mean = np.nanmean(lsr_loc_sf - sc_loc_sf, axis=0)
        lsr_err_std = np.nanstd(lsr_loc_sf - sc_loc_sf, axis=0)

    # plot didymain & didymoon
    fig1, ax = plt.subplots(figsize=(30, 19.5))
    if not is_4km:
        r = np.linspace(0, math.pi * 2, 100, endpoint=True)
        d1_r = 780/2
        d2_r = 163/2
        d1_xoff, d1_yoff, _ = np.mean(d1_loc_sf, axis=0)
        ax.plot(d2_r * np.cos(r), d2_r * np.sin(r), 'b-')
        ax.plot(d1_r * np.cos(r) + d1_xoff, d1_r * np.sin(r) + d1_yoff, 'b-')

        # plot s/c real loc & nav filter solution in synodic frame
        ax.plot(sc_loc_sf[:, 0], sc_loc_sf[:, 1], 'b--', label='real')
        ax.plot(flt_loc_sf[:, 0], flt_loc_sf[:, 1], 'r-', label='filter')
        ax.set_xlabel(c_lab[0], fontsize=18)
        ax.set_ylabel(c_lab[1], fontsize=18)
    else:
        # plot s/c real loc & nav filter solution in inertial frame
        ax.plot(sc_loc[:, 0], sc_loc[:, 1], 'b--', label='real')
        ax.plot(flt_loc[:, 0], flt_loc[:, 1], 'r-', label='filter')
        ax.set_xlabel('x', fontsize=18)
        ax.set_ylabel('y', fontsize=18)
    plt.tight_layout()

    # normal plots
    fig2, axs = plt.subplots(4, 1, sharex=True, figsize=(30, 19.5))

    # location errors
    i = 0
    for j, a in enumerate('real '+a for a in c_lab):
        axs[i].plot(time, sc_loc_sf[:, j], label=a)
    axs[i].set_prop_cycle(None)
    for j, a in enumerate('filter '+a for a in c_lab):
        axs[i].plot(time, flt_loc_sf[:, j], ':', label=a)

    axs[i].set_title('filter output\nerr μ=(%.2f, %.2f, %.2f), σ=(%.2f, %.2f, %.2f)' % (*flt_err_mean, *flt_err_std))
    axs[i].legend()

    # measurements
    i += 1
    for j, a in enumerate('real '+a for a in c_lab):
        axs[i].plot(time, sc_loc_sf[:, j], label=a)
    axs[i].set_prop_cycle(None)

    if has_lma:
        for j, a in enumerate('optical '+a for a in c_lab):
            axs[i].plot(time, lma_loc_sf[:, j], '--', label=a)
        axs[i].set_prop_cycle(None)

    if has_lsr:
        for j, a in enumerate('laser '+a for a in c_lab):
            axs[i].plot(time, lsr_loc_sf[:, j], ':', label=a)
    axs[i].set_title('measurements'
                     + ('\nopt err μ=(%.2f, %.2f, %.2f), σ=(%.2f, %.2f, %.2f)' % (*lma_err_mean, *lma_err_std) if has_lma else '')
                     + ('\nlsr err μ=(%.2f, %.2f, %.2f), σ=(%.2f, %.2f, %.2f)' % (*lsr_err_mean, *lsr_err_std) if has_lsr else '')
                    )
    axs[i].legend()

    # measurement likelihood
    i += 1
    axs[i].plot(time, meas_ll)
    axs[i].set_title('measurement likelihood')

    # delta-v used
    i += 1
    axs[i].plot(time, cum_delta_v)
    axs[i].set_title('cumulative delta-v usage')

    plt.tight_layout()
    plt.show()


def calc_transf(d1_loc, d2_loc):
    """ get transformation mx that transforms coordinates to rotating frame centered on d2 """
    def calc_q(r):
        lat, lon, r = tools.cartesian2spherical(*r)
        return tools.ypr_to_q(lat, lon, 0).conj()

    return d2_loc, np.array(list(map(calc_q, d1_loc - d2_loc)))


def apply_transf(transf, x):
    """ apply transformation calculated earlier with calc_transf """
    tr, q = transf
    return tools.q_times_mx(q, x - tr)


def plot_mission_sc(data):
    # time (days), x (m), y (m), z (m), vx (m/s), vy (m/s), vz (m/s), q1, q2, q3, q4
    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(data[:, 0], data[:, 1:4])
    axs[1].plot(data[:, 0], np.linalg.norm(data[:, 4:7], axis=1))

    plt.tight_layout()
    while not plt.waitforbuttonpress():
        pass


if __name__ == '__main__':
    main(mission_sc=False)

