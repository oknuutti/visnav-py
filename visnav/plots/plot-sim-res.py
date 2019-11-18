import csv
import math
import sys

import numpy as np
import quaternion
import matplotlib as mpl
import matplotlib.pyplot as plt
from cv2 import cv2
from scipy import optimize

from visnav.algo import tools


FONT_SIZE = 8
MARKER_SIZE = 4
LINE_WIDTH = 0.5


def main(mission_sc=False, simple=False):
    mpl.rcParams['font.size'] = FONT_SIZE
    mpl.rcParams['lines.markersize'] = MARKER_SIZE
    mpl.rcParams['lines.linewidth'] = LINE_WIDTH

    try:
        switches = {'--video'}
        args = sys.argv[1:]
        video = '--video' in args
        if video:
            args = [a for a in args if a not in switches]
        filename = args[0]
        span = tuple(map(int, args[1].split(':'))) if len(args) > 1 else None
        is_4km = '4km' in filename
        is_vo = '-vo-' in filename
        is_nac = '-nac-' in filename or '-1n-' in filename or '-2n-' in filename
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
        plot_mission_sc(data, format=3 if 'results' in filename else 2)
        quit()

    use_d2 = '-d2-' in filename.lower() or '-2n-' in filename.lower() or '-2w-' in filename.lower()
    id = filename.split('\\')[-1].split('-')[0]

    # time (1)
    time = data[:, 0]/3600

    # real D1 pose (7)
    d1_loc = data[:, 1:4]
    d1_q = data[:, 4:8]

    # real D2 pose (7)          15
    d2_loc = data[:, 8:11]
    d2_q = data[:, 11:15]

    trg_loc = d2_loc if use_d2 else d1_loc
    trg_q = quaternion.as_quat_array(d2_q if use_d2 else d1_q)

    # real s/c pose (7)
    sc_loc = data[:, 15:18]
    sc_q = quaternion.as_quat_array(data[:, 18:22])

    # init s/c pose (7)         29
    isc_loc = data[:, 22:25]
    isc_q = data[:, 25:29]

    # landmark algo loc (3)
    # TODO: check that sc_q works, seems that static
    spl_loc = trg_loc - tools.q_times_mx(sc_q, data[:, 29:32])

    # landmark algo loc ok (1)  33
    spl_loc_ok = data[:, 32:33]
    spl_loc[np.logical_not(spl_loc_ok).flatten(), :] = np.nan

    # landmark algo ori (4)
    spl_q = data[:, 33:37]

    # landmark algo ori ok (1)  38
    spl_q_ok = data[:, 37:38]
    spl_q[np.logical_not(spl_q_ok).flatten(), :] = np.nan

    # laser algo loc (3)
    lsr_loc = trg_loc - tools.q_times_mx(sc_q, data[:, 38:41])

    # laser algo loc ok (1)     42
    lsr_loc_ok = data[:, 41:42]
    lsr_loc[np.logical_not(lsr_loc_ok).flatten(), :] = np.nan

    # nav filter loc (3)
    flt_loc = data[:, 42:45] if True else np.full_like(spl_loc, np.nan)

    # measurement log likelihood (1)
    meas_ll = data[:, 45:46]

    # delta-v spent (1)
    cum_delta_v = data[:, 46:47]

    # vo loc (3)
    vo_loc = trg_loc - tools.q_times_mx(sc_q, data[:, 47:50])

    # vo ori (4)
    vo_q = data[:, 50:54]

    # vo meas sds (6)
    vo_meas_sds = data[:, 54:60]

    # vo bias sds (6)
    vo_bias_sds = data[:, 60:66]

    # vo scale drift sd (1)
    vo_bias_sds = data[:, 66:67]

    # vo ok
    vo_ok = data[:, 67:68]

    # phase angle
    phase_angle = data[:, 68:69]

    # vo scale (1)
    vo_scale = data[:, 69:70]

    # cnt location (3)
    cnt_loc = trg_loc - tools.q_times_mx(sc_q, data[:, 70:73]) if data.shape[1] >= 73 else np.full_like(spl_loc, np.nan)

    # sun-sc vect
    sun_v = data[:, 73:76] if data.shape[1] >= 76 else None

    # s/c-target distance
    distance = np.linalg.norm(sc_loc - trg_loc, axis=1)

    has_spl = not np.all(np.isnan(spl_loc))
    has_lsr = not np.all(np.isnan(lsr_loc))
    has_vo = not np.all(np.isnan(vo_loc)) and False
    has_cnt = not np.all(np.isnan(cnt_loc))
    has_flt = False

    if use_d2:
        sun_v = sun_v if sun_v is not None else {
            'id2': np.array([-0.3067, -0.9427, -0.1315]),
            'id4': np.array([-0.5252, -0.8379, -0.1485]),
            'id5': np.array([0, -1, 0]),
        }[id]
        is_d2_ecl = d2_eclipses(sun_v, d1_loc, d2_loc)
        d2_ecl = get_intervals(time, is_d2_ecl)

        is_d1_bg, is_d1_fg = d2_when_d1_in_view(sc_loc, sc_q, d1_loc, d2_loc)
        d1_bg, d1_fg = get_intervals(time, is_d1_bg), get_intervals(time, is_d1_fg)

    if not video:
        cnt_max_dist = {
            True: {             # is_nac
                True: 1300,     # - use_d2
                False: 5800,    # - not use_d2
            },
            False: {            # not is_nac
                True: 225,      # - use_d2
                False: 1050,    # - not use_d2
            },
        }[is_nac][use_d2]
        cnt_loc[phase_angle.flatten() > 100 / 180 * np.pi, :] = np.nan
        cnt_loc[distance.flatten() < cnt_max_dist, :] = np.nan
        spl_loc[phase_angle.flatten() > 135 / 180 * np.pi, :] = np.nan

    incl_for_stats = (phase_angle < 100 / 180 * np.pi).flatten()    # phase angle less than 100 deg
    if use_d2:
        incl_for_stats = np.logical_and.reduce((
            incl_for_stats,
            np.logical_not(is_d1_fg),
            np.logical_not(is_d2_ecl),
        ))

    # calculate transformation to synodic frame, apply
    tr_sf = calc_transf(d1_loc, d2_loc)

    # for error calculation and plots
    tr_stf = calc_transf(sc_loc, d2_loc if use_d2 else d1_loc)
    c_lab = ('distance', 'along orbit', 'above orbit')

    if has_vo:
        vo_loc, vo_scale, vo_loc_bias, nkf_idxs, is_mm = vo_data_prep(vo_loc, vo_scale, vo_bias_sds, sc_loc, trg_loc)
        if False:
            # correct drifting scale
            vo_loc = (vo_loc - trg_loc) / vo_scale.reshape((-1, 1)) + trg_loc

    d1_loc_sf = apply_transf(tr_sf, d1_loc)
    d2_loc_sf = apply_transf(tr_sf, d2_loc)
    trg_loc_sf = d2_loc_sf if use_d2 else d1_loc_sf
    sc_loc_sf = apply_transf(tr_sf, sc_loc)
    isc_loc_sf = apply_transf(tr_sf, isc_loc)
    spl_loc_sf = apply_transf(tr_sf, spl_loc)
    lsr_loc_sf = apply_transf(tr_sf, lsr_loc)
    flt_loc_sf = apply_transf(tr_sf, flt_loc)
    vo_loc_sf = apply_transf(tr_sf, vo_loc)
    cnt_loc_sf = apply_transf(tr_sf, cnt_loc)

    d1_loc_stf = apply_transf(tr_stf, d1_loc)
    d2_loc_stf = apply_transf(tr_stf, d2_loc)
    trg_loc_stf = d2_loc_stf if use_d2 else d1_loc_stf
    sc_loc_stf = apply_transf(tr_stf, sc_loc)
    isc_loc_stf = apply_transf(tr_stf, isc_loc)
    spl_loc_stf = apply_transf(tr_stf, spl_loc)
    lsr_loc_stf = apply_transf(tr_stf, lsr_loc)
    flt_loc_stf = apply_transf(tr_stf, flt_loc)
    vo_loc_stf = apply_transf(tr_stf, vo_loc)
    cnt_loc_stf = apply_transf(tr_stf, cnt_loc)

    if has_vo:
        vo_loc_sf = apply_transf(tr_sf, vo_loc)
        vo_loc_stf = apply_transf(tr_stf, vo_loc)
        vo_loc_bias_stf = apply_transf(tr_stf, vo_loc_bias)
        #vo_loc_sf, _, vo_loc_bias_sf, _, _ = vo_data_prep(vo_loc_sf, vo_scale, vo_bias_sds, sc_loc_sf, trg_loc_sf)
        #vo_loc_stf, vo_scale, vo_loc_bias_stf, nkf_idxs, is_mm = vo_data_prep(vo_loc_stf, vo_scale, vo_bias_sds, sc_loc_stf, trg_loc_stf)

    flt_err_mean = tools.robust_mean(flt_loc_stf - sc_loc_stf, axis=0)
    flt_err_std = tools.robust_std(flt_loc_stf - sc_loc_stf, axis=0)
    if has_lsr:
        lsr_err_mean = tools.robust_mean((lsr_loc_stf - sc_loc_stf), axis=0)
        lsr_err_std = tools.robust_std((lsr_loc_stf - sc_loc_stf), axis=0)
    if has_cnt:
        cnt_err_mean = tools.robust_mean((cnt_loc_stf - sc_loc_stf)[incl_for_stats, :], axis=0)
        cnt_err_std = tools.robust_std((cnt_loc_stf - sc_loc_stf)[incl_for_stats, :], axis=0)
    if has_spl:
        spl_err_mean = tools.robust_mean((spl_loc_stf - sc_loc_stf)[incl_for_stats, :], axis=0)
        spl_err_std = tools.robust_std((spl_loc_stf - sc_loc_stf)[incl_for_stats, :], axis=0)
    if has_vo:
        vo_err_mean = tools.robust_mean((vo_loc_stf - sc_loc_stf)[incl_for_stats, :], axis=0)
        vo_err_std = tools.robust_std((vo_loc_stf - sc_loc_stf)[incl_for_stats, :], axis=0)

        # nkf_idxs need to include a nan value between vo resets, is_mm
        vo_delta_scale_mean = tools.robust_mean(np.diff(vo_scale[nkf_idxs])[np.logical_not(is_mm[1:])])
        vo_delta_scale_std = tools.robust_std(np.diff(vo_scale[nkf_idxs])[np.logical_not(is_mm[1:])], mean=0)
        vo_delta_bias_mean = tools.robust_mean(np.diff(vo_loc_bias_stf[nkf_idxs], axis=0)[np.logical_not(is_mm[1:]), :], axis=0)
        vo_delta_bias_std = tools.robust_std(np.diff(vo_loc_bias_stf[nkf_idxs], axis=0)[np.logical_not(is_mm[1:]), :], mean=0, axis=0)
        vo_mm_delta_scale_mean = tools.robust_mean(np.diff(vo_scale[nkf_idxs])[is_mm[1:]])
        vo_mm_delta_scale_std = tools.robust_std(np.diff(vo_scale[nkf_idxs])[is_mm[1:]], mean=0)
        vo_mm_delta_bias_mean = tools.robust_mean(np.diff(vo_loc_bias_stf[nkf_idxs], axis=0)[is_mm[1:], :], axis=0)
        vo_mm_delta_bias_std = tools.robust_std(np.diff(vo_loc_bias_stf[nkf_idxs], axis=0)[is_mm[1:], :], mean=0, axis=0)

    cutoff_time = time[0] + {
        'id1': time[-1],
        'id2': 1.5*73.125,
        'id3': time[-1],
        'id4': 4*11.91,
        'id5': 2*11.91,
    }[id]
    cutoff = np.argmax(time > cutoff_time)

    # normal plots
    if simple:
        fig2, axs = plt.subplots(4 + (0 if has_vo else 0), 1, sharex=True, figsize=(8, 6))

        axs[0].plot(time, phase_angle/np.pi*180, 'C0', label='phase angle')
        axs[0].set_ylabel('phase angle', color='C0')
        axs[0].tick_params(axis='y', labelcolor='C0')
        axs[0].set_ybound(0, 180)

        ax0b = axs[0].twinx()
        ax0b.plot(time, distance, 'C1', label='distance')
        ax0b.set_ylabel('distance', color='C1')
        ax0b.tick_params(axis='y', labelcolor='C1')

        axs[-1].set_xlabel('time [h]')
        for i, lab in enumerate(c_lab):
            axs[i+1].set_ylabel(lab + ' error [m]')

#        for i, a in enumerate('real '+a for a in c_lab):
#            axs[i+1].plot(time, sc_loc_stf[:, i] - sc_loc_stf[:, i], label=a)

#        print('filter err μ=(%.2f, %.2f, %.2f), σ=(%.2f, %.2f, %.2f)' % (*flt_err_mean, *flt_err_std))
#        for i, a in enumerate('filter '+a for a in c_lab):
#            axs[i+1].plot(time, flt_loc_stf[:, i] - sc_loc_stf[:, i], label=a)

        if id in ('id3', 'id5'):
            idx = np.isclose((time * 60 * 60 - 5 + 1e-10) % 60, 0)
        else:
            idx = np.ones(len(time), dtype=np.bool)

        if has_cnt:
            print('cnt err μ=(%.2f, %.2f, %.2f), σ=(%.2f, %.2f, %.2f)' % (*cnt_err_mean, *cnt_err_std))
            for i, a in enumerate(c_lab):
                axs[i+1].plot(time[idx], cnt_loc_stf[idx, i] - sc_loc_stf[idx, i], 'C0--', label='CNT')

        if has_spl:
            print('spl err μ=(%.2f, %.2f, %.2f), σ=(%.2f, %.2f, %.2f)' % (*spl_err_mean, *spl_err_std))
            for i, a in enumerate(c_lab):
                axs[i+1].plot(time[idx], spl_loc_stf[idx, i] - sc_loc_stf[idx, i], 'C1--', label='SPL')

        if has_lsr:
            print('lsr err μ=(%.2f, %.2f, %.2f), σ=(%.2f, %.2f, %.2f)' % (*lsr_err_mean, *lsr_err_std))
            for i, a in enumerate(c_lab):
                axs[i+1].plot(time[idx], lsr_loc_stf[idx, i] - sc_loc_stf[idx, i], 'C3:', label='LSR')

        if has_vo:
            print('vo delta scale μ=%.2f, σ=%.2f' % (vo_delta_scale_mean, vo_delta_scale_std))
            print('vo delta bias μ=(%.2f, %.2f, %.2f), σ=(%.2f, %.2f, %.2f)' % (*vo_delta_bias_mean, *vo_delta_bias_std))
            print('vo mm delta scale μ=%.2f, σ=%.2f' % (vo_mm_delta_scale_mean, vo_mm_delta_scale_std))
            print('vo mm delta bias μ=(%.2f, %.2f, %.2f), σ=(%.2f, %.2f, %.2f)' % (*vo_mm_delta_bias_mean, *vo_mm_delta_bias_std))
            print('vo meas err μ=(%.2f, %.2f, %.2f), σ=(%.2f, %.2f, %.2f)' % (*vo_err_mean, *vo_err_std))

            if id == 'id5':
                idx4 = np.isclose((time * 60 * 60 - 5 + 1e-10) % 60, 0)
            else:
                idx4 = np.ones(len(time), dtype=np.bool)

            for i, a in enumerate(c_lab):
                axs[i+1].plot(time[idx4], vo_loc_stf[idx4, i] - sc_loc_stf[idx4, i], 'C2-.', label='VO')
            # for i, a in enumerate('vo bias ' + a for a in c_lab):
            #     axs[i].plot(time[idx], vo_loc_bias_stf[idx, i], 'b-', label=a)
            # axs[-1].plot(time[idx], vo_scale[idx], 'b-', label='vo scale')

        bounded = True
        bad_pa = get_intervals(time, phase_angle > 135 / 180 * np.pi)

        if id == 'id1':
            pass
            #axs[i].set_ybound(-1000, 1000)
        elif id == 'id2':
            if bounded:
                axs[1].set_ybound(-400, 400)
                axs[2].set_ybound(-40, 40)
                axs[3].set_ybound(-40, 40)
        elif id == 'id3':
            pass
            #axs[i].set_ybound(-1000, 1000)
        elif id == 'id4':
            if bounded:
                axs[1].set_ybound(-20, 20)
                axs[2].set_ybound(-40, 40)
                axs[3].set_ybound(-40, 40)
        elif id == 'id5':
            if bounded:
                axs[1].set_ybound(-5, 5)
                axs[2].set_ybound(-10, 10)
                axs[3].set_ybound(-10, 10)

        for i in range(1, 4):
            axs[i].legend(loc='lower right')
            for s, e in bad_pa:
                axs[i].axvspan(s, e, facecolor='#f7aaaa', alpha=0.5)  # pink
            if use_d2:
                for s, e in d2_ecl:
                    axs[i].axvspan(s, e, facecolor='#b0f9ef', alpha=0.5)  # turq
                for s, e in d1_bg:
                    axs[i].axvspan(s, e, facecolor='#f8f9b0', alpha=0.5)  # green
                for s, e in d1_fg:
                    axs[i].axvspan(s, e, facecolor='#f5b0f9', alpha=0.5)  # purple
        if bounded:
            ax0b.set_xbound(time[0], cutoff_time)

    else:
        fig2, axs = plt.subplots(4 if cum_delta_v[-1] > 0 else 3, 1, sharex=True, figsize=(8, 6))

        # # location errors
        # i = 0
        # for j, a in enumerate('real '+a for a in c_lab):
        #     axs[i].plot(time, sc_loc_stf[:, j], label=a)
        # axs[i].set_prop_cycle(None)
        # for j, a in enumerate('filter '+a for a in c_lab):
        #     axs[i].plot(time, flt_loc_stf[:, j], ':', label=a)
        #
        # axs[i].set_title('filter output\nerr μ=(%.2f, %.2f, %.2f), σ=(%.2f, %.2f, %.2f)' % (*flt_err_mean, *flt_err_std))
        # axs[i].legend(loc='lower right')
        #
        # # measurements
        # i += 1
        # for j, a in enumerate('real '+a for a in c_lab):
        #     axs[i].plot(time, sc_loc_stf[:, j], label=a)
        # axs[i].set_prop_cycle(None)
        #
        # if has_spl:
        #     for j, a in enumerate('spl '+a for a in c_lab):
        #         axs[i].plot(time, spl_loc_stf[:, j], 'C1--', label=a)
        #     axs[i].set_prop_cycle(None)
        #
        # if has_lsr:
        #     for j, a in enumerate('laser '+a for a in c_lab):
        #         axs[i].plot(time, lsr_loc_stf[:, j], 'r:', label=a)
        #     axs[i].set_prop_cycle(None)
        #
        # if has_vo:
        #     for j, a in enumerate('vo '+a for a in c_lab):
        #         axs[i].plot(time, vo_loc_stf[:, j], 'C2.-', label=a)
        #     axs[i].set_prop_cycle(None)
        #
        # axs[i].set_title('measurements'
        #                  + ('\nopt err μ=(%.2f, %.2f, %.2f), σ=(%.2f, %.2f, %.2f)' % (*spl_err_mean, *spl_err_std) if has_spl else '')
        #                  + ('\nlsr err μ=(%.2f, %.2f, %.2f), σ=(%.2f, %.2f, %.2f)' % (*lsr_err_mean, *lsr_err_std) if has_lsr else '')
        #                  + ('\nvo err μ=(%.2f, %.2f, %.2f), σ=(%.2f, %.2f, %.2f)' % (*vo_err_mean, *vo_err_std) if has_vo else '')
        #                  )
        # axs[i].legend(loc='lower right')

        # measurement likelihood
        i += 1
        axs[i].plot(time, meas_ll)
        axs[i].set_title('measurement likelihood')

        # delta-v used
        if cum_delta_v[-1] > 0:
            i += 1
            axs[i].plot(time, cum_delta_v)
            axs[i].set_title('cumulative delta-v usage')

        axs[i].set_xlim(np.min(time), np.max(time))
    plt.tight_layout()

    # plot didymain & didymoon
    fig1, ax = plt.subplots(figsize=(7, 7))

    if video:
        framerate = 25
        dw, dh = fig1.canvas.get_width_height()
        writer = cv2.VideoWriter(filename[:-4]+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), framerate, (dw*2, dh*2))

    try:
        skip = 2
        for c in range(skip, len(d1_loc_sf), skip):
            if video:
                tools.show_progress(len(d1_loc_sf)//skip, c//skip)
            else:
                c = cutoff or -1

            if is_vo:
                # for c in range(0, len(d1_loc_sf), 30):
                #     plot_orbit_sf(ax, d1_loc_sf, sc_loc_sf, vo_loc_sf, cutoff=c, idx1=1, static=False)
                plot_orbit_sf(ax, d1_loc_sf, sc_loc_sf, vo_loc_sf, cutoff=c, static=not video)
            elif is_4km:
                plot_orbit_sf(ax, d1_loc, d2_loc, sc_loc,
                              flt_loc if has_flt else None,
                              spl_loc=spl_loc[idx, :] if id in ('id1', 'id2', 'id3') else None,
                              vo_loc=vo_loc[idx4, :] if id in ('id1', 'id3') and has_vo else None,
                              synodic=False, cutoff=c, static=not video)
            else:
                plot_orbit_sf(ax, d1_loc_sf, d2_loc_sf, sc_loc_sf,
                              flt_loc_sf if has_flt else None,
                              spl_loc=spl_loc_sf[idx, :] if id in ('id4', 'id5') else None,
                              #vo_loc=vo_loc_sf[idx4, :] if id in ('id5',) else None,
                              synodic=True, cutoff=c, static=not video)
            if video:
                #plt.tight_layout()
                # plt.pause(0.05)
                # plt.waitforbuttonpress()
                mi = [m for m in (5760, 7593) if m < c]
                if len(mi) > 0:
                    ax.plot(spl_loc[mi, 0], spl_loc[mi, 1], 'bv', label='Maneuver', fillstyle='none')
                errtxt = 'error [m]: x=%5.1f, y=%5.1f, z=%5.1f' % tuple(spl_loc[c, :] - sc_loc[c, :])
                plt.text(2650, 9500, errtxt, family='monospace', fontsize=12, horizontalalignment='center')
                ax.set_xbound(-5200, 10500)
                ax.set_ybound(-7100, 8600)
                fig1.canvas.draw()
                img = np.frombuffer(fig1.canvas.tostring_argb(), dtype=np.uint8)
                img.shape = (dh*3, dw*3, 4)     # why need *3 ???
                # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
                img = np.roll(img, 3, axis=2)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                img = cv2.resize(img, (dw*2, dh*2))
                if False:
                    cv2.imshow('test', img)
                    cv2.waitKey()
                writer.write(img)
                ax.clear()
            else:
                plt.tight_layout()
                plt.show()
                break
    finally:
        if video:
            writer.release()


def vo_data_prep(vo_loc, vo_scale, vo_bias_sds, sc_loc, trg_loc):
    # get raw measures first
    vo_raw_loc = (vo_loc - trg_loc)/vo_scale.reshape((-1, 1))
    new_vo_scale = np.ones(len(vo_loc))*np.nan
    new_vo_loc_bias = np.ones((len(vo_loc),3))*np.nan
    new_vo_loc = np.ones((len(vo_loc),3))*np.nan

    # TODO: estimate sc and bias based on likelihood maximization, estimate hyperparams also:
    #   - delta_scale_sd (1 & 2), delta_bias_sd (1 & 2), dist_err_sd, lat_err_sd
    #   - current scheme doesnt work as measurement error can be super small so that bias can explain all and scale goes to inf

    nkf_idxs = []
    is_mm = []
    bias0 = [0., 0., 0.]
    j = 0
    ks = (i for i, sd in enumerate(vo_bias_sds[:, 0]) if sd > 0)
    while j is not None:
        k = next(ks, None)
        if vo_bias_sds[j, 0] is np.inf:
            nkf_idxs.append(0)
            is_mm.append(False)

        sc = np.nanmean(1/vo_scale[j:k])

        if np.sum(np.any(np.logical_not(np.isnan(vo_raw_loc[j:k, :])), axis=1)) >= 3:
            # need to skip keyframes with too few measurements
            #   => leads to bad scale and bias diffs
            #   => better way would be some kind of likelihood maximization thing

            # vo_raw_loc[j:k, :]/sc - bias == sc_loc[j:k, :] - trg_loc[j:k, :]
            def costfun(bias, sc):
                err = sc_loc[j:k, :] - trg_loc[j:k, :] - vo_raw_loc[j:k, :]/sc + np.array(bias)
                err = err[np.logical_not(np.any(np.isnan(err), axis=1)), :].flatten()
                return err
            (*bias,), _ = optimize.leastsq(costfun, bias0, args=(sc,))
            nkf_idxs.append(j)
        else:
            bias = bias0
            nkf_idxs.append(0)

        new_vo_scale[j:k] = sc
        new_vo_loc_bias[j:k, :] = np.array(bias).reshape((1, 3))
        new_vo_loc[j:k, :] = vo_raw_loc[j:k, :]#/sc - np.array(bias).reshape((1, 3))

        # TODO: take into account orientation bias?

        is_mm.append(vo_bias_sds[j, 0] == 0.05)
        bias0 = bias
        j = k

    new_vo_loc_bias[np.isnan(vo_loc[:, 0]), :] = np.nan
    return new_vo_loc + trg_loc, new_vo_scale, new_vo_loc_bias + trg_loc, nkf_idxs, is_mm


def plot_orbit_sf(ax, d1_loc, d2_loc, sc_loc, flt_loc=None, spl_loc=None, vo_loc=None,
                  synodic=True, labels=('x [m]', 'y [m]'), idx0=0, idx1=1, cutoff=None, static=True):
    if cutoff is not None:
        d1_loc = d1_loc[:cutoff, :]
        d2_loc = d2_loc[:cutoff, :]
        sc_loc = sc_loc[:cutoff, :]
        if flt_loc is not None:
            flt_loc = flt_loc[:cutoff, :]
        if spl_loc is not None:
            spl_loc = spl_loc[:cutoff, :]
        if vo_loc is not None:
            vo_loc = vo_loc[:cutoff, :]

    ax.axis('equal')

    if synodic:
        r = np.linspace(0, math.pi * 2, 100, endpoint=True)
        d1_r = 775/2
        d2_r = 163/2
        d1_xoff, d1_yoff, _ = np.mean(d1_loc, axis=0)
        ax.plot(d2_r * np.cos(r), d2_r * np.sin(r), 'b-')
        ax.plot(d1_r * np.cos(r) + d1_xoff, d1_r * np.sin(r) + d1_yoff, 'b-')
    else:
        ax.plot(0, 0, 'bx', label='Barycenter')
        if not static:
            ax.plot(d2_loc[:, idx0], d2_loc[:, idx1], 'C4-', label='Didymoon')
            #ax.plot(d1_loc[:, idx0], d1_loc[:, idx1], 'C5-', label='Didymain')
            ax.plot(d2_loc[-1, idx0], d2_loc[-1, idx1], 'C4o', fillstyle='none')
            #ax.plot(d1_loc[-1, idx0], d1_loc[-1, idx1], 'C5o', fillstyle='none')

    # plot s/c real loc & nav filter solution
    ax.plot(sc_loc[:, idx0], sc_loc[:, idx1], 'b--', label='APEX')
    if static:
        ax.plot(sc_loc[0, idx0], sc_loc[0, idx1], 'bo', label='Start', fillstyle='none')
    else:
        ax.plot(sc_loc[-1, idx0], sc_loc[-1, idx1], 'bo', fillstyle='none')
        ax.plot([], [], 'bv', label='Maneuver', fillstyle='none')

    if spl_loc is not None:
        ax.plot(spl_loc[:, idx0], spl_loc[:, idx1], 'C1--', label='SPL')
        if not static:
            ax.plot(spl_loc[-1, idx0], spl_loc[-1, idx1], 'C1x')
    if vo_loc is not None:
        ax.plot(vo_loc[:, idx0], vo_loc[:, idx1], 'C2--', label='VO')
        if not static:
            ax.plot(vo_loc[-1, idx0], vo_loc[-1, idx1], 'C2x')
    if flt_loc is not None:
        ax.plot(flt_loc[:, idx0], flt_loc[:, idx1], 'r--', label='filter')
        if not static:
            ax.plot(flt_loc[-1, idx0], flt_loc[-1, idx1], 'rx')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.legend(loc='lower right')


def d2_eclipses(sun_v, d1_loc, d2_loc):
    # distance to sun_v
    d, s = tools.point_vector_dist(d2_loc - d1_loc, sun_v.reshape(1, -1), dist_along_v=True)

    # joint radius, i.e. some part is eclipsed if distance to sun_v less than this
    R = (163+775)/2

    is_ecl = np.logical_and(d < R, s > 0).flatten()
    return is_ecl


def d2_when_d1_in_view(sc_v, sc_q, d1_loc, d2_loc):
    sep_a = tools.angle_between_mx(d1_loc - sc_v, d2_loc - sc_v)
    d1_ra = tools.sphere_angle_radius(d1_loc - sc_v, 775/2)
    d2_ra = tools.sphere_angle_radius(d2_loc - sc_v, 163/2)
    is_bg = np.linalg.norm(d1_loc - sc_v, axis=1) > np.linalg.norm(d2_loc - sc_v, axis=1)

    is_conj = sep_a - d1_ra - d2_ra < 0
    d1_bg = np.logical_and(is_conj, is_bg).flatten()
    d1_fg = np.logical_and(is_conj, np.logical_not(is_bg)).flatten()

    return d1_bg, d1_fg


def get_intervals(t, is_interval):
    t0 = t[0]
    intervals = []
    for i, e in enumerate(np.diff(is_interval.astype('int8'), axis=0)):
        if e > 0:
            t0 = t[i]
        elif e < 0:
            intervals.append((t0, t[i]))
            t0 = None
    if t0 is not None and t0 > t[0]:
        intervals.append((t0, t[-1]))
    return intervals


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


def plot_mission_sc(data, format=3):
    # marcus format:
    # time (days), x (m), y (m), z (m), vx (m/s), vy (m/s), vz (m/s), q1, q2, q3, q4
    #
    # own format for emil files:
    # Time (s),
    # APEX x (km), APEX y (km), APEX z (km), APEX vx (km/s), APEX vy (km/s), APEX vz (km/s),
    # APEX eclipse, SUN x, SUN y, SUN z,
    # HERA x (km), HERA y (km), HERA z (km), HERA vx (km/s), HERA vy (km/s), HERA vz (km/s),
    # D1 x (km), D1 y (km), D1 z (km), D1 vx (km/s), D1 vy (km/s), D1 vz (km/s),
    # D2 x (km), D2 y (km), D2 z (km), D2 vx (km/s), D2 vy (km/s), D2 vz (km/s)

    if format == 2:
        t = data[:, 0]/3600
        apex_r = data[:, 1:4]*1000
        apex_ecl = data[:, 7:8]
        sun_uv = data[:, 8:11]
        hera_r = data[:, 11:14]*1000
        d1_r = data[:, 17:20]*1000
        d2_r = data[:, 23:26]*1000
    elif format == 3:
        t = data[:, 0]/3600
        d1_r = data[:, 1:4]
        d2_r = data[:, 8:11]
        apex_r = data[:, 15:18]

    transf = calc_transf(d1_r, d2_r)
    d1_sf = apply_transf(transf, d1_r)
    apex_sf = apply_transf(transf, apex_r)

    fig, axs = plt.subplots(2, 1) #, sharex=True)
    plot_orbit_sf(axs[0], d1_sf, apex_sf)

    axs[1].plot(t, apex_sf)
#    axs[1].plot(t, np.linalg.norm(v, axis=1))
    plt.tight_layout()
    plt.show()

    # while not plt.waitforbuttonpress():
    #     pass


if __name__ == '__main__':
    main(mission_sc=False, simple=True)

