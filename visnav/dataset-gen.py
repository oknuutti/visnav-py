import argparse
import csv
import sys
import math
import re

import cv2
import numpy as np
import quaternion
from tqdm import tqdm

from visnav.algo import tools
from visnav.algo.keypoint import KeypointAlgo
from visnav.algo.tools import PositioningException
from visnav.batch1 import get_system_model
from visnav.iotools import cnn_export
from visnav.testloop import TestLoop
from visnav.settings import *


# TODO:
#    - Include image rendered at cropped pose (for autoencoder target)
#    - Save a config file about parameters used for dataset generation
#    - Support usage of real navcam images for generating a validation/testing dataset
#    (- Upload dataset to triton?)


def parse_arguments():
    missions = ('rose', 'orex', 'didy1n', 'didy1w', 'didy2n', 'didy2w')

    parser = argparse.ArgumentParser(description='Asteroid Image Data Generator')
    parser.add_argument('--relative', '-R', action='store_true',
                        help='generate pairs of images with a pose change instead of a single image')
    parser.add_argument('--cache', '-c', metavar='DIR', default=CACHE_DIR,
                        help='path to cache dir (default: %s), ./[mission]/[id] is added to the path' % CACHE_DIR)
    parser.add_argument('--output', '-o', metavar='DIR', default=None,
                        help='path to output dir, default: %s/[mission]/final-[id]' % CACHE_DIR)
    parser.add_argument('--log-missing', action='store_true', help='if log is missing, do without')
    parser.add_argument('--mission', '-m', metavar='M', default='rose', choices=missions,
                        help='mission: %s (default: rose)' % (' | '.join(missions)))
    parser.add_argument('--count', '-n', default='10', type=str, metavar='N',
                        help='number of images to be generated, accepts also format [start:end]')
    parser.add_argument('--crop', default=0, type=int,
                        help='detect object and crop exported images, only when "--relative" not set')
    parser.add_argument('--id', '-i', default=None, type=str, metavar='N',
                        help='a unique dataset id, defaults to a hash calculated from image generation parameters')

    parser.add_argument('--max-rot-err', default=10, type=float, metavar='A',
                        help='Max rotation error (in deg) allowed when determining pose with AKAZE-PnP-RANSAC (default: %f)' % 10)

    parser.add_argument('--res-mult', type=float, default=1.0, help="scale camera resultion, default=1.0")
    parser.add_argument('--sm-lores', default=False, help="use low resolution shape model", action='store_true')
    parser.add_argument('--sm-noise', default=0, type=float, metavar='SD',
                        help='Shape model noise level (default: %f)' % 0)
    parser.add_argument('--sm-noise-len-sc', default=SHAPE_MODEL_NOISE_LEN_SC, type=float, metavar='SC',
                        help='Shape model noise length scale (default: %f)' % SHAPE_MODEL_NOISE_LEN_SC)

    parser.add_argument('--tx-noise', default=0, type=float, metavar='SD',
                        help='Texture noise level (default: %f)' % 0)
    parser.add_argument('--tx-noise-len-sc', default=SHAPE_MODEL_NOISE_LEN_SC, type=float, metavar='SC',
                        help='Texture noise length scale (default: %f)' % SHAPE_MODEL_NOISE_LEN_SC)

    parser.add_argument('--haze', default=0.0, type=float, metavar='HZ',
                        help='Max haze brightness (uniform-dist) (default: %f)' % 0.0)

    parser.add_argument('--jets', default=0.0, type=float, metavar='JN',
                        help='Average jet count (exp-distr) (default: %f)' % 0.0)
    parser.add_argument('--jet-int-mode', '--jm', default=0.001, type=float, metavar='JM',
                        help='Jet intensity mode [0, 1], beta-distributed, (default: %f)' % 0.001)
    parser.add_argument('--jet-int-conc', '--jc', default=10, type=float, metavar='JC',
                        help='Jet intensity concentration [1, 1000] (default: %f)' % 10)

    parser.add_argument('--max-phase-angle', default=100, type=float, metavar='A',
                        help='Max phase angle allowed when generating system state (default: %f)' % 100)
    parser.add_argument('--max-sc-distance', default=1.0, type=float, metavar='A',
                        help='Max spacecraft distance as min_dist+A*(max_dist-min_dist) (default: %f)' % 1.0)
    parser.add_argument('--max-sc-lateral-disp', default=1.0, type=float, metavar='A',
                        help='Max spacecraft lateral displacement [0-1] where 0 is always centered (default: %f)' % 1.0)

    parser.add_argument('--hapke-noise', '--hn', default=0.0, type=float, metavar='SD',
                        help=('Randomize all Hapke reflection model parameters by multiplying with log normally'
                              ' distributed random variable with given SD (default: %f)') % 0.0)
    parser.add_argument('--hapke-th-sd', '--h1', default=None, type=float, metavar='SD',
                        help='Override Hapke effective roughness, th_p [deg] param noise sd (default: %f)' % 0.0)
    parser.add_argument('--hapke-w-sd', '--h2', default=None, type=float, metavar='SD',
                        help='Override Hapke single scattering albedo, w [0, 1] param noise sd (default: %f)' % 0.0)
    parser.add_argument('--hapke-b-sd', '--h3', default=None, type=float, metavar='SD',
                        help='Override Hapke SPPF asymmetry parameter, b [-1, 1] param noise sd (default: %f)' % 0.0)
    parser.add_argument('--hapke-c-sd', '--h4', default=None, type=float, metavar='SD',
                        help='Override Hapke SPPF asymmetry parameter, b [0, 1] param noise sd (default: %f)' % 0.0)
    parser.add_argument('--hapke-shoe', '--h5', default=None, type=float, metavar='SD',
                        help='Override Hapke amplitude of shadow-hiding opposition effect (SHOE), B_SH0 [>=0] param noise sd (default: %f)' % 0.0)
    parser.add_argument('--hapke-shoe-w', '--h6', default=None, type=float, metavar='SD',
                        help='Override Hapke angular half width of SHOE [rad] param noise sd (default: %f)' % 0.0)
    parser.add_argument('--hapke-cboe', '--h7', default=None, type=float, metavar='SD',
                        help='Override Hapke amplitude of coherent backscatter opposition effect (CBOE), B_CB0 [>=0] param noise sd (default: %f)' % 0.0)
    parser.add_argument('--hapke-cboe-w', '--h8', default=None, type=float, metavar='SD',
                        help='Override Hapke angular half width of CBOE [rad] param noise sd (default: %f)' % 0.0)

    # only related to "relative" data generation mode
    parser.add_argument('--noise-time', '--nt', default=0, type=float, metavar='SD',
                        help='time noise when generating second frame, noise sd [s] (default: %f)' % 0.0)
    parser.add_argument('--noise-ast-rot-axis', '--nar', default=0, type=float, metavar='SD',
                        help='asteroid axis rotation noise when generating second frame, noise sd [deg] (default: %f)' % 0.0)
    parser.add_argument('--noise-ast-phase-shift', '--nap', default=0, type=float, metavar='SD',
                        help='asteroid rotation phase shift noise when generating second frame, noise sd [deg] (default: %f)' % 0.0)
    parser.add_argument('--noise-sco-lat', '--nsa', default=0, type=float, metavar='SD',
                        help='spacecraft latitudinal orientation noise when generating second frame, noise sd [deg] (default: %f)' % 0.0)
    parser.add_argument('--noise-sco-lon', '--nso', default=0, type=float, metavar='SD',
                        help='spacecraft longitudinal orientation noise when generating second frame, noise sd [deg] (default: %f)' % 0.0)
    parser.add_argument('--noise-sco-rot', '--nsr', default=0, type=float, metavar='SD',
                        help='spacecraft roll orientation noise when generating second frame, noise sd [deg] (default: %f)' % 0.0)
    parser.add_argument('--noise-lateral', '--nl', default=0, type=float, metavar='SD',
                        help='spacecraft lateral translational noise when generating second frame, noise sd [m/m, ratio] (default: %f)' % 0.0)
    parser.add_argument('--noise-altitude', '--na', default=0, type=float, metavar='SD',
                        help='spacecraft radial translational noise when generating second frame, noise sd [m/m, ratio] (default: %f)' % 0.0)
    parser.add_argument('--noise-phase-angle', '--npa', default=0, type=float, metavar='SD',
                        help='change in phase angle when generating second frame, noise sd [deg] (default: %f)' % 0.0)
    parser.add_argument('--noise-light-dir', '--nld', default=0, type=float, metavar='SD',
                        help='change in light direction when generating second frame, noise sd [deg] (default: %f)' % 0.0)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    sm = get_system_model(args.mission, hi_res_shape_model=not args.sm_lores, res_mult=args.res_mult)
    file_prefix_mod = ''
    img_file_prefix = 'cm' if args.sm_noise > 0 else ''
    log_prefix = ('r-' if args.relative else '')+'dsg-'+args.mission+'-'+args.id+'-'
    cache_path = os.path.join(args.cache, args.mission, args.id)

    # operational zone only
    sm.min_distance = sm.min_med_distance
    sm.max_distance = sm.min_distance + args.max_sc_distance * (sm.max_med_distance - sm.min_distance)
    sm.min_elong = 180 - args.max_phase_angle    # max phase angle = 100 deg

    traj_len = 2 if args.relative else 1

    state_gen = lambda sm: sm.random_state(uniform_distance=False, opzone_only=True,
                                           max_sc_lateral_disp=args.max_sc_lateral_disp)

    print('starting to generate images')
    tl = TestLoop(sm, file_prefix_mod=file_prefix_mod, est_real_ast_orient=False,
              state_generator=state_gen, uniform_distance_gen=False, operation_zone_only=True,
              cache_path=cache_path, only_populate_cache=True,
              sm_noise=0, sm_noise_len_sc=SHAPE_MODEL_NOISE_LEN_SC,
              navcam_cache_id=img_file_prefix, save_distance=True, save_coords=True,
              traj_len=traj_len, traj_prop_dt=60,
              real_sm_noise=args.sm_noise, real_sm_noise_len_sc=args.sm_noise_len_sc,
              real_tx_noise=args.tx_noise, real_tx_noise_len_sc=args.tx_noise_len_sc,
              haze=args.haze,
              jets=args.jets, jet_int_mode=args.jet_int_mode, jet_int_conc=args.jet_int_conc,
              hapke_noise=args.hapke_noise,
              hapke_th_sd=args.hapke_th_sd, hapke_w_sd=args.hapke_w_sd,
              hapke_b_sd=args.hapke_b_sd, hapke_c_sd=args.hapke_c_sd,
              hapke_shoe=args.hapke_shoe, hapke_shoe_w=args.hapke_shoe_w,
              hapke_cboe=args.hapke_cboe, hapke_cboe_w=args.hapke_cboe_w,
              noise_time=args.noise_time, noise_ast_rot_axis=args.noise_ast_rot_axis,
              noise_ast_phase_shift=args.noise_ast_phase_shift, noise_sco_lat=args.noise_sco_lat,
              noise_sco_lon=args.noise_sco_lon, noise_sco_rot=args.noise_sco_rot,
              noise_lateral=args.noise_lateral, noise_altitude=args.noise_altitude,
              noise_phase_angle=args.noise_phase_angle, noise_light_dir=args.noise_light_dir,
              ext_noise_dist=True
          )

    # check if can skip testloop iterations in case the execution died during previous execution
    if args.log_missing:
        log_entries = make_log_entries(os.path.join(args.cache, sm.mission_id, args.id), traj_len=traj_len)
    else:
        log_entries = read_logfiles(sm, log_prefix, args.max_rot_err, traj_len=traj_len)

    if args.relative:
        entry_exists = [i for i, fs in log_entries if np.all(
            [os.path.exists(f) and f == tl.cache_file(i, postfix='%d.png' % j)
            for j, f in enumerate(fs)])
        ]
    else:
        entry_exists = [i for i, f in log_entries if os.path.exists(f) and f == tl.cache_file(i)+'.png']

    row_range = get_range(args.count, entry_exists)

    if row_range is not None:
        tl.run(row_range, log_prefix=log_prefix,
               constant_sm_noise=True, smn_cache_id='lo',
               method='keypoint' if args.max_rot_err > 0 else 'centroid', feat=1, verbose=0)

    # export
    print('starting to export images')
    img_file_prefix = (img_file_prefix + '_') if img_file_prefix else ''
    os.makedirs(args.output, exist_ok=True)  # make sure output folder exists

    # skip export of already existing images, process only images in given iteration range
    if ':' in args.count:
        start, end = map(int, args.count.split(':'))
    else:
        start, end = 0, int(args.count)

    if args.log_missing:
        imgfiles = make_log_entries(os.path.join(args.cache, sm.mission_id, args.id), traj_len=traj_len)
    else:
        imgfiles = read_logfiles(sm, log_prefix, args.max_rot_err, traj_len=traj_len)

    if args.relative:
        imgfiles = [(f, os.path.join(args.output, os.path.basename(f))) for i, fs in imgfiles if start <= i < end for f in fs]
    else:
        imgfiles = [(f, os.path.join(args.output, os.path.basename(f))) for i, f in imgfiles if start <= i < end]

    ok_files = cnn_export.get_files_with_metadata(args.output, traj_len=traj_len)
    imgfiles = [sf for sf, df in imgfiles if not os.path.exists(df) or os.path.basename(df) not in ok_files]

    if args.relative:
        # Copies images as they are, saves system states in format "traj id, frame id, sc_ast_r (3), sc_ast_q (4), sc_light_u (3)"
        # Data about keypoints and their matches generated at nn data loader, could be something like
        #     "traj id, frame id1, frame id2, ix1, iy1, x1, y1, z1, ix2p, iy2p, ix2, iy2, x2, y2, z2, d_2d, d_3d"
        cnn_export.export_relative(sm, args.output, src_imgs=imgfiles, img_prefix=img_file_prefix,
                                   title="Synthetic Image Set, mission=%s, id=%s" % (args.mission, args.id), debug=0)
    else:
        cnn_export.export(sm, args.output, src_imgs=imgfiles, trg_shape=(224, 224), img_prefix=img_file_prefix,
                          title="Synthetic Image Set, mission=%s, id=%s" % (args.mission, args.id),
                          crop=args.crop, debug=0)


def sample_mosaic():
    import cv2

    args = parse_arguments()
    files = cnn_export.get_files_with_metadata(args.output)
    s = 56
    r, c = 6, 24
    comp = np.ones(((s+1)*r-1, (s+1)*c-1), dtype=np.uint8)*255
    for i, file in enumerate([f for f in files if f][0:r*c]):
        img = cv2.imread(os.path.join(args.output, file), cv2.IMREAD_GRAYSCALE)
        k, j = i // c, i % c
        comp[k*(s+1):(k+1)*(s+1)-1, j*(s+1):(j+1)*(s+1)-1] = cv2.resize(img, (s, s))
    cv2.imwrite('mosaic.png', comp)
    cv2.imshow('mosaic.png', comp)
    cv2.waitKey()


def get_range(org_range, exists):
    if ':' in org_range:
        start, end = map(int, org_range.split(':'))
    else:
        start, end = 0, int(org_range)
    if len(exists) > 0:
        start = max(start, np.max(exists)+1)
    if start >= end:
        return None
    return '%d:%d' % (start, end)


def read_logfiles(sm, file_prefix, max_err, traj_len=1):
    # find all files with given prefix, merge in order of ascending date
    logfiles = []
    for f in os.listdir(LOG_DIR):
        if f[:len(file_prefix)] == file_prefix and f[-4:] == '.log':
            fpath = os.path.join(LOG_DIR, f)
            logfiles.append((os.path.getmtime(fpath), fpath))
    logfiles = sorted(logfiles, key=lambda x: x[0])

    imgs, roterrs = {}, {}
    for _, logfile in logfiles:
        with open(logfile, newline='') as csvfile:
            data = csv.reader(csvfile, delimiter='\t')
            first = True
            pbar = tqdm(data, mininterval=3)
            ok = 0
            for n, row in enumerate(pbar):
                if len(row) > 10:
                    if first:
                        first = False
                        lbl_i = row.index('iter')
                        img_i = row.index('imgfile')
                        rot_i = row.index('rot error')
                    else:
                        try:
                            i = int(row[lbl_i])
                            img0 = row[img_i]
                            if traj_len == 1:
                                imgs[i] = img0
                                roterrs[i] = float(row[rot_i])
                            else:
                                imgs[i] = [re.sub(r'(.*?)_(\d+)\.(png|jpg)$', r'\1_%s.\3' % j, img0) for j in range(traj_len)]
                                #roterrs[i] = min(float(row[rot_i]), relative_pose_err(sm, imgs[i]))
                                roterrs[i] = float(row[rot_i])
                                # TODO: cache result as now this is called two times
                            ok += 1 if max_err <= 0 or roterrs[i] < max_err else 0
                            pbar.set_postfix({'ratio': '%.3f' % (ok / n)}, refresh=False)
                        except ValueError as e:
                            print('Can\'t convert roterr or iter on row %s' % (row[lbl_i],))
                            raise e

    images = [(i, imgs[i]) for i, e in roterrs.items() if not math.isnan(e) and (max_err <= 0 or e < max_err)]
    images = sorted(images, key=lambda x: x[0])
    return images


def make_log_entries(path, traj_len):
    entries = []
    for file in os.listdir(path):
        tmp = re.search(r'^(.*?)(\d+)_(\d+)\.(png|jpg)$', file)
        if tmp:
            base, i, j_, ext = tmp.groups()
            if int(j_) == 0:
                entries.append((int(i), [os.path.join(path, f'{base}{i}_{j}.{ext}') for j in range(traj_len)]))

    entries = sorted(entries, key=lambda x: x[0])
    return entries


def true_relative_pose(sm, img_files):
    rel_qs = []
    for file in img_files:
        sm.load_state(file[:-4].replace('_cm', '') + '.lbl')
        sm.swap_values_with_real_vals()
        _, sc_ast_lf_q, _ = sm.get_system_scf()
        rel_qs.append(sc_ast_lf_q)
    return rel_qs[0].conj() * rel_qs[1]


def relative_pose_err(sm, imgs):
    try:
        img0, img1 = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in imgs]
        kp0, desc0, det = KeypointAlgo.detect_features(img0, KeypointAlgo.AKAZE, max_feats=1000, for_ref=True, maxmem=0)
        kp1, desc1, det = KeypointAlgo.detect_features(img1, KeypointAlgo.AKAZE, max_feats=1000, for_ref=False, maxmem=0)
        matches = KeypointAlgo.match_features(desc1, desc0, det.defaultNorm(), method='brute')

        depth0 = cv2.imread(imgs[0][:-4]+'.d.exr', cv2.IMREAD_UNCHANGED)
        kp0_3d = KeypointAlgo.inverse_project(sm, [kp0[m.trainIdx].pt for m in matches], depth0, np.nanmax(depth0), 1)
        kp1_2d = np.array([kp1[m.queryIdx].pt for m in matches], dtype='float32')
        rvec, tvec, inliers = KeypointAlgo.solve_pnp_ransac(sm, kp1_2d, kp0_3d, 8)
    except PositioningException as e:
        inliers = []

    err = np.nan
    if len(inliers) >= KeypointAlgo.MIN_FEATURES:
        rel_q = true_relative_pose(sm, imgs)
        sc2cv_q = sm.frm_conv_q(sm.SPACECRAFT_FRAME, sm.OPENCV_FRAME)

        est_rel_cv_q = tools.angleaxis_to_q(rvec)
        est_rel_q1 = sc2cv_q * est_rel_cv_q * sc2cv_q.conj()

        # solvePnPRansac has some bug that apparently randomly gives 180deg wrong answer
        est_rel_q2 = sc2cv_q * est_rel_cv_q * tools.ypr_to_q(0, math.pi, 0) * sc2cv_q.conj()

        err = math.degrees(min(tools.angle_between_q(est_rel_q1, rel_q),
                               tools.angle_between_q(est_rel_q2, rel_q)))

    return err


if __name__ == '__main__':
    if 1:
        main()
    else:
        sample_mosaic()
