import argparse
import sys

from visnav.batch1 import get_system_model
from visnav.testloop import TestLoop
from visnav.settings import *


# TODO:
#    - DEBUG
#    - At testloop.py, randomize image generation using
#       - Haze: ampl, length sc
#    - If SPL-algo succeeds, include image in dataset
#    - Call export-test-batch2 on the generated files
#       - Fix cropping so that out-of-frame parts stay out-of-frame
#       - Include depth image rendered at cropped pose (for depth target)
#       - Include image rendered at cropped pose (for autoencoder target)
#    - Save a config file about parameters used for dataset generation
#    - Support usage of real navcam images for generating a validation/testing dataset
#    (- Upload dataset to triton?)


def main():
    args = parse_arguments()

    sm = get_system_model(args.mission, hi_res_shape_model=True)
    file_prefix_mod = ''

    tl = TestLoop(sm, file_prefix_mod=file_prefix_mod, est_real_ast_orient=False,
                  state_generator=None, uniform_distance_gen=False, operation_zone_only=True,
                  cache_path=os.path.join(args.cache, args.mission, args.id),
                  sm_noise=0, sm_noise_len_sc=SHAPE_MODEL_NOISE_LEN_SC,
                  navcam_cache_id='cm' if args.sm_noise > 0 else '',
                  real_sm_noise=args.sm_noise, real_sm_noise_len_sc=args.sm_noise_len_sc,
                  real_tx_noise=args.tx_noise, real_tx_noise_len_sc=args.tx_noise_len_sc,
                  haze=args.haze, haze_len_sc=args.haze_len_sc,
                  hapke_noise=args.hapke_noise,
                  hapke_th_sd=args.hapke_th_sd, hapke_w_sd=args.hapke_w_sd,
                  hapke_b_sd=args.hapke_b_sd, hapke_c_sd=args.hapke_c_sd,
                  hapke_shoe=args.hapke_shoe, hapke_shoe_w=args.hapke_shoe_w,
                  hapke_cboe=args.hapke_cboe, hapke_cboe_w=args.hapke_cboe_w)

    tl.run(args.count, log_prefix='dsg-'+args.mission+'-'+args.id+'-',
           constant_sm_noise=True, smn_cache_id='lo',
           method='keypoint', feat=1)


def parse_arguments():
    missions = ('rose', 'didy1n', 'didy1w', 'didy2n', 'didy2w')

    parser = argparse.ArgumentParser(description='Asteroid Image Data Generator')
    parser.add_argument('--cache', '-c', metavar='DIR', default=CACHE_DIR,
                        help='path to cache dir (default: %s), ./[mission]/[id] is added to the path' % CACHE_DIR)
    parser.add_argument('--output', '-o', metavar='DIR', default=None,
                        help='path to output dir, default: %s/[mission]/final-[id]' % CACHE_DIR)
    parser.add_argument('--mission', '-m', metavar='M', default='rose', choices=missions,
                        help='mission: %s (default: rose)' % (' | '.join(missions)))
    parser.add_argument('--count', '-n', default='10', type=str, metavar='N',
                        help='number of images to be generated, accepts also format [start:end]')
    parser.add_argument('--id', '-i', default=None, type=str, metavar='N',
                        help='a unique dataset id, defaults to a hash calculated from image generation parameters')

    parser.add_argument('--sm-noise', default=0, type=float, metavar='SD',
                        help='Shape model noise level (default: %f)' % 0)
    parser.add_argument('--sm-noise-len-sc', default=SHAPE_MODEL_NOISE_LEN_SC, type=float, metavar='SC',
                        help='Shape model noise length scale (default: %f)' % SHAPE_MODEL_NOISE_LEN_SC)

    parser.add_argument('--tx-noise', default=0, type=float, metavar='SD',
                        help='Texture noise level (default: %f)' % 0)
    parser.add_argument('--tx-noise-len-sc', default=SHAPE_MODEL_NOISE_LEN_SC, type=float, metavar='SC',
                        help='Texture noise length scale (default: %f)' % SHAPE_MODEL_NOISE_LEN_SC)

    parser.add_argument('--haze', default=0.0, type=float, metavar='SD',
                        help='Haze brightness (default: %f)' % 0.0)
    parser.add_argument('--haze-len-sc', default=SHAPE_MODEL_NOISE_LEN_SC, type=float, metavar='SC',
                        help='Haze length scale (default: %f)' % SHAPE_MODEL_NOISE_LEN_SC)

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

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
