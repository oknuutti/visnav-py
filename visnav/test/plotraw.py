import argparse

import rawpy
import matplotlib.pyplot as plt
import numpy as np
import cv2

from visnav.algo.model import Camera


def main():
    parser = argparse.ArgumentParser("load raw image, plot")
    parser.add_argument('--path1', required=True)
    parser.add_argument('--path2')
    parser.add_argument('--region1', nargs=4, type=int, help="region in img1: x0 x1 y0 y1")
    parser.add_argument('--region2', nargs=4, type=int, help="region in img2: x0 x1 y0 y1")
    parser.add_argument('--demosaic', default='DHT', choices=('DHT', 'AHD', 'LINEAR', 'NAIVE'), help="demosaic algo")
    parser.add_argument('--color-corr', type=int, default=0, help="do color correction")
    parser.add_argument('--px-size', '--px', type=float, help='pixel size [um]')
    parser.add_argument('--focal-length', '--fl', type=float, help='focal length [mm]')
    parser.add_argument('--focal-length-px', '--flx', type=float, help='focal length [px]')
    parser.add_argument('--principal-point', '--pp', nargs=2, type=float, help='principal point: x y')
    args = parser.parse_args()

    with rawpy.imread(args.path1) as raw:
        img1 = raw.raw_image.copy()
        bg1 = np.mean(raw.black_level_per_channel)
        rgb = None
        if args.path2 is None:
            if args.demosaic == "NAIVE":
                rgb = naive_bilinear_demosaic(img1 - bg1, raw.raw_colors)
                if args.color_corr:
                    shape = rgb.shape
                    # rgb = rgb / np.array(raw.camera_whitebalance)[None, None, :3]
                    rgb = rgb.reshape((-1, 3)).dot(raw.color_matrix[:3, :3]).reshape(shape)
                    # rgb = rgb / np.array(raw.camera_whitebalance)[None, None, :3]
            else:
                # DHT said to be good, default is AHD, Linear is not naive bilinear..
                #  - not easy to find info at https://www.libraw.org/
                wb = None if args.color_corr else [1, 1, 1, 1]
                rgb = raw.postprocess(demosaic_algorithm=getattr(rawpy.DemosaicAlgorithm, args.demosaic), gamma=(1, 1),
                                      no_auto_bright=True, no_auto_scale=True, user_wb=wb, output_bps=16)
                if not args.color_corr:
                    rgb = rgb.reshape((-1, 3)).dot(np.linalg.inv(raw.color_matrix[:3, :3].T)).reshape(rgb.shape)

    if args.path2 is None:
        plt.imshow(np.clip(rgb/np.max(rgb), 0, 1))
        plt.show()
        return

    with rawpy.imread(args.path2) as raw:
        img2 = raw.raw_image.copy()
        bg2 = np.mean(raw.black_level_per_channel)

    (h, w), cam = img1.shape, None
    if args.px_size and args.focal_length:
        cam = Camera(w, h, sensor_size=(h*args.px_size*1e-3, w*args.px_size*1e-3), focal_length=args.focal_length)
    elif args.focal_length_px:
        px_size, fl = (args.px_size or 3.0) * 1e-3, args.focal_length_px

        cam_mx = None
        if args.principal_point:
            cam_mx = np.array([[fl, 0, args.principal_point[0]],
                               [0, fl, args.principal_point[1]],
                               [0,  0, 1]])

        cam = Camera(w, h, sensor_size=(h*px_size, w*px_size), focal_length=fl*px_size,
                     px_saturation_e=2000, cam_mx=cam_mx)

    if cam:
        c1 = np.mean(np.array(args.region1).reshape((2, 2)), axis=1)
        c2 = np.mean(np.array(args.region2).reshape((2, 2)), axis=1)
        rel_ill1 = cam.cos4(*c1)
        rel_ill2 = cam.cos4(*c2)
        print('expected ratio: %.2f%%' % (100*rel_ill2/rel_ill1,))

    plt.figure(1)
    plt.imshow(img1)
    plt.figure(2)
    plt.imshow(img2)
    plt.show()

    x10, x11, y10, y11 = args.region1
    x20, x21, y20, y21 = args.region2
    mean1 = np.mean(img1[y10:y11, x10:x11]) - bg1
    mean2 = np.mean(img2[y20:y21, x20:x21]) - bg2

    print('shape: %s' % (img1.shape,))
    print('mean1: %s' % mean1)  # c=[1350.0, 3775.0], v=84.188 (x - 64 = 20.188)
    print('mean2: %s' % mean2)  # c=[1405.0, 2020.0], v=113.138	(x - 64 = 49.138)
    print('ratio: %.2f%%' % (100*mean2/mean1,))

    # phone cam calib based on 6 photos:
    #   intrinsics (f_x, f_y, c_x, c_y): 1393.8, 1375.9, 1034.2, 765.49
    #   dist coefs (k_1, k_2, p_1, p_2, k_3): 0.17843, -1.7580, 0.00000, 0.00000, 0.00000
    #   stds (f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3): 40.9, 50.1, 24.6, 44.7, 0.188, 1.34, 0.000, 0.000, 0.000
    #   relative (f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3): 2.9%, 3.6%, 2.4%, 5.8%, 110%, 76%, nan%, nan%, nan%
    #   repr err across all imgs q=0.99: 6.229, e_r: 5.069, e_c: 5.488


def naive_bilinear_demosaic(raw, color_idxs):
    rgb = np.ones_like(raw, shape=raw.shape + (3,)) * np.nan

    for k in range(3):
        # assign measured colors
        i, j = np.where(color_idxs == k)
        rgb[i, j, k] = raw[i, j]

        # detect missing measurements, interpolate them
        isna = np.isnan(rgb[:, :, k])
        rgbz = rgb[:, :, k].copy()
        rgbz[isna] = 0
        interp = cv2.filter2D(rgbz, -1, np.ones((3, 3)))
        counts = cv2.filter2D(np.logical_not(isna).astype(np.float32), -1, np.ones((3, 3)))
        with np.errstate(divide='ignore'):
            interp = interp / counts
        i, j = np.where(isna)
        rgb[i, j, k] = interp[i, j]

    return rgb


if __name__ == '__main__':
    main()
