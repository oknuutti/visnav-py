import argparse

import numpy as np
import matplotlib.pyplot as plt
import cv2


def main():
    parser = argparse.ArgumentParser(description='Compare pixel values of two images')
    parser.add_argument('img1', nargs=1, help='path to first image')
    parser.add_argument('img2', nargs=1, help='path to second image')
    parser.add_argument('--grayscale', '-g', type=int, default=1,
                        help='convert images to grayscale images')
    parser.add_argument('--scale', '-s', type=int, default=0,
                        help='scale image brightness of second image for best match')
    parser.add_argument('--mask', '-m', type=int, default=0,
                        help='mask threshold, values below in first image are not considered for summary stats')
    parser.add_argument('--quantile', '-q', metavar='Q', type=float, default=0.5,
                        help='Quantile for summary stats [0-1], set to 0.5 for median')
    parser.add_argument('--max-err', type=float, default=50,
                        help='max absolute error for error plot')
    args = parser.parse_args()

    img1 = cv2.imread(args.img1[0], cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(args.img2[0], cv2.IMREAD_UNCHANGED)

    if args.grayscale:
        # fast and dirty
        if len(img1.shape) > 2:
            img1 = img1[:, :, 0]
        if len(img2.shape) > 2:
            img2 = img2[:, :, 0]

    if args.mask:
        def circ_kern(size):
            kern = cv2.getGaussianKernel(size, 1.0)
            kern = kern * kern.T
            _, kern = cv2.threshold(kern, kern[0, size//2], 1.0, cv2.THRESH_BINARY)
            return kern.astype(np.uint8)

        _, mask = cv2.threshold(img1, args.mask, np.max(img1), cv2.THRESH_BINARY)

        # remove stars
        kern = circ_kern(7)
        mask = cv2.erode(mask, kern, iterations=1)
        mask = cv2.dilate(mask, kern, iterations=1)

        # remove small shadows inside asteroid
        kern = circ_kern(25)
        mask = cv2.dilate(mask, kern, iterations=1)
        mask = cv2.erode(mask, kern, iterations=1)

        img = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if args.grayscale else img1
        mask_img = np.concatenate((mask[:, :, None], np.zeros((*mask.shape, 2))), axis=2).astype(img.dtype)
        img = cv2.addWeighted(img, 0.5, mask_img, 0.5, 0.0)
        plt.figure()
        plt.imshow(img)
        plt.title('mask')

        mask = mask.astype(bool).flatten()
    else:
        mask = np.ones(img1.shape[:2], dtype=bool).flatten()

    if args.scale:
        img2 = img2.astype(float)
        from scipy.optimize import minimize
        cost = lambda x: np.quantile(np.abs(x * img2.flatten()[mask] - img1.flatten()[mask]), args.quantile)
        res = minimize(cost, [1.0], method="Nelder-Mead", options={'maxiter': 100, 'xtol': 1e-4, 'ftol': 1e-4})
        img2 *= res.x
        print('Second image scaled by multiplying with %f' % res.x)

    abs_err = np.abs(img2.astype(float) - img1.astype(float))
    vmax = 255 if img1.dtype == np.uint8 else max(np.max(img1), np.max(img2))

    print('Quantile %f of abs(err): %.3f' % (args.quantile, np.quantile(abs_err.flatten()[mask], args.quantile)))

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9.5, 3))
    axs[0].imshow(img1, cmap='gray', vmin=0, vmax=vmax)
    axs[1].imshow(img2, cmap='gray', vmin=0, vmax=vmax)
    a = axs[2].imshow(abs_err, vmin=0, vmax=args.max_err)

    (x0, y0, x1, y1), hpad, vpad, frac = axs[2].figbox.extents, 0.01, 0.022, 0.1
    cax = fig.add_axes((x1+hpad, y0+vpad, (x1-x0)*frac, y1-y0-vpad*2))   # [left, bottom, width, height]
    fig.colorbar(a, cax=cax, ax=axs[2])
    plt.show()


if __name__ == '__main__':
    main()
