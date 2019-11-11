from functools import lru_cache

import math

from scipy import optimize, stats
import numpy as np
import cv2

from settings import *

class ImageProc():
    latest_opt = None
    show_fit = None

    @staticmethod
    def add_noise_to_image(image, noise_img_file):
        tmp = cv2.imread(noise_img_file, cv2.IMREAD_UNCHANGED)
        noise_img = cv2.resize(tmp, None,
                               fx=image.shape[1]/tmp.shape[1],
                               fy=image.shape[0]/tmp.shape[0],
                               interpolation=cv2.INTER_CUBIC)
                               
        return cv2.add(image, noise_img[:,:,3])
    
    @staticmethod
    def crop_and_zoom_image(img, x_off, y_off, width, height, scale, trg_w_h=None):
        if scale is None:
            scale = min(trg_w_h[1] / height, trg_w_h[0] / width)

        imgc = cv2.resize(img[y_off:y_off + height, x_off:x_off + width], None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

        if trg_w_h is not None:
            imgd = np.zeros((trg_w_h[1], trg_w_h[0]), dtype=img.dtype)
            y0, x0 = (imgd.shape[0] - imgc.shape[0]) // 2, (imgd.shape[1] - imgc.shape[1]) // 2
            imgd[y0:y0 + imgc.shape[0], x0:x0 + imgc.shape[1]] = imgc
        else:
            imgd = imgc

        return imgd

    @staticmethod
    def single_object_bounds(img, threshold, crop_marg, min_px, debug=False):
        # binary image
        _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        # remove stars
        mask = cv2.erode(mask, ImageProc.bsphkern(9), iterations=1)

        if np.sum(mask) < min_px:
            return (None,) * 4

        # detect target
        x_, y_, w_, h_ = cv2.boundingRect(mask)

        # add margin
        x, y = max(0, x_ - crop_marg), max(0, y_ - crop_marg)
        w = min(mask.shape[1] - x, w_ + 2*crop_marg - (x - x_ + crop_marg))
        h = min(mask.shape[0] - y, h_ + 2*crop_marg - (y - y_ + crop_marg))

        if debug:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_color = cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)
            img_color[y + h // 2, x + w // 2] = (0, 0, 255)
            cv2.imshow('box', cv2.resize(img_color, (512, 512)))

        return x, y, w, h



    @staticmethod
    def equalize_brightness(image, ref_image, percentile=98, image_gamma=1):
        image = ImageProc.adjust_gamma(image, 1/image_gamma)
        ip = np.percentile(image, percentile)
        rp = np.percentile(ImageProc.adjust_gamma(ref_image, 1/image_gamma), percentile)
        image = cv2.convertScaleAbs(image, None, rp/ip, 0)
        return ImageProc.adjust_gamma(image, image_gamma)
    
    @staticmethod
    def adjust_gamma(image, gamma):
        if gamma == 1:
            return image

        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma

        if image.dtype == 'uint8':
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            adj_img = cv2.LUT(image, table)
        else:
            adj_img = ((image / 255.0) ** invGamma) * 255

        # apply gamma correction using the lookup table
        return adj_img

    @staticmethod
    def apply_point_spread_fn(img, ratio):
        # ratio is how many % of power on central pixel
        sd = 1/math.sqrt(2*math.pi*ratio)
        kernel = ImageProc.gkern2d(5, sd)
        cv2.filter2D(img, -1, kernel, img)
        return img

    @staticmethod
    @lru_cache(maxsize=1)
    def gkern2d(l=5, sig=1.):
        """
        creates gaussian kernel with side length l and a sigma of sig
        """
        ax = np.arange(-l // 2 + 1., l // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sig ** 2))
        return kernel / np.sum(kernel)

    @staticmethod
    def bsphkern(l=5):
        """
        creates a binary spherical kernel
        """
        gkern = ImageProc.gkern2d(l=l, sig=l)
        limit = gkern[l//2, -1]*0.995
        return np.array(gkern >= limit, dtype=np.uint8)

    @staticmethod
    def add_stars(img, mask, coef=2, cache=False):
        # add power law distributed stars to image
        assert img.shape == img.shape[:2], 'works only with grayscale images'
        if not cache:
            ImageProc._cached_random_stars.cache_clear()
        stars = ImageProc._cached_random_stars(coef, img.shape)
        # can be over 255, will clip later
        img[mask] = np.clip(stars[mask], 0, 600)
        return img

    @staticmethod
    @lru_cache(maxsize=1)
    def _cached_random_stars(coef, shape):
        return np.random.pareto(coef, shape)

    @staticmethod
    def add_sensor_noise(img, mean=7, sd=2, cache=False):
        if not cache:
            ImageProc._cached_sensor_noise.cache_clear()
        img += ImageProc._cached_sensor_noise(mean, sd, img.shape)
        return img

    @staticmethod
    @lru_cache(maxsize=1)
    def _cached_sensor_noise(mean, sd, shape):
        return np.random.normal(mean, sd, shape)

    @staticmethod
    def process_target_image(image_src):
        hist = cv2.calcHist([image_src],[0],None,[256],[0,256])
        if False:
            threshold_value = ImageProc.optimal_threshold(hist)
        else:
            threshold_value = 50
        th, image_dst = cv2.threshold(image_src, threshold_value, 255, cv2.THRESH_TOZERO)
        return image_dst, hist, threshold_value
    
    @staticmethod
    def optimal_threshold(hist, image=None):
        if hist is None:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])

        tot_px = 256 # sum(hist) -- for some reason get error if divide with pixel count
        x = list(range(1,len(hist)+1))
        loghist = np.array(list(map(lambda x: math.log(x+1)/tot_px, hist)))
        
        def fitfun1(p, x):
            return stats.gamma.pdf(x, p[0], loc=0, scale=p[1])*p[2]

        def fitfun2(p, x):
            return stats.norm.pdf(x, p[0], p[1])*p[2]
        
        def fitfun(p, x):
            return fitfun1(p[:3], x) + fitfun2(p[3:], x)
        
        def errfun(p, x, y):
            tmp = y - fitfun(p, x)
#            assert False, 'p:%s, x:%s, y:%s, ffval:%s'%(p, x[0:50], y[0:50], fitfun(p, x[0:50]))
            return tmp
        
        shape = 1.5
        init = [
            shape, np.argmax(loghist)/(shape-1), 1, # for fitfun1
            127, 50, 1,                             # for fitfun2
        ]
        if not BATCH_MODE or DEBUG:
            print('init: %s'%init)
        
        out = optimize.leastsq(errfun, init, args=(x, loghist))
        ImageProc.latest_opt = out
        if not BATCH_MODE or DEBUG:
            print('result: %s'%list(out))
        
        # threshold value where background makes up roughly a fourth of all pixels
        bg = reversed(fitfun1(out[0][:3], x))
        ast = list(reversed(fitfun2(out[0][3:], x)))
        threshold_value = 255-next((i for i, v in enumerate(bg) if v/ast[i]>0.33), 255-100)
        
        if not BATCH_MODE or DEBUG:
            bg_ratio = out[0][:3][2] / out[0][3:][2]
            print('threshold_value: %s; bg_ratio: %s'%(threshold_value, bg_ratio))

        # plot figure with histogram and estimated distributions
        if DEBUG:
            from matplotlib import pyplot as plt
            plt.clf()
            plt.plot(x, fitfun1(out[0][:3], x), label='background')
            plt.plot(x, fitfun2(out[0][3:], x), label='foreground')
            plt.plot(x, fitfun(out[0], x), label='total fit')
            plt.plot(x, loghist, label='log(hist)')
            plt.legend()
            fig = plt.gcf()
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            data = data.reshape((h*3, w*3, 3))      # for some reason get_width_height returns 1/3 of the actual dims
            cv2.imshow('histogram fitting', data)
        
        return threshold_value

    @staticmethod
    def overlay_mask(image, mask):
        sc_img = min(image.shape[0], mask.shape[0])/image.shape[0]
        sc_mask = min(image.shape[0], mask.shape[0])/mask.shape[0]
        img_color = cv2.cvtColor(cv2.resize(image, None, fx=sc_img, fy=sc_img, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
        mask_color = cv2.cvtColor(cv2.resize((mask > 0).astype(np.uint8)*255, None, fx=sc_mask, fy=sc_mask, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
        mask_color[:, :, 0:2] = 0
        return cv2.addWeighted(img_color, 0.5, mask_color, 0.5, 0.0)

    @staticmethod
    def merge(images):
        summed_weights = 1
        summed_images = images[0]
        for i in range(1,len(images)):
            summed_images = cv2.addWeighted(summed_images, summed_weights/(summed_weights+1),
                                            images[i], 1/(summed_weights+1), 0.0)
            summed_weights += 1
        return summed_images

    @staticmethod
    def norm_xcorr(sce_img, res_img):
        """ calculate normalized cross corralation of images """
        
        if sce_img.shape[:2] != res_img.shape[:2]:
            sce_img = cv2.resize(sce_img, None,
                fx=res_img.shape[1]/sce_img.shape[1],
                fy=res_img.shape[0]/sce_img.shape[0],
                interpolation=cv2.INTER_CUBIC)
        
        sce_img = np.atleast_3d(sce_img)
        res_img = np.atleast_3d(res_img)
        
        sce_mean, sce_std = cv2.meanStdDev(sce_img)
        res_mean, res_std = cv2.meanStdDev(res_img)
        stds = sce_std * res_std
        if stds==0:
            return 0

        corr = (sce_img-sce_mean)*(res_img-res_mean)
        nxcorr = np.mean(corr)/stds
        
        if False:
            # for debugging
            tmp = np.log(corr-np.min(corr)+0.001)
            mint = np.min(tmp)
            maxt = np.max(tmp)
            tmp = (tmp-mint)*(1/(maxt-mint))

            print('sm %.3f, ss %.3f, rm %.3f, rs %.3f, min %.3f, max %.3f, res %.3f'%(sce_mean, sce_std, res_mean, res_std, mint, maxt, nxcorr))
            cv2.imshow('corr', tmp)
            cv2.waitKey()
        
        return nxcorr