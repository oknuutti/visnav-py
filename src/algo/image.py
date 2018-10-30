import math

from scipy import signal
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
    def crop_and_zoom_image(full_image, x_off, y_off, width, height, scale):
        return cv2.resize(full_image[y_off:(y_off+height), x_off:(x_off+width)],
                        None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
    @staticmethod
    def equalize_brightness(image, ref_image, percentile=98):
        ip = np.percentile(image, percentile)
        rp = np.percentile(ref_image, percentile)
        return cv2.convertScaleAbs(image, None, rp/ip, 0)
    
    @staticmethod
    def adjust_gamma(image, gamma):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    @staticmethod
    def add_stars(img, mask, coef=2):
        # add power law distributed stars to image
        assert img.shape == img.shape[:2], 'works only with grayscale images'
        stars = np.random.pareto(coef, img.shape)
        # can be over 255, will clip later
        img[mask] = np.clip(stars[mask], 0, 600)
        return img

    @staticmethod
    def apply_point_spread_fn(img, ratio):
        # ratio is how many % of power on central pixel
        sd = 1/math.sqrt(2*math.pi*ratio)
        kernel = ImageProc.gkern2d(5, sd)
        img = signal.convolve2d(img, kernel, mode='same')
        return img

    @staticmethod
    def gkern2d(l=5, sig=1.):
        """
        creates gaussian kernel with side length l and a sigma of sig
        """
        ax = np.arange(-l // 2 + 1., l // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sig ** 2))
        return kernel / np.sum(kernel)

    @staticmethod
    def add_ccd_noise(img, mean=7, sd=2):
        img += np.random.normal(mean, sd, img.shape)
        return img

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
    def optimal_threshold(hist):
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
        if True:
            from matplotlib import pyplot as plt
            plt.clf()
            plt.plot(x, fitfun1(out[0][:3], x), label='background')
            plt.plot(x, fitfun2(out[0][3:], x), label='foreground')
            plt.plot(x, fitfun(out[0], x), label='total fit')
            plt.plot(x, loghist, label='log(hist)')
            plt.legend()
            fig = plt.gcf()
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            cv2.imshow('histogram fitting', data)
        
        return threshold_value


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