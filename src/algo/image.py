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
    def crop_and_zoom_image(full_image, x_off, y_off, width, height, scale):
        return cv2.resize(full_image[y_off:(y_off+height), x_off:(x_off+width)],
                        None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
    @staticmethod
    def process_target_image(image_src):
        hist = cv2.calcHist([image_src],[0],None,[256],[0,256])
        
        if not BATCH_MODE:
            threshold_value = ImageProc.optimal_threshold(hist)
        else:
            threshold_value = 15 # imgs created by VISIT dont have background
        
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
        if False:
            from matplotlib import pyplot as plt
            plt.plot(x, fitfun1(out[0][:3], x))
            plt.plot(x, fitfun2(out[0][3:], x))
            plt.plot(x, fitfun(out[0], x))
            plt.plot(x, loghist)
            plt.show()
        
        return threshold_value

