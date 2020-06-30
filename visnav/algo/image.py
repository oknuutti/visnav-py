from functools import lru_cache

import math

from scipy import optimize, stats, integrate
import numpy as np
import quaternion
import cv2
from scipy.optimize import leastsq

from visnav.settings import *

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
    def crop_and_zoom_image(image, x_off, y_off, width, height, scale, trg_w_h=None, depth=None):
        tw, th = trg_w_h
        if scale is None:
            scale = min(th / height, tw / width)

        res = []
        for img in [image] + ([] if depth is None else [depth]):
            imgc = cv2.resize(img[y_off:y_off + height, x_off:x_off + width], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            oh, ow = img.shape
            ch, cw = imgc.shape

            if trg_w_h is not None:
                if x_off + width >= ow:
                    x0 = tw - cw
                elif x_off <= 0:
                    x0 = 0
                else:
                    x0 = (tw - cw) // 2

                if y_off + height >= oh:
                    y0 = th - ch
                elif y_off <= 0:
                    y0 = 0
                else:
                    y0 = (th - ch) // 2

                imgd = np.zeros((th, tw), dtype=img.dtype)
                imgd[y0:y0+ch, x0:x0+cw] = imgc
            else:
                imgd = imgc
            res.append(imgd)

        if depth is not None:
            return res
        return res[0]

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
    def default_preprocess(image, max=255):
        bg = np.percentile(image, 250/1024*100)
        return ImageProc.adjust_gamma(np.clip((image - bg)*max/(max-bg), 0, max), 1.8)

    @staticmethod
    def change_color_depth(img, src_bits, dst_bits):
        if src_bits == dst_bits:
            return img

        if str(img.dtype)[:4] == 'uint':
            new_type = 'uint' + str(math.ceil(dst_bits / 8) * 8)
        else:
            new_type = img.dtype

        if src_bits < dst_bits:
            img = img.astype(new_type)

        img = img * (2**(dst_bits - src_bits))

        if src_bits > dst_bits:
            img = img.astype(new_type)

        return img

    @staticmethod
    def remove_bg(img, bg_img, gain=None, max_val=None, offset=0):
        if gain is None:
            # estimate correct gain
            cost_fun = lambda g: np.var((img - g[0] * bg_img).reshape((-1, 3)), axis=0)
            x, _ = leastsq(cost_fun, np.array([1]))
            gain = x[0]
            print('estimated bg gain: %f' % gain)

        imgr = img.astype('float') - gain * bg_img
        if offset not in (None, False):
            imgr += offset - np.min(imgr)
        if max_val and offset is not False:
            return np.clip(imgr, 0, max_val).astype(img.dtype)
        return imgr

    @staticmethod
    def color_correct(img, bgr_mx, inverse=False, max_val=None):
        assert img.shape[2] == 3, 'can only do to BGR images'
        if inverse:
            bgr_mx = np.linalg.inv(bgr_mx)

        imgc = bgr_mx.dot(img.reshape((-1, 3)).T).T.reshape(img.shape)
        if max_val:
            return np.clip(imgc, 0, max_val).astype(img.dtype)
        return imgc

    @staticmethod
    def adjust_gamma(image, gamma, gamma_break=None, linear_part=True, inverse=False, max_val=255):
        if gamma == 1:
            return image

        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = gamma if inverse else 1.0 / gamma
        gamma_break = gamma_break or 0

        if image.dtype == 'uint8' and gamma_break == 0:
            # apply gamma correction using the lookup table
            max_val = min(max_val, 255)
            table = np.array([((i / max_val) ** invGamma) * max_val for i in np.arange(0, max_val+1)]).astype(image.dtype)
            adj_img = cv2.LUT(image, table)
        elif gamma_break == 0:
            adj_img = np.round(((image / max_val) ** invGamma) * max_val).astype(image.dtype)
        elif True:
            # from https://se.mathworks.com/help/vision/ref/gammacorrection.html
            b_p = gamma_break
            s_ls = 1/(gamma/b_p**(1/gamma-1) - gamma*gamma_break + gamma_break)
            f_s = gamma*s_ls/b_p**(1/gamma-1)
            c_o = f_s*b_p**(1/gamma) - s_ls*b_p

            img = image.flatten() / max_val
            I = img <= (s_ls if inverse else 1) * b_p
            nI = np.logical_not(I)

            adj_img = np.zeros(image.shape).flatten()
            adj_img[I] = (img[I] / s_ls) if inverse else (img[I] * s_ls)
            adj_img[nI] = (((img[nI] + c_o) / f_s)**gamma) if inverse else (f_s * img[nI]**(1/gamma) - c_o)
            adj_img = (adj_img * max_val).reshape(image.shape).astype(image.dtype)

        else:
            # from https://en.wikipedia.org/wiki/SRGB
            if 1:
                a = gamma_break
                K0 = a / (gamma - 1)
            else:
                K0 = gamma_break
                a = K0 * (gamma - 1)

            alpha = 1 + a
            th = alpha**gamma*(gamma-1)**(gamma-1) / a**(gamma-1) / gamma**gamma

            lim = K0 if inverse else K0/th
            img = image.flatten() / max_val
            I = img <= lim
            nI = np.logical_not(I)

            adj_img = np.zeros(image.shape).flatten()
            adj_img[I] = (img[I] / th) if inverse else (th * img[I])
            adj_img[nI] = (((img[nI] + a) / alpha)**gamma) if inverse else (alpha * img[nI]**(1/gamma) - a)
            adj_img = (adj_img * max_val).reshape(image.shape).astype(image.dtype)
            # adj_img = np.round(adj_img * max_val).reshape(image.shape).astype(image.dtype)

        return adj_img

    @staticmethod
    def apply_point_spread_fn(img, ratio):
        # ratio is how many % of power on central pixel
        sd = 1/math.sqrt(2*math.pi*ratio)
        size = 1+2*math.ceil(sd*2)
        kernel = ImageProc.gkern2d(size, sd)
        cv2.filter2D(img, -1, kernel, img)
        return img

    @staticmethod
    @lru_cache(maxsize=5)
    def gkern2d(l=5, sig=1.):
        """
        creates gaussian kernel with side length l and a sigma of sig
        """
        w, h = (l[0], l[1]) if '__iter__' in dir(l) else (l, l)
        sx, sy = (sig[0], sig[1]) if '__iter__' in dir(sig) else (sig, sig)
        ax = np.arange(-w // 2 + 1., w // 2 + 1.)
        ay = np.arange(-h // 2 + 1., h // 2 + 1.)
        xx, yy = np.meshgrid(ax, ay)
        kernel = np.exp(-((xx/sx) ** 2 + (yy/sy) ** 2) / 2)
        return kernel / np.sum(kernel)

    @staticmethod
    def bsphkern(l=5):
        """
        creates a binary spherical kernel
        """
        gkern = ImageProc.gkern2d(l=l, sig=l)
        limit = gkern[l//2 if isinstance(l, int) else l[1]//2, -1]*0.995
        return np.array(gkern >= limit, dtype=np.uint8)

    @staticmethod
    def fuzzy_kernel(kernel, sig):
        w = int(sig//2)
        skernel = np.zeros(tuple(np.array(kernel.shape[:2])+int(sig)) + kernel.shape[2:3], dtype=kernel.dtype)
        skernel[w:w+kernel.shape[0], w:w+kernel.shape[1]] = kernel
        gkrn = ImageProc.gkern2d(sig, sig/2)
        skernel = cv2.filter2D(skernel, kernel.shape[2], gkrn)
        return skernel

    @staticmethod
    def add_jets(cam, img, mask, base_locs: tuple, trunc_lens=(None,), phase_angles=(None,), directions=(None,),
                intensities=(0.1,), angular_radii=(np.pi/16,), down_scaling=1):
        """
        - The jet generated is a truncated cone that has a density proportional to (truncation_distance/distance_from_untruncated_origin)**2
        - Truncation so that base width is 0.1 of mask diameter
        - base_loc gives the coordinates of the base in camera frame (opengl type, -z camera axis, +y is up)
        - 95% of luminosity lost when distance from base is `length`
        - angular_radius [rad] of the cone, 95% of luminosity lost if this much off axis, uses normal distribution
        - intensity of the cone at truncation point (>0)
        - if phase_angle < pi/2, cone not drawn on top of masked parts of image as it starts behind object
        - direction: 0 - to the right, pi/2 - up, pi - left, -pi/2 - down
        """
        from visnav.algo import tools
        assert down_scaling >= 1, 'only values of >=1 make sense for down_scaling'
        scaling = 1/down_scaling

        maxval = ImageProc._img_max_valid(img)
        base_locs = np.array(base_locs).copy()
        phase_angles = [np.random.uniform(0, np.pi) if pa is None else pa for pa in phase_angles]
        directions = [np.random.uniform(-np.pi, np.pi) if dir is None else dir for dir in directions]
        trunc_lens = [np.linalg.norm(base_locs[i])*0.01 if tl is None else tl for i, tl in enumerate(trunc_lens)]

        axes = []
        for i in range(len(phase_angles)):
            # q = np.quaternion(math.cos(-direction / 2), 0, 0, math.sin(-direction / 2)) \
            #     * np.quaternion(math.cos(phase_angle / 2), 0, math.sin(phase_angle / 2), 0)
            q1 = np.quaternion(math.cos(-phase_angles[i] / 2), 0, math.sin(-phase_angles[i] / 2), 0)
            q2 = np.quaternion(math.cos(directions[i] / 2), 0, 0, math.sin(directions[i] / 2))
            q = q2 * q1
            axis = tools.q_times_v(q, np.array([0, 0, -1]))
            axes.append(axis)
            base_locs[i] -= axis * trunc_lens[i]

        # density function of the jet
        def density(loc_arr, base_loc, axis, d0, angular_radius, intensity):
            loc_arr = loc_arr - base_loc
            r, d = tools.dist_across_and_along_vect(loc_arr, axis)
#            r, d = tools.point_vector_dist(loc_arr, axis, dist_along_v=True)

            # get distance along axis
            coef = np.zeros((len(loc_arr), 1))
            coef[d > d0] = (d0 / d[d > d0])**2

            # get radial distance from axis, use normal dist pdf but scaled so that max val is 1
            r_sd = d[coef > 0] * np.tan(angular_radius)
            coef[coef > 0] *= np.exp((-0.5/r_sd**2)*(r[coef > 0]**2))

            return coef * intensity

        # x(r), y(r), z(r), r>0 for a ray cast from each pixel
        # construct an array of unit rays, one ray per pixel
        iK = cam.inv_intrinsic_camera_mx()
        xx, yy = np.meshgrid(np.linspace(cam.width, 0, int(cam.width*scaling)),
                             np.linspace(0, cam.height, int(cam.height*scaling)), indexing='xy')
        img_coords = np.vstack((xx.flatten()+0.5, yy.flatten()+0.5, np.ones(xx.size)))
        ray_axes = iK.dot(img_coords).T * -1
        ray_axes /= np.linalg.norm(ray_axes, axis=1).reshape((-1, 1))

        def i_fun(r, arg_arr):
            result = None
            for args in arg_arr:
                res = density(ray_axes * r, *args)
                if result is None:
                    result = res
                else:
                    result += res
            return result

        bg_args_arr, bg_near, bg_far = [], np.inf, -np.inf
        fg_args_arr, fg_near, fg_far = [], np.inf, -np.inf
        for args in zip(base_locs, axes, trunc_lens, angular_radii, intensities):
            base_loc, axis, trunc_len, angular_radius, intensity = args
            dist = np.linalg.norm(base_loc)
            if axis[2] < 0:
                # z-component of axis is negative => jet goes away from cam => starts behind object
                bg_args_arr.append(args)
                bg_near = min(bg_near, dist - trunc_len)
                bg_far = max(bg_far, 2 * dist)    # crude heuristic, is it enough?
            else:
                # jet goes towards cam
                fg_args_arr.append(args)
                fg_near = min(fg_near, 0)
                fg_far = max(fg_far, dist + trunc_len)

        bg_res, fg_res = None, None
        if bg_args_arr:
            #  integrate density along the rays (quad_vec requires at least scipy 1.4.x)
            res = integrate.quad_vec(lambda r: i_fun(r, bg_args_arr), bg_near, bg_far, limit=25)
#            bg_sc = np.max(res[0])/maxval
            bg_res = cv2.resize(res[0].reshape(xx.shape).astype(np.float32), img.shape)
#            bg_res = cv2.resize((res[0]/bg_sc).reshape(xx.shape).astype(img.dtype), img.shape).astype(np.float32)*bg_sc
            bg_res[mask] = 0
        if fg_args_arr:
            #  integrate density along the rays (quad_vec requires at least scipy 1.4.x)
            res = integrate.quad_vec(lambda r: i_fun(r, fg_args_arr), fg_near, fg_far, limit=25)
#            fg_sc = np.max(res[0])/maxval
            fg_res = cv2.resize(res[0].reshape(xx.shape).astype(np.float32), img.shape)
#            fg_res = cv2.resize((res[0]/fg_sc).reshape(xx.shape).astype(img.dtype), img.shape).astype(np.float32)*fg_sc

        result = (0 if bg_res is None else bg_res) + (0 if fg_res is None else fg_res)
        max_r = np.max(result)
        if max_r > 0:
            result = result / max_r * maxval * np.max(intensities)

        dtype = img.dtype
        img = img.astype(np.float32)
        img = np.clip(img + result, 0, maxval).astype(dtype)
        return img

    @staticmethod
    def add_haze(img, mask=None, intensity=0.05):
        max = ImageProc._img_max_valid(img)
        dtype = img.dtype
        img = img.astype(np.float32)

        intensity = max * intensity * (1 if mask is None else 0.5)
        img += intensity

        if mask is not None:
            img[np.logical_not(mask)] += intensity

        return np.clip(img, 0, max).astype(dtype)

    @staticmethod
    def _img_max_valid(img):
        max = 1.0 if 'float' in str(img.dtype) else 255
        assert max != 255 or img.dtype == np.uint8, 'wrong datatype for image: %s' % img.dtype
        return max


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
        for i in range(1, len(images)):
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