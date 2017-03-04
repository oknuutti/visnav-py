
from datetime import datetime
import math

from scipy import optimize, stats
import numpy as np
import cv2

from settings import *
import tools

class PositioningException(Exception):
	pass

class ImageProc():
    latest_opt = None
    show_fit = None
    
    @staticmethod
    def process_target_image(image_src):
        hist = cv2.calcHist([image_src],[0],None,[256],[0,256])
        
        if not BATCH_MODE:
            threshold_value = ImageProc.optimal_threshold(hist)
        else:
            threshold_value = 15 # imgs created by VISIT dont have background
        
        th, im = cv2.threshold(image_src, threshold_value, 255, 0)
        
        # black => 0 alpha & white => .5 alpha
        alpha = np.zeros(im.shape, im.dtype)
        alpha[im == 255] = 128
        image_dst = cv2.merge((im, im, im, alpha))
        
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
            print('threshold_value: %s'%threshold_value)

        # plot figure with histogram and estimated distributions
        if False:
            from matplotlib import pyplot as plt
            plt.plot(x, fitfun1(out[0][:3], x))
            plt.plot(x, fitfun2(out[0][3:], x))
            plt.plot(x, fitfun(out[0], x))
            plt.plot(x, loghist)
            plt.show()
        
        return threshold_value
    

class CentroidAlgo():
    @staticmethod
    def update_sc_pos(system_model, img):
        elong, direc = system_model.solar_elongation()
        
        if abs(math.degrees(elong)) < 15:
            if not BATCH_MODE or DEBUG:
                print(("It's not recommended to stare into the sun: "
                      + "solar elongation: %.2f deg") % math.degrees(elong))
        
        cx, cy, size = CentroidAlgo.adjusted_centroid(
                                img, solar_elongation=elong, direction=direc)
        
        if size==0:
            raise PositioningException('No asteroid found')
        
        # for cross-section currently assumes spherical object with mean rad 2km
        # TODO: somehow improve on this, now seems very crude
        r = math.sqrt(size/math.pi)
        R = math.sqrt(system_model.asteroid.mean_cross_section/math.pi)
        angle = r/CAMERA_HEIGHT * math.radians(CAMERA_FIELD_OF_VIEW)
        z_off = R/1000 / math.tan(angle) # in km
        
        cx = (cx - CAMERA_WIDTH/2)
        angle = cx/CAMERA_HEIGHT * math.radians(CAMERA_FIELD_OF_VIEW)
        x_off = z_off * math.tan(angle)
        
        cy = (cy - CAMERA_HEIGHT/2)
        angle = cy/CAMERA_HEIGHT * math.radians(CAMERA_FIELD_OF_VIEW)
        y_off = z_off * math.tan(angle)
        
        sc_v = (x_off, y_off, -z_off)
        
        if not BATCH_MODE or DEBUG:
            print('\nreal pos:\n%s\nest pos:\n%s\n' % (
                system_model.real_sc_pos, sc_v))
        
        if any(map(math.isnan, sc_v)):
            raise PositioningException('Position resulted in a NAN: %s'(sc_v,))
        
        system_model.set_spacecraft_pos(sc_v)
        
    
    @staticmethod
    def adjusted_centroid(img, solar_elongation=math.pi, direction=0):
        iw, ih, cs = np.shape(img)
        m = cv2.moments(img[:,:,1], binaryImage=True)
        if m['m00'] < 10:
            return iw/2, ih/2, 0
        
        # pixel count
        pixels = m['m00']

        # centroid
        cx = m['m10']/m['m00']
        cy = ih - m['m01']/m['m00']

        # pixel spreads
        # used to model dimensions of asteroid parts visible in image
        hr_app = math.sqrt(m['mu20']/m['m00'])
        vr_app = math.sqrt(m['mu02']/m['m00'])

        # percentage of object expected to be lit
        lit_pct = (1 - math.cos(solar_elongation)) / 2
        
        # if lit_pct -> 0% then correction -> inf 
        #   => need to cap corr when only a few pixels found
        # seems centroid moving is more susceptible than pixel count
        #   => different capping mechanisms
        
        # for pixel count adjusting 
        soft_cap_p = 1/50 # a tuning param (1/50 => corr < 51)
        se_corr_p = (1+soft_cap_p) / (lit_pct+soft_cap_p)
        
        # for centroid moving
        cap_s = 2.5  # a tuning param
        se_corr_s = min(1/lit_pct, cap_s)
        
        # size adjustment
        roughness_coef = 1.8   # a tuning param
        size_adj = se_corr_p ** roughness_coef
        
        # how many pixel spreads to move centroid
        mv_coef = 1 - se_corr_s
        
        # centroid correction based on solar elongation
        dx = hr_app * mv_coef * math.cos(direction)
        dy = vr_app * mv_coef * math.sin(direction)
        
        if not BATCH_MODE or DEBUG:
            # width & height estimate,
            # could be used for size? stars == danger?
            w = 2 * hr_app * (1+abs(xc))
            h = 2 * vr_app * (1+abs(yc))
            print(('centroid: [%.0f, %.0f, %.0f], '
                  +'correction: [%.0f, %.0f, %.1f], '
                  +'w & h: %.0f x %.0f') % (
                cx, cy, pixels, dx, dy, size_adj, w, h))        
            
            if False:
                img[-round(cy+dy),round(cx+dx),:] = np.array([0,0,255,128])
                cv2.imshow('centroid', img)
                cv2.waitKey()
        
        return round(cx+dx), round(cy+dy), pixels*size_adj


class CoveredPixelsAlgo():
    def __init__(self, system_model, render_model_with_target):
        self.system_model = system_model
        self.render_model_with_target = render_model_with_target
        self.min_options = None
        self.errval0 = None
        self.errval1 = None
        self.errval  = None
        self.iter_count = 0
        self.start_time = None
    
    def optfun(self, *args):
        if any(map(math.isnan, args)):
            raise PositioningException('Position resulted in a NAN: %s'(sc_v,))
        
        for (n, p), v in zip(self.system_model.get_params(), args):
            p.nvalue = v
            
        self.iter_count += 1
        return self.errfun()

    def errfun(self, overlay=None):
        # value that bounds the input parameters, kind of weak priors
        # for a bayasian thing
        # TODO: actual priors based on centroid algo
        m = self.system_model
        self.errval0 = (0
#            + (m.x_off.nvalue)**2
#            + (m.y_off.nvalue)**2
#            + (m.z_off.nvalue)**2
            + ((m.x_off.nvalue - m.x_off.def_val)**2 if not m.x_off.valid else 0)
            + ((m.y_off.nvalue - m.y_off.def_val)**2 if not m.y_off.valid else 0)
            + ((m.z_off.nvalue - m.z_off.def_val)**2 if not m.z_off.valid else 0)
#            + 10 * (m.time.nvalue)**2
#            ((m.x_rot.nvalue)**2 if m.x_rot.valid else 0) +
#            ((m.y_rot.nvalue)**2 if m.y_rot.valid else 0) +
#            ((m.z_rot.nvalue)**2 if m.z_rot.valid else 0)
        )

        if overlay is None:
            overlay = self.render_model_with_target(self)
        
        # TODO: do this somehow to subpixel accuracy
        self.q = cv2.calcHist([overlay],[0],None,[256],[0,256])

        # was 0.02 => resulted in wrong minimum,
        # i.e. more missed pixels & less unnecessarily covered ones,
        # e.g. 181) extlit:23, intdark:7920, while target
        #  had 189) extlit:1,  intdark:9312
        nonlit_penalty = 0.005 # TODO: high penalty if high sol elong and v.v.
        
        # 128 external lit pixel (also 127?!)
        # 153 internal dark pixel
        # 204 internal lit pixel
        self.errval1 = self.q[127] + self.q[128] + nonlit_penalty*self.q[153]
        self.errval = self.errval0 + self.errval1

        if not BATCH_MODE or DEBUG:
            print((
                'E[%.0f] | xtlit:%d | indrk:%d | %.2f | %.2f | %.2f > '+
                    '%.2f | %.2f | %.2f | %.2f < %s')%(
                    self.errval1, self.q[128]+self.q[127], self.q[153],
                    m.x_rot.nvalue, m.y_rot.nvalue, m.z_rot.nvalue,
                    m.x_off.nvalue, m.y_off.nvalue, m.z_off.nvalue, m.time.nvalue,
                    '%s'%self.iter_count if self.iter_count>0 else '-'
                )
            )

        return self.errval
    
    
    def findstate(self, method='simplex', **kwargs):
        self.start_time = datetime.now()
        init_vals = np.array(list(p.nvalue for n, p in self.system_model.get_params()))
        
        if method=='simplex':
            options={'maxiter':100, 'xtol':2e-2, 'ftol':5e-2}
            options.update(kwargs.get('min_options', {}))
            res = optimize.minimize(lambda x: self.optfun(*x), init_vals,
                                    method='Nelder-Mead', options=options)
            x = res.x
        elif method=='powell':
            options = {'maxiter':100, 'xtol':2e-2, 'ftol':5e-2,
                       'direc':np.identity(len(init_vals))*0.01}
            options.update(kwargs.get('min_options', {}))
            res = optimize.minimize(lambda x: self.optfun(*x), init_vals,
                                    method='Powell', options=options)
            x = res.x
        elif method=='cobyla':
            options = {'maxiter':100, 'rhobeg': 0.1}
            options.update(kwargs.get('min_options', {}))
            res = optimize.minimize(lambda x: self.optfun(*x), init_vals,
                                    method='COBYLA', options=options)
            x = res.x
        elif method=='cg':
            options = {'maxiter':1000, 'eps':0.01, 'gtol':1e-4}
            options.update(kwargs.get('min_options', {}))
            res = optimize.minimize(lambda x: self.optfun(*x), init_vals,
                                    method='CG', options=options)
            x = res.x
        elif method=='bfgs':
            options = {'maxiter':1000, 'eps':0.01, 'gtol':1e-4}
            options.update(kwargs.get('min_options', {}))
            res = optimize.minimize(lambda x: self.optfun(*x), init_vals,
                                    method='BFGS', options=options)
            x = res.x
        elif method=='anneal':
            options = {
                'niter':350, 'T':1.25, 'stepsize':0.3,
                'minimizer_kwargs':{
#                    'method':'Nelder-Mead',
#                    'options':{'maxiter':25, 'xtol':2e-2, 'ftol':1e-1}
                    'method':'COBYLA',
                    'options':{'maxiter':25, 'rhobeg': 0.1},
            }}
            options.update(kwargs.get('min_options', {}))
            res = optimize.basinhopping(lambda x: self.optfun(*x),
                                        init_vals, **options)
            x = res.x
        elif method=='brute':
            max_iter = kwargs.get('min_options', {}).pop('max_iter', 12000)
            init = list((-0.5, 0.5) for n, p in self.system_model.get_params())
            options = {'Ns':math.floor(math.pow(max_iter, 1/len(init))),
                       'finish':None}
            options.update(kwargs.get('min_options', {}))
            x = res = optimize.brute(lambda x: self.optfun(*x), init, **options)
            
        if not BATCH_MODE or DEBUG:
            print('%s'%res)
            print('seconds: %s'%(datetime.now()-self.start_time))
        self.iter_count = -1;
        self.optfun(*x)    