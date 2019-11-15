import math

import numpy as np
import cv2

from algo.base import AlgorithmBase
from algo.model import SystemModel
from iotools import lblloader
from missions.rosetta import RosettaSystemModel
from render.render import RenderEngine
from settings import *
from algo import tools
from algo.image import ImageProc
from algo.tools import PositioningException


class CentroidAlgo(AlgorithmBase):
    def __init__(self, system_model, render_engine, obj_idx, **kwargs):
        super(CentroidAlgo, self).__init__(system_model, render_engine, obj_idx)

        self._bg_threshold = 20                 # default thresholding value
        self._ref_img = None
        
        self.DEBUG_IMG_POSTFIX = 'c'            # fi batch mode, save result image in a file ending like this

        self.RENDER_SHADOWS = True
        self.MIN_PIXELS_FOR_DETECTION = 30      # fail if less pixels lit
        self.ASTEROID_MIN_BORDER_MARGIN = 0.04  # if less than margin at both extrames, astroid too close
        self.ASTEROID_MAX_SPAN = 0.85           # if asteroid spans more than this, it's too close
        self.MAX_ITERATIONS = 30                # max number of iterations
        self.ITERATION_TOL = 0.002              # min change % in position vector
        self.CHECK_RESULT_VALIDITY = False
        self.MIN_RESULT_XCORR = 0.3             # if result xcorr with scene is less than this, fail
    
    
    def adjust_iteratively(self, sce_img, outfile=None, **kwargs):
        self.debug_filebase = outfile
        self._bg_threshold = kwargs.get('bg_threshold', self._bg_threshold)
        sce_img = self.maybe_load_scene_image(sce_img, preproc='bg_threshold' not in kwargs)

        if DEBUG:
            sc = self.system_model.view_width/sce_img.shape[1]
            cv2.imshow('target img', cv2.resize(sce_img, None, fx=sc, fy=sc))

        self.system_model.spacecraft_pos = (0, 0, -self.system_model.min_med_distance)
        for i in range(self.MAX_ITERATIONS):
            ox, oy, oz = self.system_model.spacecraft_pos
            od = math.sqrt(ox**2 + oy**2 + oz**2)
            
            if not DEBUG:
                self.adjust(sce_img, preproc=False)
            else:
                try:
                    self.adjust(sce_img, preproc=False)
                except PositioningException as e:
                    print(str(e))
                    break
                finally:
                    cv2.imshow('rendered img', self._ref_img)
                    cv2.waitKey()
            
            nx, ny, nz = self.system_model.spacecraft_pos
            ch = math.sqrt((nx-ox)**2 + (ny-oy)**2 + (nz-oz)**2)
            if DEBUG:
                print('i%d: d0=%.2f, ch=%.2f, rel_ch=%.2f%%'%(i, od, ch, ch/od*100))
            if ch/od < self.ITERATION_TOL:
                break
        
        if self.CHECK_RESULT_VALIDITY:
            result_quality = ImageProc.norm_xcorr(sce_img, self._ref_img)
            if result_quality < self.MIN_RESULT_XCORR:
                raise PositioningException('Result failed quality test with score: %.3f'%(result_quality,))
        
        if BATCH_MODE and self.debug_filebase:
            img = self.render(shadows=self.RENDER_SHADOWS)
            cv2.imwrite(self.debug_filebase+'r.png', img)

        if DEBUG:
            cv2.waitKey()
            cv2.destroyAllWindows()
    
    
    def adjust(self, sce_img, ref_img=None, simple=False, preproc=True):
        sce_img = self.maybe_load_scene_image(sce_img, preproc=preproc)
        
        if not simple:
            if ref_img is None:
                ref_img = self.render(shadows=self.RENDER_SHADOWS)
                th, ref_img = cv2.threshold(ref_img, self._bg_threshold, 255, cv2.THRESH_TOZERO)
                ref_img = np.atleast_3d(ref_img)
                self._ref_img = ref_img
                
            sc_v = self.match_brightness_centroids(sce_img, ref_img)
        else:
            sc_v = self.simple_adjusted_centroid(sce_img)
        
        if not BATCH_MODE and DEBUG:
            print('\nreal pos:\n%s\nest pos:\n%s\n' % (
                self.system_model.real_spacecraft_pos, sc_v))
        
        if any(map(math.isnan, sc_v)):
            raise PositioningException('Position resulted in a NAN: %s'(sc_v,))
        
        self.system_model.spacecraft_pos = sc_v
        
        
    def match_brightness_centroids(self, sce_img, ref_img):
        
        # for asteroid cross section area visible, get lit pixels (so that no need to fine tune rendering)
        t1, t2, sce_px, t3, t4 = self.get_image_centroid(sce_img, is_scene=True, binary=True)
        t1, t2, ref_px, t3, t4 = self.get_image_centroid(ref_img, binary=True)

        # for centroid, use centre of brightness
        sce_icx, sce_icy, t1, t2, t3 = self.get_image_centroid(sce_img, is_scene=True, binary=False)
        ref_icx, ref_icy, t1, t2, t3 = self.get_image_centroid(ref_img, binary=False)
        
        old_x = self.system_model.x_off.value
        old_y = self.system_model.y_off.value
        old_z = self.system_model.z_off.value

        # when adjusting for distance, remember to adjust all coordinates, try to first adjust laterally
        new_z = old_z * (math.sqrt(ref_px/sce_px) ** .5)
        err_d_x = old_x * (1 - new_z/old_z)
        err_d_y = old_y * (1 - new_z/old_z)

        # calculate brightness centroid coordinates
        ref_cx, ref_cy = self._cam.calc_xy(ref_icx, ref_icy, max(old_z, new_z))  # use max so that wont move out of view
        sce_cx, sce_cy = self._cam.calc_xy(sce_icx, sce_icy, max(old_z, new_z))  # use max so that wont move out of view

        # lateral errors
        err_x = ref_cx - sce_cx
        err_y = ref_cy - sce_cy

        # try to first adjust laterally
        if abs(err_x) + abs(err_y) > .2:
            new_z = old_z
            err_d_x = 0
            err_d_y = 0

        # move x_off & y_off by centroid error
        new_x = old_x - err_x - err_d_x
        new_y = old_y - err_y - err_d_y
        
        # return new location
        return (new_x, new_y, new_z)
     
     
    def detect_asteroid(self, sce_img):
        ih, iw = sce_img.shape[0:2]
        
        # Threshold it so it becomes binary
        ret, bin_img = cv2.threshold(sce_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # connect close pixels by dilating and eroding
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
        
        # get connected regions and their stats
        n, regs, stats, c = cv2.connectedComponentsWithStats(bin_img, 8, cv2.CV_32S)
        
        # detect asteroid
        parts = list(s[cv2.CC_STAT_AREA] for i,s in enumerate(stats) if i>0)
        if not parts or stats[np.argmax(parts)+1][cv2.CC_STAT_AREA] < self.MIN_PIXELS_FOR_DETECTION:
            raise PositioningException('No asteroid found')
        
        # asteroid parts (i=0 is background)
        ast_parts = (
                i for i,s in enumerate(stats)
                if i>0 and s[cv2.CC_STAT_AREA]>=self.MIN_PIXELS_FOR_DETECTION
        )
        
        # check asteroid extent
        ast_parts = [
            (
                i, s[cv2.CC_STAT_AREA],
                s[cv2.CC_STAT_LEFT]/iw, (iw - (s[cv2.CC_STAT_LEFT]+s[cv2.CC_STAT_WIDTH]))/iw,
                s[cv2.CC_STAT_TOP]/ih, (ih - (s[cv2.CC_STAT_TOP]+s[cv2.CC_STAT_HEIGHT]))/ih
            )
            for i,s in enumerate(stats) if i>0 and s[cv2.CC_STAT_AREA]>=self.MIN_PIXELS_FOR_DETECTION]

        tot_area = sum((p[1] for p in ast_parts))/iw/ih
        t1, t2, lm, rm, tm, bm = np.min(ast_parts, axis=0)
        
        if DEBUG:
            print('Asteroid l,r,t,b margins: %.2f, %.2f, %.2f, %.2f'%(lm, rm, tm, bm), flush=True)
        
        if(False):
            lim = self.ASTEROID_MIN_BORDER_MARGIN
            if (lm < lim and rm < lim or tm < lim and bm < lim):
                raise PositioningException('Asteroid too close: margins (l,r,t,b): %.2f, %.2f, %.2f, %.2f'%(lm, rm, tm, bm))
        else:
            lim = self.ASTEROID_MAX_SPAN
            if (1-lm-rm > lim or 1-tm-bm > lim):
                raise PositioningException('Asteroid too close: span (w,h): %.2f, %.2f'%(1-lm-rm, 1-tm-bm))
        
        # set too small regions to zero in scene image (stars)
        for i in range(n):
            if i>0 and i not in [p[0] for p in ast_parts]:
                sce_img[regs==i] = 0
        
        return tot_area
    
    
    # not in use, for a reason...
    def simple_adjusted_centroid(self, img):
        img = self.maybe_load_scene_image(img)
        
        cx, cy, pixels, hr_app, vr_app = self.get_image_centroid(img, is_scene=True, binary=True)
        
        solar_elongation, direction = self.system_model.solar_elongation()

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
            w = 2 * hr_app * (1+abs(mv_coef*math.cos(direction)))
            h = 2 * vr_app * (1+abs(mv_coef*math.sin(direction)))
            print(('centroid: [%.0f, %.0f, %.0f], '
                  +'correction: [%.0f, %.0f, %.1f], '
                  +'w & h: %.0f x %.0f') % (
                cx, cy, pixels, dx, dy, size_adj, w, h))        
            
            if False:
                img[-round(cy+dy),round(cx+dx),:] = np.array([0,0,255,128])
                cv2.imshow('centroid', img)
                cv2.waitKey()
        
        cx, cy, cs = round(cx+dx), round(cy+dy), pixels*size_adj

        # for cross-section currently assumes spherical object with mean rad 2km
        # TODO: somehow improve on this, now seems very crude
        r = math.sqrt(cs/math.pi)
        R = math.sqrt(system_model.asteroid.mean_cross_section/math.pi)
        angle = r/self._cam.height * math.radians(self._cam.y_fov)
        dist = R/1000 / math.tan(angle) # in km
        
        x_off, y_off = self._cam.calc_xy(cx, cy, -dist)
        
        return (x_off, y_off, -dist)
        

    def maybe_load_scene_image(self, sce_img, detect_asteroid=False, preproc=False):
        if isinstance(sce_img, str):
            sce_img = self.load_target_image(sce_img)

        if preproc:
            sce_img, self._bg_threshold = self.remove_background(sce_img)
            if detect_asteroid:
                self.detect_asteroid(sce_img)
            if DEBUG:
                sc = self.system_model.view_width / sce_img.shape[1]
                cv2.imshow('scene th=%d'%self._bg_threshold, cv2.resize(sce_img, None, fx=sc, fy=sc))
            
        return np.atleast_3d(sce_img)

    def get_image_centroid(self, img, is_scene=False, binary=False):
        ih, iw, cs = img.shape
        m = cv2.moments(img[:,:,0], binaryImage=binary)
        
        if np.isclose(m['m00'],0):
            if is_scene:
                raise PositioningException('No asteroid found')
            else:
                raise PositioningException('Algorithm failure: model moved out of view')
        
        # image centroid
        icx = m['m10']/m['m00']/iw*self._cam.width
        icy = m['m01']/m['m00']/ih*self._cam.height
        brightness = m['m00']/iw/ih*self._cam.width*self._cam.height
        
        # pixel spreads
        # used to model dimensions of asteroid parts visible in image
        hr = math.sqrt(m['mu20']/m['m00']) if m['mu20']>0 else 1
        vr = math.sqrt(m['mu02']/m['m00']) if m['mu02']>0 else 1
        
        return icx, icy, brightness, hr, vr


if __name__ == '__main__':
    sm = RosettaSystemModel()
    lblloader.load_image_meta(sm.asteroid.sample_image_meta_file, sm)
    re = RenderEngine(sm.view_width, sm.view_height)
    obj_idx = re.load_object(sm.asteroid.real_shape_model)

    DEBUG = True
    algo = CentroidAlgo(sm, re, obj_idx)
    algo.adjust_iteratively(sm.asteroid.sample_image_file, None)