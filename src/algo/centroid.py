import math

import numpy as np
import cv2

from settings import *
from algo import tools
from algo.tools import PositioningException

class CentroidAlgo():
    def __init__(self, system_model, glWidget, **kwargs):
        self.system_model = system_model
        self.glWidget = glWidget
        
        self.debug_filebase = None
        self.bg_threshold = kwargs.get('bg_threshold', None)
        
        self.MIN_PIXELS_FOR_DETECTION = 10  # fail if less pixels lit
        self.MAX_ITERATIONS = 10            # max number of iterations       
        self.ITERATION_TOL = 0.002          # min change % in position vector
    
    
    def adjust_iteratively(self, sce_img, **kwargs):
        sce_img = self.maybe_load_scene_image(sce_img)
        for i in range(self.MAX_ITERATIONS):
            ox, oy, oz = self.system_model.spacecraft_pos
            od = math.sqrt(ox**2 + oy**2 + oz**2)
            
            self.adjust(sce_img)
            
            nx, ny, nz = self.system_model.spacecraft_pos
            ch = math.sqrt((nx-ox)**2 + (ny-oy)**2 + (nz-oz)**2)
            print('i%d: d0=%.2f, ch=%.2f, rel_ch=%.2f%%'%(i, od, ch, ch/od*100))
            if ch/od < self.ITERATION_TOL:
                break
    
    
    def adjust(self, sce_img, ref_img=None, simple=False):
        sce_img = self.maybe_load_scene_image(sce_img)
        
        if not simple:
            if ref_img is None:
                ref_img = self.glWidget.render(center=False)
                th, ref_img = cv2.threshold(ref_img, self.bg_threshold, 255, cv2.THRESH_TOZERO)
                ref_img = np.atleast_3d(ref_img)
                
            sc_v = self.match_brightness_centroids(sce_img, ref_img)
        else:
            sc_v = self.simple_adjusted_centroid(sce_img)
        
        if not BATCH_MODE and DEBUG:
            print('\nreal pos:\n%s\nest pos:\n%s\n' % (
                self.system_model.real_sc_pos, sc_v))
        
        if any(map(math.isnan, sc_v)):
            raise PositioningException('Position resulted in a NAN: %s'(sc_v,))
        
        self.system_model.set_spacecraft_pos(sc_v)
        
        
    def match_brightness_centroids(self, sce_img, ref_img):
        
        # for asteroid cross section area visible, get lit pixels (so that no need to fine tune rendering)
        t1, t2, sce_px, t3, t4 = self.get_image_centroid(sce_img, binary=True)
        t1, t2, ref_px, t3, t4 = self.get_image_centroid(ref_img, binary=True)

        # for centroid, use centre of brightness
        sce_icx, sce_icy, t1, t2, t3 = self.get_image_centroid(sce_img, binary=False)
        ref_icx, ref_icy, t1, t2, t3 = self.get_image_centroid(ref_img, binary=False)
        
        old_x = self.system_model.x_off.value
        old_y = self.system_model.y_off.value
        old_z = self.system_model.z_off.value
        old_d = math.sqrt(old_x**2 + old_y**2 + old_z**2)
        
        # estimate distance first
        new_d = old_d * math.sqrt(ref_px/sce_px)
        
        # calculate brightness centroid coordinates
        ref_cx, ref_cy = tools.calc_xy(ref_icx, ref_icy, new_d)
        sce_cx, sce_cy = tools.calc_xy(sce_icx, sce_icy, new_d)

        # move x_off & y_off by centroid error
        new_x = old_x - (ref_cx - sce_cx)
        new_y = old_y - (ref_cy - sce_cy)
        
        # use new x & y for better estimate of z
        new_z = -math.sqrt(new_d**2 - new_x**2 - new_y**2)
        
        # return new location
        return (new_x, new_y, new_z)
        
    
    # not in use, for a reason...
    def simple_adjusted_centroid(self, img):
        img = self.maybe_load_scene_image(img)
        
        cx, cy, pixels, hr_app, vr_app = self.get_image_centroid(img, binary=True)
        
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
        angle = r/CAMERA_HEIGHT * math.radians(CAMERA_Y_FOV)
        dist = R/1000 / math.tan(angle) # in km
        
        x_off, y_off = tools.calc_xy(cx, cy, dist)
        
        return (x_off, y_off, -dist)
        

    def maybe_load_scene_image(self, sce_img):
        if isinstance(sce_img, str):
            self.glWidget.loadTargetImage(sce_img, remove_bg=True)
            sce_img = self.glWidget.full_image
            self.bg_threshold = self.glWidget.image_bg_threshold
            
        return np.atleast_3d(sce_img)

    def get_image_centroid(self, img, binary=False):
        ih, iw, cs = img.shape
        m = cv2.moments(img[:,:,0], binaryImage=binary)
        
        if m['m00'] < self.MIN_PIXELS_FOR_DETECTION:
            raise PositioningException('No asteroid found')
        
        # image centroid
        icx = (m['m10']/m['m00'] - iw/2)/iw*CAMERA_WIDTH
        icy = (ih/2 - m['m01']/m['m00'])/ih*CAMERA_HEIGHT
        brightness = m['m00']/iw/ih*CAMERA_WIDTH*CAMERA_HEIGHT
        
        # pixel spreads
        # used to model dimensions of asteroid parts visible in image
        hr = math.sqrt(m['mu20']/m['m00'])
        vr = math.sqrt(m['mu02']/m['m00'])
        
        return icx, icy, brightness, hr, vr