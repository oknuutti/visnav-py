import math

import numpy as np
import cv2

from settings import *
from algo import tools
from algo.tools import PositioningException

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
        angle = r/CAMERA_HEIGHT * math.radians(CAMERA_Y_FOV)
        z_off = R/1000 / math.tan(angle) # in km
        
        x_off, y_off = tools.calc_xy(cx-CAMERA_WIDTH/2, cy-CAMERA_HEIGHT/2, z_off)
        
        sc_v = (x_off, y_off, -z_off)
        
        if not BATCH_MODE or DEBUG:
            print('\nreal pos:\n%s\nest pos:\n%s\n' % (
                system_model.real_sc_pos, sc_v))
        
        if any(map(math.isnan, sc_v)):
            raise PositioningException('Position resulted in a NAN: %s'(sc_v,))
        
        system_model.set_spacecraft_pos(sc_v)
        
    
    @staticmethod
    def adjusted_centroid(img, solar_elongation=math.pi, direction=0):
        img = np.atleast_3d(img)
        ih, iw, cs = img.shape
        m = cv2.moments(img[:,:,0], binaryImage=True)
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
        
        return round(cx+dx), round(cy+dy), pixels*size_adj

