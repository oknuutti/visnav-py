from datetime import datetime
import math

from scipy import optimize
import numpy as np
import cv2

from PyQt5.QtCore import QCoreApplication

from algo.base import AlgorithmBase
from settings import *
from algo import tools
from algo.tools import PositioningException
from algo.centroid import CentroidAlgo


class PhaseCorrelationAlgo(AlgorithmBase):
    def __init__(self, system_model, render_engine, obj_idx):
        super(PhaseCorrelationAlgo, self).__init__(system_model, render_engine, obj_idx)

        self.min_options = None
        self.errval0 = None
        self.errval1 = None
        self.errval  = None
        self.iter_count = 0
        self.start_time = None
        self.extra_values = []

        self._image = None
        self._image_file = None
        self._target_image = None
        self._hannw = None
        self._render_result = None
        self._count = 0
    
    def optfun(self, *args):
        if any(map(math.isnan, args)):
            raise PositioningException('Position resulted in a NAN: %s'(sc_v,))
        
        # disable update events fired when parameters are changed
        #self.system_model.param_change_events(False)
        
        for (n, p), v in zip(self.system_model.get_params(), args):
            p.nvalue = v
            
        self.iter_count += 1
        err = self.errfun()

        # log result
        self.extra_values.append(err)
        
        # enable events again
        #self.system_model.param_change_events(True)
        #QCoreApplication.processEvents()
        return err

    def errfun(self, render_result=None):
        # value that bounds the input parameters, kind of weak priors
        # for a bayasian thing
        # TODO: actual priors based on centroid algo
        m = self.system_model
        self.errval0 = (0
#            + (m.x_off.nvalue)**2
#            + (m.y_off.nvalue)**2
#            + (m.z_off.nvalue)**2
#            + ((m.x_off.nvalue - m.x_off.def_val)**2 if not m.x_off.valid else 0)
#            + ((m.y_off.nvalue - m.y_off.def_val)**2 if not m.y_off.valid else 0)
#            + ((m.z_off.nvalue - m.z_off.def_val)**2 if not m.z_off.valid else 0)
#            + 10 * (m.time.nvalue)**2
#            ((m.x_rot.nvalue)**2 if m.x_rot.valid else 0) +
#            ((m.y_rot.nvalue)**2 if m.y_rot.valid else 0) +
#            ((m.z_rot.nvalue)**2 if m.z_rot.valid else 0)
        )
        
        if render_result is None:
            render_result = self.render()
            #imgfile = self.debug_filebase+('i%.0f'%(self.iter_count))+'.png'
            #cv2.imwrite(imgfile, render_result)
            self._count+=1
            if self._count%5 == 0:
                QCoreApplication.processEvents()

        self._render_result = render_result
#        cv2.imshow('render_result', render_result)
#        cv2.imshow('target_image', self._img)
#        cv2.waitKey()
        
        assert render_result.shape==self._target_image.shape, (
            'phase corr params: %s, %s\n'
            + 'im_xoff:%s, im_yoff:%s, '
            + 'im_width:%s, im_height:%s, im_scale:%.3f, '
            + 'vw_width:%s, vw_height:%s\n') % (
                render_result.shape, self._target_image.shape,
                self.im_xoff, self.im_yoff,
                self.im_width, self.im_height, self.im_scale,
                m.view_width, m.view_height)
        (tcx, tcy), pwr = cv2.phaseCorrelate(
                np.float32(render_result), self._target_image, self._hannw)
        
        # if crop params set, compensate cx, cy
        xi, yi = self._original_image_coords(
                tcx + m.view_width/2,
                tcy + m.view_height/2)
        
        scx, scy = self._cam.calc_xy(xi, yi, m.z_off.value)
        m.spacecraft_pos = (scx, scy, m.z_off.value)
        
        #print('z=%.3f: (%.3f, %.3f) => (%.3f, %.3f) => (%.3f, %.3f) => (%.3f, %.3f)'%(
        #    -m.z_off.value, tcx, tcy, cxc, cyc, cx, cy, m.x_off.value, m.y_off.value))
        
        self.errval1 = -pwr
        self.errval = self.errval0 + self.errval1

        if not BATCH_MODE or DEBUG:
            print((
                'E[%.3f] | (%.2f, %.2f) | %.2f | %.2f | %.2f > '+
                    '%.2f | %.2f | %.2f | %.2f < %s')%(
                    self.errval1, -xi, -yi,
                    m.x_rot.nvalue, m.y_rot.nvalue, m.z_rot.nvalue,
                    m.x_off.nvalue, m.y_off.nvalue, m.z_off.nvalue, m.time.nvalue,
                    '%s'%self.iter_count if self.iter_count>0 else '-'
                )
            )

        return self.errval
    
    def _get_bounds(self, img, margin, max_width):

        # get x, y, w, h from img
        xs, ys, ws, hs = cv2.boundingRect(img)

        cw, ch = self._cam.width, self._cam.height

        # translate to 1:1 scale, add margin
        tmpw = round(ws/self.im_scale) +2*margin
        tmph = round(hs/self.im_scale) +2*margin
        im_width = min(cw, max(tmpw, round(tmph * cw/ch)))
        im_height = min(ch, max(tmph, round(tmpw * ch/cw)))
        
        (x, y) = self._original_image_coords(xs, ys)
        im_xoff = round(max(0, min(x - margin, cw - im_width)))
        im_yoff = round(max(0, min(y - margin, ch - im_height)))
        
        if DEBUG:
            print('(%.0f, %.0f, %.0f, %.0f) => (%.1f, %.1f, %.1f, %.1f)'%(
                xs, ys, ws, hs, im_xoff, im_yoff, im_width, im_height
            ))
        
        # see if still need to down scale
        im_scale = min(1, max_width/im_width)
        
        # adjust bounds to cover whole view
        if im_scale == 1:
            view_w = self.system_model.view_width
            view_h = self.system_model.view_height
            im_xoff = max(0, min(im_xoff-round((view_w-im_width)/2), cw-view_w))
            im_yoff = max(0, min(im_yoff-round((view_h-im_height)/2), ch-view_h))
            im_width = view_w
            im_height = view_h
        
        return im_xoff, im_yoff, im_width, im_height, im_scale
    
    def _original_image_coords(self, cx, cy):
        return tuple(c/self.im_scale + o for c, o in
                    ((cx, self.im_xoff), (cy, self.im_yoff)))

    def findstate(self, imgfile, outfile, min_options={}, **kwargs):
        self.debug_filebase = outfile        
        min_options = dict(min_options)
        
        if self._image_file != imgfile or self._image is None:
            self._image = self.load_target_image(imgfile)
            self._image_file = imgfile

        self._target_image = np.float32(self._image)
        
        # consider if useful at all
        if kwargs.pop('centroid_init', False):
            try:
                CentroidAlgo.update_sc_pos(
                        self.system_model, self._image)
                ok = True
            except PositioningException as e:
                if not e.args or e.args[0] != 'No asteroid found':
                    print(str(e))
                print('|', end='', flush=True)
                ok = False
            if not ok:
                return False
        
        self.extra_values = []
        
        #hwsize = kwargs.get('hwin_size', 4)
        #tmp = cv2.createHanningWindow((hwsize, hwsize), cv2.CV_32F)
        #sd = int((self._cam.height - hwsize)/2)
        #self._hannw = cv2.copyMakeBorder(tmp,
        #        sd, sd, sd, sd, cv2.BORDER_CONSTANT, 0)
        
        self._hannw = None
        self.start_time = datetime.now()
        method = min_options.pop('method', False)
        init_vals = np.array(list(p.nvalue for n, p in
                                  self.system_model.get_params()))
        
        if method=='simplex':
            options={'maxiter':100, 'xtol':2e-2, 'ftol':5e-2}
            options.update(min_options)
            res = optimize.minimize(lambda x: self.optfun(*x), init_vals,
                                    method='Nelder-Mead', options=options)
            x = res.x
        elif method=='powell':
            options = {'maxiter':100, 'xtol':2e-2, 'ftol':5e-2,
                       'direc':np.identity(len(init_vals))*0.01}
            options.update(min_options)
            res = optimize.minimize(lambda x: self.optfun(*x), init_vals,
                                    method='Powell', options=options)
            x = res.x
        elif method=='cobyla':
            options = {'maxiter':100, 'rhobeg': 0.1}
            options.update(min_options)
            res = optimize.minimize(lambda x: self.optfun(*x), init_vals,
                                    method='COBYLA', options=options)
            x = res.x
        elif method=='cg':
            options = {'maxiter':1000, 'eps':0.01, 'gtol':1e-4}
            options.update(min_options)
            res = optimize.minimize(lambda x: self.optfun(*x), init_vals,
                                    method='CG', options=options)
            x = res.x
        elif method=='bfgs':
            options = {'maxiter':1000, 'eps':0.01, 'gtol':1e-4}
            options.update(min_options)
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
            options.update(min_options)
            res = optimize.basinhopping(lambda x: self.optfun(*x),
                                        init_vals, **options)
            x = res.x
        elif method=='brute':
            min_opts = dict(min_options)
            max_iter = min_opts.pop('max_iter', 1000)
            init = list((-0.5, 0.5) for n, p in self.system_model.get_params())
            options = {'Ns':math.floor(math.pow(max_iter, 1/len(init))),
                       'finish':None}
            options.update(min_opts)
            optfun = self.optfun if len(init)==1 else lambda x: self.optfun(*x)
            x = res = optimize.brute(optfun, init, **options)
            x = (x,) if len(init)==1 else x
                
        elif method=='two-step-brute':
            min_opts = dict(min_options)
            
            ## PHASE I
            ## >> 
            first_opts = dict(min_opts.pop('first'))

            im_scale = first_opts.pop('scale', self.im_def_scale)
            self.set_image_zoom_and_resolution(im_scale=im_scale)

            max_iter = first_opts.pop('max_iter', 50)
            init = list((-0.5, 0.5) for n, p in self.system_model.get_params())
            optfun = self.optfun if len(init)==1 else lambda x: self.optfun(*x)            
            options = {'Ns':math.floor(math.pow(max_iter, 1/len(init))),
                       'finish':None}
            options.update(first_opts)
            x = res = optimize.brute(optfun, init, **options)
            x = (x,) if len(init)==1 else x
            self.iter_count = -1
            self.optfun(*x) # set sc params to result values
            ## <<
            ## PHASE I
            
            QCoreApplication.processEvents()
            img = self.render()
            cv2.imwrite(self.debug_filebase+'.png', img)

            if DEBUG:
                print('Phase I: (%.3f, %.3f, %.3f)\n'%(
                    self.system_model.x_off.value,
                    self.system_model.y_off.value,
                    -self.system_model.z_off.value,
                ))
            
            ## PHASE II
            ## >>
            second_opts = dict(min_opts.pop('second'))
            margin = second_opts.pop('margin', 50)
            distance_margin = second_opts.pop('distance_margin', 0.2)
            max_width = second_opts.pop('max_width', -self.system_model.view_width)
            
            # calculate min_dist and max_dist
            min_dist = (-self.system_model.z_off.value) * (1-distance_margin)
            max_dist = (-self.system_model.z_off.value) * (1+distance_margin)

            # calculate im_xoff, im_xoff, im_width, im_height
            render_result = self.render()
            xo, yo, w, h, sc = self._get_bounds(render_result, margin, max_width)
            
            if DEBUG:
                print('min_d, max_d, xo, yo, w, h, sc: %.3f, %.3f, %.0f, %.0f, %.0f, %.0f, %.3f\n'%(
                    min_dist, max_dist, xo, yo, w, h, sc
                ))
            
            if True:
                self.debug_filebase = outfile+'r2'
                
                # limit distance range
                self.system_model.z_off.range = (-max_dist, -min_dist)
                
                # set cropping & zooming
                self.set_image_zoom_and_resolution(
                                            im_xoff=xo, im_yoff=yo,
                                            im_width=w, im_height=h, im_scale=sc)
                self._target_image = np.float32(self._image)
                img = self.render()
                cv2.imwrite(self.debug_filebase + '.png', img)

                # set optimization params
                max_iter = second_opts.pop('max_iter', 18)
                init = list((-0.5, 0.5) for n, p in self.system_model.get_params())
                optfun = self.optfun if len(init)==1 else lambda x: self.optfun(*x)
                options = {'Ns':math.floor(math.pow(max_iter, 1/len(init))),
                           'finish':None}
                options.update(second_opts)
            
                # optimize
                x = res = optimize.brute(optfun, init, **options)
                x = (x,) if len(init)==1 else x
            
        self.iter_count = -1;
        self.optfun(*x)
        outfile = imgfile[0:-4]+'r3'+'.png'
        img = self.render()
        cv2.imwrite(outfile, img)

        if method=='two-step-brute':
            self.system_model.z_off.range = (-self.system_model.max_distance, -self.system_model.min_med_distance)
            if DEBUG:
                print('Phase II: (%s, %s, %s)\n'%(
                    self.system_model.x_off.value,
                    self.system_model.y_off.value,
                    -self.system_model.z_off.value,
                ))
        
        if not BATCH_MODE or DEBUG:
            print('%s'%res)
            print('seconds: %s'%(datetime.now()-self.start_time))
            
        if self.im_def_scale != self.im_scale:
            self.set_image_zoom_and_resolution(im_scale=self.im_def_scale)
        
        return True
