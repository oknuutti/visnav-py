from settings import *

import math
from math import degrees as deg
from math import radians as rad
import os
import shutil
import pickle
import threading
from datetime import datetime as dt

import numpy as np
import quaternion
from astropy.coordinates import spherical_to_cartesian

from iotools.visitclient import VisitClient
from iotools import objloader

from algo.model import Asteroid
import algo.tools as tools
from algo.tools import (ypr_to_q, q_to_ypr, q_times_v, q_to_unitbase, normalize_v,
                   wrap_rads, solar_elongation, angle_between_ypr)
from algo.tools import PositioningException

#from memory_profiler import profile
#import tracemalloc
#import gc
# TODO: fix suspected memory leaks at
#   - quaternion.as_float_array (?)
#   - cv2.solvePnPRansac (ref_kp_3d?)
#   - astropy, many places



class TestLoop():
    FILE_PREFIX = 'iteration'
    
    def __init__(self, window):
        self.window = window
        self.exit = False
        self._visit_client = VisitClient()
        self._algorithm_finished = None
        
        # gaussian sd in seconds
        self._noise_time = 30/2     # 95% within +-30s
        
        # uniform, max dev in deg
        self._noise_ast_rot_axis = 10
        self._noise_ast_phase_shift = 10/2  # 95% within 10 deg
        
        # s/c orientation noise, gaussian sd in deg
        self._noise_sco_lat = 2/2   # 95% within 2 deg
        self._noise_sco_lon = 2/2   # 95% within 2 deg
        self._noise_sco_rot = 2/2   # 95% within 2 deg
        
        # minimum allowed elongation in deg
        self._min_elong = 45
        
        # transients
        self._smn_cache_id = ''
        self._iter_dir = None
        self._logfile = None
        self._fval_logfile = None
        self._run_times = []
        self._laterrs = []
        self._disterrs = []
        self._roterrs = []
        self._shifterrs = []
        self._fails = 0        
        self._timer = None
        
        def handle_close():
            self.exit = True
            if self._algorithm_finished:
                self._algorithm_finished.set()
            
        self.window.closing.append(handle_close)
        
        
    # main method
    def run(self, times, log_prefix='test-', cleanup=True, **kwargs):
        self._smn_cache_id = kwargs.pop('smn_type', '')

        # write logfile header
        self._init_log(log_prefix)
        
        li = 0
        sm = self.window.systemModel
        
        for i in range(times):
            # maybe generate new noise for shape model
            sm_noise = 0
            if ADD_SHAPE_MODEL_NOISE:
                sm_noise = self.load_noisy_shape_model(sm, i)
                if sm_noise is None:
                    if DEBUG:
                        print('generating new noisy shape model')
                    sm_noise = self.generate_noisy_shape_model(sm, i)
                    self._maybe_exit()

            # try to load system state
            initial = self.load_state(sm, i)
            if initial is None:
                if DEBUG:
                    print('generating new state')
                
                # generate system state
                self.generate_system_state(sm)

                # add noise to current state, wipe sc pos
                initial = self.add_noise(sm)

                # save state to lbl file
                sm.save_state(self._cache_file(i))
                imgfile = None
            else:
                # successfully loaded system state,
                # try to load related navcam image
                imgfile = self.load_navcam_image(i)
            
            # maybe new system state or no previous image, if so, render
            if imgfile is None:
                if DEBUG:
                    print('generating new navcam image')
                imgfile = self.render_navcam_image(sm, i)
                self._maybe_exit()
            
            # run algorithm
            ok, rtime = self._run_algo(imgfile, self._iter_file(i), **kwargs)
            
            # calculate results
            results = self.calculate_result(sm, i, ok, initial, **kwargs)
            
            # write log entry
            self._write_log_entry(i, rtime, sm_noise, *results)
            self._maybe_exit()

            # print out progress
            if math.floor(100*i/times) > li:
                print('.', end='', flush=True)
                li += 1

        self._close_log(times)
        
        if cleanup:
            self._cleanup()
    

    def generate_system_state(self, sm):
        # reset asteroid axis to true values
        sm.asteroid = Asteroid()
        sm.asteroid_rotation_from_model()
        
        for i in range(100):
            ## sample params from suitable distributions
            ##
            # datetime dist: uniform, based on rotation period
            time = np.random.uniform(*sm.time.range)

            # spacecraft position relative to asteroid in ecliptic coords:
            sc_lat = np.random.uniform(-math.pi/2, math.pi/2)
            sc_lon = np.random.uniform(-math.pi, math.pi)

            # s/c distance as inverse uniform distribution
            #max_r, min_r = MAX_DISTANCE, MIN_DISTANCE
            max_r, min_r = MAX_MED_DISTANCE, MIN_MED_DISTANCE
            sc_r = 1/np.random.uniform(1/max_r, 1/min_r)

            # same in cartesian coord
            sc_ex_u, sc_ey_u, sc_ez_u = spherical_to_cartesian(sc_r, sc_lat, sc_lon)
            sc_ex, sc_ey, sc_ez = sc_ex_u.value, sc_ey_u.value, sc_ez_u.value

            # s/c to asteroid vector
            sc_ast_v = -np.array([sc_ex, sc_ey, sc_ez])

            # sc orientation: uniform, center of asteroid at edge of screen - some margin
            da = np.random.uniform(0, rad(CAMERA_Y_FOV/2))
            dd = np.random.uniform(0, 2*math.pi)
            sco_lat = wrap_rads(-sc_lat + da*math.sin(dd))
            sco_lon = wrap_rads(math.pi + sc_lon + da*math.cos(dd))
            sco_rot = np.random.uniform(-math.pi, math.pi) # rotation around camera axis
            sco_q = ypr_to_q(sco_lat, sco_lon, sco_rot)
            
            # sc_ast_p ecliptic => sc_ast_p open gl -z aligned view
            sc_pos = q_times_v((sco_q * sm.sc2gl_q).conj(), sc_ast_v)
            
            # get asteroid position so that know where sun is
            # *actually barycenter, not sun
            as_v = sm.asteroid.position(time)
            elong, direc = solar_elongation(as_v, sco_q)

            # limit elongation to always be more than set elong
            if elong > rad(self._min_elong):
                break
        
        if elong <= rad(self._min_elong):
            assert False, 'probable infinite loop'
        
        # put real values to model
        sm.time.value = time
        sm.spacecraft_pos = sc_pos
        sm.spacecraft_rot = (deg(sco_lat), deg(sco_lon), deg(sco_rot))

        # save real values so that can compare later
        sm.time.real_value = sm.time.value
        sm.real_spacecraft_pos = sm.spacecraft_pos
        sm.real_spacecraft_rot = sm.spacecraft_rot
        sm.real_asteroid_axis = sm.asteroid_axis

        # get real relative position of asteroid model vertices
        sm.real_sc_ast_vertices = sm.sc_asteroid_vertices()
        
        
    def add_noise(self, sm):
        
        ## add measurement noise to
        # - datetime (seconds)
        meas_time = sm.time.real_value + np.random.normal(0, self._noise_time)
        sm.time.value = meas_time;
        assert np.isclose(sm.time.value, meas_time), 'Failed to set time value'

        # - asteroid state estimate
        ax_lat, ax_lon, ax_phs = map(rad, sm.real_asteroid_axis)
        noise_da = np.random.uniform(0, rad(self._noise_ast_rot_axis))
        noise_dd = np.random.uniform(0, 2*math.pi)
        meas_ax_lat = ax_lat + noise_da*math.sin(noise_dd)
        meas_ax_lon = ax_lon + noise_da*math.cos(noise_dd)
        meas_ax_phs = ax_phs + np.random.normal(0, rad(self._noise_ast_phase_shift))
        sm.asteroid_axis = map(deg, (meas_ax_lat, meas_ax_lon, meas_ax_phs))

        # - spacecraft orientation measure
        sc_lat, sc_lon, sc_rot = map(rad, sm.real_spacecraft_rot)
        meas_sc_lat = max(-math.pi/2, min(math.pi/2, sc_lat
                + np.random.normal(0, rad(self._noise_sco_lat))))
        meas_sc_lon = wrap_rads(sc_lon 
                + np.random.normal(0, rad(self._noise_sco_lon)))
        meas_sc_rot = wrap_rads(sc_rot
                + np.random.normal(0, rad(self._noise_sco_rot)))
        sm.spacecraft_rot = map(deg, (meas_sc_lat, meas_sc_lon, meas_sc_rot))
        
        # wipe spacecraft position clean
        sm.spacecraft_pos = (0, 0, -MIN_MED_DISTANCE)

        # return this initial state
        return self._initial_state(sm)
        
    def _initial_state(self, sm):
        return {
            'time': sm.time.value,
            'ast_axis': sm.asteroid_axis,
            'sc_rot': sm.spacecraft_rot,
        }
    
    def load_state(self, sm, i):
        try:
            sm.load_state(self._cache_file(i)+'.lbl', sc_ast_vertices=True)
            initial = self._initial_state(sm)
        except FileNotFoundError:
            initial = None
        return initial
        
        
    def generate_noisy_shape_model(self, sm, i):
        sup = objloader.OBJ(SHAPE_MODEL_NOISE_SUPPORT)
        noisy_vertices, sm_noise = tools.apply_noise(
                np.array(sm.real_shape_model.vertices),
                support=np.array(sup.vertices),
                len_sc=SHAPE_MODEL_NOISE_LEN_SC,
                noise_lv=SHAPE_MODEL_NOISE_LV)
        
        prefix = 'shapemodel_'+self._smn_cache_id
        fname = self._cache_file(i, prefix=prefix)+'.nsm'
        with open(fname, 'wb') as fh:
            pickle.dump((noisy_vertices, sm_noise), fh, -1)
        
        self._widget_load_obj(noisy_vertices)
        return sm_noise
    
    def load_noisy_shape_model(self, sm, i):
        try:
            prefix = 'shapemodel_'+self._smn_cache_id
            fname = self._cache_file(i, prefix=prefix)+'.nsm'
            with open(fname, 'rb') as fh:
                noisy_vertices, sm_noise = pickle.load(fh)
            self._widget_load_obj(noisy_vertices)
        except FileNotFoundError:
            sm_noise = None
        return sm_noise
        
    def _widget_load_obj(self, noisy_vertices):
        self._run_on_qt(
                lambda x, y: x.loadObject(noisy_vertices=y),
                self.window.glWidget, noisy_vertices)
    
    
    def render_navcam_image(self, sm, i):
        """ 
        based on real system state, call VISIT to make a target image
        """
        
        ast_q = sm.real_asteroid_q()
        as_pos = sm.asteroid.position(sm.time.real_value)
        
        sc_q = sm.real_spacecraft_q()
        ast_sc_v = -q_times_v((sc_q * sm.sc2gl_q), np.array(sm.real_spacecraft_pos))
        
        light = normalize_v(q_times_v(ast_q.conj(), np.array(as_pos)))
        focus = q_times_v(ast_q.conj(), ast_sc_v)
        view_x, ty, view_z = q_to_unitbase(ast_q.conj() * sc_q)
        
        # in VISIT focus & view_normal are vectors pointing out from the object,
        # light however points into the object
        view_x = -view_x
        focus += -23*view_x     # 23km bias in view_normal direction?!
        
        # all params VISIT needs
        visit_params = {
            'out_file':         'visitout',
            'out_dir':          self._iter_dir.replace('\\', '\\\\'),
            'view_angle':       CAMERA_Y_FOV,
            'out_width':        min(MAX_TEST_X_RES, CAMERA_WIDTH),
            'out_height':       min(MAX_TEST_Y_RES, CAMERA_HEIGHT),
            'max_distance':     -sm.z_off.range[0]+10,
            'light_direction':  tuple(light),    # vector into the object
            'focus':            tuple(focus),    # vector from object to camera
            'view_normal':      tuple(view_x),   # reverse camera borehole direction
            'up_vector':        tuple(view_z),   # camera up direction
        }

        # call VISIT
        imgfile = self._visit_client.render(visit_params)
        cache_file = self._cache_file(i)+'.png'
        shutil.move(imgfile, cache_file)
        return cache_file

    def load_navcam_image(self, i):
        fname = self._cache_file(i)+'.png'
        return fname if os.path.isfile(fname) else None


    def calculate_result(self, sm, i, ok, initial, **kwargs):
        # save function values from optimization
        fvals = self.window.phasecorr.optfun_values \
                if ok and kwargs.get('method', False)=='phasecorr' \
                else None
        final_fval = fvals[-1] if fvals else None

        real_rel_rot = q_to_ypr(sm.real_sc_asteroid_rel_q())
        elong, direc = sm.solar_elongation(real=True)
        r_ast_axis = sm.real_asteroid_axis
        
        # real system state
        params = (sm.time.real_value, *r_ast_axis,
                *sm.real_spacecraft_rot, deg(elong), deg(direc),
                *sm.real_spacecraft_pos, *map(deg, real_rel_rot),
                self._iter_file(i), final_fval)
        
        # calculate added noise
        #
        time_noise = initial['time'] - sm.time.real_value
        ast_rot_noise = (
            initial['ast_axis'][0]-r_ast_axis[0],
            initial['ast_axis'][1]-r_ast_axis[1],
            360*time_noise/sm.asteroid.rotation_period
                + (initial['ast_axis'][2]-r_ast_axis[2])
        )
        sc_rot_noise = tuple(np.subtract(initial['sc_rot'], sm.real_spacecraft_rot))
        
        dev_angle = deg(angle_between_ypr(map(rad, ast_rot_noise),
                                          map(rad, sc_rot_noise)))
        
        noise = (time_noise,) + ast_rot_noise + sc_rot_noise + (dev_angle,)
        
        if not ok:
            pos = float('nan')*np.ones(3)
            rel_rot = float('nan')*np.ones(3)
            err = float('nan')*np.ones(4)
            
        else:
            pos = sm.spacecraft_pos
            rel_rot = q_to_ypr(sm.sc_asteroid_rel_q())
            est_vertices = sm.sc_asteroid_vertices()
            max_shift = tools.sc_asteroid_max_shift_error(
                    sm.real_sc_ast_vertices, est_vertices)
            
            err = (
                *np.subtract(pos, sm.real_spacecraft_pos),
                deg(angle_between_ypr(rel_rot, real_rel_rot)),
                max_shift,
            )
        
        return params, noise, pos, map(deg, rel_rot), fvals, err
    
    
    def _init_log(self, log_prefix):
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        logbody = log_prefix + dt.now().strftime('%Y%m%d-%H%M%S')
        self._iter_dir = os.path.join(LOG_DIR, logbody)
        os.mkdir(self._iter_dir)
        
        self._fval_logfile = LOG_DIR + logbody + '-fvals.log'
        self._logfile = LOG_DIR + logbody + '.log'
        with open(self._logfile, 'w') as file:
            file.write('\t'.join((
                'iter', 'date', 'execution time',
                'time', 'ast lat', 'ast lon', 'ast rot',
                'sc lat', 'sc lon', 'sc rot', 
                'sol elong', 'light dir', 'x sc pos', 'y sc pos', 'z sc pos',
                'rel yaw', 'rel pitch', 'rel roll', 
                'imgfile', 'optfun val', 'shape model noise',
                'time dev', 'ast lat dev', 'ast lon dev', 'ast rot dev',
                'sc lat dev', 'sc lon dev', 'sc rot dev', 'total dev angle',
                'x est sc pos', 'y est sc pos', 'z est sc pos',
                'yaw rel est', 'pitch rel est', 'roll rel est',
                'x err sc pos', 'y err sc pos', 'z err sc pos', 'rot error',
                'shift error km', 'lat error', 'dist error', 'rel shift error',
            ))+'\n')
            
        self._run_times = []
        self._laterrs = []
        self._disterrs = []
        self._roterrs = []
        self._shifterrs = []
        self._fails = 0
        self._timer = tools.Stopwatch()
        self._timer.start()
        
        
    def _write_log_entry(self, i, rtime, sm_noise, params, noise, pos, rel_rot, fvals, err):
            
            # save execution time
            self._run_times.append(rtime)

            # calculate errors
            dist = abs(params[-6])
            if not math.isnan(err[0]):
                lerr = math.sqrt(err[0]**2 + err[1]**2) / dist
                derr = abs(err[2]) / dist
                rerr = abs(err[3])
                serr = err[4] / dist
                self._laterrs.append(lerr)
                self._disterrs.append(derr)
                self._roterrs.append(rerr)
                self._shifterrs.append(serr)
            else:
                lerr = derr = rerr = serr = float('nan')
                self._fails += 1

            # log all parameter values, timing & errors into a file
            with open(self._logfile, 'a') as file:
                file.write('\t'.join(map(str, (
                    i, dt.now().strftime("%Y-%m-%d %H:%M:%S"), rtime, *params,
                    sm_noise, *noise, *pos, *rel_rot, *err, lerr, derr, serr
                )))+'\n')
                
            # log opt fun values in other file
            with open(self._fval_logfile, 'a') as file:
                file.write('\t'.join(map(str, fvals or []))+'\n')
        
        
    def _close_log(self, times):
        prctls = (50, 68, 95) + ((99.7,) if times>=2000 else tuple())
        calc_prctls = lambda errs: \
                ', '.join('%.2f' % p
                for p in 100*np.nanpercentile(errs, prctls))
        try:
            laterr_pctls = calc_prctls(self._laterrs)
            disterr_pctls = calc_prctls(self._disterrs)
            shifterr_pctls = calc_prctls(self._shifterrs)
            roterr_pctls = ', '.join(
                    ['%.2f'%p for p in np.nanpercentile(self._roterrs, prctls)])
        except Exception as e:
            print('Error calculating quantiles: %s'%e)
            laterr_pctls = 'error'
            disterr_pctls = 'error'
            shifterr_pctls = 'error'
            roterr_pctls = 'error'
        
        self._timer.stop()
        
        # a summary line
        summary = (
            '%s - t: %.1fmin (%.1fms), '
            + 'Le: %.2f%% (%s), '
            + 'De: %.2f%% (%s), '
            + 'Se: %.2f%% (%s), '
            + 'Re: %.2f° (%s), '
            + 'fail: %.1f%% \n'
        ) % (
            dt.now().strftime("%Y-%m-%d %H:%M:%S"),
            self._timer.elapsed/60,
            1000*np.nanmean(self._run_times),
            100*sum(self._laterrs)/len(self._laterrs),
            laterr_pctls,
            100*sum(self._disterrs)/len(self._disterrs),
            disterr_pctls,
            100*sum(self._shifterrs)/len(self._shifterrs),
            shifterr_pctls,
            sum(self._roterrs)/len(self._roterrs),
            roterr_pctls,
            100*self._fails/times,
        )
        
        with open(self._logfile, 'r') as org: data = org.read()
        with open(self._logfile, 'w') as mod: mod.write(summary + data)
        print("\n" + summary)


    def _run_algo(self, imgfile_, outfile_, **kwargs_):
        def run_this_from_qt_thread(glWidget, imgfile, outfile, **kwargs):
            ok, rtime = False, False
            timer = tools.Stopwatch()
            timer.start()
            method = kwargs.pop('method', False)
            if method == 'keypoint+':
                try:
                    glWidget.parent().mixed.run(imgfile, outfile, **kwargs)
                    ok = True
                except PositioningException as e:
                    print(str(e))
            if method == 'keypoint':
                try:
                    # try using pympler to find memory leaks, fail: crashes always
                    #    from pympler import tracker
                    #    tr = tracker.SummaryTracker()
                    glWidget.parent().keypoint.solve_pnp(imgfile, outfile, **kwargs)
                    #    tr.print_diff()
                    ok = True
                    rtime = glWidget.parent().keypoint.timer.elapsed
                except PositioningException as e:
                    print(str(e))
            elif method == 'centroid':
                try:
                    glWidget.parent().centroid.adjust_iteratively(imgfile, outfile, **kwargs)
                    ok = True
                except PositioningException as e:
                    print(str(e))
            elif method == 'phasecorr':
                ok = glWidget.parent().phasecorr.findstate(imgfile, outfile, **kwargs)
            timer.stop()
            rtime = rtime if rtime else timer.elapsed
            return ok, rtime
        
        res = self._run_on_qt(run_this_from_qt_thread, 
                              self.window.glWidget, imgfile_, outfile_, **kwargs_)
        return res
    
    
    def _run_on_qt(self, target_func, *args_, **kwargs_):
        self._algorithm_finished = threading.Event()
        def runthis(*args, **kwargs):
            if PROFILE:
                import cProfile
                pr = cProfile.Profile()
                pr.enable()
            
            res = target_func(*args, **kwargs)
            
            if PROFILE:
                pr.disable()
                for i in range(100):
                    if not os.path.isfile(PROFILE_OUT_FILE+str(i)):
                        break
                pr.dump_stats(PROFILE_OUT_FILE+str(i))
            
            self._algorithm_finished.set()
            return res
        
        self.window.tsRun.emit((runthis, args_, kwargs_))
        self._algorithm_finished.wait()
        return self.window.tsRunResult
    
    def _iter_file(self, i, prefix=FILE_PREFIX):
        return os.path.normpath(
                os.path.join(self._iter_dir, prefix+'%04d'%i))
    
    def _cache_file(self, i, prefix=FILE_PREFIX):
        return os.path.normpath(
                os.path.join(CACHE_DIR, prefix+'%04d'%i))
    
    def _cleanup(self):
        self._send('quit')

    def _maybe_exit(self):
        if self.exit:
            print('Exiting...')
            self._cleanup()
            quit()

