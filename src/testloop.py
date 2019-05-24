import math
from math import degrees as deg, radians as rad
import os
import sys
import pickle
from datetime import datetime as dt
from decimal import *

import numpy as np
import quaternion
from astropy.coordinates import spherical_to_cartesian
import cv2

from algo.centroid import CentroidAlgo
from algo.image import ImageProc
from algo.keypoint import KeypointAlgo
from algo.mixed import MixedAlgo
from algo.phasecorr import PhaseCorrelationAlgo
from missions.rosetta import RosettaSystemModel
from render.render import RenderEngine
from iotools import objloader, lblloader
import algo.tools as tools
from algo.tools import (ypr_to_q, q_to_ypr, q_times_v, q_to_unitbase, normalize_v,
                   wrap_rads, solar_elongation, angle_between_ypr)
from algo.tools import PositioningException

from settings import *

#from memory_profiler import profile
#import tracemalloc
#import gc
# TODO: fix suspected memory leaks at
#   - quaternion.as_float_array (?)
#   - cv2.solvePnPRansac (ref_kp_3d?)
#   - astropy, many places


class TestLoop:
    UNIFORM_DISTANCE_GENERATION = True

    def __init__(self, system_model, far=False, est_real_ast_orient=False, operation_zone_only=False):
        self.system_model = system_model
        self.est_real_ast_orient = est_real_ast_orient

        self.exit = False
        self._algorithm_finished = None
        self._smooth_faces = self.system_model.asteroid.render_smooth_faces
        self._opzone_only = operation_zone_only

        file_prefix_mod = ''
        if far and self.system_model.max_distance > self.system_model.max_med_distance:
            file_prefix_mod = 'far_'
            self.max_r = self.system_model.max_distance
            self.min_r = self.system_model.min_distance
        else:
            self.max_r = self.system_model.max_med_distance
            self.min_r = self.system_model.min_distance
        self.file_prefix = system_model.mission_id+'_'+file_prefix_mod
        self.noisy_sm_prefix = system_model.mission_id
        self.cache_path = os.path.join(CACHE_DIR, system_model.mission_id)
        os.makedirs(self.cache_path, exist_ok=True)

        self.render_engine = RenderEngine(system_model.view_width, system_model.view_height)
        self.obj_idx = self.render_engine.load_object(self.system_model.asteroid.real_shape_model, smooth=self._smooth_faces)

        self.keypoint = KeypointAlgo(self.system_model, self.render_engine, self.obj_idx, est_real_ast_orient=est_real_ast_orient)
        self.centroid = CentroidAlgo(self.system_model, self.render_engine, self.obj_idx)
        self.phasecorr = PhaseCorrelationAlgo(self.system_model, self.render_engine, self.obj_idx)
        self.mixedalgo = MixedAlgo(self.centroid, self.keypoint)

        # init later if needed
        self._synth_navcam = None
        self._hires_obj_idx = None

        # gaussian sd in seconds
        self._noise_time = 0     # disabled as _noise_ast_phase_shift does same thing, was 95% within +-30s
        
        # uniform, max dev in deg
        self._noise_ast_rot_axis = 10       # 0 - 10 deg uniform
        self._noise_ast_phase_shift = 10/2  # 95% within 10 deg

        # s/c orientation noise, gaussian sd in deg
        self._noise_sco_lat = 2/2   # 95% within 2 deg
        self._noise_sco_lon = 2/2   # 95% within 2 deg
        self._noise_sco_rot = 2/2   # 95% within 2 deg

        # s/c position noise, gaussian sd in km per km of distance
        self.enable_initial_location = True
        self._unknown_sc_pos = (0, 0, -self.system_model.min_med_distance)
        self._noise_lateral = 0.3     # sd in deg, 0.298 calculated using centroid algo AND 5 deg fov
        self._noise_altitude = 0.10   # 0.131 when calculated using centroid algo AND 5 deg fov

        if self._opzone_only:
            # uniform, max dev in deg
            self._noise_ast_rot_axis = 5         # 0 - 5 deg uniform
            self._noise_ast_phase_shift = 5 / 2  # 95% within 5 deg

            # s/c orientation noise, gaussian sd in deg
            self._noise_sco_lat = 1 / 2  # 95% within 1 deg
            self._noise_sco_lon = 1 / 2  # 95% within 1 deg
            self._noise_sco_rot = 1 / 2  # 95% within 1 deg

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
        self._L = None
        self._state_list = None
        self._rotation_noise = None
        self._loaded_sm_noise = None

        def handle_close():
            self.exit = True
            if self._algorithm_finished:
                self._algorithm_finished.set()


    # main method
    def run(self, times, log_prefix='test-', smn_type='', constant_sm_noise=True, state_db_path=None,
            rotation_noise=True, **kwargs):
        self._smn_cache_id = smn_type
        self._state_db_path = state_db_path
        self._rotation_noise = rotation_noise
        self._constant_sm_noise = constant_sm_noise

        skip = 0
        if isinstance(times, str):
            if ':' in times:
                skip, times = map(int, times.split(':'))
            else:
                times = int(times)
        
        if state_db_path is not None:
            n = self._init_state_db()
            times = min(n, times)
        
        # write logfile header
        self._init_log(log_prefix)
        
        li = 0
        sm = self.system_model
        
        for i in range(skip, times):
            #print('%s'%self._state_list[i])
            
            # maybe generate new noise for shape model
            sm_noise = 0
            if self._smn_cache_id or self._constant_sm_noise:
                sm_noise = self.load_noisy_shape_model(sm, i)
                if sm_noise is None:
                    if DEBUG:
                        print('generating new noisy shape model')
                    sm_noise = self.generate_noisy_shape_model(sm, i)
                    self._maybe_exit()

            # try to load system state
            initial = self.load_state(sm, i) if self._rotation_noise else None
            if initial or self._state_list:
                # successfully loaded system state,
                # try to load related navcam image
                imgfile = self.load_navcam_image(i)
            else:
                imgfile = None

            if initial is None:
                if DEBUG:
                    print('generating new state')
                
                # generate system state
                self.generate_system_state(sm, i)

                # add noise to current state, wipe sc pos
                initial = self.add_noise(sm)

                # save state to lbl file
                if self._rotation_noise:
                    sm.save_state(self._cache_file(i))
            
            # maybe new system state or no previous image, if so, render
            if imgfile is None:
                if DEBUG:
                    print('generating new navcam image')
                imgfile = self.render_navcam_image(sm, i)
                self._maybe_exit()

            if ONLY_POPULATE_CACHE:
                # TODO: fix synthetic navcam generation to work with onboard rendering
                #       current work-around:
                #           1) first generate cache files and skip _run_algo
                #           2) run again, this time not skipping _run_algo
                continue

            # run algorithm
            ok, rtime = self._run_algo(imgfile, self._iter_file(i), **kwargs)

            if kwargs.get('use_feature_db', False) and kwargs.get('add_noise', False):
                sm_noise = self.keypoint.sm_noise
            
            # calculate results
            results = self.calculate_result(sm, i, imgfile, ok, initial, **kwargs)
            
            # write log entry
            self._write_log_entry(i, rtime, sm_noise, *results)
            self._maybe_exit()

            # print out progress
            if DEBUG:
                print('\niteration i=%d:'%(i+1), flush=True)
            elif math.floor(100*i/(times - skip)) > li:
                print('.', end='', flush=True)
                li += 1

        self._close_log(times-skip)


    def generate_system_state(self, sm, i):
        # reset asteroid axis to true values
        sm.asteroid.reset_to_defaults()
        sm.asteroid_rotation_from_model()
        
        if self._state_list is not None:
            lblloader.load_image_meta(
                os.path.join(self._state_db_path, self._state_list[i]+'.LBL'), sm)

            return
        
        for i in range(100):
            ## sample params from suitable distributions
            ##
            # datetime dist: uniform, based on rotation period
            time = np.random.uniform(*sm.time.range)

            # spacecraft position relative to asteroid in ecliptic coords:
            sc_lat = np.random.uniform(-math.pi/2, math.pi/2)
            sc_lon = np.random.uniform(-math.pi, math.pi)

            # s/c distance as inverse uniform distribution
            if TestLoop.UNIFORM_DISTANCE_GENERATION:
                sc_r = np.random.uniform(self.min_r, self.max_r)
            else:
                sc_r = 1/np.random.uniform(1/self.max_r, 1/self.min_r)

            # same in cartesian coord
            sc_ex_u, sc_ey_u, sc_ez_u = spherical_to_cartesian(sc_r, sc_lat, sc_lon)
            sc_ex, sc_ey, sc_ez = sc_ex_u.value, sc_ey_u.value, sc_ez_u.value

            # s/c to asteroid vector
            sc_ast_v = -np.array([sc_ex, sc_ey, sc_ez])

            # sc orientation: uniform, center of asteroid at edge of screen
            if self._opzone_only:
                # always get at least 50% of astroid in view, 5% of the time maximum offset angle
                max_angle = rad(min(sm.cam.x_fov, sm.cam.y_fov)/2)
                da = min(max_angle, np.abs(np.random.normal(0, max_angle/2)))
                dd = np.random.uniform(0, 2*math.pi)
                sco_lat = wrap_rads(-sc_lat + da*math.sin(dd))
                sco_lon = wrap_rads(math.pi + sc_lon + da*math.cos(dd))
                sco_rot = np.random.uniform(-math.pi, math.pi)  # rotation around camera axis
            else:
                # follows the screen edges so that get more partial views, always at least 25% in view
                # TODO: add/subtract some margin
                sco_lat = wrap_rads(-sc_lat)
                sco_lon = wrap_rads(math.pi + sc_lon)
                sco_rot = np.random.uniform(-math.pi, math.pi)  # rotation around camera axis
                sco_q = ypr_to_q(sco_lat, sco_lon, sco_rot)

                ast_ang_r = math.atan(sm.asteroid.mean_radius/1000/sc_r)  # if asteroid close, allow s/c to look at limb
                dx = max(rad(sm.cam.x_fov/2), ast_ang_r)
                dy = max(rad(sm.cam.y_fov/2), ast_ang_r)
                disturbance_q = ypr_to_q(np.random.uniform(-dy, dy), np.random.uniform(-dx, dx), 0)
                sco_lat, sco_lon, sco_rot = q_to_ypr(sco_q * disturbance_q)

            sco_q = ypr_to_q(sco_lat, sco_lon, sco_rot)
            
            # sc_ast_p ecliptic => sc_ast_p open gl -z aligned view
            sc_pos = q_times_v((sco_q * sm.sc2gl_q).conj(), sc_ast_v)
            
            # get asteroid position so that know where sun is
            # *actually barycenter, not sun
            as_v = sm.asteroid.position(time)
            elong, direc = solar_elongation(as_v, sco_q)

            # limit elongation to always be more than set elong
            if elong > rad(sm.min_elong):
                break
        
        if elong <= rad(sm.min_elong):
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
        sm.asteroid.real_sc_ast_vertices = sm.sc_asteroid_vertices()
        
        
    def add_noise(self, sm):
        rotation_noise = True if self._state_list is None else self._rotation_noise
        
        ## add measurement noise to
        # - datetime (seconds)
        if rotation_noise:
            meas_time = sm.time.real_value + np.random.normal(0, self._noise_time)
            sm.time.value = meas_time
            assert np.isclose(sm.time.value, meas_time), 'Failed to set time value'

        # - asteroid state estimate
        ax_lat, ax_lon, ax_phs = map(rad, sm.real_asteroid_axis)
        noise_da = np.random.uniform(0, rad(self._noise_ast_rot_axis))
        noise_dd = np.random.uniform(0, 2*math.pi)
        meas_ax_lat = ax_lat + noise_da*math.sin(noise_dd)
        meas_ax_lon = ax_lon + noise_da*math.cos(noise_dd)
        meas_ax_phs = ax_phs + np.random.normal(0, rad(self._noise_ast_phase_shift))
        if rotation_noise:
            sm.asteroid_axis = map(deg, (meas_ax_lat, meas_ax_lon, meas_ax_phs))

        # - spacecraft orientation measure
        sc_lat, sc_lon, sc_rot = map(rad, sm.real_spacecraft_rot)
        meas_sc_lat = max(-math.pi/2, min(math.pi/2, sc_lat
                + np.random.normal(0, rad(self._noise_sco_lat))))
        meas_sc_lon = wrap_rads(sc_lon 
                + np.random.normal(0, rad(self._noise_sco_lon)))
        meas_sc_rot = wrap_rads(sc_rot
                + np.random.normal(0, rad(self._noise_sco_rot)))
        if rotation_noise:
            sm.spacecraft_rot = map(deg, (meas_sc_lat, meas_sc_lon, meas_sc_rot))
        
        # - spacecraft position noise
        if self.enable_initial_location:
            sm.spacecraft_pos = self._noisy_sc_position(sm)
        else:
            sm.spacecraft_pos = self._unknown_sc_pos

        # return this initial state
        return self._initial_state(sm)

    def _noisy_sc_position(self, sm):
        x, y, z = sm.real_spacecraft_pos
        d = np.linalg.norm((x, y, z))
        return (
            x + d * np.random.normal(0, math.tan(math.radians(self._noise_lateral))),
            y + d * np.random.normal(0, math.tan(math.radians(self._noise_lateral))),
            z + d * np.random.normal(0, self._noise_altitude),
        )

    def _initial_state(self, sm):
        return {
            'time': sm.time.value,
            'ast_axis': sm.asteroid_axis,
            'sc_rot': sm.spacecraft_rot,
            'sc_pos': sm.spacecraft_pos,
        }
    
    def load_state(self, sm, i):
        if self.est_real_ast_orient:
            return None

        try:
            sm.load_state(self._cache_file(i)+'.lbl', sc_ast_vertices=True)
            self._fill_or_censor_init_sc_pos(sm, self._cache_file(i)+'.lbl')
            initial = self._initial_state(sm)
        except FileNotFoundError:
            initial = None
        return initial

    def _fill_or_censor_init_sc_pos(self, sm, state_file):
        # generate and save if missing
        if sm.spacecraft_pos == self._unknown_sc_pos:
            sm.spacecraft_pos = self._noisy_sc_position(sm)
            sm.save_state(state_file)

        # maybe censor
        if not self.enable_initial_location:
            sm.spacecraft_pos = self._unknown_sc_pos


    def generate_noisy_shape_model(self, sm, i):
        #sup = objloader.ShapeModel(fname=SHAPE_MODEL_NOISE_SUPPORT)
        noisy_model, sm_noise, self._L = \
                tools.apply_noise(sm.asteroid.real_shape_model,
                                  #support=np.array(sup.vertices),
                                  L=self._L,
                                  len_sc=SHAPE_MODEL_NOISE_LEN_SC,
                                  noise_lv=SHAPE_MODEL_NOISE_LV[self._smn_cache_id])

        fname = self._cache_file(i, prefix=self.noisy_sm_prefix)+'_'+self._smn_cache_id+'.nsm'
        with open(fname, 'wb') as fh:
            pickle.dump((noisy_model.as_dict(), sm_noise), fh, -1)
        
        self.render_engine.load_object(noisy_model, self.obj_idx, smooth=self._smooth_faces)
        return sm_noise
    
    def load_noisy_shape_model(self, sm, i):
        try:
            if self._constant_sm_noise:
                if self._loaded_sm_noise is not None:
                    return self._loaded_sm_noise
                fname = sm.asteroid.constant_noise_shape_model[self._smn_cache_id]
            else:
                fname = self._cache_file(i, prefix=self.noisy_sm_prefix)+'_'+self._smn_cache_id+'.nsm'

            with open(fname, 'rb') as fh:
                noisy_model, self._loaded_sm_noise = pickle.load(fh)
            self.render_engine.load_object(objloader.ShapeModel(data=noisy_model), self.obj_idx, smooth=self._smooth_faces)
        except (FileNotFoundError, EOFError):
            self._loaded_sm_noise = None

        return self._loaded_sm_noise

    @staticmethod
    def render_navcam_image_static(sm, renderer, obj_idxs, rel_pos_v=None, rel_rot_q=None, light_v=None,
                                   use_shadows=True, use_textures=False):

        if rel_pos_v is None:
            rel_pos_v = sm.spacecraft_pos
        if rel_rot_q is None:
            rel_rot_q, _ = sm.gl_sc_asteroid_rel_q()
        if light_v is None:
            light_v, _ = sm.gl_light_rel_dir()

        model = RenderEngine.REFLMOD_HAPKE
        RenderEngine.REFLMOD_PARAMS[model] = sm.asteroid.reflmod_params[model]
        img, depth = renderer.render(obj_idxs, rel_pos_v, rel_rot_q, light_v, get_depth=True,
                                     shadows=use_shadows, textures=use_textures, reflection=model)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # do same gamma correction as the available rosetta navcam images have
        img = ImageProc.adjust_gamma(img, 1.8)

        # coef=2 gives reasonably many stars, star brightness was tuned without gamma correction
        img = ImageProc.add_stars(img.astype('float'), mask=depth>=sm.max_distance-0.1, coef=2.5)

        # ratio seems too low but blurring in images match actual Rosetta navcam images
        img = ImageProc.apply_point_spread_fn(img, ratio=0.2)

        # add background noise
        img = ImageProc.add_ccd_noise(img, mean=7, sd=2)
        img = np.clip(img, 0, 255).astype('uint8')

        if False:
            cv2.imshow('test', img)
            cv2.waitKey()
            quit()

        return img

    def render_navcam_image(self, sm, i):
        if self._synth_navcam is None:
            self._synth_navcam = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=16)
            self._synth_navcam.set_frustum(sm.cam.x_fov, sm.cam.y_fov, sm.min_altitude, sm.max_distance)
            self._hires_obj_idx = self._synth_navcam.load_object(sm.asteroid.hires_target_model_file)

        use_textures = sm.asteroid.hires_target_model_file_textures
        sm.swap_values_with_real_vals()
        img = TestLoop.render_navcam_image_static(sm, self._synth_navcam, self._hires_obj_idx, use_textures=use_textures)
        sm.swap_values_with_real_vals()

        cache_file = self._cache_file(i)+'.png'
        cv2.imwrite(cache_file, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        return cache_file

    def load_navcam_image(self, i):
        if self._state_list is None:
            fname = self._cache_file(i)+'.png'
        else:
            fname = os.path.join(self._state_db_path, self._state_list[i]+'_P.png')
        return fname if os.path.isfile(fname) else None

    def calculate_result(self, sm, i, imgfile, ok, initial, **kwargs):
        # save function values from optimization
        fvals = getattr({
            'phasecorr': self.phasecorr,
            'keypoint+': self.mixedalgo,
            'keypoint': self.keypoint,
            'centroid': self.centroid,
        }[kwargs['method']], 'extra_values', None)
        final_fval = fvals[-1] if fvals else None

        real_rel_rot = q_to_ypr(sm.real_sc_asteroid_rel_q())
        elong, direc = sm.solar_elongation(real=True)
        r_ast_axis = sm.real_asteroid_axis
        
        # real system state
        params = (sm.time.real_value, *r_ast_axis,
                *sm.real_spacecraft_rot, deg(elong), deg(direc),
                *sm.real_spacecraft_pos, sm.real_spacecraft_altitude, *map(deg, real_rel_rot),
                imgfile, final_fval)
        
        # calculate added noise
        #
        getcontext().prec = 6
        time_noise = float(Decimal(initial['time']) - Decimal(sm.time.real_value))
        
        ast_rot_noise = (
            initial['ast_axis'][0]-r_ast_axis[0],
            initial['ast_axis'][1]-r_ast_axis[1],
            360*time_noise/sm.asteroid.rotation_period
                + (initial['ast_axis'][2]-r_ast_axis[2])
        )
        sc_rot_noise = tuple(np.subtract(initial['sc_rot'], sm.real_spacecraft_rot))
        
        dev_angle = deg(angle_between_ypr(map(rad, ast_rot_noise),
                                          map(rad, sc_rot_noise)))

        if self.enable_initial_location:
            sc_loc_noise = tuple(np.array(initial['sc_pos']) - np.array(sm.real_spacecraft_pos))
        else:
            sc_loc_noise = ('', '', '')

        noise = sc_loc_noise + (time_noise,) + ast_rot_noise + sc_rot_noise + (dev_angle,)

        if np.all(ok):
            ok_pos, ok_rot = True, True
        elif not np.any(ok):
            ok_pos, ok_rot = False, False
        else:
            ok_pos, ok_rot = ok

        if ok_pos:
            pos = sm.spacecraft_pos
            pos_err = tuple(np.subtract(pos, sm.real_spacecraft_pos))
        else:
            pos = float('nan')*np.ones(3)
            pos_err = tuple(float('nan')*np.ones(3))

        if ok_rot:
            rel_rot = q_to_ypr(sm.sc_asteroid_rel_q())
            rot_err = (deg(wrap_rads(angle_between_ypr(rel_rot, real_rel_rot))),)
        else:
            rel_rot = float('nan')*np.ones(3)
            rot_err = (float('nan'),)

        alt = float('nan')
        if ok_pos and ok_rot:
            est_vertices = sm.sc_asteroid_vertices()
            max_shift = tools.sc_asteroid_max_shift_error(
                    est_vertices, sm.asteroid.real_sc_ast_vertices)
            alt = sm.spacecraft_altitude
            both_err = (max_shift, alt - sm.real_spacecraft_altitude)
        else:
            both_err = (float('nan'), float('nan'),)

        err = pos_err + rot_err + both_err
        return params, noise, pos, alt, map(deg, rel_rot), fvals, err

    def _init_state_db(self):
        try:
            with open(os.path.join(self._state_db_path, 'ignore_these.txt'), 'rb') as fh:
                ignore = tuple(l.decode('utf-8').strip() for l in fh)
        except FileNotFoundError:
            ignore = tuple()
        self._state_list = sorted([f[:-4] for f in os.listdir(self._state_db_path)
                                          if f[-4:]=='.LBL' and f[:-4] not in ignore])
        return len(self._state_list)
    
    @staticmethod
    def log_columns():
        return (
                'iter', 'date', 'execution time',
                'time', 'ast lat', 'ast lon', 'ast rot',
                'sc lat', 'sc lon', 'sc rot',
                'sol elong', 'light dir', 'x sc pos', 'y sc pos', 'z sc pos', 'sc altitude',
                'rel yaw', 'rel pitch', 'rel roll',
                'imgfile', 'extra val', 'shape model noise',
                'sc pos x dev', 'sc pos y dev', 'sc pos z dev',
                'time dev', 'ast lat dev', 'ast lon dev', 'ast rot dev',
                'sc lat dev', 'sc lon dev', 'sc rot dev', 'total dev angle',
                'x est sc pos', 'y est sc pos', 'z est sc pos', 'altitude est sc',
                'yaw rel est', 'pitch rel est', 'roll rel est',
                'x err sc pos', 'y err sc pos', 'z err sc pos', 'rot error',
                'shift error km', 'altitude error', 'lat error (m/km)', 'dist error (m/km)', 'rel shift error (m/km)',
            )

    def _init_log(self, log_prefix):
        os.makedirs(LOG_DIR, exist_ok=True)

        logbody = log_prefix + dt.now().strftime('%Y%m%d-%H%M%S')
        self._iter_dir = os.path.join(LOG_DIR, logbody)
        os.mkdir(self._iter_dir)
        
        self._fval_logfile = LOG_DIR + logbody + '-fvals.log'
        self._logfile = LOG_DIR + logbody + '.log'
        with open(self._logfile, 'w') as file:
            file.write(' '.join(sys.argv)+'\n'+ '\t'.join(self.log_columns())+'\n')
            
        self._run_times = []
        self._laterrs = []
        self._disterrs = []
        self._roterrs = []
        self._shifterrs = []
        self._fails = 0
        self._timer = tools.Stopwatch()
        self._timer.start()
        
        
    def _write_log_entry(self, i, rtime, sm_noise, params, noise, pos, alt, rel_rot, fvals, err):

        # save execution time
        self._run_times.append(rtime)

        # calculate errors
        dist = abs(params[-7])
        if not math.isnan(err[0]):
            lerr = 1000*math.sqrt(err[0]**2 + err[1]**2) / dist     # m/km
            derr = 1000*err[2] / dist                               # m/km
            rerr = abs(err[3])
            serr = 1000*err[4] / dist                               # m/km
            self._laterrs.append(lerr)
            self._disterrs.append(abs(derr))
            self._roterrs.append(rerr)
            self._shifterrs.append(serr)
        else:
            lerr = derr = rerr = serr = float('nan')
            self._fails += 1

        # log all parameter values, timing & errors into a file
        with open(self._logfile, 'a') as file:
            file.write('\t'.join(map(str, (
                i, dt.now().strftime("%Y-%m-%d %H:%M:%S"), rtime, *params,
                sm_noise, *noise, *pos, alt, *rel_rot, *err, lerr, derr, serr
            )))+'\n')

        # log opt fun values in other file
        if fvals:
            with open(self._fval_logfile, 'a') as file:
                file.write(str(i)+'\t'+'\t'.join(map(str, fvals))+'\n')

    def _close_log(self, samples):
        self._timer.stop()

        summary = self.calc_err_summary(self._timer.elapsed, samples, self._fails, self._run_times,
                                        self._laterrs, self._disterrs, self._shifterrs, self._roterrs)
        
        with open(self._logfile, 'r') as org: data = org.read()
        with open(self._logfile, 'w') as mod: mod.write(summary + data)
        print("\n" + summary)

    @staticmethod
    def calc_err_summary(runtime, samples, fails, runtimes, laterrs, disterrs, shifterrs, roterrs):
        disterrs = np.abs(disterrs)
        if len(laterrs):
            # percentiles matching the 0*sd, 1*sd, 2*sd, 3*sd limits of a normal distribution
            ignore_worst = 99.87 if samples >= 2000 else 97.72
            prctls = (50, 84.13, 97.72) + ((99.87,) if samples >= 2000 else tuple())
            def calc_prctls(errs):
                errs = np.array(errs)[np.logical_not(np.isnan(errs))]
                lim = np.percentile(errs, ignore_worst)
                errs = errs[errs < lim]
                m = np.mean(errs)
                sd = np.std(errs)
                return ', '.join('%.2f/%.2f' % (m+i*sd, p) for i, p in enumerate(np.percentile(errs, prctls)))

            try:
                laterr_pctls = calc_prctls(laterrs)
                disterr_pctls = calc_prctls(disterrs)
                shifterr_pctls = calc_prctls(shifterrs)
                roterr_pctls = calc_prctls(roterrs)
            except Exception as e:
                print('Error calculating quantiles: %s' % e)
                laterr_pctls = 'error'
                disterr_pctls = 'error'
                shifterr_pctls = 'error'
                roterr_pctls = 'error'

            # a summary line
            summary_data = (
                laterr_pctls,
                disterr_pctls,
                shifterr_pctls,
                roterr_pctls,
            )
        else:
            summary_data = tuple(np.ones(8) * float('nan'))

        return (
              '%s - t: %.1fmin (%.0fms), '
              + 'Le m/km: (%s), '
              + 'De m/km: (%s), '
              + 'Se m/km: (%s), '
              + 'Re m/km: (%s), '
              + 'fail: %.2f%% \n'
        ) % (
              dt.now().strftime("%Y-%m-%d %H:%M:%S"),
              runtime / 60,
              1000 * np.nanmean(runtimes) if len(runtimes) else float('nan'),
              *summary_data,
              100 * fails / samples,
        )


    def _run_algo(self, imgfile, outfile, **kwargs):
        ok, rtime = False, False
        timer = tools.Stopwatch()
        timer.start()
        method = kwargs.pop('method', False)
        sm = self.system_model
        if method == 'keypoint+':
            try:
                ok = self.mixedalgo.run(imgfile, outfile, **kwargs)
            except PositioningException as e:
                print(str(e))
        if method == 'keypoint':
            try:
                # try using pympler to find memory leaks, fail: crashes always
                #    from pympler import tracker
                #    tr = tracker.SummaryTracker()
                if self.enable_initial_location:
                    x, y, z = sm.spacecraft_pos
                    ix_off, iy_off = sm.cam.calc_img_xy(x, y, z)
                    uncertainty_radius = math.tan(math.radians(self._noise_lateral) * 2) \
                                         * abs(z) * (1 + self._noise_altitude * 2)
                    kwargs['match_mask_params'] = ix_off-sm.cam.width/2, iy_off-sm.cam.height/2, z, uncertainty_radius

                self.keypoint.solve_pnp(imgfile, outfile, **kwargs)
                #    tr.print_diff()

                # TODO: if fdb, log discretization errors from
                #  - self.keypoint.latest_discretization_err_q
                #  - self.keypoint.latest_discretization_light_err_angle

                ok = True
                rtime = self.keypoint.timer.elapsed
            except PositioningException as e:
                print(str(e))
        elif method == 'centroid':
            try:
                self.centroid.adjust_iteratively(imgfile, outfile, **kwargs)
                ok = True
            except PositioningException as e:
                print(str(e))
        elif method == 'phasecorr':
            ok = self.phasecorr.findstate(imgfile, outfile, **kwargs)
        timer.stop()
        rtime = rtime if rtime else timer.elapsed
        return ok, rtime

    def _iter_file(self, i, prefix=None):
        if prefix is None:
            prefix = self.file_prefix
        return os.path.normpath(
                os.path.join(self._iter_dir, prefix+'%04d'%i))
    
    def _cache_file(self, i, prefix=None):
        if prefix is None:
            prefix = self.file_prefix
        return os.path.normpath(
            os.path.join(self.cache_path, (prefix+'%04d'%i) if self._state_list is None
                                    else self._state_list[i]))
    
    def _maybe_exit(self):
        if self.exit:
            print('Exiting...')
            quit()


if __name__ == '__main__':
    sm = RosettaSystemModel(rosetta_batch='mtp006')
    img = 'ROS_CAM1_20140822T020718'

    lblloader.load_image_meta(os.path.join(sm.asteroid.image_db_path, img + '.LBL'), sm)
    sm.swap_values_with_real_vals()

    renderer = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=16)
    renderer.set_frustum(sm.cam.x_fov, sm.cam.y_fov, sm.min_altitude * .1, sm.max_distance)
    obj_idx = renderer.load_object(sm.asteroid.hires_target_model_file, smooth=sm.asteroid.render_smooth_faces)
    use_textures = sm.asteroid.hires_target_model_file_textures

    img = TestLoop.render_navcam_image_static(sm, renderer, obj_idx, use_textures=use_textures)
    cv2.imshow('synthetic navcam image', img)
    cv2.waitKey()
