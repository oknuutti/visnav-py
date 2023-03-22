import random

import math
from math import degrees as deg, radians as rad
import os
import sys
import pickle
from datetime import datetime as dt
from decimal import *

import numpy as np
import quaternion
import cv2
from tqdm import tqdm

from visnav.algo.centroid import CentroidAlgo
from visnav.algo.image import ImageProc
from visnav.algo.keypoint import KeypointAlgo
from visnav.algo.mixed import MixedAlgo
from visnav.algo.model import SystemModel
from visnav.algo.phasecorr import PhaseCorrelationAlgo
from visnav.missions.didymos import DidymosSystemModel
from visnav.missions.rosetta import RosettaSystemModel
from visnav.render.particles import Particles
from visnav.render.render import RenderEngine
from visnav.iotools import objloader, lblloader
import visnav.algo.tools as tools
from visnav.algo.tools import (ypr_to_q, q_to_ypr, q_times_v, q_to_unitbase, normalize_v,
                               wrap_rads, solar_elongation, angle_between_ypr, save_float_img)
from visnav.algo.tools import PositioningException
from visnav.render.stars import Stars
from visnav.render.sun import Sun

from visnav.settings import *

#from memory_profiler import profile
import tracemalloc
#tracemalloc.start()
#import gc
# TODO: fix suspected memory leaks at
#   - quaternion.as_float_array (?)
#   - cv2.solvePnPRansac (ref_kp_3d?)
#   - astropy, many places


class TestLoop:
    UNIFORM_DISTANCE_GENERATION = True

    def __init__(self, system_model, file_prefix_mod, est_real_ast_orient=False,
                 uniform_distance_gen=UNIFORM_DISTANCE_GENERATION,
                 operation_zone_only=False, state_generator=None, cache_path=None,
                 max_sc_lateral_disp=1.0,
                 sm_noise=0, sm_noise_len_sc=SHAPE_MODEL_NOISE_LEN_SC,
                 navcam_cache_id='', save_distance=False, save_depth=False, save_coords=False,
                 traj_len=1, traj_prop_dt=60, only_populate_cache=ONLY_POPULATE_CACHE,
                 real_sm_noise=0, real_sm_noise_len_sc=SHAPE_MODEL_NOISE_LEN_SC,
                 real_tx_noise=0, real_tx_noise_len_sc=SHAPE_MODEL_NOISE_LEN_SC,
                 haze=0, jets=0, jet_int_mode=0.001, jet_int_conc=10,
                 hapke_noise=0,
                 hapke_th_sd=None, hapke_w_sd=None,
                 hapke_b_sd=None, hapke_c_sd=None,
                 hapke_shoe=None, hapke_shoe_w=None,
                 hapke_cboe=None, hapke_cboe_w=None,
                 noise_time=0, noise_ast_rot_axis=0, noise_ast_phase_shift=0, noise_sco_lat=0,
                 noise_sco_lon=0, noise_sco_rot=0, noise_lateral=0, noise_altitude=0,
                 noise_phase_angle=0, noise_light_dir=0, ext_noise_dist=False
                 ):

        self.system_model = system_model
        self.est_real_ast_orient = est_real_ast_orient
        self.only_populate_cache = only_populate_cache

        self.exit = False
        self._algorithm_finished = None
        self._smooth_faces = self.system_model.asteroid.render_smooth_faces
        self._opzone_only = operation_zone_only

        self._state_generator = state_generator if state_generator is not None else \
                                lambda sm: sm.random_state(uniform_distance=uniform_distance_gen,
                                                           opzone_only=self._opzone_only,
                                                           max_sc_lateral_disp=max_sc_lateral_disp)

        self.sm_noise = sm_noise
        self.sm_noise_len_sc = sm_noise_len_sc
        self.navcam_cache_id = navcam_cache_id
        self.real_sm_noise = real_sm_noise
        self.real_sm_noise_len_sc = real_sm_noise_len_sc
        self.real_tx_noise = real_tx_noise
        self.real_tx_noise_len_sc = real_tx_noise_len_sc
        self.haze = haze
        self.jets = jets
        self.jet_int_mode = jet_int_mode
        self.jet_int_conc = jet_int_conc
        self.hapke_noise = hapke_noise
        self.hapke_th_sd = hapke_th_sd
        self.hapke_w_sd = hapke_w_sd
        self.hapke_b_sd = hapke_b_sd
        self.hapke_c_sd = hapke_c_sd
        self.hapke_shoe = hapke_shoe
        self.hapke_shoe_w = hapke_shoe_w
        self.hapke_cboe = hapke_cboe
        self.hapke_cboe_w = hapke_cboe_w
        self.save_distance = save_distance
        self.save_depth = save_depth
        self.save_coords = save_coords

        self.traj_len = traj_len
        self.traj_prop_dt = traj_prop_dt

        self.file_prefix = system_model.mission_id+'_'+file_prefix_mod
        self.noisy_sm_prefix = system_model.mission_id
        self.cache_path = cache_path if cache_path else os.path.join(CACHE_DIR, system_model.mission_id)
        os.makedirs(self.cache_path, exist_ok=True)

        self.render_engine = RenderEngine(system_model.view_width, system_model.view_height)
        self.render_engine.load_object(self.system_model.asteroid.real_shape_model, smooth=self._smooth_faces)

        self.keypoint = KeypointAlgo(self.system_model, self.render_engine, 0, est_real_ast_orient=est_real_ast_orient)
        self.keypoint.RENDER_TEXTURES = self.system_model.asteroid.hires_target_model_file_textures
        self.centroid = CentroidAlgo(self.system_model, self.render_engine, 0)
        self.centroid.RENDER_TEXTURES = self.system_model.asteroid.hires_target_model_file_textures
        self.phasecorr = PhaseCorrelationAlgo(self.system_model, self.render_engine, 0)
        self.mixedalgo = MixedAlgo(self.centroid, self.keypoint)
        self.absnet = None  # lazy load

        # init later if needed
        self._synth_navcam = None
        self._hires_obj_idx = None
        self.obj_idx = None

        # instead of sampling ast, s/c orient and light direction from gaussians with given sd,
        # sample uniformly: y ~ [-2*sd, 2*sd], x = sqrt(y)
        self._ext_noise_dist = ext_noise_dist

        # gaussian sd in seconds
        self._noise_time = noise_time     # _noise_ast_phase_shift does practically same thing
        
        # uniform, max dev in deg
        self._noise_ast_rot_axis = noise_ast_rot_axis        # in deg, uniform
        self._noise_ast_phase_shift = noise_ast_phase_shift  # in deg, 95% within 2x this

        # s/c orientation noise, gaussian sd in deg
        self._noise_sco_lat = noise_sco_lat   # in deg, 95% within 2x this
        self._noise_sco_lon = noise_sco_lon   # in deg, 95% within 2x this
        self._noise_sco_rot = noise_sco_rot   # in deg, 95% within 2x this

        # light direction noise, gaussian sd in degrees
        self._noise_phase_angle = noise_phase_angle
        self._noise_light_dir = noise_light_dir

        # s/c position noise, gaussian sd in km per km of distance
        self.enable_initial_location = True
        self._unknown_sc_pos = (0, 0, -self.system_model.min_med_distance)
        self._noise_lateral = noise_lateral    # sd in deg, 0.298 calculated using centroid algo AND 5 deg fov
        self._noise_altitude = noise_altitude   # 0.131 when calculated using centroid algo AND 5 deg fov

        # transients
        self._smn_cache_id = ''
        self._iter_dir = None
        self._logfile = None
        self._fval_logfile = None
        self.image_files = []
        self.run_times = []
        self.laterrs = []
        self.disterrs = []
        self.roterrs = []
        self.shifterrs = []
        self.fails = []
        self._timer = None
        self._L = (None, None)
        self._shape_model = None
        self._hires_L = (None, None)
        self._lo_res_support = None
        self._hi_res_support = None
        self._state_list = None
        self._rotation_noise = None
        self._loaded_sm_noise = None

        def handle_close():
            self.exit = True
            if self._algorithm_finished:
                self._algorithm_finished.set()


    # main method
    def run(self, times, log_prefix='test-', smn_cache_id='', constant_sm_noise=True, state_db_path=None,
            rotation_noise=True, resynth_cam_image=False, method='akaze', **kwargs):
        self._smn_cache_id = smn_cache_id
        self._state_db_path = state_db_path
        self._resynth_cam_image = resynth_cam_image
        self._rotation_noise = rotation_noise
        self._constant_sm_noise = constant_sm_noise
        sm = self.system_model

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

        for i in tqdm(list(range(skip, times))):
            # maybe generate new noise for shape model
            sm_noise = 0
            if self._smn_cache_id or self._constant_sm_noise:
                sm_noise = self.load_noisy_shape_model(sm, i)
                if sm_noise is None:
                    if self._smn_cache_id:
                        if DEBUG:
                            print('generating new noisy shape model')
                        sm_noise = self.generate_noisy_shape_model(sm, i)
                    else:
                        sm_noise = float('nan')
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
                
                if self._state_list is not None:
                    # load system state from image meta data
                    lblloader.load_image_meta(os.path.join(self._state_db_path, self._state_list[i] + '.LBL'), sm)
                else:
                    # generate system state
                    self._state_generator(sm)

                # add noise to current state, wipe sc pos
                if self.traj_len == 1:
                    initial = self.add_noise(sm)
                else:
                    initial = self._initial_state(sm)

                # save state to lbl file
                if self._rotation_noise:
                    sm.save_state(self.cache_file(i, skip_cache_id=True))
            
            # maybe new system state or no previous image, if so, render
            if imgfile is None:
                if DEBUG:
                    print('generating new navcam image')

                # snapshot1 = tracemalloc.take_snapshot()
                imgfile = self.render_navcam_image(sm, i, traj_len=self.traj_len, dt=self.traj_prop_dt)

                if self.traj_len > 1:
                    # run current selection of absolute navigation algos with first image only
                    imgfile = imgfile[0]

                # snapshot2 = tracemalloc.take_snapshot()
                # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
                # print("[ Top 10 differences ]")
                # for stat in top_stats[:10]:
                #     print(stat)
                # input("Press Enter to continue...")
                self._maybe_exit()

            # if isinstance(imgfile, str):
            #     img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
            #     if img.dtype == np.uint16:
            #         img = (img / 256).astype(np.uint8)
            #     img = cv2.resize(img, (sm.view_width, sm.view_height), interpolation=cv2.INTER_AREA)
            #     outfile = os.path.join(self.cache_path, os.path.basename(imgfile))
            #     cv2.imwrite(outfile, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            if self.only_populate_cache:
                continue

            # run algorithm
            #tracemalloc.start()
            #current, peak = tracemalloc.get_traced_memory()
            #print("before: %.0fMB" % ((peak - current)/1024/1024))
            ok, rtime = self._run_algo(imgfile, self._iter_file(i), method=method, **kwargs)
            #current, peak = tracemalloc.get_traced_memory()
            #print("after: %.0fMB" % ((peak - current)/1024/1024))
            #tracemalloc.stop()

            if kwargs.get('use_feature_db', False) and kwargs.get('add_noise', False):
                sm_noise = self.keypoint.sm_noise
            
            # calculate results
            results = self.calculate_result(sm, i, imgfile, ok, initial, method=method, **kwargs)
            
            # write log entry
            self._write_log_entry(i, rtime, sm_noise, *results)
            self._maybe_exit()

        self._close_log(times-skip)

    @staticmethod
    def _ext_rand(min_v, max_v=None, zero=0, ext_coef=2):
        if max_v is None:
            max_v = min_v
            min_v = zero - max_v

        x = random.random() ** (1 / ext_coef)  # get values that are closer to extremes
        a = (min_v - zero) if random.random() > 0.5 else (max_v - zero)
        return a * x

    def add_noise(self, sm):
        rotation_noise = True if self._state_list is None else self._rotation_noise
        
        ## add measurement noise to
        # - direction of light by rotating the whole system around the sun
        if self._noise_phase_angle > 0 or self._noise_light_dir > 0:
            ph, noise_ph, noise_na = sm.phase_angle(), 0, 0
            for i in range(100):
                if self._ext_noise_dist:
                    noise_ph = self._ext_rand(2 * rad(self._noise_phase_angle))
                    noise_na = self._ext_rand(2 * rad(self._noise_light_dir))
                else:
                    noise_ph = np.random.normal(0, rad(self._noise_phase_angle))
                    noise_na = np.random.normal(0, rad(self._noise_light_dir))

                if 0 < ph + noise_ph < np.pi - rad(sm.min_elong):
                    # generate new phase angles until resulting angle is in the accepted range
                    break
            assert 0 < ph + noise_ph < np.pi - rad(sm.min_elong), 'some strange problem'

            cam_axis = tools.q_times_v(sm.spacecraft_q, np.array([1, 0, 0]))
            light_v = tools.normalize_v(sm.asteroid.position(sm.time.value))
            axis_ph = np.cross(cam_axis, light_v)
            ph_q = tools.angleaxis_to_q(np.array([noise_ph, *axis_ph]))
            sm.rotate_light(ph_q)

            axis_na = tools.q_times_v(sm.spacecraft_q, np.array([1, 0, 0]))
            na_q = tools.angleaxis_to_q(np.array([noise_na, *axis_na]))
            sm.rotate_light(na_q)

        # - datetime (seconds)
        if rotation_noise:
            if self._ext_noise_dist:
                noise_t = self._ext_rand(2*self._noise_time)
            else:
                noise_t = np.random.normal(0, self._noise_time)
            meas_time = sm.time.real_value + noise_t
            sm.time.value = meas_time
            assert np.isclose(sm.time.value, meas_time), 'Failed to set time value'

        # - asteroid state estimate
        noise_da = np.random.uniform(0, rad(self._noise_ast_rot_axis))
        noise_dd = np.random.uniform(0, 2 * math.pi)
        if self._ext_noise_dist:
            noise_da = np.sqrt(noise_da)
            noise_ph = self._ext_rand(2*rad(self._noise_ast_phase_shift))
        else:
            noise_ph = np.random.normal(0, rad(self._noise_ast_phase_shift))
        ax_lat, ax_lon, ax_phs = map(rad, sm.real_asteroid_axis)
        meas_ax_lat = ax_lat + noise_da*math.sin(noise_dd)
        meas_ax_lon = ax_lon + noise_da*math.cos(noise_dd)
        meas_ax_phs = ax_phs + noise_ph
        if rotation_noise:
            sm.asteroid_axis = map(deg, (meas_ax_lat, meas_ax_lon, meas_ax_phs))

        # - spacecraft orientation measure
        if self._ext_noise_dist:
            noise_lat = self._ext_rand(2*rad(self._noise_sco_lat))
            noise_lon = self._ext_rand(2*rad(self._noise_sco_lon))
            noise_rot = self._ext_rand(2*rad(self._noise_sco_rot))
        else:
            noise_lat = np.random.normal(0, rad(self._noise_sco_lat))
            noise_lon = np.random.normal(0, rad(self._noise_sco_lon))
            noise_rot = np.random.normal(0, rad(self._noise_sco_rot))

        sc_lat, sc_lon, sc_rot = map(rad, sm.real_spacecraft_rot)
        meas_sc_lat = max(-math.pi/2, min(math.pi/2, sc_lat + noise_lat))
        meas_sc_lon = wrap_rads(sc_lon + noise_lon)
        meas_sc_rot = wrap_rads(sc_rot + noise_rot)
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

        if self._ext_noise_dist:
            nx = self._ext_rand(2 * math.tan(math.radians(self._noise_lateral)))
            ny = self._ext_rand(2 * math.tan(math.radians(self._noise_lateral)))
            nz = self._ext_rand(2 * self._noise_altitude)
        else:
            nx = np.random.normal(0, math.tan(math.radians(self._noise_lateral)))
            ny = np.random.normal(0, math.tan(math.radians(self._noise_lateral)))
            nz = np.random.normal(0, self._noise_altitude)

        return x + d * nx, y + d * ny, z + d * nz

    def _initial_state(self, sm):
        return {
            'time': sm.time.value,
            'ast_axis': sm.asteroid_axis,
            'sc_rot': sm.spacecraft_rot,
            'sc_pos': sm.spacecraft_pos,
            'ast_pos': sm.asteroid.real_position,
        }
    
    def load_state(self, sm, i):
        if self.est_real_ast_orient:
            return None

        try:
            sm.load_state(self.cache_file(i, skip_cache_id=True) + '.lbl', sc_ast_vertices=True)
            self._fill_or_censor_init_sc_pos(sm, self.cache_file(i, skip_cache_id=True) + '.lbl')
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
                tools.apply_noise(self._shape_model,
                                  #support=np.array(sup.vertices),
                                  L=self._L,
                                  len_sc=self.sm_noise_len_sc,
                                  noise_lv=self.sm_noise)

        fname = self.cache_file(i, prefix=self.noisy_sm_prefix, postfix=self._smn_cache_id) + '.nsm'
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
                fname = self.cache_file(i, prefix=self.noisy_sm_prefix, postfix=self._smn_cache_id) + '.nsm'

            with open(fname, 'rb') as fh:
                noisy_model, self._loaded_sm_noise = pickle.load(fh)
            self._shape_model = objloader.ShapeModel(data=noisy_model)

            supports = {'': ['_hi_res_support'], 'lo': ['_lo_res_support']}
            supports[self._smn_cache_id].append(self._shape_model)
            for k, p in supports.items():
                if len(p) == 1:
                    with open(sm.asteroid.constant_noise_shape_model[k], 'rb') as fh:
                        noisy_model, self._loaded_sm_noise = pickle.load(fh)
                    p.append(objloader.ShapeModel(data=noisy_model))
                setattr(self, p[0], p[1])

            i = self.render_engine.load_object(self._shape_model, self.obj_idx, smooth=self._smooth_faces)
            self.obj_idx = i if self.obj_idx is None else self.obj_idx
        except (FileNotFoundError, EOFError):
            print('cant find shape model "%s"' % fname)
            self._loaded_sm_noise = None

        return self._loaded_sm_noise


    @staticmethod
    def render_navcam_image_static(sm, renderer, obj_idxs, rel_pos_v=None, rel_rot_q=None, light_v=None, sc_q=None,
                                   sun_distance=None, exposure=None, gain=None, gamma=1.8, auto_gain=True,
                                   use_shadows=True, use_textures=False, reflmod_params=None, cam=None, fluxes_only=False,
                                   stars=True, lens_effects=True, star_db=None, particles=None, return_depth=False,
                                   return_coords=False):

        if rel_pos_v is None:
            rel_pos_v = sm.spacecraft_pos
        if rel_rot_q is None:
            rel_rot_q, _ = sm.gl_sc_asteroid_rel_q()
        if light_v is None:
            light_v, _ = sm.gl_light_rel_dir()
        if sc_q is None:
            sc_q = sm.spacecraft_q  # for correct stars
        if star_db:
            Stars.STARDB = star_db

        light_v = tools.normalize_v(light_v)
        sun_sc_distance = sun_distance or np.linalg.norm(sm.asteroid.position(sm.time.value))  # in meters

        model = RenderEngine.REFLMOD_HAPKE
        RenderEngine.REFLMOD_PARAMS[model] = sm.asteroid.reflmod_params[model] if reflmod_params is None else reflmod_params

        dist_au = sun_sc_distance / 1.496e+11  # in au
        solar_flux_density = 1360.8 / dist_au**2

        if auto_gain:
            # don't use K correction factor if use automatically scaled image
            RenderEngine.REFLMOD_PARAMS[model][9] = 0

        object_flux, depth = renderer.render(obj_idxs, rel_pos_v, rel_rot_q, light_v, get_depth=True,
                                             shadows=use_shadows, textures=use_textures, reflection=model,
                                             gamma=1.0, flux_density=False if auto_gain else solar_flux_density)

        if return_coords:
            coords = renderer.render_extra_data(obj_idxs, rel_pos_v, rel_rot_q, light_v)

        if np.any(np.isnan(object_flux)):
            print('NaN(s) encountered in rendered image!')
            object_flux[np.isnan(object_flux)] = 0

        cam = cam or sm.cam
        exposure = exposure or cam.def_exposure
        gain = gain or cam.def_gain
        mask = depth >= renderer.frustum_far - 0.1

        if not auto_gain:
            sun_lf = - tools.q_times_v(SystemModel.sc2gl_q, sun_sc_distance * light_v)
            # TODO: investigate if need *cam.px_sr or something else!!
            downsample = 2
            lens_effect = Sun.flux_density(cam, sun_lf, mask) if lens_effects else np.array([0])

            # radiance given in W/(m2*sr) => needed in W/m2
            object_flux *= cam.px_sr
        else:
            object_flux = object_flux.astype('f4')/255/cam.sensitivity/exposure/gain
            lens_effect = np.array([0])

        # add flux density of stars from Tycho-2 catalog
        star_flux = Stars.flux_density(sc_q, cam, mask=mask) if stars else np.array([0])

        # render particle effects such as jets and haze
        if particles is not None:
            particles.cam = cam or sm.cam
            try:
                rel_rot_q, rel_pos_v = rel_rot_q[0], rel_pos_v[0]
            except:
                pass
            particle_flux = particles.flux_density(object_flux, depth, np.logical_not(mask), rel_pos_v, rel_rot_q,
                                                   light_v, solar_flux_density)
        else:
            particle_flux = np.array([0])

        total_flux = object_flux.astype(np.float64) + lens_effect.astype(np.float64) + \
                     star_flux.astype(np.float64) + particle_flux.astype(np.float64)

        if fluxes_only:
            return total_flux

        # do the sensing
        img = cam.sense(total_flux, exposure=exposure, gain=gain)

        # do same gamma correction as the available rosetta navcam images have
        img = np.clip(img*255, 0, 255)
        img = ImageProc.adjust_gamma(img, gamma).astype('uint8')

        if False:
            cv2.imshow('test', img)
            cv2.waitKey()
            quit()

        ret = [img]
        if return_depth:
            ret.append(depth)
        if return_coords:
            ret.append(coords)
        return ret

    def render_navcam_image(self, sm, i, traj_len=1, dt=60):
        use_textures = sm.asteroid.hires_target_model_file_textures or self.real_tx_noise > 0
        tx_randomize = self.real_tx_noise > 0
        update_model = False
        tx_hf_noise = 1

        if self._synth_navcam is None:
            update_model = True
            self._synth_navcam = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=16,
                                              enable_extra_data=self.save_coords)
            self._synth_navcam.set_frustum(sm.cam.x_fov, sm.cam.y_fov, sm.min_altitude*.2, sm.max_distance*1.1)
            if use_textures and not sm.asteroid.hires_target_model_file_textures:
                sm.asteroid.real_shape_model.texfile = None
                # TODO: customize the size, affects performance quite a lot
                sm.asteroid.real_shape_model.tex = np.ones((2048, 2048) if tx_hf_noise else (100, 100))

        # NOTICE: Randomizing the vertices takes a lot of time, texture not so
        #         - Also: caching the vertices would take too much space, so no caching
        if self.real_sm_noise > 0 or tx_randomize:
            update_model = True
            Sv = (np.array(self._lo_res_support.vertices),
                  np.array(self._hi_res_support.vertices)) if self.real_sm_noise > 0 else None
            tx_sup = self._lo_res_support if tx_randomize else None
            model, _, self._hires_L = \
                tools.apply_noise(sm.asteroid.real_shape_model, support=(Sv, tx_sup), L=self._hires_L,
                                  noise_lv=self.real_sm_noise, len_sc=self.real_sm_noise_len_sc,
                                  tx_noise=self.real_tx_noise if use_textures else 0,
                                  tx_noise_len_sc=self.real_tx_noise_len_sc,
                                  tx_hf_noise=tx_hf_noise)
        else:
            model = sm.asteroid.real_shape_model

        if update_model:
            self._hires_obj_idx = self._synth_navcam.load_object(model, self._hires_obj_idx)

        reflmod_params = self.randomized_hapke_params(sm)

        particles = None
        cache_files = []
        single = traj_len == 1
        px_s = 2*math.tan(max(math.radians(sm.cam.x_fov) / sm.cam.width, math.radians(sm.cam.y_fov) / sm.cam.height)/2)

        for j in range(traj_len):
            sm.swap_values_with_real_vals()

            if particles is None and (self.haze > 0 or self.jets > 0):
                haze = 0
                if self.haze > 0:
                    dist = np.linalg.norm(sm.spacecraft_pos)
                    haze = np.random.uniform(0, (sm.min_distance / dist)**2 * self.haze)

                cones = None
                if self.jets > 0:
                    cones = {
                        'n': int(np.random.exponential(self.jets)) if self.jets > 0 else 0,
                        'trunc_len_m': sm.asteroid.max_radius * 0.1,
                        'jet_int_mode': self.jet_int_mode,
                        'jet_int_conc': self.jet_int_conc,
                    }

                particles = Particles(sm.cam, None, None, cones=cones, haze=haze)

            ret = TestLoop.render_navcam_image_static(sm, self._synth_navcam, self._hires_obj_idx, gamma=1.0,
                                                      use_textures=use_textures, reflmod_params=reflmod_params,
                                                      particles=particles, return_depth=True,
                                                      return_coords=self.save_coords)
            if self.save_coords:
                img, depth, coords = ret
            else:
                img, depth = ret
            depth[depth >= 0.99 * sm.max_distance] = np.nan
            img = ImageProc.normalize_brightness(img, gamma=1.8)

            sm.swap_values_with_real_vals()

            fname = self.cache_file(i, postfix=('' if single else ('%d' % j)))
            cache_files.append(fname + '.png')
            cv2.imwrite(cache_files[j], img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            if self.save_distance:
                # saves distance instead of depth
                ixy = tools.unit_aflow(sm.cam.width, sm.cam.height).reshape((-1, 2))
                xyz = sm.cam.backproject(ixy, z_off=depth.flatten().astype(np.float32))
                dist = np.linalg.norm(xyz, axis=1).reshape((sm.cam.height, sm.cam.width))
                save_float_img(fname + '.d', dist)
            elif self.save_depth:
                save_float_img(fname + '.d', depth.astype(np.float32))

            if self.save_coords:
                save_float_img(fname + '.xyz', coords.astype(np.float32))
                save_float_img(fname + '.s', (depth * px_s).astype(np.float32))

            if not single:
                sm.save_state(self.cache_file(i, skip_cache_id=True, postfix=str(j)))

                if j+1 < traj_len:
                    # propagate if not last state
                    sm.propagate(0)     # should be dt instead of 0, but now propagation is just done using add_noise

                    # TODO: remove following when propagate() does more than just save current state
                    if 1:
                        tmp = sm.get_vals(real=True)     # save current real values
                        self.add_noise(sm)               # add noise on top of current real values
                        sm.swap_values_with_real_vals()  # make new real values be the noisy values
                        sm.set_vals(tmp, real=False)     # set noisy values to previous real values

        return cache_files[0] if single else cache_files

    def randomized_hapke_params(self, sm):
        reflmod_params = list(sm.asteroid.reflmod_params[RenderEngine.REFLMOD_HAPKE])
        params = {
            1: ('hapke_th_sd', 0, 90),
            2: ('hapke_w_sd', 0, 1),
            3: ('hapke_b_sd', -1 if reflmod_params[4] == 0 else 0, 1),
            4: ('hapke_c_sd', 0, 1),
            5: ('hapke_shoe', 0, np.inf),
            6: ('hapke_shoe_w', 0, np.pi),
            7: ('hapke_cboe', 0, np.inf),
            8: ('hapke_cboe_w', 0, np.pi),
        }
        for i, (param, lo, hi) in params.items():
            sd = getattr(self, param)
            if sd is None and self.hapke_noise > 0:
                # multiplicative noise
                reflmod_params[i] *= np.random.lognormal(0, self.hapke_noise)
            elif sd is not None and sd > 0:
                # additive noise
                reflmod_params[i] += np.random.normal(0, sd)
            reflmod_params[i] = np.clip(reflmod_params[i], lo, hi)

        return reflmod_params

    def load_navcam_image(self, i):
        if self._state_list is None or self._resynth_cam_image:
            fname = self.cache_file(i) + '.png'
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
            'absnet': self.absnet,
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
        self.image_files.append(imgfile)

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
            
        self.run_times = []
        self.laterrs = []
        self.disterrs = []
        self.roterrs = []
        self.shifterrs = []
        self.fails = []
        self._timer = tools.Stopwatch()
        self._timer.start()
        
        
    def _write_log_entry(self, i, rtime, sm_noise, params, noise, pos, alt, rel_rot, fvals, err):

        # save execution time
        self.run_times.append(rtime)

        # calculate errors
        dist = abs(params[-7])
        if not math.isnan(err[0]):
            lerr = 1000*math.sqrt(err[0]**2 + err[1]**2) / dist     # m/km
            derr = 1000*err[2] / dist                               # m/km
            rerr = abs(err[3])                                      # deg
            serr = 1000*err[4] / dist                               # m/km
            fail = 0
        else:
            lerr = derr = rerr = serr = float('nan')
            fail = 1
        self.laterrs.append(lerr)
        self.disterrs.append(abs(derr))
        self.roterrs.append(rerr)
        self.shifterrs.append(serr)
        self.fails.append(fail)

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

        summary = self.calc_err_summary(self._timer.elapsed, samples, self.fails, self.run_times,
                                        self.laterrs, self.disterrs, self.shifterrs, self.roterrs)
        
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
            summary_data = tuple(np.ones(4) * float('nan'))

        return (
              '%s - t: %.1fmin (%.0fms), '
              + 'Le m/km: (%s), '
              + 'De m/km: (%s), '
              + 'Se m/km: (%s), '
              + 'Re deg: (%s), '
              + 'fail: %.2f%% \n'
        ) % (
              dt.now().strftime("%Y-%m-%d %H:%M:%S"),
              runtime / 60,
              1000 * np.nanmean(runtimes) if len(runtimes) else float('nan'),
              *summary_data,
              100 * np.sum(fails) / samples,
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
                if kwargs.get('verbose', 0):
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
                if kwargs.get('verbose', 0):
                    print(str(e))
        elif method == 'centroid':
            try:
                self.centroid.adjust_iteratively(imgfile, outfile, **kwargs)
                ok = True
            except PositioningException as e:
                if kwargs.get('verbose', 0):
                    print(str(e))
        elif method == 'absnet':
            if self.absnet is None:
                from visnav.algo.absnet import AbsoluteNavigationNN
                self.absnet = AbsoluteNavigationNN(self.system_model, self.render_engine, self.obj_idx, verbose=True)
            try:
                self.absnet.process(imgfile, outfile, **kwargs)
                ok = True
            except PositioningException as e:
                if kwargs.get('verbose', 0):
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
    
    def cache_file(self, i, prefix=None, postfix=None, skip_cache_id=False):
        if prefix is None:
            prefix = self.file_prefix

        full_prefix = prefix + ('' if skip_cache_id or not self.navcam_cache_id else '%s_' % self.navcam_cache_id)
        postfix = ('' if not postfix else '_%s' % postfix)

        return os.path.normpath(
            os.path.join(self.cache_path,
                         (full_prefix + '%04d' % i) if self._state_list is None
                         else self._state_list[i])) + postfix

    def _maybe_exit(self):
        if self.exit:
            print('Exiting...')
            quit()


def test_rosetta_navcam():
    ast = True
    if ast:
        gain = 1.7
        fa = True
        if 1:
            batch = 'mtp006'
            imgf = 'ROS_CAM1_20140822T020718'  # ok
            exp = 2.0
        elif 0:
            batch = 'mtp006'
            imgf = 'ROS_CAM1_20140801T150717'  # too dark
            exp = 1.0
        elif 0:
            batch = 'mtp006'
            imgf = 'ROS_CAM1_20140902T062253'  # ok
            exp = 2.8
        elif 0:
            batch = 'mtp007'
            imgf = 'ROS_CAM1_20140916T064502'  # ok
            exp = 0.9
        elif 0:
            batch = 'mtp017'
            imgf = 'ROS_CAM1_20150603T060217'  # ok
            exp = 1.16
        elif 0:
            batch = 'mtp017'
            imgf = 'ROS_CAM1_20150603T192546'  # ok
            exp = 0.01
            gain = 1.0
            fa = False
        elif 0:
            batch = 'mtp026'
            imgf = 'ROS_CAM1_20160303T182855'  # too bright
            exp = 3.27
        elif 1:
            batch = 'mtp026'
            imgf = 'ROS_CAM1_20160305T233002'  # ok
            exp = 0.01
            gain = 1.0
            fa = False

        # TODO: check if should take distance into account as currently near objects too bright and far ones too dim
    else:
        batch = 'mtp003'
        imgf = 'ROS_CAM1_20140531T114923'  # mtp003
        exp = 2.5
        gain = 1.0
        fa = False

    sm = RosettaSystemModel(rosetta_batch=batch, focused_attenuated=fa)
    lblloader.load_image_meta(os.path.join(sm.asteroid.image_db_path, imgf + '.LBL'), sm)
    sm.swap_values_with_real_vals()

    if False:
        s = 16
        mf = sm.asteroid.hires_target_model_file
    else:
        s = 0
        mf = sm.asteroid.target_model_file

    renderer = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=s)
    renderer.set_frustum(sm.cam.x_fov, sm.cam.y_fov, sm.min_altitude * .1, sm.max_distance)
    obj_idx = renderer.load_object(mf, smooth=sm.asteroid.render_smooth_faces)
    use_textures = sm.asteroid.hires_target_model_file_textures

    img = TestLoop.render_navcam_image_static(sm, renderer, obj_idx, use_textures=use_textures,
                                              exposure=exp, gain=gain, gamma=1.0, auto_gain=False)

    real = cv2.imread(os.path.join(sm.asteroid.image_db_path, imgf + '_P.png'), cv2.IMREAD_GRAYSCALE)
    real = ImageProc.adjust_gamma(real, (1 / 1.8))
    sc = 768 / img.shape[0]
    img2 = np.hstack((real, img))
    cv2.imwrite('D:/temp/test.png', img2)
    cv2.imshow('synthetic navcam image', cv2.resize(img2, None, fx=sc, fy=sc))
    cv2.waitKey()


def test_apex_navcam():
    stars = 0
    if 1:
        nac = False
        exp = 3.5 if stars else 0.001
        gain = 6.0 if stars else 1
        dist = 1.5
    elif 1:
        nac = True
        exp = 3.5 if stars else 0.010
        gain = 3.0 if stars else 1
        dist = 7

    #    sm = RosettaSystemModel(rosetta_batch='mtp007', focused_attenuated=True)
    sm = DidymosSystemModel(target_primary=True, use_narrow_cam=nac)

    if False:
        s = 16
        mf = sm.asteroid.hires_target_model_file
    else:
        s = 0
        mf = sm.asteroid.target_model_file

    renderer = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=s)
#    renderer.set_frustum(sm.cam.x_fov, sm.cam.y_fov, sm.min_altitude * .1, sm.max_distance)
    renderer.set_frustum(sm.cam.x_fov, sm.cam.y_fov, 1e-2, 1e6)
    obj_idx = renderer.load_object(mf, smooth=sm.asteroid.render_smooth_faces)
    use_textures = sm.asteroid.hires_target_model_file_textures

    sm.random_state(uniform_distance=True, opzone_only=True)
    #lblloader.load_image_meta(os.path.join(DATA_DIR, 'rosetta-mtp007', 'ROS_CAM1_20140916T064502' + '.LBL'), sm)
    au = 1.496e+8  # in km
    sm.asteroid.real_position = 1.2 * au * np.array([-1, 0, 0])  #np.array([0.42024187, -0.78206733, -0.46018199])
    sm.swap_values_with_real_vals()
    sm.spacecraft_pos = [0, 0, -dist]
    sm.asteroid_q = quaternion.one

    for i in range(-10, 10):
        sm.spacecraft_q = tools.ypr_to_q(math.radians(2*i), 0, 0) * tools.ypr_to_q(0, math.radians(180), 0)  # tools.ypr_to_q(math.radians(-89), 0, 0)  # [52.5, -64, 2]
        img = TestLoop.render_navcam_image_static(sm, renderer, obj_idx, use_textures=use_textures,
                                                  exposure=exp, gain=gain, gamma=1.0, auto_gain=False)

        cv2.imwrite('D:/temp/test2.png', img)
        sc = 768 / img.shape[0]
        cv2.imshow('synthetic navcam image', cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA))
        cv2.waitKey()


if __name__ == '__main__':
    if False:
        test_rosetta_navcam()
    else:
        test_apex_navcam()
