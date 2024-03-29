import re
import subprocess
import time
from subprocess import TimeoutExpired
from shutil import which

import math
import socket
import json
import traceback
from datetime import datetime
from json import JSONDecodeError

import cv2
import numpy as np
import quaternion
import sys
import pytz
#import tracemalloc

import settings
from visnav.algo.centroid import CentroidAlgo
from visnav.algo.image import ImageProc
from visnav.algo.odometry import VisualOdometry, Pose

settings.LOG_DIR = sys.argv[1]
settings.CACHE_DIR = settings.LOG_DIR
from visnav.settings import *

from visnav.algo import tools
from visnav.algo.base import AlgorithmBase
from visnav.algo.keypoint import KeypointAlgo
from visnav.algo.model import SystemModel
from visnav.algo.tools import PositioningException
from visnav.batch1 import get_system_model
from visnav.missions.didymos import DidymosPrimary, DidymosSystemModel, DidymosSecondary
from visnav.render.render import RenderEngine
from visnav.testloop import TestLoop


def main():
    port = int(sys.argv[2])

    if len(sys.argv) > 3:
        server = ApiServer(sys.argv[3], port=port, hires=True, result_rendering=False)
    else:
        server = SpawnMaster(port=port, max_count=20000)

    try:
        server.listen()
    except QuitException:
        server.print('quit received, exiting')
    finally:
        server.close()
    quit()


class QuitException(Exception):
    pass


class ApiServer:
    SERVER_READY_NOTICE = 'server started, waiting for connections'
    MAX_ORI_DIFF_ANGLE = 360  # in deg

    (
        FRAME_GLOBAL,
        FRAME_LOCAL,
    ) = range(2)

    def __init__(self, mission, hires=False, addr='127.0.0.1', port=50007, result_rendering=True,
                 result_frame=FRAME_LOCAL):
        self._pid = os.getpid()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._addr = addr
        self._port = port
        self._sock.bind(('', port))
        self._sock.listen(1)
        self._sock.settimeout(5)

        self._result_rendering = result_rendering
        self._result_frame = result_frame
        self._hires = hires

        self._mission = mission
        self._sm = sm = get_system_model(mission, hires)
        self._target_d2 = '2' in mission
        self._use_nac = mission[-1] == 'n'
        self._autolevel = True  # not self._use_nac
        self._current_level = False
        self._level_lambda = 0.9

        if not self._target_d2:
            # so that D2 always contained in frustrum
            sm.max_distance += 1.3
        self._renderer = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=16 if hires else 0)
        self._renderer.set_frustum(sm.cam.x_fov, sm.cam.y_fov, sm.min_altitude*.1, sm.max_distance)
        if isinstance(sm, DidymosSystemModel):
            self.asteroids = [
                DidymosPrimary(hi_res_shape_model=hires),
                DidymosSecondary(hi_res_shape_model=hires),
            ]
        else:
            self.asteroids = [sm.asteroid]

        self._obj_idxs = []
        self._wireframe_obj_idxs = [
            self._renderer.load_object(os.path.join(DATA_DIR, 'ryugu+tex-%s-100.obj'%ast), wireframe=True)
            for ast in ('d1', 'd2')] if result_rendering else []

        self._logpath = os.path.join(LOG_DIR, 'api-server', self._mission)
        os.makedirs(self._logpath, exist_ok=True)

        self._onboard_renderer = RenderEngine(sm.view_width, sm.view_height, antialias_samples=0)
        self._target_obj_idx = self._onboard_renderer.load_object(sm.asteroid.target_model_file, smooth=sm.asteroid.render_smooth_faces)
        self._keypoint = KeypointAlgo(sm, self._onboard_renderer, self._target_obj_idx)
        self._centroid = CentroidAlgo(sm, self._onboard_renderer, self._target_obj_idx)
        self._odometry = {}

        # laser measurement range given in dlem-20 datasheet
        self._laser_min_range, self._laser_max_range, self._laser_nominal_max_range = 10, 2100, 5000
        self._laser_fail_prob = 0.05   # probability of measurement fail because e.g. asteroid low albedo
        self._laser_false_prob = 0.01  # probability for random measurement even if no target
        self._laser_err_sigma = 0.5    # 0.5m sigma1 accuracy given in dlem-20 datasheet

        # laser algo params
        self._laser_adj_loc_weight = 1
        self._laser_meas_err_weight = 0.3
        self._laser_max_adj_dist = 100  # in meters

    def _maybe_load_objects(self):
        if len(self._obj_idxs) == 0:
            for ast in self.asteroids:
                file = ast.hires_target_model_file if self._hires else ast.target_model_file
                cache_file = os.path.join(CACHE_DIR, os.path.basename(file).split('.')[0] + '.pickle')
                self._obj_idxs.append(self._renderer.load_object(file, smooth=ast.render_smooth_faces, cache_file=cache_file))
            self._obj_idxs.append(self._renderer.load_object(self._sm.sc_model_file, smooth=False))

    @staticmethod
    def _parse_poses(params, offset):
        d1_v = np.array(params[offset][:3])*0.001
        d1_q = np.quaternion(*(params[offset][3:7])).normalized()
        d2_v = np.array(params[offset+1][:3])*0.001
        d2_q = np.quaternion(*(params[offset+1][3:7])).normalized()
        sc_v = np.array(params[offset+2][:3])*0.001
        sc_q = np.quaternion(*(params[offset+2][3:7])).normalized()
        return d1_v, d1_q, d2_v, d2_q, sc_v, sc_q

    def _laser_meas(self, params):
        time = params[0]
        d1_v, d1_q, d2_v, d2_q, sc_v, sc_q = self._parse_poses(params, offset=1)

        d1, d2 = self.asteroids
        q = SystemModel.sc2gl_q.conj() * sc_q.conj()
        rel_rot_q = np.array([q * d1_q * d1.ast2sc_q.conj(), q * d2_q * d2.ast2sc_q.conj()])
        rel_pos_v = np.array([tools.q_times_v(q, d1_v - sc_v), tools.q_times_v(q, d2_v - sc_v)])

        self._maybe_load_objects()
        dist = self._renderer.ray_intersect_dist(self._obj_idxs[0:2], rel_pos_v, rel_rot_q)

        if dist is None:
            if np.random.uniform(0, 1) < self._laser_false_prob:
                noisy_dist = np.random.uniform(self._laser_min_range, self._laser_nominal_max_range)
            else:
                noisy_dist = None
        else:
            if np.random.uniform(0, 1) < self._laser_fail_prob:
                noisy_dist = None
            else:
                noisy_dist = dist * 1000 + np.random.normal(0, self._laser_err_sigma)
                if noisy_dist < self._laser_min_range or noisy_dist > self._laser_max_range:
                    noisy_dist = None

        return json.dumps(noisy_dist)

    def _laser_algo(self, params):
        time = params[0]
        dist_meas = params[1]
        if not dist_meas or dist_meas < self._laser_min_range or dist_meas > self._laser_max_range:
            raise PositioningException('invalid laser distance measurement: %s' % dist_meas)

        d1_v, d1_q, d2_v, d2_q, sc_v, sc_q = self._parse_poses(params, offset=2)

        q = SystemModel.sc2gl_q.conj() * sc_q.conj()
        d1, d2 = self.asteroids
        ast = d2 if self._target_d2 else d1
        ast_v = d2_v if self._target_d2 else d1_v
        ast_q = d2_q if self._target_d2 else d1_q

        rel_rot_q = q * ast_q * ast.ast2sc_q.conj()
        rel_pos_v = tools.q_times_v(q, ast_v - sc_v) * 1000
        max_r = ast.max_radius
        max_diam = 2*max_r/1000

        # set orthographic projection
        self._onboard_renderer.set_orth_frustum(max_diam, max_diam, -max_diam/2, max_diam/2)

        # render orthographic depth image
        _, zz = self._onboard_renderer.render(self._target_obj_idx, [0, 0, 0], rel_rot_q, [1, 0, 0],
                                              get_depth=True, shadows=False, textures=False)

        # restore regular perspective projection
        self._onboard_renderer.set_frustum(self._sm.cam.x_fov, self._sm.cam.y_fov,
                                           self._sm.min_altitude*.1, self._sm.max_distance)

        zz[zz > max_diam/2*0.999] = float('nan')
        zz = zz*1000 - rel_pos_v[2]
        xx, yy = np.meshgrid(np.linspace(-max_r, max_r, self._sm.view_width) - rel_pos_v[0],
                             np.linspace(-max_r, max_r, self._sm.view_height) - rel_pos_v[1])

        x_expected = np.clip((rel_pos_v[0]+max_r)/max_r/2*self._sm.view_width + 0.5, 0, self._sm.view_width - 1.001)
        y_expected = np.clip((rel_pos_v[1]+max_r)/max_r/2*self._sm.view_height + 0.5, 0, self._sm.view_height - 1.001)
        dist_expected = tools.interp2(zz, x_expected, y_expected, discard_bg=True)

        # mse cost function balances between adjusted location and measurement error
        adj_dist_sqr = (zz - dist_meas)**2 + xx**2 + yy**2
        cost = self._laser_adj_loc_weight * adj_dist_sqr \
             + (self._laser_meas_err_weight - self._laser_adj_loc_weight) * (zz - dist_meas)**2

        j, i = np.unravel_index(np.nanargmin(cost), cost.shape)
        if np.isnan(zz[j, i]):
            raise PositioningException('laser algo results in off asteroid pointing')
        if math.sqrt(adj_dist_sqr[j, i]) >= self._laser_max_adj_dist:
            raise PositioningException('laser algo solution too far (%.0fm, limit=%.0fm), spurious measurement assumed'
                                       % (math.sqrt(adj_dist_sqr[j, i]), self._laser_max_adj_dist))

        dx, dy, dz = xx[0, i], yy[j, 0], zz[j, i] - dist_meas

        if self._result_frame == ApiServer.FRAME_GLOBAL:
            # return global ast-sc vector
            est_sc_ast_v = ast_v * 1000 - tools.q_times_v(q.conj(), rel_pos_v + np.array([dx, dy, dz]))
        else:
            # return local sc-ast vector
            est_sc_ast_v = tools.q_times_v(SystemModel.sc2gl_q, rel_pos_v + np.array([dx, dy, dz]))
        dist_expected = float(dist_expected) if not np.isnan(dist_expected) else -1.0
        return json.dumps([list(est_sc_ast_v), dist_expected])

    def _render(self, params):
        time = params[0]
        sun_distance = np.linalg.norm(np.array(params[1][:3]))  # in meters
        sun_ast_v = tools.normalize_v(np.array(params[1][:3]))
        d1_v, d1_q, d2_v, d2_q, sc_v, sc_q = self._parse_poses(params, offset=2)

        d1, d2 = self.asteroids
        q = SystemModel.sc2gl_q.conj() * sc_q.conj()
        rel_rot_q = np.array([q * d1_q * d1.ast2sc_q.conj(), q * d2_q * d2.ast2sc_q.conj(),
                              np.quaternion(1, 0, 1, 0).normalized()])  # last one is the for the spacecraft
        rel_pos_v = np.array([tools.q_times_v(q, d1_v - sc_v), tools.q_times_v(q, d2_v - sc_v), [0, 0, 0]])
        light_v = tools.q_times_v(q, sun_ast_v)

        self._maybe_load_objects()  # lazy load objects

        exp_range = (0.001, 3.5)
        for i in range(20):
            if self._autolevel and self._current_level:
                level = self._current_level
            else:
                level = 3*2.5*1.3e-3 if self._use_nac else 1.8*2.5*1.3e-3

            exp, gain = self._sm.cam.level_to_exp_gain(level, exp_range)

            img = TestLoop.render_navcam_image_static(self._sm, self._renderer, self._obj_idxs,
                                                      rel_pos_v, rel_rot_q, light_v, sc_q, sun_distance,
                                                      exposure=exp, gain=gain, auto_gain=False,
                                                      gamma=1.0, use_shadows=True, use_textures=True)
            img = img[0]
            if self._autolevel:
                v = np.percentile(img, 100 - 0.0003)
                level_trg = level * 170 / v
                print('autolevel (max_v=%.1f, e=%.3f, g=%.1f) current: %.3f, target: %.3f' % (v, exp, gain, level, level_trg))

                self._current_level = level_trg if not self._current_level else \
                        (self._current_level*self._level_lambda + level_trg*(1-self._level_lambda))

                if v < 85 or (v == 255 and level > exp_range[0]):
                    level = level_trg if v < 85 else level * 70 / v
                    self._current_level = level
                    continue
            break

        if False:
            img = ImageProc.default_preprocess(img)

        date = datetime.fromtimestamp(time, pytz.utc)  # datetime.now()
        fname = os.path.join(self._logpath, date.isoformat()[:-6].replace(':', '')) + '.png'
        cv2.imwrite(fname, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        return fname

    def _get_pose(self, params, algo_id=1):
        # load target navcam image
        fname = params[0]
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

        # asteroid position relative to the sun
        self._sm.asteroid.real_position = np.array(params[1][:3])

        d1_v, d1_q, d2_v, d2_q, sc_v, init_sc_q = self._parse_poses(params, offset=2)

        # set asteroid orientation
        d1, d2 = self.asteroids
        ast = d2 if self._target_d2 else d1
        init_ast_q = d2_q if self._target_d2 else d1_q
        self._sm.asteroid_q = init_ast_q * ast.ast2sc_q.conj()

        # set spacecraft orientation
        self._sm.spacecraft_q = init_sc_q

        # initial rel q
        init_rel_q = init_sc_q.conj() * init_ast_q

        # sc-asteroid relative location
        ast_v = d2_v if self._target_d2 else d1_v   # relative to barycenter
        init_sc_pos = tools.q_times_v(SystemModel.sc2gl_q.conj() * init_sc_q.conj(), ast_v - sc_v)
        self._sm.spacecraft_pos = init_sc_pos

        # run keypoint algo
        err = None
        rot_ok = True
        try:
            if algo_id == 1:
                self._keypoint.solve_pnp(img, fname[:-4], KeypointAlgo.AKAZE, verbose=1 if self._result_rendering else 0)
            elif algo_id == 2:
                self._centroid.adjust_iteratively(img, fname[:-4])
                rot_ok = False
            else:
                assert False, 'invalid algo_id=%s' % algo_id
        except PositioningException as e:
            err = e

        rel_gf_q = np.quaternion(*([np.nan]*4))
        if err is None:
            sc_q = self._sm.spacecraft_q

            # resulting sc-ast relative orientation
            rel_lf_q = self._sm.asteroid_q * ast.ast2sc_q if rot_ok else np.quaternion(*([np.nan]*4))
            rel_gf_q = sc_q.conj() * rel_lf_q

            # sc-ast vector in meters
            rel_lf_v = tools.q_times_v(SystemModel.sc2gl_q, np.array(self._sm.spacecraft_pos) * 1000)
            rel_gf_v = tools.q_times_v(sc_q, rel_lf_v)

            # collect to one result list
            if self._result_frame == ApiServer.FRAME_GLOBAL:
                result = [list(rel_gf_v), list(quaternion.as_float_array(rel_gf_q))]
            else:
                result = [list(rel_lf_v), list(quaternion.as_float_array(rel_lf_q))]

            diff_angle = math.degrees(tools.angle_between_q(init_rel_q, rel_gf_q))
            if diff_angle > ApiServer.MAX_ORI_DIFF_ANGLE:
                err = PositioningException('Result orientation too different than initial one, diff %.1f°, max: %.1f°'
                                           % (diff_angle, ApiServer.MAX_ORI_DIFF_ANGLE))

        # render a result image
        if self._result_rendering:
            self._render_result([fname]
                + [list(np.array(self._sm.spacecraft_pos)*1000) + list(quaternion.as_float_array(rel_gf_q))]
                + [list(np.array(init_sc_pos)*1000) + list(quaternion.as_float_array(init_rel_q))])

        if err is not None:
            raise err

        # send back in json format
        return json.dumps(result)

    def _odometry_track(self, params):
        VO_EST_CAM_POSE = False

        session = params[0]

        fname = params[1]
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

        orig_time = params[2]
        time = datetime.fromtimestamp(orig_time, pytz.utc)

        sun_ast_v = tools.normalize_v(np.array(params[3][:3]))
        d1_v, d1_q, d2_v, d2_q, sc_v, sc_q = self._parse_poses(params, offset=4)
        ast_q = d2_q if self._target_d2 else d1_q
        ast_v = d2_v if self._target_d2 else d1_v

        cv2sc_q = SystemModel.cv2gl_q * SystemModel.sc2gl_q.conj()
        sc_ast_cvf_q = cv2sc_q * sc_q.conj() * ast_q * cv2sc_q.conj()
        sc_ast_cvf_v = tools.q_times_v(cv2sc_q * sc_q.conj(), ast_v - sc_v) * 1000

        if VO_EST_CAM_POSE:
            # still from global to old ast-sc frame
            sc_ast_cvf_q = cv2sc_q * ast_q.conj() * sc_q * cv2sc_q.conj()
            sc_ast_cvf_v = tools.q_times_v(cv2sc_q * ast_q.conj(), sc_v - ast_v) * 1000
#            sc_ast_cvf_v = tools.q_times_v(sc_ast_cvf_q * cv2sc_q, sc_v - ast_v)

        # TODO: modify so that uncertainties are given (or not)
        prior = Pose(sc_ast_cvf_v, sc_ast_cvf_q, np.ones((3,))*0.1, np.ones((3,))*0.01)

        # tracemalloc.start()
        # current0, peak0 = tracemalloc.get_traced_memory()
        # print("before: %.0fMB" % (current0/1024/1024))
        if session not in self._odometry:
            self._odometry[session] = VisualOdometry(self._sm.cam, self._sm.view_width*2, verbose=1,
                                                     use_scale_correction=False, est_cam_pose=VO_EST_CAM_POSE)
        post, bias_sds, scale_sd = self._odometry[session].process(img, time, prior, sc_q)
        # current, peak = tracemalloc.get_traced_memory()
        # print("transient: %.0fMB" % ((peak - current)/1024/1024))
        # tracemalloc.stop()

        if post is not None:
            est_sc_ast_scf_q = cv2sc_q.conj() * post.quat * cv2sc_q
            est_sc_ast_scf_v = tools.q_times_v(cv2sc_q.conj(), post.loc)

            if VO_EST_CAM_POSE:
                # still from old ast-sc frame to local sc-ast frame
                est_sc_ast_scf_q = cv2sc_q.conj() * post.quat.conj() * cv2sc_q
                # NOTE: we use prior orientation instead of posterior one as the error there grows over time
                est_sc_ast_scf_v = -tools.q_times_v(cv2sc_q.conj() * prior.quat.conj(), post.loc)

            if self._result_frame == ApiServer.FRAME_GLOBAL:
                est_sc_ast_scf_q = sc_q * est_sc_ast_scf_q
                est_sc_ast_scf_v = tools.q_times_v(sc_q, est_sc_ast_scf_v)

            # TODO: (1) return in sc local frame
            #   - distance err sd
            #   - lateral err sd
            #   - orientation err sd (pitch & yaw)
            #   - orientation err sd (roll)
            #   - distance bias drift sd
            #   - lateral bias drift sd
            #   - orientation bias drift sd (pitch & yaw)
            #   - orientation bias drift sd (roll)
            #   - scale drift sd
            dist = np.linalg.norm(sc_ast_cvf_v)
            bias_sds = bias_sds * dist
            est_sc_ast_lf_v_s2 = post.loc_s2 * dist
            est_sc_ast_lf_so3_s2 = post.so3_s2

            result = [
                list(est_sc_ast_scf_v),
                list(quaternion.as_float_array(est_sc_ast_scf_q)),
                list(est_sc_ast_lf_v_s2) + list(est_sc_ast_lf_so3_s2) + list(bias_sds) + [scale_sd],
                orig_time,
            ]
        else:
            raise PositioningException('No tracking result')

        # send back in json format
        return json.dumps(result)

    def _render_result(self, params):
        fname = params[0]
        img = cv2.imread(fname, cv2.IMREAD_COLOR)

        if np.all(np.logical_not(np.isnan(params[1]))):
            rel_pos_v = np.array(params[1][:3]) * 0.001
            rel_rot_q = np.quaternion(*(params[1][3:7]))
            color = np.array((0, 1, 0))*0.6
        else:
            rel_pos_v = np.array(params[2][:3]) * 0.001
            rel_rot_q = np.quaternion(*(params[2][3:7]))
            color = np.array((0, 0, 1))*0.6

        # ast_v = np.array(params[1][:3])
        # ast_q = np.quaternion(*(params[1][3:7]))
        # sc_v = np.array(params[2][:3])
        # sc_q = np.quaternion(*(params[2][3:7]))
        #
        ast_idx = 1 if self._target_d2 else 0
        ast = self.asteroids[ast_idx]
        # q = SystemModel.sc2gl_q.conj() * sc_q.conj()
        # rel_rot_q = q * ast_q * ast.ast2sc_q.conj()
        # rel_pos_v = tools.q_times_v(q, ast_v - sc_v) * 0.001

        rel_rot_q = SystemModel.sc2gl_q.conj() * rel_rot_q * ast.ast2sc_q.conj()
        #rel_pos_v = tools.q_times_v(SystemModel.sc2gl_q.conj(), rel_pos_v) * 0.001

        overlay = self._renderer.render_wireframe(self._wireframe_obj_idxs[ast_idx], rel_pos_v, rel_rot_q, color)
        overlay = cv2.resize(overlay, (img.shape[1], img.shape[0]))

        blend_coef = 0.6
        alpha = np.zeros(list(img.shape[:2]) + [1])
        alpha[np.any(overlay > 0, axis=2)] = blend_coef
        result = (overlay * alpha + img * (1 - alpha)).astype('uint8')

        fout = fname[:-4] + '-res.png'
        cv2.imwrite(fout, result, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        return fout

    def _handle(self, call):
        if len(call) == 0:
            return None
        if call == 'quit':
            raise QuitException()
        elif call == 'ping':
            return 'pong'

        error = False
        rval = False

        idx = call.find(' ')
        mission, command = (call[:idx] if idx >= 0 else call).split('|')
        if mission != self._mission:
            assert False, 'wrong mission for this server instance, expected %s but got %s, command was %s'%(self._mission, mission, command)

        params = []
        try:
            params = json.loads(call[idx + 1:]) if idx >= 0 else []
        except JSONDecodeError as e:
            error = 'Invalid parameters: ' + str(e) + ' "' + call[idx + 1:] + '"'

        if command == 'quit':
            raise QuitException()

        last_exception = None
        if not error:
            try:
                get_pose = re.match(r'^get_pose(\d+)$', command)
                if get_pose:
                    try:
                        rval = self._get_pose(params, algo_id=int(get_pose[1]))
                    except PositioningException as e:
                        error = 'algo failed: ' + str(e)
                elif command == 'odometry':
                    try:
                        rval = self._odometry_track(params)
                    except PositioningException as e:
                        error = str(e)
                elif command == 'render':
                    rval = self._render(params)
                elif command == 'laser_meas':
                    # return a laser measurement
                    rval = self._laser_meas(params)
                elif command == 'laser_algo':
                    # return new sc-target vector based on laser measurement
                    try:
                        rval = self._laser_algo(params)
                    except PositioningException as e:
                        error = 'algo failed: ' + str(e)
                else:
                    error = 'invalid command: ' + command
            except (ValueError, TypeError) as e:
                error = 'invalid args: ' + str(e)
                self.print(str(e) + ''.join(traceback.format_exception(*sys.exc_info())))
            except Exception as e:
                last_exception = ''.join(traceback.format_exception(*sys.exc_info()))
                self.print('Exception: %s' % last_exception)

        if last_exception is not None:
            error = 'Exception encountered: %s' % last_exception

        out = ' '.join((('0' if error else '1'),) + ((error,) if error else (rval,) if rval else tuple()))
        return out

    def listen(self):
        # main loop here
        self.print('%s on port %d' % (self.SERVER_READY_NOTICE, self._port))
        while True:
            # outer loop accepting connections (max 1)
            try:
                conn, addr = self._sock.accept()
                try:
                    with conn:
                        while True:
                            # inner loop accepting multiple requests on same connection
                            req = self._receive(conn).strip(' \n\r\t')
                            if req != '':
                                for call in req.split('\n'):
                                    # in case multiple calls in one request
                                    out = self._handle(call.strip(' \n\r\t'))
                                    if out is not None:
                                        out = out.strip(' \n\r\t') + '\n'
                                        conn.sendall(out.encode('utf-8'))
                except ConnectionAbortedError:
                    self.print('client closed the connection')
            except socket.timeout:
                pass

    def print(self, msg, start=False, finish=False):
        prefix = '%s [%d]' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self._pid)
        if start:
            print(' '.join((prefix, msg)), end='', flush=True)
        elif finish:
            print(msg)
        else:
            print(' '.join((prefix, msg)))

    def _receive(self, conn):
        chunks = []
        buffer_size = 1024
        while True:
            try:
                chunk = conn.recv(buffer_size)
                chunks.append(chunk)
                if chunk == b'':
                    if len(chunks) > 1:
                        break
                    else:
                        raise ConnectionAbortedError()
                elif len(chunk) < buffer_size:
                    break
            except socket.timeout:
                pass

        return (b''.join(chunks)).decode('utf-8')

    def close(self):
        self._sock.close()


class SpawnMaster(ApiServer):
    MAX_WAIT = 600  # in secs

    def __init__(self, addr='127.0.0.1', port=50007, max_count=2000):
        self._pid = os.getpid()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._addr = addr
        self._port = port
        self._sock.bind(('', port))
        self._sock.listen(1)
        self._sock.settimeout(5)

        self._max_count = max_count
        self._children = {}
        if which('singularity_wrapper'):
            self._python_cmds = ['singularity_wrapper', 'exec', 'xvfb-run',
                                 '--server-args="-screen 0 1x1x24"',
                                 '/scratch/work/knuutto1/conda/envs/visnav/bin/python']
        elif which('xvfb-run'):
            self._python_cmds = ['xvfb-run', '--server-args="-screen 0 1x1x24"', 'python']
        else:
            self._python_cmds = ['python']

        self._spawn_cmd = sys.argv[0]

    def _spawn(self, mission, verbose=True):
        if verbose:
            self.print('Starting api-server for %s ... ' % mission, start=True)

        if mission not in self._children:
            self._children[mission] = {
                'port': max((self._port,) + tuple(v['port'] for v in self._children.values())) + 1,
                'proc': None,
                'client': None,
                'count': 0,
            }
        for i in range(5):
            port = self._children[mission]['port']

            # spawn new api-server
            cmdarr = self._python_cmds + [self._spawn_cmd, LOG_DIR, str(port), mission]
            if verbose:
                self.print('using: ' + ' '.join(cmdarr) + ' ... ', start=True)
            try:
                self._children[mission]['proc'] = subprocess.Popen(cmdarr) #,# shell=True, close_fds=True,
                                                                #stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                                #encoding='utf8')#, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NEW_CONSOLE)
                break
            except OSError as e:
                if 'port' in str(e):
                    self.print('%s => trying next port')
                    self._children[mission]['port'] += 1
                else:
                    raise e

        time.sleep(5)
        # wait until get acknowledgement of all ready
        # success = False
        # for i in range(SpawnMaster.MAX_WAIT):
        #     try:
        #         out, err = self._children[mission]['proc'].communicate(timeout=1)
        #         if self.SERVER_READY_NOTICE in out:
        #             success = True
        #             break
        #         if len(out.strip()) > 0 and verbose:
        #             print('child process out: %s' % out)
        #         if len(err.strip()) > 0:
        #             print('child process err: %s' % err)
        #             break
        #     except TimeoutExpired:
        #         pass
        #assert success, 'Spawn for %s failed to start in %ds' % (mission, SpawnMaster.MAX_WAIT)

        # open connection to newly spawned sub-process
        self._subproc_conn(mission)
        if verbose:
            self.print('done', finish=True)

    def _subproc_conn(self, mission, max_wait=MAX_WAIT):
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._children[mission]['client'] = client
        client.settimeout(self.MAX_WAIT)
        for i in range(max_wait//5):
            try:
                client.connect((self._addr, self._children[mission]['port']))
                break
            except ConnectionRefusedError:
                time.sleep(5)
            except OSError as e:
                if 'already connected socket' in str(e):
                    break
                else:
                    raise e
        return client

    def _reset(self, mission):
        self.print('Restarting %s api-server to avoid running out of memory due to leaks ... ' % mission, start=True)
        self.send(mission, 'quit')
        time.sleep(20)
        self._children[mission]['count'] = 0
        self._spawn(mission, verbose=False)
        self.print('done', finish=True)

    def _hard_reset(self, mission):
        client = self._children[mission]['client']
        proc = self._children[mission]['proc']
        if client is not None:
            client.close()
        if proc is not None:
            proc.kill()
        time.sleep(5)

    def _shutdown(self):
        for m, d in self._children.items():
            self.send(m, 'quit')
        raise QuitException()

    def _handle(self, call):
        if call == 'quit':
            self._shutdown()
        elif call == 'ping':
            return 'pong'

        idx = call.find('|')
        mission = call[:idx]
        if mission not in self._children:
            self._spawn(mission)
        if self._children[mission]['count'] > self._max_count:
            self._reset(mission)

        response = self.send(mission, call)

        # detect if child process is out of memory, reset and try again once
        if 'Insufficient memory' in response \
                or 'MemoryError' in response \
                or 'cv::Mat::create' in response \
                or 'memory allocation failed' in response:
            self.print('Child process out of memory detected')
            self._reset(mission)
            response = self.send(mission, call)

        self._children[mission]['count'] += 1
        return response

    def send(self, mission, call):
        rec = None
        for i in range(2):
            client = self._children[mission]['client']
            try:
                if client._closed:
                    client = self._subproc_conn(mission)
                client.sendall(call.encode('utf-8'))
                rec = self._receive(client)
                break
            except (ConnectionAbortedError, socket.timeout, OSError) as err:
                if call == 'quit' and isinstance(err, ConnectionAbortedError):
                    return ''  # quit executed successfully, it doesn't return anything
                # reset and try again
                self.print('Can\'t reach %s api-server, trying hard reset' % mission)
                self._hard_reset(mission)
                if call == 'quit':
                    return ''  # if call was quit, just kill process and return
                self._spawn(mission)
        assert rec is not None, 'could not receive anything from api-server even if tried twice'
        return rec


if __name__ == '__main__':
    main()
