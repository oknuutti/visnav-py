import math
import socket
import json
from datetime import datetime
from json import JSONDecodeError

import cv2
import numpy as np
import quaternion
import sys
import pytz

from algo import tools
from algo.base import AlgorithmBase
from algo.keypoint import KeypointAlgo
from algo.model import SystemModel
from algo.tools import PositioningException
from batch1 import get_system_model
from missions.didymos import DidymosPrimary, DidymosSystemModel, DidymosSecondary
from render.render import RenderEngine

from settings import *
from testloop import TestLoop


class ApiServer:
    class QuitException(Exception):
        pass

    def __init__(self, mission, hires=True, addr='127.0.0.1', port=50007):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._addr = addr
        self._port = port
        self._sock.bind(('', port))
        self._sock.listen(1)
        self._sock.settimeout(5)

        self._mission = mission
        self._sm = sm = get_system_model(mission, hires)
        self._target_d2 = '2' in mission
        self._renderer = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=16 if hires else 0)
        self._renderer.set_frustum(sm.cam.x_fov, sm.cam.y_fov, sm.min_altitude*.1, sm.max_distance)
        if isinstance(sm, DidymosSystemModel):
            self.asteroids = [
                DidymosPrimary(hi_res_shape_model=hires),
                DidymosSecondary(hi_res_shape_model=hires),
            ]
        else:
            self.asteroids = [sm.asteroid]

        self._obj_idxs = [
            self._renderer.load_object(
                ast.hires_target_model_file if hires else ast.target_model_file,
                smooth=ast.render_smooth_faces)
            for ast in self.asteroids]

        self._wireframe_obj_idxs = [
            self._renderer.load_object(os.path.join(BASE_DIR, 'data/ryugu+tex-%s-100.obj'%ast), wireframe=True)
            for ast in ('d1', 'd2')]

        self._logpath = os.path.join(LOG_DIR, 'api-server', self._mission)
        os.makedirs(self._logpath, exist_ok=True)

        self._onboard_renderer = RenderEngine(sm.view_width, sm.view_height, antialias_samples=0)
        self._target_obj_idx = self._onboard_renderer.load_object(sm.asteroid.target_model_file, smooth=sm.asteroid.render_smooth_faces)
        self._keypoint = KeypointAlgo(sm, self._onboard_renderer, self._target_obj_idx)

        # laser measurement range given in dlem-20 datasheet
        self._laser_min_range, self._laser_max_range, self._laser_nominal_max_range = 10, 1200, 5000
        self._laser_fail_prob = 0.05   # probability of measurement fail because e.g. asteroid low albedo
        self._laser_false_prob = 0.01  # probability for random measurement even if no target
        self._laser_err_sigma = 0.5    # 0.5m sigma1 accuracy given in dlem-20 datasheet

        # laser algo params
        self._laser_adj_loc_weight = 1
        self._laser_meas_err_weight = 0.3
        self._laser_max_adj_dist = 100  # in meters

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

        dist = self._renderer.ray_intersect_dist(self._obj_idxs, rel_pos_v, rel_rot_q)

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

        # TODO: check nav filter init state

        # mse cost function balances between adjusted location and measurement error
        adj_dist_sqr = (zz - dist_meas)**2 + xx**2 + yy**2
        cost = self._laser_adj_loc_weight * adj_dist_sqr \
             + (self._laser_meas_err_weight - self._laser_adj_loc_weight) * (zz - dist_meas)**2

        j, i = np.unravel_index(np.nanargmin(cost), cost.shape)
        if np.isnan(zz[j, i]):
            raise PositioningException('laser algo results in off asteroid pointing')
        if math.sqrt(adj_dist_sqr[j, i]) >= self._laser_max_adj_dist:
            # TODO: check that can handle spurious measurements
            raise PositioningException('laser algo solution too far (%.0fm, limit=%.0fm), spurious measurement assumed'
                                       % (math.sqrt(adj_dist_sqr[j, i]), self._laser_max_adj_dist))

        dx, dy, dz = xx[0, i], yy[j, 0], zz[j, i] - dist_meas
        est_sc_v = ast_v*1000 - tools.q_times_v(q.conj(), rel_pos_v + np.array([dx, dy, dz]))
        dist_expected = float(dist_expected) if not np.isnan(dist_expected) else -1.0
        return json.dumps([list(est_sc_v), dist_expected])

    def _render(self, params):
        time = params[0]
        sun_ast_v = tools.normalize_v(np.array(params[1][:3]))
        d1_v, d1_q, d2_v, d2_q, sc_v, sc_q = self._parse_poses(params, offset=2)

        d1, d2 = self.asteroids
        q = SystemModel.sc2gl_q.conj() * sc_q.conj()
        rel_rot_q = np.array([q * d1_q * d1.ast2sc_q.conj(), q * d2_q * d2.ast2sc_q.conj()])
        rel_pos_v = np.array([tools.q_times_v(q, d1_v - sc_v), tools.q_times_v(q, d2_v - sc_v)])
        light_v = tools.q_times_v(q, sun_ast_v)

        img = TestLoop.render_navcam_image_static(self._sm, self._renderer, self._obj_idxs, rel_pos_v, rel_rot_q, light_v,
                                                  use_shadows=True, use_textures=True)

        date = datetime.fromtimestamp(time, pytz.utc)  # datetime.now()
        fname = os.path.join(self._logpath, date.isoformat()[:-6].replace(':', '')) + '.png'
        cv2.imwrite(fname, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        return fname

    def _get_pose(self, params):
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

        # sc-asteroid relative location
        ast_v = d2_v if self._target_d2 else d1_v   # relative to barycenter
        init_sc_pos = tools.q_times_v(SystemModel.sc2gl_q.conj() * init_sc_q.conj(), ast_v - sc_v)
        self._sm.spacecraft_pos = init_sc_pos

        # run keypoint algo
        err = None
        try:
            self._keypoint.solve_pnp(img, fname[:-4], KeypointAlgo.AKAZE)
        except PositioningException as e:
            err = e

        if err is None:
            # resulting sc-ast relative orientation
            sc_q = self._sm.spacecraft_q
            rel_q = sc_q.conj() * self._sm.asteroid_q * ast.ast2sc_q

            # sc-ast vector in meters
            rel_v = tools.q_times_v(sc_q * SystemModel.sc2gl_q, np.array(self._sm.spacecraft_pos)*1000)

            # collect to one result list
            result = [list(rel_v), list(quaternion.as_float_array(rel_q))]

        # render a result image
        self._render_result([fname]
            + [list(np.array(self._sm.spacecraft_pos)*1000) + (result[1] if err is None else [float('nan')]*4)]
            + [list(np.array(init_sc_pos)*1000) + list(quaternion.as_float_array(init_sc_q.conj() * init_ast_q))])

        if err is not None:
            raise err

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
        cv2.imwrite(fout, result, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        return fout

    def _handle(self, call):
        if len(call) == 0:
            return None

        error = False
        rval = False

        idx = call.find(' ')
        command = call[:idx] if idx >= 0 else call
        params = []
        try:
            params = json.loads(call[idx + 1:]) if idx >= 0 else []
        except JSONDecodeError as e:
            error = 'Invalid parameters: ' + str(e) + ' "' + call[idx + 1:] + '"'

        if command == 'quit':
            raise self.QuitException()

        ok = False
        last_exception = None

        if not error:
            for i in range(3):
                try:
                    if command == 'render':
                        rval = self._render(params)
                        ok = True
                    elif command == 'get_pose':
                        try:
                            rval = self._get_pose(params)
                            ok = True
                        except PositioningException as e:
                            error = 'algo failed: ' + str(e)
                    elif command == 'laser_meas':
                        # return a laser measurement
                        rval = self._laser_meas(params)
                        ok = True
                    elif command == 'laser_algo':
                        # return new sc-target vector based on laser measurement
                        try:
                            rval = self._laser_algo(params)
                            ok = True
                        except PositioningException as e:
                            error = 'algo failed: ' + str(e)
                        pass
                    else:
                        error = 'invalid command: ' + command
                        break
                except (ValueError, TypeError) as e:
                    error = 'invalid args: ' + str(e)
                    break
                except Exception as e:
                    print('Trying to open compute engine again because of: %s' % e)
                    last_exception = e
                    self._reset()
                if ok:
                    break

        if not ok and last_exception is not None:
            error = 'Exception encountered: %s' % last_exception

        out = ' '.join((('0' if error else '1'),) + ((error,) if error else (rval,) if rval else tuple()))
        return out

    def listen(self):
        # main loop here
        print('server started, waiting for connections')
        since_reset = 0
        while True:
            # outer loop accepting connections (max 1)
            try:
                conn, addr = self._sock.accept()
                try:
                    with conn:
                        while True:
                            # inner loop accepting multiple requests on same connection
                            req = self._receive(conn)
                            for call in req.strip(' \n\r\t').split('\n'):
                                # in case multiple calls in one request
                                out = self._handle(call.strip(' \n\r\t'))
                                if out is not None:
                                    out = out.strip(' \n\r\t') + '\n'
                                    conn.sendall(out.encode('utf-8'))
                                    since_reset += 1
                                    if since_reset >= 1000:
                                        self._reset()
                                        since_reset = 0
                except ConnectionAbortedError:
                    print('client closed the connection')
            except socket.timeout:
                pass


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

    def _reset(self):
        print('Restarting computing engine to avoid memory leak [NOT IMPLEMENTED]')
        # CloseComputeEngine("localhost", "")
        # OpenComputeEngine("localhost", ("-l", "srun", "-np", "1"))
        # RestoreSession("data/default-visit.session", 0)


if __name__ == '__main__':
    server = ApiServer(sys.argv[1], hires=True)
    try:
        server.listen()
    finally:
        server.close()
    quit()
