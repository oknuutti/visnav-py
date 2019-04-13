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

        re = RenderEngine(sm.view_width, sm.view_height, antialias_samples=0)
        oi = re.load_object(sm.asteroid.target_model_file, smooth=sm.asteroid.render_smooth_faces)
        self._keypoint = KeypointAlgo(sm, re, oi)


    def _render(self, params):
        time = params[0]
        sun_ast_v = tools.normalize_v(np.array(params[1][:3]))
        d1_v = np.array(params[2][:3])
        d1_q = np.quaternion(*(params[2][3:7]))
        d2_v = np.array(params[3][:3])
        d2_q = np.quaternion(*(params[3][3:7]))
        sc_v = np.array(params[4][:3])
        sc_q = np.quaternion(*(params[4][3:7]))

        d1, d2 = self.asteroids
        q = SystemModel.sc2gl_q.conj() * sc_q.conj()
        rel_rot_q = np.array([q * d1_q * d1.ast2sc_q.conj(), q * d2_q * d2.ast2sc_q.conj()])
        rel_pos_v = np.array([tools.q_times_v(q, d1_v - sc_v), tools.q_times_v(q, d2_v - sc_v)]) * 0.001
        light_v = tools.q_times_v(q, sun_ast_v)

        img = TestLoop.render_navcam_image_static(self._sm, self._renderer, self._obj_idxs,
                                                  rel_pos_v, rel_rot_q, light_v)

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

        # set asteroid orientation
        TargetD2 = False
        d1, d2 = self.asteroids
        ast = d2 if TargetD2 else d1
        init_ast_q = np.quaternion(*(params[3 if TargetD2 else 2][3:7])).normalized()
        self._sm.asteroid_q = init_ast_q * ast.ast2sc_q.conj()

        # set spacecraft orientation
        init_sc_q = np.quaternion(*(params[4][3:7])).normalized()
        self._sm.spacecraft_q = init_sc_q

        # sc-asteroid relative location
        ast_v = np.array(params[3 if TargetD2 else 2][:3])*0.001   # relative to barycenter
        sc_v = np.array(params[4][:3])*0.001  # relative to barycenter
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
            rel_q = sc_q.conj() * self._sm.asteroid_q

            # sc-ast vector in meters
            rel_v = tools.q_times_v(sc_q, np.array(self._sm.spacecraft_pos)*1000)

            # collect to one result list
            result = [list(rel_v), list(quaternion.as_float_array(rel_q*ast.ast2sc_q))]

        # render a result image
        self._render_result([fname]
            + [list(np.array(self._sm.spacecraft_pos)*1000) + (result[1] if err is None else [float('nan')]*4)]
            + [list(np.array(init_sc_pos)*1000) + list(quaternion.as_float_array(init_sc_q.conj() * init_ast_q))], TargetD2)

        if err is not None:
            raise err

        # send back in json format
        return json.dumps(result)

    def _render_result(self, params, TargetD2):
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
        ast_idx = 1 if TargetD2 else 0
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
    server = ApiServer(sys.argv[1], hires=False)
    try:
        server.listen()
    finally:
        server.close()
    quit()
