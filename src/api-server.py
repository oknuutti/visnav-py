import socket
import json
from datetime import datetime
from json import JSONDecodeError

import cv2
import numpy as np
import quaternion
import sys

from algo import tools
from algo.base import AlgorithmBase
from algo.model import SystemModel
from batch1 import get_system_model
from missions.didymos import DidymosPrimary
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

        self._mission = mission
        self._sm = sm = get_system_model(mission, hires)
        self._renderer = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=16 if hires else 0)
        self._renderer.set_frustum(sm.cam.x_fov, sm.cam.y_fov, sm.min_altitude*.1, sm.max_distance)
        self._obj_idx = self._renderer.load_object(
                sm.asteroid.hires_target_model_file if hires else sm.asteroid.target_model_file,
                smooth=sm.asteroid.render_smooth_faces)

    def _render(self, params):
        sun_ast_v = tools.normalize_v(np.array(params[0][:3]))
        d1_v = np.array(params[1][:3])
        d1_q = np.quaternion(*(params[1][3:7]))
        d2_v = np.array(params[2][:3])
        d2_q = np.quaternion(*(params[2][3:7]))
        sc_v = np.array(params[3][:3])
        sc_q = np.quaternion(*(params[3][3:7]))

        q = SystemModel.sc2gl_q.conj() * sc_q.conj()

        ast_q = d1_q if isinstance(self._sm.asteroid, DidymosPrimary) else d2_q
        rel_rot_q = q * ast_q

        ast_v = d1_v if isinstance(self._sm.asteroid, DidymosPrimary) else d2_v
        rel_pos_v = tools.q_times_v(q, ast_v - sc_v)*0.001

        light_v = tools.q_times_v(q, -sun_ast_v)

        img = TestLoop.render_navcam_image_static(self._sm, self._renderer, self._obj_idx,
                                                  rel_pos_v, rel_rot_q, light_v)

        path = os.path.join(LOG_DIR, 'api-server', self._mission)
        os.makedirs(path, exist_ok=True)
        fname = os.path.join(path, datetime.now().isoformat()[:-7].replace(':', '')) + '.png'
        cv2.imwrite(fname, img, [cv2.IMWRITE_PNG_COMPRESSION, 7])

        return fname

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

        last_exception = None
        for i in range(3):
            try:
                if command == 'render':
                    try:
                        rval = self._render(params)
                    except (ValueError, TypeError) as e:
                        error = 'invalid args: ' + str(e)
                else:
                    error = 'invalid command: ' + command
                    break
                ok = True
            except Exception as e:
                print('Trying to open compute engine again because of: %s' % e)
                last_exception = e
                self._reset()
                ok = False
            if ok:
                break
        if not ok and last_exception is not None:
            error = 'Exception encountered: %s' % last_exception

        out = ' '.join((('1' if error else '0'),) + ((error,) if error else (rval,) if rval else tuple()))
        return out

    def listen(self):
        # main loop here

        since_reset = 0
        while True:
            # outer loop accepting connections (max 1)
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
                                conn.sendall(out.encode('utf-8'))
                                since_reset += 1
                                if since_reset >= 1000:
                                    self._reset()
                                    since_reset = 0
            except ConnectionAbortedError:
                print('client closed the connection')


    def _receive(self, conn):
        chunks = []
        buffer_size = 1024
        while True:
            chunk = conn.recv(buffer_size)
            chunks.append(chunk)
            if chunk == b'':
                if len(chunks) > 1:
                    break
                else:
                    raise ConnectionAbortedError()
            elif len(chunk) < buffer_size:
                break

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
