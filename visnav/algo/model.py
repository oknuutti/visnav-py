import pickle
import sys
import math
from abc import ABC
from functools import lru_cache
from math import degrees as deg, radians as rad

import numpy as np
import quaternion # adds to numpy
from astropy.time import Time
from astropy import constants as const
from astropy import units
from astropy.coordinates import SkyCoord, spherical_to_cartesian
import scipy.integrate as integrate
import configparser

from visnav.algo.image import ImageProc
from visnav.iotools import objloader
from visnav.settings import *
from visnav.algo import tools


class Parameter():
    def __init__(self, min_val, max_val, def_val=None, estimate=True, is_gl_z=False):
        self._min_val = min_val
        self._max_val = max_val
        self.estimate = estimate
        self._def_val = def_val
        self._value = self.def_val
        self.is_gl_z = is_gl_z
        self.real_value = None
        self.change_callback = None
        self.fire_change_events = True
        self.debug = False
    
    @property
    def range(self):
        return (self._min_val, self._max_val)

    @range.setter
    def range(self, range):
        min_val, max_val = range
        self._min_val = min_val
        self._max_val = max_val

        # NOTE: need fine rtol as time is in seconds (e.g. 1407258438)
        if not np.isclose(self._min_val, min_val, rtol=1e-9) \
                or not np.isclose(self._max_val, max_val, rtol=1e-9):
            if self.fire_change_events:
                try:
                    self.change_callback(self._value, self._min_val, self._max_val)
                except TypeError:
                    pass

    @property
    def scale(self):
        return abs(self._max_val - self._min_val)
    
    @property
    def def_val(self):
        return (self._min_val + self._max_val)/2 \
                if self._def_val is None \
                else self._def_val

    @def_val.setter
    def def_val(self, def_val):
        self._def_val = def_val

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        if self.debug:
            print('o: %s, n: %s'%(self._value, value), flush=True)

        # NOTE: need fine rtol as time is in seconds (e.g. 1407258438)
        if not np.isclose(self._value, value, rtol=1e-9):
            if self.debug:
                print('value set: %s'%self._value, flush=True)
            if self.fire_change_events:
                try:
                    self.change_callback(value)
                except TypeError:
                    pass
    
    @property
    def nvalue(self):
        if self.is_gl_z:
            scale = abs(1/self._min_val - 1/self._max_val)
            offset = (1/self._min_val + 1/self._max_val)/2
            return (-1/(self._value or 1e-6) + offset)/scale
        return (self._value - self.def_val)/self.scale
    
    @nvalue.setter
    def nvalue(self, nvalue):
        if self.is_gl_z:
            scale = abs(1/self._min_val - 1/self._max_val)
            offset = (1/self._min_val + 1/self._max_val)/2
            self.value = -1/((nvalue or 1e-6)*scale - offset)
        else:
            self.value = nvalue*self.scale + self.def_val
    
    def valid(self):
        return self._value >= self._min_val and self._value < self._max_val
    
    def __str__(self):
        return '%.2f (%.2f) in [%.2f, %.2f]' % (
            self._value,
            self.real_value if self.real_value is not None else float('nan'),
            self._min_val,
            self._max_val,
        )
    

class SystemModel(ABC):
    (
        OPENGL_FRAME,
        SPACECRAFT_FRAME,
        ASTEROID_FRAME,
        OPENCV_FRAME,
    ) = range(4)
    
    # from sc cam frame (axis: +x, up: +z) to opengl (axis -z, up: +y)
    sc2gl_q = np.quaternion(0.5, 0.5, -0.5, -0.5)

    # from opencv cam frame (axis: +z, up: -y) to opengl (axis -z, up: +y)
    cv2gl_q = np.quaternion(0, 1, 0, 0)
    
    def __init__(self, asteroid, camera, limits, *args, **kwargs):
        super(SystemModel, self).__init__()

        self.asteroid = asteroid
        self.cam = camera

        # mission limits
        (
            self.min_distance,      # min_distance in km
            self.min_med_distance,  # min_med_distance in km
            self.max_med_distance,  # max_med_distance in km
            self.max_distance,      # max_distance in km
            self.min_elong,         # min_elong in deg
            self.min_time           # min time instant as astropy.time.Time
        ) = limits

        assert self.min_altitude > 0, \
            'min distance %.2fkm too small, possible collision as asteroid max_radius=%.0fm'%(self.min_distance, self.asteroid.max_radius)

        self.mission_id = None      # overridden by particular missions
        self.view_width = VIEW_WIDTH

        # spacecraft position relative to asteroid, z towards spacecraft,
        #   x towards right when looking out from s/c camera, y up
        self.x_off = Parameter(-4, 4, estimate=False)
        self.y_off = Parameter(-4, 4, estimate=False)
        
        # whole view: 1.65km/tan(2.5deg) = 38km
        # can span ~30px: 1.65km/tan(2.5deg * 30/1024) = 1290km
        self.z_off = Parameter(-self.max_distance, -self.min_distance, def_val=-self.min_med_distance, is_gl_z=True) # was 120, 220

        # spacecraft orientation relative to stars
        self.x_rot = Parameter(-90, 90, estimate=False) # axis latitude
        self.y_rot = Parameter(-180, 180, estimate=False) # axis longitude
        self.z_rot = Parameter(-180, 180, estimate=False) # rotation

        # asteroid zero orientation relative to stars
        self.ast_x_rot = Parameter(-90, 90, estimate=False) # axis latitude
        self.ast_y_rot = Parameter(-180, 180, estimate=False) # axis longitude
        self.ast_z_rot = Parameter(-180, 180, estimate=False) # rotation
        self.asteroid_rotation_from_model()

        # time in seconds since 1970-01-01 00:00:00
        self.time = Parameter(
            self.min_time.unix,
            self.min_time.unix + self.asteroid.rotation_period,
            estimate=False
        )
        
        # override any default params
        for n, v in kwargs.items():
            setattr(self, n, v)
        
        # set default values to params
        for n, p in self.get_params():
            p.value = p.def_val


    @property
    def min_altitude(self):
        """ in km """
        return self.min_distance - self.asteroid.max_radius/1000

    @property
    def view_height(self):
        return int(self.cam.height * self.view_width/self.cam.width)

    def get_params(self, all=False):
        return (
            (n, getattr(self, n))
            for n in sorted(self.__dict__)
            if isinstance(getattr(self, n), Parameter)
                and (all or getattr(self, n).estimate)
        )
        
    def param_change_events(self, enabled):
        for n, p in self.get_params(all=True):
            p.fire_change_events = enabled
    
    @property
    def spacecraft_pos(self):
        return self.x_off.value, self.y_off.value, self.z_off.value

    @spacecraft_pos.setter
    def spacecraft_pos(self, pos):
        self.z_off.value = pos[2]

        half_range = abs(pos[2] / 170 * 4)
        self.x_off.range = (pos[0] - half_range, pos[0] + half_range)
        self.x_off.value = pos[0]

        self.y_off.range = (pos[1] - half_range, pos[1] + half_range)
        self.y_off.value = pos[1]

    @property
    def spacecraft_rot(self):
        return self.x_rot.value, self.y_rot.value, self.z_rot.value

    @spacecraft_rot.setter
    def spacecraft_rot(self, r):
        self.x_rot.value, self.y_rot.value, self.z_rot.value = r

    @property
    def asteroid_axis(self):
        return self.ast_x_rot.value, self.ast_y_rot.value, self.ast_z_rot.value
    
    @asteroid_axis.setter
    def asteroid_axis(self, r):
        self.ast_x_rot.value, self.ast_y_rot.value, self.ast_z_rot.value = r
        self.update_asteroid_model()

    @property
    def spacecraft_dist(self):
        return math.sqrt(sum(x**2 for x in self.spacecraft_pos))

    @property
    def spacecraft_altitude(self):
        sc_ast_v = tools.normalize_v(np.array(self.spacecraft_pos))
        ast_vx = self.sc_asteroid_vertices()
        min_distance = np.min(sc_ast_v.dot(ast_vx.T))
        return min_distance

    @property
    def real_spacecraft_altitude(self):
        sc_ast_v = tools.normalize_v(np.array(self.real_spacecraft_pos))
        ast_vx = self.sc_asteroid_vertices(real=True)
        min_distance = np.min(sc_ast_v.dot(ast_vx.T))
        return min_distance

    def asteroid_rotation_from_model(self):
        self.ast_x_rot.value = math.degrees(self.asteroid.axis_latitude)
        self.ast_y_rot.value = math.degrees(self.asteroid.axis_longitude)
        self.ast_z_rot.value = (math.degrees(self.asteroid.rotation_pm) + 180) % 360 - 180

    def update_asteroid_model(self):
        self.asteroid.axis_latitude = math.radians(self.ast_x_rot.value)
        self.asteroid.axis_longitude = math.radians(self.ast_y_rot.value)
        self.asteroid.rotation_pm = math.radians(self.ast_z_rot.value)

    def pixel_extent(self, distance=None):
        distance = abs(self.z_off) if distance is None else distance
        return self.cam.width * math.atan(self.asteroid.mean_radius/1000/distance)*2 / math.radians(self.cam.x_fov)

    @property
    def real_spacecraft_pos(self):
        return self.x_off.real_value, self.y_off.real_value, self.z_off.real_value
    
    @real_spacecraft_pos.setter
    def real_spacecraft_pos(self, rv):
        self.x_off.real_value, self.y_off.real_value, self.z_off.real_value = rv

    @property
    def real_spacecraft_rot(self):
        return self.x_rot.real_value, self.y_rot.real_value, self.z_rot.real_value
    
    @real_spacecraft_rot.setter
    def real_spacecraft_rot(self, rv):
        self.x_rot.real_value, self.y_rot.real_value, self.z_rot.real_value = rv

    @property
    def real_asteroid_axis(self):
        return self.ast_x_rot.real_value, self.ast_y_rot.real_value, self.ast_z_rot.real_value
    
    @real_asteroid_axis.setter
    def real_asteroid_axis(self, rv):
        self.ast_x_rot.real_value, self.ast_y_rot.real_value, self.ast_z_rot.real_value = rv

    @property
    def spacecraft_q(self):
        return tools.ypr_to_q(*list(map(
                math.radians,
                (self.x_rot.value, self.y_rot.value, self.z_rot.value)
        )))

    @spacecraft_q.setter
    def spacecraft_q(self, new_q):
        self.x_rot.value, self.y_rot.value, self.z_rot.value = \
            list(map(math.degrees, tools.q_to_ypr(new_q)))

    @property
    def real_spacecraft_q(self):
        return tools.ypr_to_q(*list(map(
                math.radians,
                (self.x_rot.real_value, self.y_rot.real_value, self.z_rot.real_value)
        )))

    @real_spacecraft_q.setter
    def real_spacecraft_q(self, new_q):
        self.x_rot.real_value, self.y_rot.real_value, self.z_rot.real_value = \
            list(map(math.degrees, tools.q_to_ypr(new_q)))

    @property
    def asteroid_q(self):
        return self.asteroid.rotation_q(self.time.value)

    @asteroid_q.setter
    def asteroid_q(self, new_q):
        ast = self.asteroid
        sc2ast_q = SystemModel.frm_conv_q(SystemModel.SPACECRAFT_FRAME, SystemModel.ASTEROID_FRAME, ast=ast)

        ast.axis_latitude, ast.axis_longitude, new_theta = tools.q_to_ypr(new_q * sc2ast_q)

        old_theta = ast.rotation_theta(self.time.value)
        ast.rotation_pm = tools.wrap_rads(ast.rotation_pm + new_theta - old_theta)

        self.asteroid_rotation_from_model()

    @property
    def real_asteroid_q(self):
        org_ast_axis = self.asteroid_axis
        self.asteroid_axis = self.real_asteroid_axis
        q = self.asteroid.rotation_q(self.time.real_value)
        self.asteroid_axis = org_ast_axis
        return q

    @real_asteroid_q.setter
    def real_asteroid_q(self, new_q):
        org_ast_axis = self.asteroid_axis
        self.asteroid_axis = self.real_asteroid_axis
        self.asteroid_q = new_q
        self.asteroid_axis = org_ast_axis

    def gl_sc_asteroid_rel_q(self, discretize_tol=False):
        """ rotation of asteroid relative to spacecraft in opengl coords """
        assert not discretize_tol, 'discretize_tol deprecated at gl_sc_asteroid_rel_q function'
        self.update_asteroid_model()
        sc_ast_rel_q = SystemModel.sc2gl_q.conj() * self.sc_asteroid_rel_q()

        if discretize_tol:
            qq, _ = tools.discretize_q(sc_ast_rel_q, discretize_tol)
            err_q = sc_ast_rel_q * qq.conj()
            sc_ast_rel_q = qq
        
        if not BATCH_MODE and DEBUG:
            print('asteroid x-axis: %s'%tools.q_times_v(sc_ast_rel_q, np.array([1, 0, 0])))
        
        return sc_ast_rel_q, err_q if discretize_tol else False
    
    
    def sc_asteroid_rel_q(self, time=None):
        """ rotation of asteroid relative to spacecraft in spacecraft coords """
        ast_q = self.asteroid.rotation_q(time or self.time.value)
        sc_q = self.spacecraft_q
        return sc_q.conj() * ast_q


    def real_sc_asteroid_rel_q(self):
        org_sc_rot = self.spacecraft_rot
        org_ast_axis = self.asteroid_axis
        self.spacecraft_rot = self.real_spacecraft_rot
        self.asteroid_axis = self.real_asteroid_axis
        
        q_tot = self.sc_asteroid_rel_q(time=self.time.real_value)
        
        self.spacecraft_rot = org_sc_rot
        self.asteroid_axis = org_ast_axis
        return q_tot

    def rotate_spacecraft(self, q):
        new_q = self.spacecraft_q * q
        self.x_rot.value, self.y_rot.value, self.z_rot.value = \
            list(map(math.degrees, tools.q_to_ypr(new_q)))

    def rotate_asteroid(self, q):
        """ rotate asteroid in spacecraft frame """
        # global rotation q on asteroid in sc frame, followed by local rotation to asteroid frame
        new_q = q * self.asteroid.rotation_q(self.time.value)
        self.asteroid_q = new_q

    def reset_to_real_vals(self):
        for n, p in self.get_params(True):
            assert p.real_value is not None, 'real value missing for %s'%n
            p.value = p.real_value


    def swap_values_with_real_vals(self):
        for n, p in self.get_params(True):
            assert p.real_value is not None, 'real value missing for %s'%n
            assert p.value is not None, 'current value missing %s'%n
            tmp = p.value
            p.value = p.real_value
            p.real_value = tmp

    def calc_shift_err(self):
        est_vertices = self.sc_asteroid_vertices()
        self.swap_values_with_real_vals()
        target_vertices = self.sc_asteroid_vertices()
        self.swap_values_with_real_vals()
        return tools.sc_asteroid_max_shift_error(est_vertices, target_vertices)
    
    
    def sc_asteroid_vertices(self, real=False):
        """ asteroid vertices rotated and translated to spacecraft frame """
        if self.asteroid.real_shape_model is None:
            return None

        sc_ast_q = self.real_sc_asteroid_rel_q() if real else self.sc_asteroid_rel_q()
        sc_pos = self.real_spacecraft_pos if real else self.spacecraft_pos
        
        return tools.q_times_mx(sc_ast_q, np.array(self.asteroid.real_shape_model.vertices)) + sc_pos
    
    def gl_light_rel_dir(self, err_q=False, discretize_tol=False):
        """ direction of light relative to spacecraft in opengl coords """
        assert not discretize_tol, 'discretize_tol deprecated at gl_light_rel_dir function'

        light_v, err_angle = self.light_rel_dir(err_q=False, discretize_tol=False)

        err_q = (err_q or np.quaternion(1, 0, 0, 0))
        light_gl_v = tools.q_times_v(err_q.conj() * SystemModel.sc2gl_q.conj(), light_v)

        # new way to discretize light, consistent with real fdb inplementation
        if discretize_tol:
            dlv, _ = tools.discretize_v(light_gl_v, discretize_tol, lat_range=(-math.pi/2, math.radians(90 - self.min_elong)))
            err_angle = tools.angle_between_v(light_gl_v, dlv)
            light_gl_v = dlv

        return light_gl_v, err_angle

    def light_rel_dir(self, err_q=False, discretize_tol=False):
        """ direction of light relative to spacecraft in s/c coords """
        assert not discretize_tol, 'discretize_tol deprecated at light_rel_dir function'

        light_v = tools.normalize_v(self.asteroid.position(self.time.value))
        sc_q = self.spacecraft_q
        err_q = (err_q or np.quaternion(1, 0, 0, 0))

        # old, better way to discretize light, based on asteroid rotation axis, now not in use
        if discretize_tol:
            ast_q = self.asteroid_q
            light_ast_v = tools.q_times_v(ast_q.conj(), light_v)
            dlv, _ = tools.discretize_v(light_ast_v, discretize_tol)
            err_angle = tools.angle_between_v(light_ast_v, dlv)
            light_v = tools.q_times_v(ast_q, dlv)

        return tools.q_times_v(err_q.conj() * sc_q.conj(), light_v),\
               err_angle if discretize_tol else False

    def solar_elongation(self, real=False):
        ast_v = self.asteroid.position(self.time.real_value if real else self.time.value)
        sc_q = self.real_spacecraft_q if real else self.spacecraft_q
        elong, direc = tools.solar_elongation(ast_v, sc_q)
        if not BATCH_MODE and DEBUG:
            print('elong: %.3f | dir: %.3f' % (
                math.degrees(elong), math.degrees(direc)))
        return elong, direc
    
    
    def rel_rot_err(self):
        return tools.angle_between_q(
            self.sc_asteroid_rel_q(),
            self.real_sc_asteroid_rel_q())
    
    def lat_pos_err(self):
        real_pos = self.real_spacecraft_pos
        err = np.subtract(self.spacecraft_pos, real_pos)
        return math.sqrt(err[0]**2 + err[1]**2) / abs(real_pos[2])

    def dist_pos_err(self):
        real_d = self.real_spacecraft_pos[2]
        return abs(self.spacecraft_pos[2] - real_d) / abs(real_d)

    def calc_visibility(self, pos=None):
        if pos is None:
            pos = self.spacecraft_pos

        if isinstance(pos, np.ndarray):
            pos = pos.reshape((-1, 3))
            return_array = True
        else:
            pos = np.array([pos], shape=(1, 3))
            return_array = False

        rad = self.asteroid.mean_radius * 0.001
        xt = np.abs(pos[:, 2]) * math.tan(math.radians(self.cam.x_fov) / 2)
        yt = np.abs(pos[:, 2]) * math.tan(math.radians(self.cam.y_fov) / 2)

        # xm = np.clip((xt - (abs(pos[0])-rad))/rad/2, 0, 1)
        # ym = np.clip((yt - (abs(pos[1])-rad))/rad/2, 0, 1)
        xm = 1 - np.minimum(1, (np.maximum(0, pos[:, 0] + rad - xt) + np.maximum(0, rad - pos[:, 0] - xt)) / rad / 2)
        ym = 1 - np.minimum(1, (np.maximum(0, pos[:, 1] + rad - yt) + np.maximum(0, rad - pos[:, 1] - yt)) / rad / 2)
        visib = xm * ym * 100

        return visib if return_array else visib[0]

    def get_cropped_system_scf(self, x, y, w, h):
        sc_ast_lf_r = tools.q_times_v(SystemModel.sc2gl_q, self.spacecraft_pos)
        sc_ast_lf_q = self.spacecraft_q.conj() * self.asteroid.rotation_q(self.time.value)
        ast_sun_lf_u = tools.q_times_v(self.spacecraft_q.conj(), -tools.normalize_v(self.asteroid.position(self.time.value)))
        sc, dq = self.cropped_system_tf(x, y, w, h)

        # adjust position
        sc_ast_lf_r = tools.q_times_v(dq.conj(), sc_ast_lf_r)
        sc_ast_lf_r[0] *= sc

        # adjust rotation
        sc_ast_lf_q = dq.conj() * sc_ast_lf_q

        # adjust sun vect
        ast_sun_lf_u = tools.q_times_v(dq.conj(), ast_sun_lf_u)

        return sc_ast_lf_r, sc_ast_lf_q, ast_sun_lf_u

    def set_cropped_system_scf(self, x, y, w, h, sc_ast_lf_r, sc_ast_lf_q, rotate_sc=False):
        sc, dq = self.cropped_system_tf(x, y, w, h)

        # adjust and set position
        sc_ast_lf_r[0] /= sc
        self.spacecraft_pos = tools.q_times_v(SystemModel.sc2gl_q.conj() * dq, sc_ast_lf_r)

        # adjust and set rotation
        if rotate_sc:
            self.spacecraft_q = self.asteroid_q * sc_ast_lf_q.conj()     # TODO: check that valid
        else:
            self.asteroid_q = self.spacecraft_q * sc_ast_lf_q

    def cropped_system_tf(self, x, y, w, h):
        # for rotation adjustment
        if False:
            # px are on a plane
            dx = math.atan(
                ((x + w / 2) - self.cam.width // 2) / (self.cam.width / 2) * math.tan(math.radians(sm.cam.x_fov / 2)))
            dy = math.atan(
                ((y + h / 2) - self.cam.height // 2) / (self.cam.height / 2) * math.tan(math.radians(sm.cam.y_fov / 2)))
        else:
            # px are on a curved surface
            dx = ((x + w / 2) - self.cam.width // 2) / self.cam.width * math.radians(self.cam.x_fov)
            dy = ((y + h / 2) - self.cam.height // 2) / self.cam.height * math.radians(self.cam.y_fov)
        dq = tools.ypr_to_q(-dy, -dx, 0)

        # for distance adjustment
        sc = max(h / self.cam.height, w / self.cam.width)
        return sc, dq

    def random_state(self, uniform_distance=True, opzone_only=False):
        # reset asteroid axis to true values
        self.asteroid.reset_to_defaults()
        self.asteroid_rotation_from_model()

        for i in range(100):
            ## sample params from suitable distributions
            ##
            # datetime dist: uniform, based on rotation period
            time = np.random.uniform(*self.time.range)

            # spacecraft position relative to asteroid in ecliptic coords:
            sc_lat = np.random.uniform(-math.pi / 2, math.pi / 2)
            sc_lon = np.random.uniform(-math.pi, math.pi)

            # s/c distance as inverse uniform distribution
            if uniform_distance:
                sc_r = np.random.uniform(self.min_distance, self.max_distance)
            else:
                sc_r = 1 / np.random.uniform(1 / self.max_distance, 1 / self.min_distance)

            # same in cartesian coord
            sc_ex_u, sc_ey_u, sc_ez_u = spherical_to_cartesian(sc_r, sc_lat, sc_lon)
            sc_ex, sc_ey, sc_ez = sc_ex_u.value, sc_ey_u.value, sc_ez_u.value

            # s/c to asteroid vector
            sc_ast_v = -np.array([sc_ex, sc_ey, sc_ez])

            # sc orientation: uniform, center of asteroid at edge of screen
            if opzone_only:
                # always get at least 50% of astroid in view, 5% of the time maximum offset angle
                max_angle = rad(min(self.cam.x_fov, self.cam.y_fov) / 2)
                da = min(max_angle, np.abs(np.random.normal(0, max_angle / 2)))
                dd = np.random.uniform(0, 2 * math.pi)
                sco_lat = tools.wrap_rads(-sc_lat + da * math.sin(dd))
                sco_lon = tools.wrap_rads(math.pi + sc_lon + da * math.cos(dd))
                sco_rot = np.random.uniform(-math.pi, math.pi)  # rotation around camera axis
            else:
                # follows the screen edges so that get more partial views, always at least 25% in view
                # TODO: add/subtract some margin
                sco_lat = tools.wrap_rads(-sc_lat)
                sco_lon = tools.wrap_rads(math.pi + sc_lon)
                sco_rot = np.random.uniform(-math.pi, math.pi)  # rotation around camera axis
                sco_q = tools.ypr_to_q(sco_lat, sco_lon, sco_rot)

                ast_ang_r = math.atan(
                    self.asteroid.mean_radius / 1000 / sc_r)  # if asteroid close, allow s/c to look at limb
                dx = max(rad(self.cam.x_fov / 2), ast_ang_r)
                dy = max(rad(self.cam.y_fov / 2), ast_ang_r)
                disturbance_q = tools.ypr_to_q(np.random.uniform(-dy, dy), np.random.uniform(-dx, dx), 0)
                sco_lat, sco_lon, sco_rot = tools.q_to_ypr(sco_q * disturbance_q)

            sco_q = tools.ypr_to_q(sco_lat, sco_lon, sco_rot)

            # sc_ast_p ecliptic => sc_ast_p open gl -z aligned view
            sc_pos = tools.q_times_v((sco_q * self.sc2gl_q).conj(), sc_ast_v)

            # get asteroid position so that know where sun is
            # *actually barycenter, not sun
            as_v = self.asteroid.position(time)
            elong, direc = tools.solar_elongation(as_v, sco_q)

            # limit elongation to always be more than set elong
            if elong > rad(self.min_elong):
                break

        if elong <= rad(self.min_elong):
            assert False, 'probable infinite loop'

        # put real values to model
        self.time.value = time
        self.spacecraft_pos = sc_pos
        self.spacecraft_rot = (deg(sco_lat), deg(sco_lon), deg(sco_rot))

        # save real values so that can compare later
        self.time.real_value = self.time.value
        self.real_spacecraft_pos = self.spacecraft_pos
        self.real_spacecraft_rot = self.spacecraft_rot
        self.real_asteroid_axis = self.asteroid_axis

        # get real relative position of asteroid model vertices
        self.asteroid.real_sc_ast_vertices = self.sc_asteroid_vertices()

    def export_state(self, filename):
        """ saves state in an easy to access format """

        qn = ('w', 'x', 'y', 'z')
        vn = ('x', 'y', 'z')
        lines = [['type'] + ['ast_q' + i for i in qn] + ['sc_q' + i for i in qn]
                 + ['ast_sc_v' + i for i in vn] + ['sun_ast_v' + i for i in vn]]

        for t in ('initial', 'real'):
            # if settings.USE_ICRS, all in solar system barycentric equatorial frame
            ast_q = self.asteroid.rotation_q(self.time.value)
            sc_q = self.spacecraft_q
            ast_sc_v = tools.q_times_v(sc_q, self.spacecraft_pos)
            sun_ast_v = self.asteroid.position(self.time.value)

            lines.append((t,) + tuple('%f'%f for f in (tuple(ast_q.components) + tuple(sc_q.components)
                      + tuple(ast_sc_v) + tuple(sun_ast_v))))
            self.swap_values_with_real_vals()

        with open(filename, 'w') as f:
            f.write('\n'.join(['\t'.join(l) for l in lines]))


    def save_state(self, filename, printout=False):
        config = configparser.ConfigParser()
        filename = filename+('.lbl' if len(filename)<5 or filename[-4:]!='.lbl' else '')
        config.read(filename)
        if not config.has_section('main'):
            config.add_section('main')
        if not config.has_section('real'):
            config.add_section('real')
        
        for n, p in self.get_params(all=True):
            config.set('main', n, str(p.value))
            if p.real_value is not None:
                config.set('real', n, str(p.real_value))

        if self.asteroid.real_position is not None:
            config.set('real', 'sun_asteroid_pos', str(self.asteroid.real_position))

        if not printout:
            with open(filename, 'w') as f:
                config.write(f)
        else:
            config.write(sys.stdout)
    
    def load_state(self, filename, sc_ast_vertices=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)

        self.asteroid.reset_to_defaults()

        config = configparser.ConfigParser()
        filename = filename+('.lbl' if len(filename)<5 or filename[-4:]!='.lbl' else '')
        config.read(filename)
        
        for n, p in self.get_params(all=True):
            v = float(config.get('main', n))
            if n == 'time':
                rp = self.asteroid.rotation_period
                p.range = (v-rp/2, v+rp/2)
            p.value = v
            
            rv = config.get('real', n, fallback=None)
            if rv is not None:
                p.real_value = float(rv)
        
        rv = config.get('real', 'sun_asteroid_pos', fallback=None)
        if rv is not None:
            self.asteroid.real_position = np.fromstring(rv[1:-1], dtype=np.float, sep=' ')
        
        assert np.isclose(self.time.value, float(config.get('main', 'time'))), \
               'Failed to set time value: %s vs %s'%(self.time.value, float(config.get('main', 'time')))
               
        self.update_asteroid_model()
        
        if sc_ast_vertices:
            # get real relative position of asteroid model vertices
            self.asteroid.real_sc_ast_vertices = self.sc_asteroid_vertices(real=True)
    
    @staticmethod
    def frm_conv_q(fsrc, fdst, ast=None):
        fqm = {
            SystemModel.OPENGL_FRAME:np.quaternion(1,0,0,0),
            SystemModel.OPENCV_FRAME:SystemModel.cv2gl_q,
            SystemModel.SPACECRAFT_FRAME:SystemModel.sc2gl_q,
            SystemModel.ASTEROID_FRAME: None if ast is None else ast.ast2sc_q*SystemModel.sc2gl_q,
        }
        return fqm[fsrc]*fqm[fdst].conj()

    def __repr__(self):
        return (
              'system state:\n\t%s\n'
            + '\nsolar elongation: %s\n'
            + '\nasteroid rotation: %.2f\n'
        ) % (
            '\n\t'.join('%s = %s'%(n, p) for n, p in self.get_params(all=True)), 
            tuple(map(math.degrees, self.solar_elongation())),
            math.degrees(self.asteroid.rotation_theta(self.time.value)),
        )
        

class Camera:
    def __init__(self, width, height, x_fov, y_fov,
                 sensor_size=None, focal_length=None, f_stop=None, aperture=None,
                 quantum_eff=None, px_saturation_e=None, emp_coef=1,
                 lambda_min=None, lambda_eff=None, lambda_max=None,
                 dark_noise_mu=None, dark_noise_sd=None, readout_noise_sd=None,
                 point_spread_fn=None, scattering_coef=None,
                 exclusion_angle_x=90, exclusion_angle_y=90):
        self.width = width      # in pixels
        self.height = height    # in pixels
        self.x_fov = x_fov      # in deg
        self.y_fov = y_fov      # in deg

        self.focal_length = None
        self.sensor_width = None
        self.sensor_height = None
        self.f_stop = None
        self.aperture = None

        self.emp_coef = emp_coef
        self.quantum_eff = quantum_eff      # average in range 400-800nm => 375e12-750e12 Hz
        self.px_saturation_e = px_saturation_e
        self.lambda_min = lambda_min
        self.lambda_eff = lambda_eff
        self.lambda_max = lambda_max
        self.dark_noise_mu = dark_noise_mu or 250/1e5 * self.px_saturation_e
        self.dark_noise_sd = dark_noise_sd or 60/1e5 * self.px_saturation_e
        self.readout_noise_sd = readout_noise_sd or 2/1e5 * self.px_saturation_e
        self.gain = None
        self.point_spread_fn = point_spread_fn
        self.scattering_coef = scattering_coef
        self.exclusion_angle_x = exclusion_angle_x
        self.exclusion_angle_y = exclusion_angle_y

        if px_saturation_e is not None:
            self.gain = 1 / self.px_saturation_e

        if sensor_size is not None:
            sw, sh = sensor_size    # in mm
            self.focal_length = min(sw / math.tan(math.radians(x_fov) / 2) / 2,
                                    sh / math.tan(math.radians(y_fov) / 2) / 2)
            self.sensor_width = sw
            self.sensor_height = sh
        if focal_length is not None:
            assert sensor_size is None, 'either give sensor_size or focal_length, not both'
            self.focal_length = focal_length
            self.sensor_width = math.tan(math.radians(x_fov) / 2) * 2 * focal_length
            self.sensor_height = math.tan(math.radians(y_fov) / 2) * 2 * focal_length

        if self.focal_length is not None:
            self.pixel_size = 1e3*min(self.sensor_width/self.width, self.sensor_height/self.width)  # in um
            if f_stop is not None:
                self.f_stop = f_stop
                self.aperture = self.focal_length / f_stop
            if aperture is not None:
                assert f_stop is None, 'either give fstop or aperture, not both'
                self.f_stop = self.focal_length / aperture
                self.aperture = aperture

    @property
    def aperture_area(self):
        return np.pi * (self.aperture*1e-3/2)**2

    @property
    def sensitivity(self):
        return self.gain * self.aperture_area * self.electrons_per_solar_irradiance * self.emp_coef

    @property
    def px_sr(self):
        return math.radians(self.x_fov) / self.width * math.radians(self.y_fov) / self.height

    @property
    def def_exposure(self):
        return min(5e-9/self.sensitivity, 5)

    @property
    def def_gain(self):
        return 5e-9/(self.def_exposure * self.sensitivity)

    @property
    def electrons_per_solar_irradiance(self):
        return Camera.electrons_per_solar_irradiance_s(self.quantum_eff, self.lambda_min, self.lambda_max)

    @staticmethod
    def level_to_exp_gain(level, exp_range):
        exp = 0.001 * np.floor(np.clip(level, *exp_range) * 1000)
        gain = 0.001 * np.floor(max(1, level/exp) * 1000)
        return exp, gain

    @staticmethod
    @lru_cache(maxsize=1)
    def electrons_per_solar_irradiance_s(quantum_eff, lambda_min, lambda_max):
        """
        Returns electrons per total solar irradiance [W/m2] assuming spectrum of the sun & sensor
        """

        h = 6.626e-34  # planck constant (m2kg/s)
        c = 3e8  # speed of light
        T = 5778  # temperature of sun
        k = 1.380649e-23  # Boltzmann constant
        # sun_sr = 6.807e-5  # sun steradians from earth

        def qeff(f):
            # sensor quantum efficiency
            return quantum_eff

        def phi(f):
            # planck's law of black body radiation [W/s/m2/Hz/sr]
            r = 2*h*f**3/c**2/(math.exp(h*f/k/T) - 1)
            return r

        def spectral_electrons(f):
            E = f * h             # energy per photon
            return qeff(f) * phi(f) / E

        tphi = integrate.quad(phi, c/1e-2, c/1e-8)
        telec = integrate.quad(spectral_electrons, c/lambda_max, c/lambda_min)
        return telec[0]/tphi[0]

    def electrons(self, flux_density, exposure=1):
        # electrons from total solar irradiance [J/(s*m2)]
        electrons = flux_density * self.aperture_area * exposure * self.electrons_per_solar_irradiance * self.emp_coef
        return electrons

    def sense(self, flux_density, exposure=1, gain=1, add_noise=True):
        flux_density = ImageProc.apply_point_spread_fn(flux_density, ratio=self.point_spread_fn)
        electrons = self.electrons(flux_density, exposure=exposure)

        if add_noise:
            # shot noise (should be based on electrons, but figured that electrons should be fine)
            # - also, poisson distributed with lambda=sqrt(electrons)
            #   here approximated using normal distribution with mu=electrons, sd=sqrt(electrons)
            mu = exposure*self.dark_noise_mu + electrons
            sigma2 = exposure*self.dark_noise_sd**2 + electrons + self.readout_noise_sd**2

            # shot noise, dark current and readout noise
            electrons = np.random.normal(mu, np.sqrt(sigma2))

        return np.clip(gain * self.gain * np.floor(electrons), 0, 1)

    def intrinsic_camera_mx(self, legacy=True):
        return Camera._intrinsic_camera_mx(self.width, self.height, self.x_fov, self.y_fov, legacy=legacy)

    def inv_intrinsic_camera_mx(self, legacy=True):
        return Camera._inv_intrinsic_camera_mx(self.width, self.height, self.x_fov, self.y_fov, legacy=legacy)

    @staticmethod
    def _intrinsic_camera_mx(width, height, x_fov, y_fov, legacy=True):
        x = width/2
        y = height/2
        fl_x = x / math.tan(math.radians(x_fov)/2)
        fl_y = y / math.tan(math.radians(y_fov)/2)
        return np.array([[fl_x * (1 if legacy else -1), 0, x],
                        [0, fl_y, y],
                        [0, 0, 1]], dtype="float")

    @staticmethod
    @lru_cache(maxsize=1)
    def _inv_intrinsic_camera_mx(w, h, xfov, yfov, legacy=True):
        return np.linalg.inv(Camera._intrinsic_camera_mx(w, h, xfov, yfov, legacy=legacy))

    def calc_xy(self, xi, yi, z_off):
        """ xi and yi are unaltered image coordinates, z_off is usually negative  """

        xh = xi + 0.5
        # yh = height - (yi+0.5)
        yh = yi + 0.5
        # zh = -z_off

        if True:
            iK = self.inv_intrinsic_camera_mx(legacy=False)
            x_off, y_off, _ = iK.dot(np.array([xh, yh, 1])) * z_off

        else:
            cx = xh / self.width - 0.5
            cy = yh / self.height - 0.5

            h_angle = cx * math.radians(self.x_fov)
            x_off = zh * math.tan(h_angle)

            v_angle = cy * math.radians(self.y_fov)
            y_off = zh * math.tan(v_angle)

        # print('%.3f~%.3f, %.3f~%.3f, %.3f~%.3f'%(ax, x_off, ay, y_off, az, z_off))
        return x_off, y_off

    def calc_img_xy(self, x, y, z):
        """ x, y, z are in camera frame (z typically negative),  return image coordinates  """
        K = self.intrinsic_camera_mx(legacy=False)
        ix, iy, iw = K.dot(np.array([x, y, z]))
        return ix / iw, iy / iw

    def calc_img_R(self, R):
        """
        R is a matrix where each row is a point in camera frame (z typically negative),
        returns a matrix where each row corresponds to points in image space """
        K = self.intrinsic_camera_mx(legacy=False)
        iRh = R.dot(K.T)
        return iRh[:, 0:2]/iRh[:, 2].reshape((-1, 1))


class Asteroid(ABC):
    ast2sc_q = None  # depends on the shape model coordinate frame

    (
        SM_NOISE_NO,
        SM_NOISE_LOW,
        SM_NOISE_HIGH,
    ) = ('', 'lo', 'hi')

    def __init__(self, *args, shape_model=None, **kwargs):
        super(Asteroid, self).__init__()

        self.name = None                # (NOT IN USE)

        self.image_db_path = None
        self.target_model_file = None
        self.hires_target_model_file = None
        self.hires_target_model_file_textures = False

        # shape model related
        self.render_smooth_faces = False    # when rendering shape model, smooth faces instead of angular ones
        self.real_shape_model = None        # loaded at overriding class __init__
        self.real_sc_ast_vertices = None
        self.reflmod_params = None

        self.real_position = None       # in km, transient, loaded from image metadata at iotools.lblloader
        
        self.max_radius = None          # in meters, maximum extent of object from asteroid frame coordinate origin
        self.mean_radius = None         # in meters

        # for cross section (probably not in (good) use)
        self.mean_cross_section = None  # in m2
        
        # epoch for orbital elements
        self.oe_epoch = None            # as astropy.time.Time

        # orbital elements
        self.eccentricity = None                    # unitless
        self.semimajor_axis = None                  # with astropy.units
        self.inclination = None                     # in rads
        self.longitude_of_ascending_node = None     # in rads
        self.argument_of_periapsis = None           # in rads
        self.mean_anomaly = None                    # in rads

        # other
        self.aphelion = None        # in rads
        self.perihelion = None      # in rads
        self.orbital_period = None  # in seconds

        # rotation period
        self.rot_epoch = None           # as astropy.time.Time
        self.rotation_velocity = None   # in rad/s
        self.rotation_pm = None         # in rads
        self.axis_latitude = None       # in rads
        self.axis_longitude = None      # in rads

        self.precession_cone_radius = None      # in rads       (NOT IN USE)
        self.precession_period = None           # in seconds    (NOT IN USE)
        self.precession_pm = None               # in rads       (NOT IN USE)

        # default values of axis that gets changed during monte-carlo simulation at testloop.py
        self.def_rotation_pm = None
        self.def_axis_latitude = None
        self.def_axis_longitude = None

    def set_defaults(self):
        self.def_rotation_pm = self.rotation_pm          # in rads
        self.def_axis_latitude = self.axis_latitude      # in rads
        self.def_axis_longitude = self.axis_longitude    # in rads

    def reset_to_defaults(self):
        self.rotation_pm = self.def_rotation_pm          # in rads
        self.axis_latitude = self.def_axis_latitude      # in rads
        self.axis_longitude = self.def_axis_longitude    # in rads

    @property
    def rotation_period(self):
        return 2 * math.pi / self.rotation_velocity

    @lru_cache(maxsize=1)
    def rot_epoch_unix(self):
        return (self.rot_epoch - Time(0, format='unix')).sec

    def rotation_theta(self, timestamp):
        dt = timestamp - self.rot_epoch_unix()
        theta = (self.rotation_pm + self.rotation_velocity * dt) % (2 * math.pi)
        return theta

    def rotation_q(self, timestamp):
        theta = self.rotation_theta(timestamp)

        # TODO: use precession info

        # orient z axis correctly, rotate around it
        return tools.ypr_to_q(self.axis_latitude, self.axis_longitude, theta) \
               * self.ast2sc_q

    def position(self, timestamp):
        if self.real_position is not None:
            return self.real_position

        # from http://space.stackexchange.com/questions/8911/determining-\
        #                           orbital-position-at-a-future-point-in-time

        # convert unix seconds to seconds since oe_epoch
        dt = (Time(timestamp, format='unix') - self.oe_epoch).sec

        # mean anomaly M
        M = (self.mean_anomaly + 2 * math.pi * dt / self.orbital_period) % (2 * math.pi)

        # eccentric anomaly E, orbit plane coordinates P & Q
        ecc = self.eccentricity
        E = tools.eccentric_anomaly(ecc, M)
        P = self.semimajor_axis * (math.cos(E) - ecc)
        Q = self.semimajor_axis * math.sin(E) * math.sqrt(1 - ecc ** 2)

        # rotate by argument of periapsis
        w = self.argument_of_periapsis
        x = math.cos(w) * P - math.sin(w) * Q
        y = math.sin(w) * P + math.cos(w) * Q

        # rotate by inclination
        z = math.sin(self.inclination) * x
        x = math.cos(self.inclination) * x

        # rotate by longitude of ascending node
        W = self.longitude_of_ascending_node
        xtemp = x
        x = math.cos(W) * xtemp - math.sin(W) * y
        y = math.sin(W) * xtemp + math.cos(W) * y

        # corrections for ROS_CAM1_20150720T113057
        if (False):
            x += 1.5e9 * units.m
            y += -1e9 * units.m
            z += -26.55e9 * units.m

        v_ba = np.array([x.value, y.value, z.value])
        if not USE_ICRS:
            sc = SkyCoord(x=x, y=y, z=z, frame='icrs',
                          representation_type='cartesian', obstime='J2000') \
                .transform_to('heliocentrictrueecliptic') \
                .represent_as('cartesian')
            v_ba = np.array([sc.x.value, sc.y.value, sc.z.value])

        return v_ba

    def load_noisy_shape_model(self, type, return_noise_sd=False):
        fname = self.constant_noise_shape_model[type]
        with open(fname, 'rb') as fh:
            noisy_model, sm_noise = pickle.load(fh)
        obj = objloader.ShapeModel(data=noisy_model)
        return obj if not return_noise_sd else (obj, sm_noise)
