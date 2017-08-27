
import math

import numpy as np
import quaternion # adds to numpy
from astropy.time import Time
from astropy import constants as const

from settings import *
from algo import tools

class Parameter():
    def __init__(self, min_val, max_val, def_val=None, estimate=True, is_gl_z=False):
        self._min_val = min_val
        self._max_val = max_val
        self.estimate = estimate
        self._def_val = def_val
        self._value = self.def_val
        self.is_gl_z = is_gl_z
        self.change_callback = None
        self.fire_change_events = True
    
    @property
    def range(self):
        return (self._min_val, self._max_val)

    @range.setter
    def range(self, range):
        min_val, max_val = range
        if self._min_val != min_val or self._max_val != max_val:
            self._min_val = min_val
            self._max_val = max_val
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
        if self._value != value:
            self._value = value
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
        return '%.2f in [%.2f, %.2f]' % (self._value, self._min_val, self._max_val)
    
    def __repr__(self):
        return 'value %s (%s), [%s, %s], def: %s, sc: %s'%(
            self._value, self.nvalue, self._min_val, self._max_val,
            self.def_val, self.scale,
        )
    

class SystemModel():
    # from sc cam frame (axis: +x, up: +z) to opengl (axis -z, up: +y)
    q_sc2gl = np.quaternion(0.5, 0.5, -0.5, -0.5)
    
    def __init__(self, *args, **kwargs):
        self.asteroid = Asteroid()
        
        self.real_sc_pos = None
        
        # spacecraft position relative to asteroid, z towards spacecraft,
        #   x towards right when looking out from s/c camera, y up
        self.x_off = Parameter(-4, 4, estimate=False)
        self.y_off = Parameter(-4, 4, estimate=False)
        
        # whole view: 1.65km/tan(2.5deg) = 38km
        # can span ~30px: 1.65km/tan(2.5deg * 30/1024) = 1290km
        self.z_off = Parameter(-MAX_DISTANCE, -MIN_DISTANCE, def_val=-200, is_gl_z=True) # was 120, 220

        # spacecraft orientation relative to stars
        self.x_rot = Parameter(-90, 90, estimate=False) # axis latitude
        self.y_rot = Parameter(-180, 180, estimate=False) # axis longitude
        self.z_rot = Parameter(-180, 180, estimate=False) # rotation

        # asteroid zero orientation relative to stars
        self.ast_x_rot = Parameter(-90, 90, estimate=False) # axis latitude
        self.ast_y_rot = Parameter(-180, 180, estimate=False) # axis longitude
        self.ast_z_rot = Parameter(-180, 180, estimate=False) # rotation
        self.ast_x_rot.value = math.degrees(self.asteroid.axis_latitude)
        self.ast_y_rot.value = math.degrees(self.asteroid.axis_longitude)
        self.ast_z_rot.value = \
                (math.degrees(self.asteroid.rotation_pm) + 180) % 360 - 180

        # time in seconds since 1970-01-01 00:00:00
        self.time = Parameter(
            Time('2015-01-01 00:00:00').unix,
            Time('2015-01-01 00:00:00').unix + self.asteroid.rotation_period,
            estimate=False
        )
        
        # override any default params
        for n, v in kwargs.items():
            setattr(self, n, v)
        
        # set default values to params
        for n, p in self.get_params():
            p.value = p.def_val
        
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
        
    def set_spacecraft_pos(self, pos):
        self.z_off.value = pos[2]

        half_range = abs(pos[2] / 170 * 4)
        self.x_off.range = (pos[0] - half_range, pos[0] + half_range)
        self.x_off.value = pos[0]

        self.y_off.range = (pos[1] - half_range, pos[1] + half_range)
        self.y_off.value = pos[1]

    def rotate_spacecraft(self, q):
        new_q = self.spacecraft_q() * q
        self.x_rot.value, self.y_rot.value, self.z_rot.value = \
            list(map(math.degrees, tools.q_to_spherical(new_q)))

    def spacecraft_q(self):
        return tools.spherical_to_q(*list(map(
                math.radians,
                (self.x_rot.value, self.y_rot.value, self.z_rot.value)
        )))
    
    def sc_asteroid_rel_rot(self):
        """ rotation of asteroid relative to spacecraft in opengl coords """
        
        self.asteroid.axis_latitude = math.radians(self.ast_x_rot.value)
        self.asteroid.axis_longitude = math.radians(self.ast_y_rot.value)
        self.asteroid.rotation_pm = math.radians(self.ast_z_rot.value)
        
        ast_q = self.asteroid.rotation_q(self.time.value)
        sc_q = self.spacecraft_q()
        
        # ast => eq => sc => opengl
        q_tot = ast_q.conj() * sc_q * SystemModel.q_sc2gl
        
        if not BATCH_MODE and DEBUG:
            print('asteroid x-axis: %s'%tools.q_times_v(q_tot.conj(), np.array([1, 0, 0])))
        
        return q_tot.conj()
    
    def light_rel_dir(self):
        """ direction of light relative to spacecraft in opengl coords """
        ast_v = tools.normalize_v(self.asteroid.position(self.time.value))
        sc_q = self.spacecraft_q()
        return tools.q_times_v(SystemModel.q_sc2gl.conj() * sc_q.conj(), ast_v)
        
    def solar_elongation(self):
        ast_v = self.asteroid.position(self.time.value)
        sc_q = self.spacecraft_q()
        elong, direc = tools.solar_elongation(ast_v, sc_q)
        if not BATCH_MODE or DEBUG:
            print('elong: %.3f | dir: %.3f' % (
                math.degrees(elong), math.degrees(direc)))
        return elong, direc
    
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
        

class Asteroid():
    def __init__(self, *args, **kwargs):
        self.name = '67P/Churyumov-Gerasimenko'
        
        # for cross section, assume spherical object and 2km radius
        self.mean_cross_section = math.pi*2000**2
        
        # epoch for orbital elements, 2017-Jan-10 00:00:00.0000 TDB
        self.oe_epoch = Time(2457763.5, format='jd')

        #orbital elements (from http://ssd.jpl.nasa.gov/horizons.cgi)
        self.eccentricity = .6424327680961353
        self.semimajor_axis = 3.463077807329411 * const.au
        self.inclination = math.radians(7.060539942545541)
        self.longitude_of_ascending_node = math.radians(50.01776564570947)
        self.argument_of_periapsis = math.radians(12.82332717180694)
        self.mean_anomaly = math.radians(78.81269978123633)

        #other
        self.aphelion = 5.687258171672372 * const.au
        self.perihelion = 1.242352456168495 * const.au
        self.orbital_period = 6.4495062735684*365.25*24*3600 # seconds
        self.true_anomaly = math.radians(145.5260853202137)
  
        # rotation
        # from http://www.aanda.org/articles/aa/full_html/2015/11/aa26349-15/aa26349-15.html
        self.rot_epoch = Time('J2000')
        #self.rotation_velocity = 2*math.pi/12.4043/3600 # prograde, in rad/s
        # --- above seems incorrect based on the pics, own estimate
        # based on ROS_CAM1_20150720T165249 - ROS_CAM1_20150721T075733
        self.rotation_velocity = 0.000203926
        
        # will use as ecliptic longitude of
        # asteroid zero longitude (cheops) at J2000, based on 20150720T165249
        # papar had 114deg in it..
        self.rotation_pm = math.radians(150.594)
        
        # precession cone center (J2000), paper had 69.54, 64.11, own corrected
        # values used instead
        self.axis_latitude, self.axis_longitude = \
                tools.equatorial_to_ecliptic(69.54, 64.11)
                
        self.axis_latitude += math.radians(12.93)
        self.axis_longitude += math.radians(-227.35)
        
        self.precession_cone_radius = math.radians(0.14)
        self.precession_period = 10.7*24*3600
        self.precession_pm = math.radians(0.288)
        
    @property
    def rotation_period(self):
        return 2*math.pi/self.rotation_velocity
    
    def rotation_theta(self, timestamp):
        dt = (Time(timestamp, format='unix') - self.rot_epoch).sec
        theta = (self.rotation_pm + self.rotation_velocity*dt) % (2*math.pi)
        return theta
        
    def rotation_q(self, timestamp):
        theta = self.rotation_theta(timestamp)
        # TODO: use precession info
        q_x2z = np.quaternion(1, 0, 1, 0).normalized()
        return q_x2z \
                * tools.spherical_to_q(self.axis_latitude, self.axis_longitude, theta) \
                * q_x2z.conj()
    
    def position(self, timestamp):
        # from http://space.stackexchange.com/questions/8911/determining-\
        #                           orbital-position-at-a-future-point-in-time
        
        # convert unix seconds to seconds since oe_epoch
        dt = (Time(timestamp, format='unix') - self.oe_epoch).sec
        
        # mean anomaly M
        M = (self.mean_anomaly + 2*math.pi*dt/self.orbital_period) % (2*math.pi)
        
        # eccentric anomaly E, orbit plane coordinates P & Q
        ecc = self.eccentricity
        E = tools.eccentric_anomaly(ecc, M)
        P = self.semimajor_axis * (math.cos(E) - ecc)
        Q = self.semimajor_axis * math.sin(E) * math.sqrt(1 - ecc**2)
        
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
        
        return np.array([x.value, y.value, z.value])
    