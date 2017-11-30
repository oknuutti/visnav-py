
import math
import time

import numpy as np
import quaternion
from astropy.coordinates import SkyCoord

from settings import *


class PositioningException(Exception):
	pass

class Stopwatch:
    # from https://www.safaribooksonline.com/library/view/python-cookbook-3rd/9781449357337/ch13s13.html
    
    def __init__(self, func=time.perf_counter):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        self._start = None
        
    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        

def intrinsic_camera_mx(w=CAMERA_WIDTH, h=CAMERA_HEIGHT):
    x = w/2
    y = h/2
    fl_x = x / math.tan( math.radians(CAMERA_X_FOV)/2 )
    fl_y = y / math.tan( math.radians(CAMERA_Y_FOV)/2 )
    return np.array([[fl_x, 0, x],
                    [0, fl_y, y],
                    [0, 0, 1]], dtype = "float")


def inv_intrinsic_camera_mx(w=CAMERA_WIDTH, h=CAMERA_HEIGHT):
    return np.linalg.inv(intrinsic_camera_mx(w, h))


def sc_asteroid_max_shift_error(A, B):
    """
    Calculate max error between two set of vertices when projected to camera
    """
    
    # a dot b vector product
    diagAB = (A*B).sum(-1).reshape((-1,1))
    
    # length of a
    normA = np.sqrt((A**2).sum(-1)).reshape((-1,1))
    
    # diff vector across projection
    D = B - (diagAB / normA**2)*A

    # diff vector lengths
    normD = np.sqrt((D**2).sum(-1)).reshape((-1,1))
    
    # max length of diff vectors
    return np.max(normD)
    

def calc_xy(xi, yi, z_off, width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
    """ xi and yi are unaltered image coordinates, z_off is usually negative  """
    
    xh = xi+0.5
    yh = height - (yi+0.5)
    zh = -z_off
    
    if True:
        iK = inv_intrinsic_camera_mx(w=width, h=height)
        x_off, y_off, dist = iK.dot(np.array([xh, yh, 1]))*zh
        
    else:
        cx = xh/width - 0.5
        cy = yh/height - 0.5

        h_angle = cx * math.radians(CAMERA_X_FOV)
        x_off = zh * math.tan(h_angle)

        v_angle = cy * math.radians(CAMERA_Y_FOV)
        y_off = zh * math.tan(v_angle)
        
    # print('%.3f~%.3f, %.3f~%.3f, %.3f~%.3f'%(ax, x_off, ay, y_off, az, z_off))
    return x_off, y_off


def surf_normal(x1, x2, x3):
    a, b, c = tuple(map(np.array, (x1, x2, x3)))
    return normalize_v(np.cross(b-a, c-a))


def angle_between_v(v1, v2):
    # Notice: only returns angles between 0 and 180 deg
    
    try:
        v1 = np.reshape(v1, (1,-1))
        v2 = np.reshape(v2, (-1,1))
        
        n1 = v1/np.linalg.norm(v1)
        n2 = v2/np.linalg.norm(v2)
        
        cos_angle = n1.dot(n2)
    except TypeError as e:
        raise Exception('Bad vectors:\n\tv1: %s\n\tv2: %s'%(v1, v2)) from e
    
    return math.acos(np.clip(cos_angle, -1, 1))


def angle_between_q(q1, q2):
    # from  https://chrischoy.github.io/research/measuring-rotation/
    qd = q1.conj()*q2
    return 2*math.acos(qd.normalized().w)


def angle_between_ypr(ypr1, ypr2):
    q1 = ypr_to_q(*ypr1)
    q2 = ypr_to_q(*ypr2)
    return angle_between_q(q1, q2)


def q_to_unitbase(q):
    U0 = quaternion.as_quat_array([[0,1,0,0], [0,0,1,0], [0,0,0,1.]])
    Uq = q * U0 * q.conj()
    return quaternion.as_float_array(Uq)[:, 1:]


def equatorial_to_ecliptic(ra, dec):
    """ translate from equatorial ra & dec to ecliptic ones """
    sc = SkyCoord(ra, dec, unit='deg', frame='icrs', obstime='J2000') \
            .transform_to('barycentrictrueecliptic')
    return math.radians(sc.lat.value), math.radians(sc.lon.value)
    
    
def q_to_angleaxis(q, compact=False):
    theta = math.acos(q.w) * 2.0
    v = np.array([q.x, q.y, q.z])
    return (theta,) + tuple(normalize_v(v) if sum(v)>0 else v)


def angleaxis_to_q(rv):
    if len(rv)==4:
        theta = rv[0]
        v = normalize_v(np.array(rv[1:]))
    elif len(rv)==3:
        theta = math.sqrt(sum(x**2 for x in rv))
        v = np.array(rv)/theta
    else:
        raise Exception('Invalid angle-axis vector: %s'%(rv,))
    
    w = math.cos(theta/2)
    v = v*math.sin(theta/2)
    return np.quaternion(w, *v).normalized()


def ypr_to_q(lat, lon, roll):
    # Tait-Bryan angles, aka yaw-pitch-roll, nautical angles, cardan angles
    # intrinsic euler rotations z-y'-x'', pitch=-lat, yaw=lon
    return (
          np.quaternion(math.cos(lon/2), 0, 0, math.sin(lon/2))
        * np.quaternion(math.cos(-lat/2), 0, math.sin(-lat/2), 0)
        * np.quaternion(math.cos(roll/2), math.sin(roll/2), 0, 0)
    )
    
    
def q_to_ypr(q):
    # from https://math.stackexchange.com/questions/687964/getting-euler-tait-bryan-angles-from-quaternion-representation
    q0,q1,q2,q3 = quaternion.as_float_array(q)[0]
    roll = np.arctan2(q2*q3+q0*q1, .5-q1**2-q2**2)
    lat = -np.arcsin(-2*(q1*q3-q0*q2))
    lon  = np.arctan2(q1*q2+q0*q3, .5-q2**2-q3**2)
    return lat, lon, roll
    
    
def q_times_v(q, v):
    qv = np.quaternion(0, *v)
    qv2 = q * qv * q.conj()
    return np.array([qv2.x, qv2.y, qv2.z])

def q_times_mx(q, mx):
    qqmx = q * mx2qmx(mx) * q.conj()
    aqqmx = quaternion.as_float_array(qqmx)
    return aqqmx[:,1:]

def mx2qmx(mx):
    qmx = np.zeros((mx.shape[0],4))
    qmx[:,1:] = mx
    return quaternion.as_quat_array(qmx)

def normalize_v(v):
    return v/math.sqrt(sum(map(lambda x: x**2, v)))


def wrap_rads(a):
    return (a+math.pi)%(2*math.pi)-math.pi

def wrap_degs(a):
    return (a+180)%360-180

def eccentric_anomaly(eccentricity, mean_anomaly, tol=1e-6):
    # from http://www.jgiesen.de/kepler/kepler.html
    
    E = mean_anomaly if eccentricity < 0.8 else math.pi
    F = E - eccentricity * math.sin(mean_anomaly) - mean_anomaly;
    for i in range(30):
        if abs(F) < tol:
            break
        E = E - F / (1.0 - eccentricity*math.cos(E))
        F = E - eccentricity*math.sin(E) - mean_anomaly

    return round(E/tol)*tol

def solar_elongation(ast_v, sc_q):
    sco_x, sco_y, sco_z = q_to_unitbase(sc_q)
    
    if USE_ICRS:
        sc = SkyCoord(x=ast_v[0], y=ast_v[1], z=ast_v[2], frame='icrs',
                      unit='m', representation='cartesian', obstime='J2000')\
            .transform_to('hcrs')\
            .represent_as('cartesian')
        ast_v = np.array([sc.x.value, sc.y.value, sc.z.value])
    
    # angle between camera axis and the sun, 0: right ahead, pi: behind
    elong = angle_between_v(-ast_v, sco_x)

    # direction the sun is at when looking along camera axis
    nvec = np.cross(sco_x, ast_v)
    direc = angle_between_v(nvec, sco_z)

    # decide if direction needs to be negative or not
    if np.cross(nvec, sco_z).dot(sco_x) < 0:
        direc = -direc

    return elong, direc


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def discretize_q(q, tol):
    """
    simulate feature database by giving closest lat & lon with given tolerance
    and set roll to zero as feature detectors are rotation invariant
    """
    
    lat, lon, roll = q_to_ypr(q.conj())
    
    dblat, dblon = bf_lat_lon(tol)
    nlat = find_nearest(dblat, lat)
    nlon = find_nearest(dblon, lon)
    
    nq0 = ypr_to_q(nlat, nlon, 0).conj()
    return nq0
    

def bf_lat_lon(tol):
    # tol**2 == (step/2)**2 + (step/2)**2   -- 7deg is quite nice in terms of len(lon)*len(lat) == 1260
    step = math.sqrt(2)*tol
    lat_steps = np.linspace(-math.pi/2, math.pi/2, num=math.ceil(math.pi/step), endpoint=False)[1:]
    lon_steps = np.linspace(-math.pi, math.pi, num=math.ceil(2*math.pi/step), endpoint=False)
    return lat_steps, lon_steps


def apply_noise(model, support=None, len_sc=SHAPE_MODEL_NOISE_LEN_SC, 
                noise_lv=SHAPE_MODEL_NOISE_LV, only_z=False):
    
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    import random
    import time
    from iotools import objloader
    
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
    except:
        print('Requires scikit-learn, install using "conda install scikit-learn"')
        sys.exit()
    
    points = np.array(model.vertices)
    if support is None:
        support = points[random.sample(list(range(len(points))), min(3000,len(points)))]
    
    mean = np.mean(points,axis=0)
    max_rng = np.max(np.ptp(points,axis=0))
    kernel = 0.7*noise_lv*Matern(length_scale=len_sc*max_rng, nu=1.5) + 0.2*noise_lv*Matern(length_scale=0.1*len_sc*max_rng, nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel)
    
    # sample gp
    err = np.exp(gp.sample_y(support-mean, 1, int(time.time())))

    if len(err) == len(points):
        full_err = err
        print('using orig gp sampled err')
    else:
        # interpolate
        interp = LinearNDInterpolator((support-mean)*1.0015, err)
        full_err = interp(points-mean)

        # maybe extrapolate
        nanidx = tuple(np.isnan(full_err).flat)
        if np.any(nanidx):
            if DEBUG or not BATCH_MODE:
                print('%sx nans'%np.sum(nanidx))
            naninterp = NearestNDInterpolator(support, err)
            full_err[nanidx,] = naninterp(points[nanidx,:])

    # extra high frequency noise
    white_noise = np.random.normal(scale=0.2*noise_lv*max_rng, size=(len(full_err),1))

    if only_z:
        noisy_points = np.concatenate((points[:,0:2], (points[:,2]-mean[2])*full_err +white_noise +mean[2]))
    else:
        noisy_points = (points-mean)*full_err +mean +white_noise
    
    devs = np.sqrt(np.sum((points-noisy_points)**2,axis=-1)/np.sum(points**2,axis=-1))
    if DEBUG or not BATCH_MODE:
        print('noise: %.3f, %.3f; avg=%.3f'%(tuple(np.percentile(devs, (68,95)))+(np.mean(devs),)))
    
    if False:
        import matplotlib.pyplot as plt
        plt.figure(1, figsize=(8, 8))
        #plt.plot(np.concatenate((points[:,0], err0[:,0], err[:,0], points[:,0]*err[:,0])))
        plt.subplot(2, 2, 1)
        plt.plot(points[:,0])
        plt.title('original', fontsize=12)

        plt.subplot(2, 2, 2)
        plt.plot(err0[:,0])
        plt.title('norm-err', fontsize=12)

        plt.subplot(2, 2, 3)
        plt.plot(err[:,0])
        plt.title('exp-err', fontsize=12)

        plt.subplot(2, 2, 4)
        plt.plot(noisy_points[:,0])
        plt.title('noisy', fontsize=12)

        plt.tight_layout()
        plt.show()
        assert False, 'exiting'
    
    data = model.as_dict()
    data['vertices'] = noisy_points
    noisy_model = objloader.ShapeModel(data=data)
    noisy_model.recalc_norms()
    return noisy_model, np.mean(devs)


def interp2(array, x, y):
    assert y<array.shape[0] and x<array.shape[1], 'out of bounds %s: %s'%(array.shape, (y, x))
    
    v = array[int(y):int(y)+2, int(x):int(x)+2]
    xf = x-int(x)
    yf = y-int(y)
    w = np.array((
        ((1-yf)*(1-xf), (1-yf)*xf),
        (yf*(1-xf),     yf*xf),
    ))
    w = w/np.sum(w)
    return np.sum(w*v)
    

def solve_rotation(src_q, dst_q):
    """ q*src_q*q.conj() == dst_q, solve for q """
    # based on http://web.cs.iastate.edu/~cs577/handouts/quaternion.pdf
    # and https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Pairs_of_unit_quaternions_as_rotations_in_4D_space
    
    # NOTE: not certain if works..
    
    M = np.zeros((4,4))
    for i in range(len(src_q)):
        si = src_q[i]
        Pi = np.array((
            (si.w, -si.x, -si.y, -si.z),
            (si.x, si.w, si.z, -si.y),
            (si.y, -si.z, si.w, si.x),
            (si.z, si.y, -si.x, si.w),
        ))

        qi = dst_q[i]
        Qi = np.array((
            (qi.w, -qi.x, -qi.y, -qi.z),
            (qi.x, qi.w, -qi.z, qi.y),
            (qi.y, qi.z, qi.w, -qi.x),
            (qi.z, -qi.y, qi.x, qi.w),
        ))
    
        M += Pi.T * Qi
    
    w, v = np.linalg.eig(M)
    i = np.argmax(w)
    res_q = np.quaternion(*v[:,i])
#    alt = v.dot(w)
#    print('%s,%s'%(res_q, alt))
#    res_q = np.quaternion(*alt).normalized()
    return res_q

def solve_q_bf(src_q, dst_q):
    qs = []
    d = []
    for res_q in (
        np.quaternion(0,0,0,1).normalized(),
        np.quaternion(0,0,1,0).normalized(),
        np.quaternion(0,0,1,1).normalized(),
        np.quaternion(0,0,-1,1).normalized(),
        np.quaternion(0,1,0,0).normalized(),
        np.quaternion(0,1,0,1).normalized(),
        np.quaternion(0,1,0,-1).normalized(),
        np.quaternion(0,1,1,0).normalized(),
        np.quaternion(0,1,-1,0).normalized(),
        np.quaternion(0,1,1,1).normalized(),
        np.quaternion(0,1,1,-1).normalized(),
        np.quaternion(0,1,-1,1).normalized(),
        np.quaternion(0,1,-1,-1).normalized(),
        np.quaternion(1,0,0,1).normalized(),
        np.quaternion(1,0,0,-1).normalized(),
        np.quaternion(1,0,1,0).normalized(),
        np.quaternion(1,0,-1,0).normalized(),
        np.quaternion(1,0,1,1).normalized(),
        np.quaternion(1,0,1,-1).normalized(),
        np.quaternion(1,0,-1,1).normalized(),
        np.quaternion(1,0,-1,-1).normalized(),
        np.quaternion(1,1,0,0).normalized(),
        np.quaternion(1,-1,0,0).normalized(),
        np.quaternion(1,1,0,1).normalized(),
        np.quaternion(1,1,0,-1).normalized(),
        np.quaternion(1,-1,0,1).normalized(),
        np.quaternion(1,-1,0,-1).normalized(),
        np.quaternion(1,1,1,0).normalized(),
        np.quaternion(1,1,-1,0).normalized(),
        np.quaternion(1,-1,1,0).normalized(),
        np.quaternion(1,-1,-1,0).normalized(),
        np.quaternion(1,1,1,-1).normalized(),
        np.quaternion(1,1,-1,1).normalized(),
        np.quaternion(1,1,-1,-1).normalized(),
        np.quaternion(1,-1,1,1).normalized(),
        np.quaternion(1,-1,1,-1).normalized(),
        np.quaternion(1,-1,-1,1).normalized(),
        np.quaternion(1,-1,-1,-1).normalized(),
    ):
        tq = res_q * src_q * res_q.conj()
        qs.append(res_q)
        d.append(1-np.array((tq.w, tq.x, tq.y, tq.z)).dot(np.array((dst_q.w, dst_q.x, dst_q.y, dst_q.z)))**2)
    i = np.argmin(d)
    return qs[i]


#
# Not sure if unitbase_to_q works, haven't deleted just in case still need:
#
#def unitbase_to_q(b_dst, b_src = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
#    # based on http://stackoverflow.com/questions/16648452/calculating-\
#    #   quaternion-for-transformation-between-2-3d-cartesian-coordinate-syst
#    # , which is based on http://dx.doi.org/10.1117/12.57955
#    
#    M = np.zeros((3, 3))
#
#    for i, v in enumerate(b_src):
#        x = np.matrix(np.outer(v, b_dst[i]))
#        M = M + x
#
#    N11 = float(M[0][:,0] + M[1][:,1] + M[2][:,2])
#    N22 = float(M[0][:,0] - M[1][:,1] - M[2][:,2])
#    N33 = float(-M[0][:,0] + M[1][:,1] - M[2][:,2])
#    N44 = float(-M[0][:,0] - M[1][:,1] + M[2][:,2])
#    N12 = float(M[1][:,2] - M[2][:,1])
#    N13 = float(M[2][:,0] - M[0][:,2])
#    N14 = float(M[0][:,1] - M[1][:,0])
#    N21 = float(N12)
#    N23 = float(M[0][:,1] + M[1][:,0])
#    N24 = float(M[2][:,0] + M[0][:,2])
#    N31 = float(N13)
#    N32 = float(N23)
#    N34 = float(M[1][:,2] + M[2][:,1])
#    N41 = float(N14)
#    N42 = float(N24)
#    N43 = float(N34)
#
#    N=np.matrix([[N11, N12, N13, N14],\
#                 [N21, N22, N23, N24],\
#                 [N31, N32, N33, N34],\
#                 [N41, N42, N43, N44]])
#
#    values, vectors = np.linalg.eig(N)
#    quat = vectors[:, np.argmax(values)]
#    #quat = np.array(quat).reshape(-1,).tolist()
#    
#    return np.quaternion(*quat)

import tracemalloc
import os
import linecache

def display_top(top_stats, key_type='lineno', limit=10):
#    snapshot = snapshot.filter_traces((
#        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
#        tracemalloc.Filter(False, "<unknown>"),
#    ))
#    top_stats = snapshot.statistics(key_type, cumulative=True)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f MB (x%.0f)"
              % (index, filename, frame.lineno, stat.size/1024/1024, stat.count))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f MB" % (len(other), size/1024/1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f MB" % (total/1024/1024))