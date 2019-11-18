
import math
import time

import numpy as np
import numba as nb
import quaternion
import sys
from astropy.coordinates import SkyCoord
from numba.cgutils import printf

from visnav.settings import *


class PositioningException(Exception):
	pass


class Stopwatch:
    # from https://www.safaribooksonline.com/library/view/python-cookbook-3rd/9781449357337/ch13s13.html
    
    def __init__(self, elapsed=0.0, func=time.perf_counter):
        self._elapsed = elapsed
        self._func = func
        self._start = None

    @property
    def elapsed(self):
        return self._elapsed + ((self._func() - self._start) if self.running else 0)

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self._elapsed += end - self._start
        self._start = None
        
    def reset(self):
        self._elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def sphere_angle_radius(loc, r):
    return np.arcsin(r / np.linalg.norm(loc, axis=1))


def point_vector_dist(A, B, dist_along_v=False):
    """ A: point, B: vector """

    # (length of b)**2
    normB2 = (B ** 2).sum(-1).reshape((-1, 1))

    # a dot b vector product (project a on b but also times length of b)
    diagAB = (A * B).sum(-1).reshape((-1, 1))

    # A projected along B (projection = a dot b/||b|| * b/||b||)
    A_B = (diagAB / normB2) * B

    # vector from projected A to A, it is perpendicular to B
    AB2A = A - A_B

    # diff vector lengths
    normD = np.sqrt((AB2A ** 2).sum(-1)).reshape((-1, 1))
    return (normD, diagAB/np.sqrt(normB2)) if dist_along_v else normD


def sc_asteroid_max_shift_error(A, B):
    """
    Calculate max error between two set of vertices when projected to camera,
    A = estimated vertex positions
    B = true vertex positions
    Error is a vector perpendicular to B, i.e. A - A||
    """

    # diff vector lengths
    normD = point_vector_dist(A, B)
    
    # max length of diff vectors
    return np.max(normD)
    

def surf_normal(x1, x2, x3):
    a, b, c = tuple(map(np.array, (x1, x2, x3)))
    return normalize_v(np.cross(b-a, c-a))


def angle_between_v(v1, v2):
    # Notice: only returns angles between 0 and 180 deg
    
    try:
        v1 = np.reshape(v1, (1, -1))
        v2 = np.reshape(v2, (-1, 1))
        
        n1 = v1/np.linalg.norm(v1)
        n2 = v2/np.linalg.norm(v2)

        cos_angle = n1.dot(n2)
    except TypeError as e:
        raise Exception('Bad vectors:\n\tv1: %s\n\tv2: %s'%(v1, v2)) from e
    
    return math.acos(np.clip(cos_angle, -1, 1))


def angle_between_v_mx(a, B):
    cos_angles = (B/np.linalg.norm(B, axis=1).reshape((-1, 1))).dot(normalize_v(a).reshape((-1, 1)))
    return np.arccos(np.clip(cos_angles, -1.0, 1.0))


def angle_between_mx(A, B):
    return angle_between_rows(A, B)


def angle_between_rows(A, B, normalize=True):
    assert A.shape[1] == 3 and B.shape[1] == 3, 'matrices need to be of shape (n, 3) and (m, 3)'
    if A.shape[0] == B.shape[0]:
        # from https://stackoverflow.com/questions/50772176/calculate-the-angle-between-the-rows-of-two-matrices-in-numpy/50772253
        cos_angles = np.einsum('ij,ij->i', A, B)
        if normalize:
            p2 = np.einsum('ij,ij->i', A, A)
            p3 = np.einsum('ij,ij->i', B, B)
            cos_angles /= np.sqrt(p2 * p3)
    else:
        if normalize:
            A = A / np.linalg.norm(A, axis=1).reshape((-1, 1))
            B = B / np.linalg.norm(B, axis=1).reshape((-1, 1))
        cos_angles = B.dot(A.T)

    return np.arccos(np.clip(cos_angles, -1.0, 1.0))


def rand_q(angle):
    r = normalize_v(np.random.normal(size=3))
    return angleaxis_to_q(np.hstack((angle, r)))


def angle_between_q(q1, q2):
    # from  https://chrischoy.github.io/research/measuring-rotation/
    qd = q1.conj()*q2
    return abs(wrap_rads(2*math.acos(qd.normalized().w)))


def angle_between_q_arr(q1, q2):
    qd = quaternion.as_float_array(q1.conj()*q2)
    qd = qd / np.linalg.norm(qd, axis=1).reshape((-1, 1))
    return np.abs(wrap_rads(2 * np.arccos(qd[:, 0])))


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
    return sc.lat.value, sc.lon.value


def q_to_angleaxis(q, compact=False):
    theta = math.acos(np.clip(q.w, -1, 1)) * 2.0
    v = normalize_v(np.array([q.x, q.y, q.z]))
    if compact:
        return theta * v
    else:
        return np.array((theta,) + tuple(v))


def angleaxis_to_q(rv):
    """ first angle, then axis """
    if len(rv)==4:
        theta = rv[0]
        v = normalize_v(np.array(rv[1:]))
    elif len(rv)==3:
        theta = math.sqrt(sum(x**2 for x in rv))
        v = np.array(rv) / (1 if theta == 0 else theta)
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
    q0,q1,q2,q3 = quaternion.as_float_array(q)
    roll = np.arctan2(q2*q3+q0*q1, .5-q1**2-q2**2)
    lat = -np.arcsin(-2*(q1*q3-q0*q2))
    lon  = np.arctan2(q1*q2+q0*q3, .5-q2**2-q3**2)
    return lat, lon, roll
    

def mean_q(qs, ws=None):
    """
    returns a (weighted) mean of a set of quaternions
    idea is to rotate a bit in the direction of new quaternion from the sum of previous rotations
    NOTE: not tested properly, might not return same mean quaternion if order of input changed
    """
    wtot = 0
    qtot = quaternion.one
    for q, w in zip(qs, np.ones((len(qs),)) if ws is None else ws):
        ddaa = q_to_angleaxis(qtot.conj() * q)
        ddaa[0] = wrap_rads(ddaa[0]) * w / (w + wtot)
        qtot = angleaxis_to_q(ddaa) * qtot
        wtot += w
    return qtot


def q_times_v(q, v):
    qv = np.quaternion(0, *v)
    qv2 = q * qv * q.conj()
    return np.array([qv2.x, qv2.y, qv2.z])

def q_times_mx(q, mx):
    qqmx = q * mx2qmx(mx) * q.conj()
    aqqmx = quaternion.as_float_array(qqmx)
    return aqqmx[:, 1:]

def mx2qmx(mx):
    qmx = np.zeros((mx.shape[0],4))
    qmx[:,1:] = mx
    return quaternion.as_quat_array(qmx)

def normalize_v(v):
    norm = np.linalg.norm(v)
    return v/norm if norm != 0 else v

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
                      unit='m', representation_type='cartesian', obstime='J2000')\
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

@nb.njit(nb.f8[:](nb.f8[:], nb.f8[:]))
def cross3d(left, right):
    # for short vectors cross product is faster in pure python than with numpy.cross
    x = ((left[1] * right[2]) - (left[2] * right[1]))
    y = ((left[2] * right[0]) - (left[0] * right[2]))
    z = ((left[0] * right[1]) - (left[1] * right[0]))
    return np.array((x, y, z))

def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx

def find_nearest_arr(array, value, ord=None, fun=None):
    diff = array - value
    idx = np.linalg.norm(diff if fun is None else list(map(fun, diff)), ord=ord, axis=1).argmin()
    return array[idx], idx

def find_nearest_n(array, value, r, ord=None, fun=None):
    diff = array - value
    d = np.linalg.norm(diff if fun is None else list(map(fun, diff)), ord=ord, axis=1)
    idxs = np.where(d < r)
    return idxs[0]

def cartesian2spherical(x, y, z):
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.acos(z/r)
    phi = math.atan2(y, x)
    lat = math.pi/2 - theta
    lon = phi
    return np.array([lat, lon, r])


def spherical2cartesian(lat, lon, r):
    theta = math.pi/2 - lat
    phi = lon
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return np.array([x, y, z])


def spherical2cartesian_arr(A):
    theta = math.pi/2 - A[:, 0]
    phi = A[:, 1]
    x = A[:, 2] * np.sin(theta) * np.cos(phi)
    y = A[:, 2] * np.sin(theta) * np.sin(phi)
    z = A[:, 2] * np.cos(theta)
    return np.vstack([x, y, z]).T


def discretize_v(v, tol=None, lat_range=(-math.pi/2, math.pi/2), points=None):
    """
    simulate feature database by giving closest light direction with given tolerance
    """

    if tol is not None and points is not None or tol is None and points is None:
        assert False, 'Give either tol or points'
    elif tol is not None:
        points = bf2_lat_lon(tol, lat_range=lat_range)

    lat, lon, r = cartesian2spherical(*v)

    (nlat, nlon), idx = find_nearest_arr(
        points,
        np.array((lat, lon)),
        ord=2,
        fun=wrap_rads,
    )

    ret = spherical2cartesian(nlat, nlon, r)
    return ret, idx


def discretize_q(q, tol=None, lat_range=(-math.pi/2, math.pi/2), points=None):
    """
    simulate feature database by giving closest lat & roll with given tolerance
    and set lon to zero as feature detectors are rotation invariant (in opengl coords)
    """

    if tol is not None and points is not None or tol is None and points is None:
        assert False, 'Give either tol or points'
    elif tol is not None:
        points = bf2_lat_lon(tol, lat_range=lat_range)

    lat, lon, roll = q_to_ypr(q)
    (nlat, nroll), idx = find_nearest_arr(
        points,
        np.array((lat, roll)),
        ord=2,
        fun=wrap_rads,
    )
    nq0 = ypr_to_q(nlat, 0, nroll)
    return nq0, idx
    

def bf_lat_lon(tol, lat_range=(-math.pi/2, math.pi/2)):
    # tol**2 == (step/2)**2 + (step/2)**2   -- 7deg is quite nice in terms of len(lon)*len(lat) == 1260
    step = math.sqrt(2)*tol
    lat_steps = np.linspace(*lat_range, num=math.ceil((lat_range[1] - lat_range[0])/step), endpoint=False)[1:]
    lon_steps = np.linspace(-math.pi, math.pi, num=math.ceil(2*math.pi/step), endpoint=False)
    return lat_steps, lon_steps


def bf2_lat_lon(tol, lat_range=(-math.pi/2, math.pi/2)):
    # tol**2 == (step/2)**2 + (step/2)**2   -- 7deg is quite nice in terms of len(lon)*len(lat) == 1260
    step = math.sqrt(2)*tol
    lat_steps = np.linspace(*lat_range, num=math.ceil((lat_range[1] - lat_range[0])/step), endpoint=False)[1:]

    # similar to https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    points = []
    for lat in lat_steps:
        Mphi = math.ceil(2*math.pi*math.cos(lat)/step)
        lon_steps = np.linspace(-math.pi, math.pi, num=Mphi, endpoint=False)
        points.extend(zip([lat]*len(lon_steps), lon_steps))

    return points


def robust_mean(arr, discard_percentile=0.2, ret_n=False, axis=None):
    J = np.logical_not(np.isnan(arr))
    if axis is not None:
        J = np.all(np.isnan(arr), axis=1 if axis == 0 else 0)
    if axis == 0:
        arr = arr[J, :]
    elif axis == 1:
        arr = arr[:, J]
    else:
        arr = arr[J]

    low = np.percentile(arr, discard_percentile, axis=axis)
    high = np.percentile(arr, 100 - discard_percentile, axis=axis)
    I = np.logical_and(low < arr, arr < high)
    if axis is not None:
        I = np.all(I, axis=1 if axis == 0 else 0)
    m = np.mean(arr[:, I] if axis == 1 else arr[I], axis=axis)
    return (m, np.sum(I, axis=axis)) if ret_n else m


def robust_std(arr, discard_percentile=0.2, mean=None, axis=None):
    corr = 1
    if mean is None:
        mean, n = robust_mean(arr, discard_percentile=discard_percentile, ret_n=True, axis=axis)
        corr = n/(n-1)
    return np.sqrt(robust_mean((arr-mean)**2, discard_percentile=discard_percentile, axis=axis) * corr)


def mv_normal(mean, cov=None, L=None, size=None):
    if size is None:
        final_shape = []
    elif isinstance(size, (int, np.integer)):
        final_shape = [size]
    else:
        final_shape = size
    final_shape = list(final_shape[:])
    final_shape.append(mean.shape[0])
    
    if L is None and cov is None \
          or L is not None and cov is not None:
        raise ValueError("you must provide either cov or L (cholesky decomp result)")
    if len(mean.shape) != 1:
        raise ValueError("mean must be 1 dimensional")
    
    if L is not None:
        if (len(L.shape) != 2) or (L.shape[0] != L.shape[1]):
            raise ValueError("L must be 2 dimensional and square")
        if mean.shape[0] != L.shape[0]:
            raise ValueError("mean and L must have same length")

    if cov is not None:
        if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
            raise ValueError("cov must be 2 dimensional and square")
        if mean.shape[0] != cov.shape[0]:
            raise ValueError("mean and cov must have same length")
        L = np.linalg.cholesky(cov)

    from numpy.random import standard_normal
    z = standard_normal(final_shape).reshape(mean.shape[0],-1)
    
    x = L.dot(z).T
    x += mean
    x.shape = tuple(final_shape)
    
    return x, L


def point_cloud_vs_model_err(points: np.ndarray, model) -> np.ndarray:
    faces = np.array([f[0] for f in model.faces], dtype='uint')
    vertices = np.array(model.vertices)
    errs = get_model_errors(points, vertices, faces)
    return errs


#@nb.njit(nb.f8[:](nb.f8[:, :], nb.f8[:, :]), nogil=True)
@nb.njit(nb.f8(nb.f8[:, :], nb.f8[:, :]), nogil=True, cache=True)
def poly_line_intersect(poly, line):
#    extend_line = True
    eps = 1e-6
    none = np.inf  # np.zeros(1)

    v0v1 = poly[1, :] - poly[0, :]
    v0v2 = poly[2, :] - poly[0, :]

    dir = line[1, :] - line[0, :]
    line_len = math.sqrt(np.sum(dir**2))
    if line_len < eps:
        return none

    dir = dir/line_len
    pvec = cross3d(dir, v0v2).ravel()
    det = np.dot(v0v1, pvec)
    if abs(det) < eps:
        return none

    # backface culling
    if False and det < 0:
        return none

    # frontface culling
    if False and det > 0:
        return none

    inv_det = 1.0 / det
    tvec = line[0, :] - poly[0, :]
    u = tvec.dot(pvec) * inv_det

    if u + eps < 0 or u - eps > 1:
        return none

    qvec = cross3d(tvec, v0v1).ravel()
    v = dir.dot(qvec) * inv_det

    if v + eps < 0 or u + v - eps > 1:
        return none

    t = v0v2.dot(qvec) * inv_det
    if True:
        # return error directly
        return t - line_len
    else:
        # return actual 3d intersect point
        if not extend_line and t - eps > line_len:
            return none
        return line[0, :] + t*dir


# INVESTIGATE: parallel = True does not speed up at all (or marginally) for some reason even though all cores are in use
@nb.njit(nb.f8(nb.u4[:, :], nb.f8[:, :], nb.f8[:, :]), nogil=True, parallel=False, cache=True)
def intersections(faces, vertices, line):
    #pts = np.zeros((10, 3))
    #i = 0
    min_err = np.ones(faces.shape[0])*np.inf
    for k in nb.prange(1, faces.shape[0]):
        err = poly_line_intersect(vertices[faces[k, :], :], line)
        min_err[k] = err
#        if abs(err) < min_err:
#            min_err = err
        # if len(pt) == 3:
        #     pts[i, :] = pt
        #     i += 1
        #     if i >= pts.shape[0]:
        #         print('too many intersects')
        #         i -= 1

    i = np.argmin(np.abs(min_err))
    return min_err[i]  # pts[0:i, :]


#@nb.jit(nb.f8[:](nb.f8[:, :], nb.f8[:, :], nb.i4[:, :]), nogil=True, parallel=False)
def get_model_errors(points, vertices, faces):
    count = len(points)
    show_progress(count//10, 0)
    j = 0

    devs = np.empty(points.shape[0])
    for i in nb.prange(count):
        vx = points[i, :]
        err = intersections(faces, vertices, np.array(((0, 0, 0), vx)))
        if math.isinf(err):  # len(pts) == 0:
            print('no intersections!')
            continue

        if False:
            idx = np.argmin([np.linalg.norm(pt-vx) for pt in pts])
            err = np.linalg.norm(pts[idx]) - np.linalg.norm(vx)

        devs[i] = err
        if j < i//10:
            show_progress(count//10, i//10)
            j = i//10

    return devs


def crop_model(model, cam_v, cam_q, x_fov, y_fov):
    assert False, 'not implemented'


def augment_model(model, multiplier=3, length_scales=(0, 0.1, 1), sds=(1e-5, 1.6e-4, 2.4e-4)):
    assert multiplier > 1 and multiplier % 1 == 0, 'multiplier must be integer and >1'
    from scipy.interpolate import LinearNDInterpolator
    try:
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    except:
        print('Requires scikit-learn, install using "conda install scikit-learn"')
        sys.exit()

    points = np.array(model.vertices)
    max_rng = np.max(np.ptp(points, axis=0))

    # white noise to ensure positive definite covariance matrix
    ls = dict(zip(length_scales, sds))
    sd0 = ls.pop(0, 1e-5)
    kernel = WhiteKernel(noise_level=sd0*max_rng)

    for l, s in ls.items():
        kernel += s**2 * Matern(length_scale=l*max_rng, nu=1.5)

    assert False, 'not implemented'

    # TODO: how is the covariance mx constructed again?
    y_cov = kernel(points)

    # TODO: sample gp ??? how to tie existing points and generate the new points in between?
    aug_points, L = mv_normal(points, cov=y_cov)

    # TODO: how to interpolate faces?
    pass

    # interpolate texture
    # TODO: augment texture
    interp = LinearNDInterpolator(points, model.texcoords)
    aug_texcoords = interp(aug_points)

    data = model.as_dict()
    data['faces'] = aug_faces
    data['vertices'] = aug_points
    data['texcoords'] = aug_texcoords
    from visnav.iotools import objloader
    aug_model = objloader.ShapeModel(data=data)
    aug_model.recalc_norms()

    return aug_model, L


def apply_noise(model, support=None, L=None, len_sc=SHAPE_MODEL_NOISE_LEN_SC, 
                noise_lv=SHAPE_MODEL_NOISE_LV['lo'], only_z=False):
    
    noisy_points, avg_dev, L = points_with_noise(points=np.array(model.vertices),
            support=support, L=L, len_sc=len_sc, noise_lv=noise_lv, only_z=only_z)

    data = model.as_dict()
    data['vertices'] = noisy_points
    from visnav.iotools import objloader
    noisy_model = objloader.ShapeModel(data=data)
    noisy_model.recalc_norms()
    
    return noisy_model, avg_dev, L
    

def points_with_noise(points, support=None, L=None, len_sc=SHAPE_MODEL_NOISE_LEN_SC, 
                noise_lv=SHAPE_MODEL_NOISE_LV['lo'], max_rng=None, only_z=False):
    
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    import random
    import time
    
    try:
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    except:
        print('Requires scikit-learn, install using "conda install scikit-learn"')
        sys.exit()
    
    if support is None:
        support = points #[random.sample(list(range(len(points))), min(3000,len(points)))]
    
    n = len(support)
    mean = np.mean(points, axis=0)
    max_rng = np.max(np.ptp(points, axis=0)) if max_rng is None else max_rng
    
    y_cov = None
    if L is None:
        kernel = 0.6*noise_lv * Matern(length_scale=len_sc*max_rng, nu=1.5) \
               + 0.4*noise_lv * Matern(length_scale=0.1*len_sc*max_rng, nu=1.5) \
               + WhiteKernel(noise_level=1e-5*noise_lv*max_rng) # white noise for positive definite covariance matrix only
        y_cov = kernel(support - mean)

    # sample gp
    e0, L = mv_normal(np.zeros(n), cov=y_cov, L=L)
    err = np.exp(e0).reshape((-1, 1))

    if len(err) == len(points):
        full_err = err
        if DEBUG:
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
            try:
                full_err[nanidx,] = naninterp(points[nanidx,:])
            except IndexError as e:
                raise IndexError('%s,%s,%s'%(err.shape, full_err.shape, points.shape)) from e

    # extra high frequency noise
    # white_noise = 1 if True else np.exp(np.random.normal(scale=0.2*noise_lv*max_rng, size=(len(full_err),1)))

    if only_z:
        add_err_z = (max_rng/2) * (full_err - 1)
        add_err = np.concatenate((np.zeros((len(full_err), 2)), add_err_z), axis=1)
        noisy_points = points + add_err
        devs = np.abs(noisy_points[:, 2] - points[:, 2]) / (max_rng/2)
        assert np.isclose(devs.flatten(), np.abs(full_err - 1).flatten()).all(), 'something wrong'
    else:
        # noisy_points = (points-mean)*full_err*white_noise +mean
        #r = np.sqrt(np.sum((points - mean)**2, axis=-1)).reshape(-1, 1)
        #noisy_points = (points - mean) * (1 + np.log(full_err)/r) + mean
        noisy_points = (points - mean) * full_err + mean
        devs = np.sqrt(np.sum((noisy_points - points)**2, axis=-1) / np.sum((points - mean)**2, axis=-1))
    
    if DEBUG or not BATCH_MODE:
        print('noise (lv=%.3f): %.3f, %.3f; avg=%.3f'%((noise_lv,)+tuple(np.percentile(devs, (68, 95)))+(np.mean(devs),)))
    
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
    
    return noisy_points, np.mean(devs), L


def foreground_idxs(array, max_val=None):
    iy, ix = np.where(array < max_val)
    idxs = np.concatenate(((iy,), (ix,)), axis=0).T
    return idxs

def interp2(array, x, y, max_val=None, max_dist=30, idxs=None, discard_bg=False):
    assert y<array.shape[0] and x<array.shape[1], 'out of bounds %s: %s'%(array.shape, (y, x))

    v = array[int(y):int(y)+2, int(x):int(x)+2]
    xf = x-int(x)
    yf = y-int(y)
    w = np.array((
        ((1-yf)*(1-xf), (1-yf)*xf),
        (yf*(1-xf),     yf*xf),
    ))
    
    # ignore background depths
    if max_val is not None:
        idx = v.reshape(1,-1) < max_val*0.999
    else:
        idx = ~np.isnan(v.reshape(1,-1))
    
    w_sum = np.sum(w.reshape(1,-1)[idx])
    if w_sum>0:
        # ignore background values
        val = np.sum(w.reshape(1,-1)[idx] * v.reshape(1,-1)[idx]) / w_sum

    elif discard_bg:
        return float('nan')

    else:
        # no foreground values in 2x2 matrix, find nearest foreground value
        if idxs is None:
            idxs = foreground_idxs(array, max_val)

        fallback = len(idxs)==0
        if not fallback:
            dist = np.linalg.norm(idxs - np.array((y, x)), axis=1)
            i = np.argmin(dist)
            val = array[idxs[i, 0], idxs[i, 1]]
            #print('\n%s, %s, %s, %s, %s, %s, %s'%(v, x,y,dist[i],idxs[i,1],idxs[i,0],val))
            fallback = dist[i] > max_dist
            
        if fallback:
            val = np.sum(w*v)/np.sum(w)
        
    return val


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
        #d.append(1-np.array((tq.w, tq.x, tq.y, tq.z)).dot(np.array((dst_q.w, dst_q.x, dst_q.y, dst_q.z)))**2)
        d.append(angle_between_q(tq, dst_q))
    i = np.argmin(d)
    return qs[i]


def hover_annotate(fig, ax, line, annotations):
    annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        idx = ind["ind"][0]
        try:
            # for regular plots
            x, y = line.get_data()
            annot.xy = (x[idx], y[idx])
        except AttributeError:
            # for scatter plots
            annot.xy = tuple(line.get_offsets()[idx])
        text = ", ".join([annotations[n] for n in ind["ind"]])
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = line.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


def plot_vectors(pts3d, scatter=True, conseq=True, neg_z=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)

    if scatter:
        ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
    else:
        if conseq:
            ax.set_prop_cycle('color', map(lambda c: '%f' % c, np.linspace(1, 0, len(pts3d))))
        for i, v1 in enumerate(pts3d):
            if v1 is not None:
                ax.plot((0, v1[0]), (0, v1[1]), (0, v1[2]))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if neg_z:
        ax.view_init(90, -90)
    else:
        ax.view_init(-90, -90)
    plt.show()


def numeric(s):
    try:
        float(s)
    except ValueError:
        return False
    return True


def plot_quats(quats, conseq=True, wait=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if conseq:
        ax.set_prop_cycle('color', map(lambda c: '%f' % c, np.linspace(1, 0, len(quats))))
    for i, q in enumerate(quats):
        if q is not None:
            lat, lon, _ = q_to_ypr(q)
            v1 = spherical2cartesian(lat, lon, 1)
            v2 = (v1 + normalize_v(np.cross(np.cross(v1, np.array([0, 0, 1])), v1))*0.1)*0.85
            v2 = q_times_v(q, v2)
            ax.plot((0, v1[0], v2[0]), (0, v1[1], v2[1]), (0, v1[2], v2[2]))

    while(wait and not plt.waitforbuttonpress()):
        pass


def plot_poses(poses, conseq=True, wait=True, arrow_len=1):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if conseq:
        plt.hsv()
        #ax.set_prop_cycle('color', map(lambda c: '%f' % c, np.linspace(.7, 0, len(poses))))
    for i, pose in enumerate(poses):
        if pose is not None:
            q = np.quaternion(*pose[3:])
            lat, lon, _ = q_to_ypr(q)
            v1 = spherical2cartesian(lat, lon, 1)*arrow_len
            v2 = (v1 + normalize_v(np.cross(np.cross(v1, np.array([0, 0, 1])), v1))*0.1*arrow_len)*0.85
            v2 = q_times_v(q, v2)
            ax.plot((pose[0], v1[0], v2[0]), (pose[1], v1[1], v2[1]), (pose[2], v1[2], v2[2]))

    while(wait and not plt.waitforbuttonpress()):
        pass


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


def show_progress(tot, i):
    digits = int(math.ceil(math.log10(tot+1)))
    if i == 0:
        print('%s/%d' % ('0' * digits, tot), end='', flush=True)
    else:
        print(('%s%0' + str(digits) + 'd/%d') % ('\b' * (digits * 2 + 1), i + 1, tot), end='', flush=True)


def smooth1d(xt, x, Y, weight_fun=lambda d: 0.9**abs(d)):
    if xt.ndim != 1 or x.ndim != 1:
        raise ValueError("smooth1d only accepts 1 dimension arrays for location")
    if x.shape[0] != Y.shape[0]:
        raise ValueError("different lenght x and Y")

    D = np.repeat(np.expand_dims(xt, 1), len(x), axis=1) - np.repeat(np.expand_dims(x, 0), len(xt), axis=0)
    weights = np.array(list(map(weight_fun, D.flatten()))).reshape(D.shape)
    Yt = np.sum(Y*weights, axis=1) / np.sum(weights, axis=1)

    return Yt