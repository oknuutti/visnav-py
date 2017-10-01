
import math

import numpy as np
import quaternion
from astropy.coordinates import SkyCoord

from settings import *


class PositioningException(Exception):
	pass


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
    return 2*math.atan(qd.w/np.linalg.norm(qd.vec))


def angle_between_ypr(ypr1, ypr2):
    q1 = spherical_to_q(*ypr1)
    q2 = spherical_to_q(*ypr2)
    return angle_between_q(q1, q2)


def q_to_unitbase(q):
    U0 = quaternion.as_quat_array([[0,1,0,0], [0,0,1,0], [0,0,0,1.]])
    Uq = q * U0 * q.conj()
    return quaternion.as_float_array(Uq)[:, 1:]


def equatorial_to_ecliptic(ra, dec):
    """ translate from equatorial coordinates to ecliptic ones """
    sc = SkyCoord(ra, dec, frame='icrs', unit='deg',
            obstime='J2000').transform_to('barycentrictrueecliptic')
    return math.radians(sc.lat.value), math.radians(sc.lon.value)
    
    
def q_to_angleaxis(q):
    theta = math.acos(q.w) * 2.0
    return theta, normalize_v(np.array([q.x, q.y, q.z]))


def angleaxis_to_q(theta, v):
    w = math.cos(theta/2)
    try:
        sc = 1/math.sqrt(sum(map(lambda x: x**2, v)))
        v = list(map(lambda x: x*math.sin(theta/2)*sc, v))
    except Exception as e:
        raise Exception('%s'%v) from e
    return np.quaternion(w, *v).normalized()


def spherical_to_q(lat, lon, roll):
    # Tait-Bryan angles, aka yaw-pitch-roll, nautical angles, cardan angles
    # intrinsic euler rotations z-y'-x'', pitch=-lat, yaw=lon
    return (
          np.quaternion(math.cos(lon/2), 0, 0, math.sin(lon/2))
        * np.quaternion(math.cos(-lat/2), 0, math.sin(-lat/2), 0)
        * np.quaternion(math.cos(roll/2), math.sin(roll/2), 0, 0)
    )
    
    
def q_to_spherical(q):
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


def normalize_v(v):
    return v/math.sqrt(sum(map(lambda x: x**2, v)))


def wrap_rads(a):
    return (a+math.pi)%(2*math.pi)-math.pi


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
    
    # angle between camera axis and the sun, 0: right ahead, pi: behind
    elong = angle_between_v(-ast_v, sco_x)

    # direction the sun is at when looking along camera axis
    nvec = np.cross(sco_x, ast_v)
    direc = angle_between_v(nvec, sco_z)

    # decide if direction needs to be negative or not
    if np.cross(nvec, sco_z).dot(sco_x) < 0:
        direc = -direc

    return elong, direc

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
