"""
Based on Scipy's cookbook:
http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""
import sys
import logging

import numpy as np

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


def bundle_adj(poses: np.ndarray, pts3d: np.ndarray, pts2d: np.ndarray,
               cam_idxs: np.ndarray, pt3d_idxs: np.ndarray, K: np.ndarray,
               max_nfev=None, skip_pose0=False):
    """
    Returns the bundle adjusted parameters, in this case the optimized rotation and translation vectors.

    poses with shape (n_cameras, 6) contains initial estimates of parameters for all cameras.
            First 3 components in each row form a rotation vector,
            next 3 components form a translation vector

    pts3d with shape (n_points, 3)
            contains initial estimates of point coordinates in the world frame.

    pts2d with shape (n_observations, 2)
            contains measured 2-D coordinates of points projected on images in each observations.

    cam_idxs with shape (n_observations,)
            contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.

    pt3d_idxs with shape (n_observations,)
            contains indices of points (from 0 to n_points - 1) involved in each observation.

    """
    a = 1 if skip_pose0 else 0
    assert not skip_pose0, 'some bug with skipping first pose optimization => for some reason cost stays high, maybe problem with A?'

    n_cams = poses.shape[0]
    n_pts = pts3d.shape[0]
    A = _bundle_adjustment_sparsity(n_cams-a, n_pts, cam_idxs, pt3d_idxs)
    x0 = np.hstack((poses[a:].ravel(), pts3d.ravel()))
    pose0 = poses[0:a].ravel()

    if False:
        err = _costfun(x0, pose0, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K)
        print('ERR: %.4e' % (np.sum(err**2)/2))

    tmp = sys.stdout
    sys.stdout = LogWriter()
    res = least_squares(_costfun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(pose0, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K), max_nfev=max_nfev)
    sys.stdout = tmp

    new_poses, new_pts3d = _optimized_params(np.hstack((pose0, res.x)), n_cams, n_pts)
    return new_poses, new_pts3d


class LogWriter:
    def write(self, msg):
        if msg.strip() != '':
            logging.info(msg)


def _rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def _project(pts3d, poses, K):
    """Convert 3-D points to 2-D by projecting onto images."""
    pts2d_proj = _rotate(pts3d, poses[:, :3])
    pts2d_proj += poses[:, 3:6]

    # pts2d_proj = -pts2d_proj[:, :2] / pts2d_proj[:, 2, np.newaxis]
    # f = poses[:, 6]
    # k1 = poses[:, 7]
    # k2 = poses[:, 8]
    # n = np.sum(pts2d_proj ** 2, axis=1)
    # r = 1 + k1 * n + k2 * n ** 2
    # pts2d_proj *= (r * f)[:, np.newaxis]

    pts2d_proj = K.dot(pts2d_proj.T).T                              # own addition
    pts2d_proj = pts2d_proj[:, :2] / pts2d_proj[:, 2, np.newaxis]   # own addition

    return pts2d_proj


def _costfun(params, pose0, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K):
    """
    Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    params = np.hstack((pose0, params))
    poses = params[:n_cams * 6].reshape((n_cams, 6))
    points_3d = params[n_cams * 6:].reshape((n_pts, 3))
    points_proj = _project(points_3d[pt3d_idxs], poses[cam_idxs], K)
    return (points_proj - pts2d).ravel()


def _bundle_adjustment_sparsity(n_cams, n_pts, cam_idxs, pt3d_idxs):
    m = cam_idxs.size * 2
    n = n_cams * 6 + n_pts * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(cam_idxs.size)
    for s in range(6):
        A[2 * i, cam_idxs * 6 + s] = 1
        A[2 * i + 1, cam_idxs * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cams * 6 + pt3d_idxs * 3 + s] = 1
        A[2 * i + 1, n_cams * 6 + pt3d_idxs * 3 + s] = 1

    return A


def _optimized_params(params, n_cams, n_pts):
    """
    Retrieve camera parameters and 3-D coordinates.
    """
    cam_params = params[:n_cams * 6].reshape((n_cams, 6))
    pts3d = params[n_cams * 6:].reshape((n_pts, 3))

    return cam_params, pts3d
