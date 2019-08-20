"""
Based on Scipy's cookbook:
http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""


import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


def bundleAdjust(poses, pts3d, pts2d, cam_idxs, pt2d_idxs):
    """
    Returns the bundle adjusted parameters, in this case the optimized rotation and translation vectors.

    poses with shape (n_cameras, 9) contains initial estimates of parameters for all cameras.
            First 3 components in each row form a rotation vector,
            next 3 components form a translation vector,
            then a focal distance and two distortion parameters.

    pts3d with shape (n_points, 3)
            contains initial estimates of point coordinates in the world frame.

    pts2d with shape (n_observations, 2)
            contains measured 2-D coordinates of points projected on images in each observations.

    cam_idxs with shape (n_observations,)
            contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.

    pt2d_idxs with shape (n_observations,)
            contatins indices of points (from 0 to n_points - 1) involved in each observation.

    """
    n_cams = poses.shape[0]
    n_pts = pts3d.shape[0]

    x0 = np.hstack((poses.ravel(), pts3d.ravel()))
    A = _bundle_adjustment_sparsity(n_cams, n_pts, cam_idxs, pt2d_idxs)
    res = least_squares(_costfun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                         args=(n_cams, n_pts, cam_idxs, pt2d_idxs, pts2d))
    params = _optimized_params(res.x, n_cams, n_pts, cam_idxs, pt2d_idxs, pts2d)

    return params


def _rotate(self, points, rot_vecs):
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


def _project(self, pts3d, poses):
    """Convert 3-D points to 2-D by projecting onto images."""
    pts2d_proj = self.rotate(pts3d, poses[:, :3])
    pts2d_proj += poses[:, 3:6]
    pts2d_proj = -pts2d_proj[:, :2] / pts2d_proj[:, 2, np.newaxis]
    f = poses[:, 6]
    k1 = poses[:, 7]
    k2 = poses[:, 8]
    n = np.sum(pts2d_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    pts2d_proj *= (r * f)[:, np.newaxis]
    return pts2d_proj


def _costfun(self, params, n_cams, n_pts, cam_idxs, pt2d_idxs, pts2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cams * 9].reshape((n_cams, 9))
    points_3d = params[n_cams * 9:].reshape((n_pts, 3))
    points_proj = self.project(points_3d[pt2d_idxs], camera_params[cam_idxs])
    return (points_proj - pts2d).ravel()


def _bundle_adjustment_sparsity(self, n_cams, n_pts, cam_idxs, pt2d_idxs):
    m = cam_idxs.size * 2
    n = n_cams * 9 + n_pts * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(cam_idxs.size)
    for s in range(9):
        A[2 * i, cam_idxs * 9 + s] = 1
        A[2 * i + 1, cam_idxs * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cams * 9 + pt2d_idxs * 3 + s] = 1
        A[2 * i + 1, n_cams * 9 + pt2d_idxs * 3 + s] = 1

    return A


def _optimized_params(self, params, n_cams, n_pts):
    """
    Retrieve camera parameters and 3-D coordinates.
    """
    cam_params = params[:n_cams * 9].reshape((n_cams, 9))
    pts3d = params[n_cams * 9:].reshape((n_pts, 3))

    return cam_params, pts3d






