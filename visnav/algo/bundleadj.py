"""
Based on Scipy's cookbook:
http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""
import math
import sys
import logging
from functools import lru_cache

import numpy as np
import quaternion

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

from visnav.algo import tools

# logger = logging.getLogger(__name__)


CHECK_JACOBIAN = 0
USE_WORLD_CAM_FRAME = 0
USE_OWN_PSEUDO_HUBER_LOSS = 1


def bundle_adj(poses: np.ndarray, pts3d: np.ndarray, pts2d: np.ndarray,
               cam_idxs: np.ndarray, pt3d_idxs: np.ndarray, K: np.ndarray, px_err_sd: np.ndarray=None,
               dtype=np.float64, log_writer=None, max_nfev=None, skip_pose_n=1, poses_only=False, huber_coef=False):
    """
    Returns the bundle adjusted parameters, in this case the optimized rotation and translation vectors.

    basepose with shape (6,) is the pose of the first frame that is used as an anchor

    poses with shape (n_cameras, 6) contains initial estimates of parameters for all cameras.
            First 3 components in each row form a rotation vector,
            next 3 components form a translation vector

    pts3d with shape (n_points, 3)
            contains initial estimates of point coordinates in the world frame.

    pts2d with shape (n_observations, 2)
            contains measured 2-D coordinates of points projected on images in each observations.

    pose_idxs with shape (n_observations,)
            contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.

    pt3d_idxs with shape (n_observations,)
            contains indices of points (from 0 to n_points - 1) involved in each observation.

    """
    assert len(poses.shape) == 2 and poses.shape[1] == 6, 'wrong shape poses: %s' % (poses.shape,)
    assert len(pts3d.shape) == 2 and pts3d.shape[1] == 3, 'wrong shape pts3d: %s' % (pts3d.shape,)
    assert len(pts2d.shape) == 2 and pts2d.shape[1] == 2, 'wrong shape pts2d: %s' % (pts2d.shape,)
    assert len(cam_idxs.shape) == 1, 'wrong shape pose_idxs: %s' % (cam_idxs.shape,)
    assert len(pt3d_idxs.shape) == 1, 'wrong shape pt3d_idxs: %s' % (pt3d_idxs.shape,)
    assert K.shape == (3, 3), 'wrong shape K: %s' % (K.shape,)

    n_cams = poses.shape[0]
    n_pts = pts3d.shape[0]

    pose0 = poses[0:skip_pose_n].ravel()
    x0 = [poses[skip_pose_n:].ravel()]

    if poses_only:
        fixed_pt3d = pts3d
    else:
        fixed_pt3d = np.zeros((0, 3))
        x0.append(pts3d.ravel())

    x0 = np.hstack(x0)
    I = np.stack((np.ones((len(poses) - skip_pose_n, 3), dtype=bool),
                  np.zeros((len(poses) - skip_pose_n, 3), dtype=bool)), axis=1).ravel()
    x0 = Manifold(x0.shape, buffer=x0.astype(dtype), dtype=dtype)
    x0.set_so3_groups(I)

    px_err_sd = np.ones((len(pts2d),)) if px_err_sd is None else px_err_sd

    m = cam_idxs.size * 2
    orig_dtype = pts2d.dtype
    if dtype != orig_dtype:
        pose0, fixed_pt3d, pts2d, K, px_err_sd = map(
            lambda x: x.astype(dtype), (pose0, fixed_pt3d, pts2d, K, px_err_sd))

    if 0:
        err = _costfun(x0, pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K, px_err_sd,
                       meas_r, meas_aa, meas_idxs, loc_err_sd, ori_err_sd)
        print('ERR: %.4e' % (np.sum(err**2)/2))
    if CHECK_JACOBIAN:
        if 1:
            jac = _jacobian(x0, pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K, px_err_sd).toarray()
            jac_ = numerical_jacobian(lambda x: _costfun(x, pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs,
                                                         pts2d, K, px_err_sd), x0, 1e-4)
        else:
            jac = _rot_pt_jac(x0, pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K, px_err_sd)
            jac_ = numerical_jacobian(lambda x: _rotated_points(x, pose0, fixed_pt3d, n_cams, n_pts, cam_idxs,
                                                                pt3d_idxs, pts2d, K, px_err_sd), x0, 1e-4)

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        axs[0].imshow(np.sign(jac) * np.abs(jac) ** (1/8))
        axs[0].set_title('analytical')
        axs[1].imshow(np.sign(jac_) * np.abs(jac_) ** (1/8))
        axs[1].set_title('numerical')
        plt.show()

    @lru_cache(1)
    def _cfun(x):
        return _costfun(x, pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K, px_err_sd)

    def cfun(x):
        err = _cfun(tuple(x))
        if USE_OWN_PSEUDO_HUBER_LOSS:
            sqrt_phl_err = np.sqrt(tools.pseudo_huber_loss(err, huber_coef)) if huber_coef is not False else err
        else:
            sqrt_phl_err = err
        return sqrt_phl_err

    def jac(x):
        # apply pseudo huber loss derivative
        J = _jacobian(x, pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K, px_err_sd)

        if USE_OWN_PSEUDO_HUBER_LOSS and huber_coef is not False:
            ## derivative of pseudo huber loss can be applied afterwards by multiplying with the err derivative
            # NOTE: need extra (.)**0.5 term as least_squares applies extra (.)**2 on top of our cost function
            # - d (weight * phl(e))**0.5/dksi = d (weight * phl(e))**0.5/d phl(e) * d phl(e)/de * de/dksi
            # - d (weight * phl(e))**0.5/d phl(e) = weight**0.5  * 0.5 * phl(e)**(-0.5)
            # - d phl(e)/de = e/sqrt(1+e**2/delta**2)
            # - de/dksi is solved next
            err = _cfun(tuple(x))
            phl_err = tools.pseudo_huber_loss(err, huber_coef)
            dhuber = 0.5 * np.sqrt(1 / phl_err) * err / np.sqrt(1 + err ** 2 / huber_coef ** 2)
            J = J.multiply(dhuber.reshape((-1, 1)))

        return J

    tmp = sys.stdout
    if log_writer is not None:
        sys.stdout = log_writer
    res = least_squares(cfun, x0, verbose=2, ftol=1e-4, xtol=1e-5, method='trf',
                        x_scale='jac', jac=jac, max_nfev=max_nfev,
                        loss='linear' if USE_OWN_PSEUDO_HUBER_LOSS or huber_coef is None else 'huber',
                        f_scale=huber_coef or 1.0)
    sys.stdout = tmp

    # TODO (1): return res.fun somewhat processed (per frame: mean px err, meas errs),
    #           so that can follow the errors and notice/debug problems

    assert isinstance(res.x, Manifold), 'somehow manifold lost during optimization'
    new_poses, new_pts3d = _unpack(res.x.to_array(), n_cams - skip_pose_n, 0 if poses_only else n_pts)

    if new_poses.dtype != orig_dtype:
        new_poses = new_poses.astype(orig_dtype)
        new_pts3d = new_pts3d.astype(orig_dtype)

    return new_poses, new_pts3d


def _costfun(params, pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K, px_err_sd):
    """
    Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    if isinstance(params, (tuple, list)):
        params = np.array(params)

    params = np.hstack((pose0, params))
    poses, pts3d = _unpack(params, n_cams, n_pts if len(fixed_pt3d) == 0 else 0)

    points_3d = fixed_pt3d if len(pts3d) == 0 else pts3d
    points_proj = _project(points_3d[pt3d_idxs], poses[cam_idxs], K)

    px_err = ((pts2d - points_proj) / px_err_sd[:, None]).ravel()
    return px_err


def _jacobian(params, pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K, px_err_sd):
    """
    cost function jacobian, from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6891346/
    """

    params = np.hstack((pose0, params))
    poses, pts3d = _unpack(params, n_cams, n_pts if len(fixed_pt3d) == 0 else 0)

    poses_only = len(fixed_pt3d) > 0
    points_3d = fixed_pt3d if poses_only else pts3d
    points_3d_rot = _rotate(points_3d[pt3d_idxs], poses[cam_idxs, :3])
    points_3d_cf = points_3d_rot + poses[cam_idxs, 3:6]

    # error term count (2d reprojection errors)
    m = cam_idxs.size * 2

    # parameter count (6d poses, n_p x 3d keypoint locations, 1d time offset)
    n1 = n_cams * 6
    n2 = (0 if poses_only else n_pts * 3)
    n = n1 + n2

    J = lil_matrix((m, n), dtype=np.float32)
    i = np.arange(cam_idxs.size)

    # poses affect reprojection error terms
    ########################################
    #  - NOTE: for the following to work, need the poses in cam-to-world, i.e. rotation of world origin first,
    #    then translation in rotated coordinates
    #  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6891346/  equation (11)
    #  - also https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf, section 10.3.5, eq 10.23
    #  - ksi: first three location, second three rotation
    #  - de/dksi = de/dU' * dU'/dksi
    #  - de/dU' = -[[fx/Zc, 0, -fx*Xc/Zc**2],
    #               [0, fy/Zc, -fy*Yc/Zc**2]]
    #  - dU'/dw = [I3 | -[U']^] = [[1, 0, 0, 0, Zr, -Yr],
    #                              [0, 1, 0, -Zr, 0, Xr],
    #                              [0, 0, 1, Yr, -Xr, 0]]
    #  - -[[fx/Zc, 0, -Xc*fx/Zc**2, | -Xc*Yr*fx/Zc**2, Xc*Xr*fx/Zc**2 + fx*Zr/Zc, -Yr*fx/Zc],
    #      [0, fy/Zc, -Yc*fy/Zc**2, | -Yc*Yr*fy/Zc**2 - fy*Zr/Zc, Xr*Yc*fy/Zc**2, Xr*fy/Zc]]    / px_err_sd

    fx, fy = K[0, 0], K[1, 1]
    Xr, Yr, Zr = points_3d_rot[:, 0], points_3d_rot[:, 1], points_3d_rot[:, 2]
    Xc, Yc, iZc = points_3d_cf[:, 0], points_3d_cf[:, 1], 1 / points_3d_cf[:, 2]
    px_err_sd = px_err_sd.flatten()

    # camera location
    J[2 * i + 0, cam_idxs*6 + 3] = -(fx*iZc) / px_err_sd
    J[2 * i + 0, cam_idxs*6 + 5] = -(-Xc*fx*iZc**2) / px_err_sd
    J[2 * i + 1, cam_idxs*6 + 4] = -(fy*iZc) / px_err_sd
    J[2 * i + 1, cam_idxs*6 + 5] = -(-Yc*fy*iZc**2) / px_err_sd

    # camera rotation
    J[2 * i + 0, cam_idxs*6 + 0] = -(-Xc*Yr*fx*iZc**2) / px_err_sd
    J[2 * i + 0, cam_idxs*6 + 1] = -(Xc*Xr*fx*iZc**2 + fx*Zr*iZc) / px_err_sd
    J[2 * i + 0, cam_idxs*6 + 2] = -(-Yr*fx*iZc) / px_err_sd
    J[2 * i + 1, cam_idxs*6 + 0] = -(-Yc*Yr*fy*iZc**2 - fy*Zr*iZc) / px_err_sd
    J[2 * i + 1, cam_idxs*6 + 1] = -(Xr*Yc*fy*iZc**2) / px_err_sd
    J[2 * i + 1, cam_idxs*6 + 2] = -(Xr*fy*iZc) / px_err_sd

    # keypoint locations affect reprojection error terms
    ####################################################
    # similar to above, first separate de/dU => de/dU' * dU'/dU,
    # first one (de/dU') is [[fx/Z', 0, -fx*X'/Z'**2],
    #                        [0, fy/Z', -fy*Y'/Z'**2]]
    # second one (dU'/dU => d/dU (RU + P) = R) is the camera rotation matrix R
    # => i.e. rotate de/dU' by R^-1
    if not poses_only:
        dEu = -np.stack((fx * iZc, np.zeros((len(iZc),)), -fx * Xc * iZc**2), axis=1) / px_err_sd[:, None]
        dEv = -np.stack((np.zeros((len(iZc),)), fy * iZc, -fy * Yc * iZc**2), axis=1) / px_err_sd[:, None]
        dEuc = _rotate(dEu, -poses[cam_idxs, :3])
        dEvc = _rotate(dEv, -poses[cam_idxs, :3])
        J[2 * i + 0, n1 + pt3d_idxs * 3 + 0] = dEuc[:, 0]
        J[2 * i + 0, n1 + pt3d_idxs * 3 + 1] = dEuc[:, 1]
        J[2 * i + 0, n1 + pt3d_idxs * 3 + 2] = dEuc[:, 2]
        J[2 * i + 1, n1 + pt3d_idxs * 3 + 0] = dEvc[:, 0]
        J[2 * i + 1, n1 + pt3d_idxs * 3 + 1] = dEvc[:, 1]
        J[2 * i + 1, n1 + pt3d_idxs * 3 + 2] = dEvc[:, 2]

    # maybe skip first poses
    if pose0.size > 0:
        J = J[:, pose0.size:]

    # for debugging
    if 0:
        import matplotlib.pyplot as plt
        plt.figure(2)
        plt.imshow((J.toarray() != 0).astype(int))
        plt.show()

    return J.tocsr()


def numerical_jacobian(fun, x0, eps=1e-4):
    if 1:
        from tqdm import tqdm
        y0 = fun(x0)
        J = np.zeros((len(y0), len(x0)))
        for j in tqdm(range(len(x0))):
            d = np.zeros((len(x0),))
            d[j] = eps
            x1 = x0 + d
            y1 = fun(x1)
            yd = (y1 - y0) / eps
            J[:, j] = yd
    elif 0:
        from scipy.optimize._numdiff import approx_derivative
        J = approx_derivative(fun, x0, method='2-point')
    else:
        from scipy.optimize import approx_fprime
        y0 = fun(x0)
        J = np.zeros((len(y0), len(x0)))
        for i in tqdm(range(10)):  #len(y0))):
            J[i, :] = approx_fprime(x0, lambda x: fun(x)[i], epsilon=eps)
    return J


def _rotated_points(params, pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K, px_err_sd):
    """
    Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    if isinstance(params, (tuple, list)):
        params = np.array(params)

    params = np.hstack((pose0, params))
    poses, pts3d, t_off = _unpack(params, n_cams, n_pts if len(fixed_pt3d) == 0 else 0)

    points_3d = fixed_pt3d if len(pts3d) == 0 else pts3d
    points_3d_cf = _rotate(points_3d[pt3d_idxs], poses[cam_idxs, :3]) + poses[cam_idxs, 3:6]
    return points_3d_cf.flatten()


def _rot_pt_jac(params, pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K, px_err_sd):

    params = np.hstack((pose0, params))
    poses, pts3d, t_off = _unpack(params, n_cams, n_pts if len(fixed_pt3d) == 0 else 0)

    poses_only = len(fixed_pt3d) > 0
    points_3d = fixed_pt3d if poses_only else pts3d
    points_3d_cf = _rotate(points_3d[pt3d_idxs], poses[cam_idxs, :3]) #+ poses[pose_idxs, 3:6]

    # output term count (rotated coords)
    m = cam_idxs.size * 3

    # parameter count (6d poses)
    n = n_cams * 6

    J = np.zeros((m, n))
    i = np.arange(cam_idxs.size)

    # poses affect reprojection error terms
    ########################################
    #  - NOTE: for the following to work, need the poses in cam-to-world, i.e. rotation of world origin first,
    #    then translation in rotated coordinates
    #  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6891346/  equation (11)
    #  - also https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf, section 10.3.5, eq 10.23
    #  - ksi: first three location, second three rotation
    #  - de/dksi = de/dU' * dU'/dksi
    #  - de/dU' = -[[fx/Zc, 0, -fx*Xc/Zc**2],
    #               [0, fy/Zc, -fy*Yc/Zc**2]]
    #  - dU'/dw = [I3 | -[U']^] = [[1, 0, 0, 0, Zc, -Yc],
    #                              [0, 1, 0, -Zc, 0, Xc],
    #                              [0, 0, 1, Yc, -Xc, 0]]
    #  - -[[fx/Zc, 0, -Xc*fx/Zc**2, | -Xc*Yc*fx/Zc**2, Xc**2*fx/Zc**2 + fx, -Yc*fx/Zc],
    #      [0, fy/Zc, -Yc*fy/Zc**2, | -Yc**2*fy/Zc**2 - fy, Xc*Yc*fy/Zc**2, Xc*fy/Zc]]    / px_err_sd

    Xc, Yc, Zc = points_3d_cf[:, 0], points_3d_cf[:, 1], points_3d_cf[:, 2]

    # camera location
    J[3 * i + 0, cam_idxs*6 + 3] = 1
    J[3 * i + 0, cam_idxs*6 + 4] = 0
    J[3 * i + 0, cam_idxs*6 + 5] = 0
    J[3 * i + 1, cam_idxs*6 + 3] = 0
    J[3 * i + 1, cam_idxs*6 + 4] = 1
    J[3 * i + 1, cam_idxs*6 + 5] = 0
    J[3 * i + 2, cam_idxs*6 + 3] = 0
    J[3 * i + 2, cam_idxs*6 + 4] = 0
    J[3 * i + 2, cam_idxs*6 + 5] = 1

    # camera rotation
    J[3 * i + 0, cam_idxs*6 + 0] = 0
    J[3 * i + 0, cam_idxs*6 + 1] = Zc
    J[3 * i + 0, cam_idxs*6 + 2] = -Yc
    J[3 * i + 1, cam_idxs*6 + 0] = -Zc
    J[3 * i + 1, cam_idxs*6 + 1] = 0
    J[3 * i + 1, cam_idxs*6 + 2] = Xc
    J[3 * i + 2, cam_idxs*6 + 0] = Yc
    J[3 * i + 2, cam_idxs*6 + 1] = -Xc
    J[3 * i + 2, cam_idxs*6 + 2] = 0

    return J


def _unpack(params, n_cams, n_pts):
    """
    Retrieve camera parameters and 3-D coordinates.
    """
    a = n_cams * 6
    b = a + n_pts * 3

    poses = params[:a].reshape((n_cams, 6))
    # poses[:, :3] = poses[:, :3]/180*n_p.pi

    pts3d = params[a:b].reshape((n_pts, 3))

    return poses, pts3d


def _rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
#     return _cached_rotate(tuple(points.flatten()), tuple(rot_vecs.flatten()))
#
#@lru_cache(maxsize=1)
# def _cached_rotate(points, rot_vecs):
#    points = n_p.array(points).reshape((-1, 3))
#    rot_vecs = n_p.array(rot_vecs).reshape((-1, 3))

    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        k = rot_vecs / theta
        k = np.nan_to_num(k)
    dot = np.sum(points * k, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rot_points = cos_theta * points + sin_theta * np.cross(k, points) + dot * (1 - cos_theta) * k
    return rot_points


def _rotate_rotations(rot_vecs_adj, rot_vecs_base):
    """Rotate rotation vectors by given rotation vectors.

    Rodrigues' rotation formula is used, from
      https://math.stackexchange.com/questions/382760/composition-of-two-axis-angle-rotations
    """

    a = np.linalg.norm(rot_vecs_adj, axis=1)[:, np.newaxis]
    b = np.linalg.norm(rot_vecs_base, axis=1)[:, np.newaxis]

    with np.errstate(invalid='ignore'):
        sin_a, cos_a = np.sin(0.5 * a), np.cos(0.5 * a)
        sin_b, cos_b = np.sin(0.5 * b), np.cos(0.5 * b)
        va = rot_vecs_adj / a
        vb = rot_vecs_base / b
        va = np.nan_to_num(va)
        vb = np.nan_to_num(vb)

    c = 2 * np.arccos(np.clip(cos_a * cos_b - sin_a * sin_b * np.sum(va * vb, axis=1)[:, None], -1, 1))
    sin_c = np.sin(0.5 * c)
    with np.errstate(invalid='ignore'):
        res = (c / sin_c) * (sin_a * cos_b * va +
                             cos_a * sin_b * vb +
                             sin_a * sin_b * np.cross(va, vb))
        res = np.nan_to_num(res)

    return res


def _project(pts3d, poses, K):
    """Convert 3-D points to 2-D by projecting onto images."""
    if USE_WORLD_CAM_FRAME:
        pts3d_rot = _rotate(pts3d - poses[:, 3:6], -poses[:, 0:3])
    else:
        pts3d_rot = _rotate(pts3d, poses[:, 0:3]) + poses[:, 3:6]

    pts2d_proj = K.dot(pts3d_rot.T).T                    # own addition
    pts2d_proj = pts2d_proj[:, 0:2] / pts2d_proj[:, 2:3]   # own addition

    return pts2d_proj


def project(pts3d, poses, K):
    return _project(pts3d, poses, K)


def wedge(w):
    is_mx = len(w.shape) == 2 and w.shape[1] != 1
    w_ = w.reshape((-1, 3))
    zero = np.zeros((len(w_),))
    what = np.array([[zero, -w_[:, 2], w_[:, 1]],
                     [w_[:, 2], zero, -w_[:, 0]],
                     [-w_[:, 1], w_[:, 0], zero]]).transpose((2, 0, 1))
    return what if is_mx else what.squeeze()


def vee(R):
    R_ = R.reshape((-1, 3, 3))
    x = (R_[:, 2, 1] - R_[:, 1, 2]) / 2
    y = (R_[:, 0, 2] - R_[:, 2, 0]) / 2
    z = (R_[:, 1, 0] - R_[:, 0, 1]) / 2
    w = np.array([x, y, z]).T
    R_ -= wedge(w)
#    assert n_p.any(n_p.isclose(n_p.linalg.norm(R_, axis=(1, 2)) / n_p.linalg.norm(w, axis=1), 0))
    return w if len(R.shape) == 3 else w.squeeze()


def logR(R):
    ax1, ax2 = list(range(len(R.shape)))[-2:]
    c = (np.trace(R, axis1=ax1, axis2=ax2) - 1) / 2
    s, th = np.sqrt(1 - c ** 2), np.arccos(c)
    with np.errstate(invalid='ignore'):
        tmp = (0.5 * th / s).reshape((-1, *((1,) * ax2)))
        tmp = np.nan_to_num(tmp)
    log_R = tmp * (R - R.swapaxes(ax1, ax2))
    w = vee(log_R)
    return w


def dlogR_dR(R):
    # from section 10.3.2, eq 10.11, https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
    assert R.shape == (3, 3), 'wrong size rotation matrix'

    c = (np.trace(R) - 1) / 2
    if np.isclose(c, 1):
        S1 = np.array([[0, 0, 0],
                       [0, 0, -.5],
                       [0, .5, 0]])
        S2 = np.array([[0, 0, .5],
                       [0, 0, 0],
                       [-.5, 0, 0]])
        S3 = np.array([[0, -.5, 0],
                       [.5, 0, 0],
                       [0, 0, 0]])
        return np.hstack((S1, S2, S3))

    s, th = np.sqrt(1 - c**2), np.arccos(c)
    a1, a2, a3 = np.array([R[2, 1] - R[1, 2],
                           R[0, 2] - R[2, 0],
                           R[1, 0] - R[0, 1]]) * (0.25 * (th*c - s) / s**3)
    b = 0.5 * th / s
    S1 = np.array([[a1, 0, 0],
                   [a2, 0, -b],
                   [a3, b, 0]])
    S2 = np.array([[0, a1, b],
                   [0, a2, 0],
                   [-b, a3, 0]])
    S3 = np.array([[0, -b, a1],
                   [b, 0, a2],
                   [0, 0, a3]])
    return np.hstack((S1, S2, S3))


class Manifold(np.ndarray):
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        # for some strange reason, super().__init__ is called without specifying it
        self.so3 = None
        self.not_so3 = None
        return self

    def set_so3_groups(self, so3):
        assert isinstance(so3, np.ndarray) and len(so3.shape) == 1 and so3.dtype == bool, \
                'invalid index array type, must be 1-d bool ndarray'
        self.so3 = so3.copy()
        self.not_so3 = np.logical_not(self.so3)

    def copy(self, order='C'):
        new = super(Manifold, self).copy(order)
        new.so3 = self.so3.copy()
        new.not_so3 = self.not_so3.copy()
        return new

    def __add__(self, other):
        assert type(other) == np.ndarray, 'Can only sum an ndarray to this Manifold implementation'
        new = self.to_array(new=True)
        new[self.not_so3] += other[self.not_so3]
        new[self.so3] = _rotate_rotations(other[self.so3].reshape((-1, 3)), new[self.so3].reshape((-1, 3))).flatten()
        new = Manifold(new.shape, buffer=new, dtype=self.dtype)
        new.set_so3_groups(self.so3)
        return new

    def __sub__(self, other):
        assert type(other) in (np.ndarray, Manifold)
        if type(other) == Manifold:
            assert np.all(self.so3 == other.so3)
            return self.__add__(-other.to_array())
        else:
            return self.__add__(-other)

    def to_array(self, new=False):
        return np.frombuffer(np.copy(self) if new else self, dtype=self.dtype)

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        if dtype in (float, np.float32, np.float64):
            return self
        return self.to_array().astype(dtype, order, casting, subok, copy)
