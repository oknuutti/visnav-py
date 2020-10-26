import math
import random

import numpy as np
import quaternion
from scipy import integrate
import cv2
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator
from scipy.interpolate.interpnd import LinearNDInterpolator

from visnav.algo.image import ImageProc
from visnav.algo import tools


class Particles:
    # TODO: figure out a better way than the following coef (maybe px_sr involved still?)
    CONE_INTENSITY_COEF = 3e-8      # heuristical coef that scales the jet intensities

    (
        TYPE_HAZE_ONLY,
        TYPE_CONES,
        TYPE_VOXELS,
    ) = range(3)

    def __init__(self, cam, density, scale, voxels=None, cones=None, haze=0.0):
        self.cam = cam
        self.density = density
        self.scale = scale
        self.voxels = voxels
        self.cones = cones
        self.haze = haze
        self.type = Particles.TYPE_HAZE_ONLY
        if cones is not None:
            self.type = Particles.TYPE_CONES
        elif voxels is not None:
            self.type = Particles.TYPE_VOXELS
        else:
            assert haze > 0, 'Need to give either cones, voxels, or haze. Not both cones and voxels though.'

    def flux_density(self, img, dist, mask, lf_ast_v, lf_ast_q, lf_light_v, solar_flux):
        if self.voxels is not None:
            voxel_res = self.flux_density_voxels(lf_ast_v, lf_ast_q, mask, solar_flux, down_scaling=10, quad_lim=5)
        else:
            voxel_res = np.array([0])

        if self.cones is not None:
            if isinstance(self.cones, dict):
                n = self.cones.pop('n')
                self.cones = Cone.random(n, self.cam, img, dist, mask, lf_ast_v, lf_ast_q, **self.cones)

            cone_res = self.flux_density_cones(lf_ast_v, lf_ast_q, mask, solar_flux, down_scaling=6)
        else:
            cone_res = np.array([0])

        if self.haze > 0:
            haze_res = self.flux_density_haze(mask, solar_flux)
        else:
            haze_res = np.array([0])

        return voxel_res + cone_res + haze_res

    def _px_ray_axes(self, scaling, dq):  # TODO: use dq
        # construct an array of unit rays, one ray per pixel
        iK = self.cam.inv_intrinsic_camera_mx()
        xx, yy = np.meshgrid(np.linspace(self.cam.width, 0, int(self.cam.width * scaling)),
                             np.linspace(0, self.cam.height, int(self.cam.height * scaling)), indexing='xy')
        img_coords = np.vstack((xx.flatten() + 0.5, yy.flatten() + 0.5, np.ones(xx.size)))
        ray_axes = iK.dot(img_coords).T * -1
        ray_axes /= np.linalg.norm(ray_axes, axis=1).reshape((-1, 1))
        return tools.q_times_mx(dq.conj(), ray_axes), xx.shape

    def flux_density_voxels(self, lf_ast_v, lf_ast_q, mask, solar_flux, down_scaling=1, quad_lim=15):
        assert down_scaling >= 1, 'only values of >=1 make sense for down_scaling'
        dq = lf_ast_q * self.voxels.lf_ast_q.conj()
        dv = tools.q_times_v(dq.conj(), lf_ast_v - self.voxels.lf_ast_v)
        ny, nx, nz = self.voxels.voxel_data.shape
        dx, dy, dz = [self.voxels.cell_size] * 3
        gx = np.linspace(-(nx-1) * dx / 2, (nx-1) * dx / 2, nx)
        gy = np.linspace(-(ny-1) * dy / 2, (ny-1) * dy / 2, ny)
        gz = np.linspace(-(nz-1) * dz / 2, (nz-1) * dz / 2, nz)

        interp3d = RegularGridInterpolator((gx, gy, gz), self.voxels.voxel_data, bounds_error=False, fill_value=0.0)
        #interp3d = NearestNDInterpolator((gx, gy, gz), self.voxels.voxel_data)

        ray_axes, sc_shape = self._px_ray_axes(1 / down_scaling, dq)# tools.ypr_to_q(0, np.pi, 0) * dq)

        margin = 0.0
        dist = np.linalg.norm(dv)
        fg_near = dist - nz*dz/2
        fg_far = dist - margin/2
        bg_near = dist + margin/2
        bg_far = dist + nz*dz/2

        def quad_fn(interp3d, ray_axes, near, far):
            points = None  #np.linspace(near, far, nx/2)
            res = integrate.quad_vec(lambda r: interp3d(ray_axes * r - dv), near, far, points=points, limit=quad_lim)
            return res[0]

        #  integrate density along the rays (quad_vec requires at least scipy 1.4.x)
        res = quad_fn(interp3d, ray_axes, bg_near, bg_far)
        bg_res = cv2.resize(res.reshape(sc_shape).astype(np.float32), mask.shape)
        bg_res[mask] = 0

        #  integrate density along the rays (quad_vec requires at least scipy 1.4.x)
        res = quad_fn(interp3d, ray_axes, fg_near, fg_far)
        fg_res = cv2.resize(res.reshape(sc_shape).astype(np.float32), mask.shape)

        result = ((0 if bg_res is None else bg_res) + (0 if fg_res is None else fg_res)) \
                 * Particles.CONE_INTENSITY_COEF * solar_flux * self.voxels.intensity

        return result

    def flux_density_cones(self, lf_ast_v, lf_ast_q, mask, solar_flux, down_scaling=1, quad_lim=25):
        """
        - The jet generated is a truncated cone that has a density proportional to (truncation_distance/distance_from_untruncated_origin)**2
        - Truncation so that base width is 0.1 of mask diameter
        - base_loc gives the coordinates of the base in camera frame (opengl type, -z camera axis, +y is up)
        - 95% of luminosity lost when distance from base is `length`
        - angular_radius [rad] of the cone, 95% of luminosity lost if this much off axis, uses normal distribution
        - intensity of the cone at truncation point (>0)
        - if phase_angle < pi/2, cone not drawn on top of masked parts of image as it starts behind object
        - direction: 0 - to the right, pi/2 - up, pi - left, -pi/2 - down
        """
        assert down_scaling >= 1, 'only values of >=1 make sense for down_scaling'
        scaling = 1 / down_scaling

        base_locs = [c.base_loc for c in self.cones]
        phase_angles = [c.phase_angle for c in self.cones]
        directions =[c.direction for c in self.cones]
        trunc_lens = [c.trunc_len for c in self.cones]
        angular_radii = [c.angular_radius for c in self.cones]
        intensities = [c.intensity for c in self.cones]

        axes = []
        for i in range(len(self.cones)):
            # q = np.quaternion(math.cos(-direction / 2), 0, 0, math.sin(-direction / 2)) \
            #     * np.quaternion(math.cos(phase_angle / 2), 0, math.sin(phase_angle / 2), 0)
            q1 = np.quaternion(math.cos(-phase_angles[i] / 2), 0, math.sin(-phase_angles[i] / 2), 0)
            q2 = np.quaternion(math.cos(directions[i] / 2), 0, 0, math.sin(directions[i] / 2))
            q = q2 * q1
            axis = tools.q_times_v(q, np.array([0, 0, -1]))
            axes.append(axis)
            base_locs[i] -= axis * trunc_lens[i]

        # density function of the jet
        def density(loc_arr, base_loc, axis, d0, angular_radius, intensity):
            loc_arr = loc_arr - base_loc
            r, d = tools.dist_across_and_along_vect(loc_arr, axis)
            #            r, d = tools.point_vector_dist(loc_arr, axis, dist_along_v=True)

            # get distance along axis
            coef = np.zeros((len(loc_arr), 1))
            coef[d > d0] = (d0 / d[d > d0]) ** 2

            # get radial distance from axis, use normal dist pdf but scaled so that max val is 1
            r_sd = d[coef > 0] * np.tan(angular_radius)
            coef[coef > 0] *= np.exp((-0.5 / r_sd ** 2) * (r[coef > 0] ** 2))

            return coef * intensity

        dq = lf_ast_q * self.cones.lf_ast_q.conj()
        dv = tools.q_times_v(dq.conj(), lf_ast_v - self.cones.lf_ast_v)
        ray_axes, sc_shape = self._px_ray_axes(scaling, dq)

        def i_fun(r, arg_arr):
            result = None
            for args in arg_arr:
                res = density(ray_axes * r - dv, *args)
                if result is None:
                    result = res
                else:
                    result += res
            return result

        bg_args_arr, bg_near, bg_far = [], np.inf, -np.inf
        fg_args_arr, fg_near, fg_far = [], np.inf, -np.inf
        for args in zip(base_locs, axes, trunc_lens, angular_radii, intensities):
            base_loc, axis, trunc_len, angular_radius, intensity = args
            dist = np.linalg.norm(base_loc)
            if axis[2] < 0:
                # z-component of axis is negative => jet goes away from cam => starts behind object
                bg_args_arr.append(args)
                bg_near = min(bg_near, dist - trunc_len)
                bg_far = max(bg_far, 2 * dist)  # crude heuristic, is it enough?
            else:
                # jet goes towards cam
                fg_args_arr.append(args)
                fg_near = min(fg_near, 0)
                fg_far = max(fg_far, dist + trunc_len)

        bg_res, fg_res = None, None
        if bg_args_arr:
            #  integrate density along the rays (quad_vec requires at least scipy 1.4.x)
            res = integrate.quad_vec(lambda r: i_fun(r, bg_args_arr), bg_near, bg_far, limit=quad_lim)
            #            bg_sc = np.max(res[0])/maxval
            bg_res = cv2.resize(res[0].reshape(sc_shape).astype(np.float32), mask.shape)
            #            bg_res = cv2.resize((res[0]/bg_sc).reshape(xx.shape).astype(img.dtype), img.shape).astype(np.float32)*bg_sc
            bg_res[mask] = 0
        if fg_args_arr:
            #  integrate density along the rays (quad_vec requires at least scipy 1.4.x)
            res = integrate.quad_vec(lambda r: i_fun(r, fg_args_arr), fg_near, fg_far, limit=quad_lim)
            #            fg_sc = np.max(res[0])/maxval
            fg_res = cv2.resize(res[0].reshape(sc_shape).astype(np.float32), mask.shape)
        #            fg_res = cv2.resize((res[0]/fg_sc).reshape(xx.shape).astype(img.dtype), img.shape).astype(np.float32)*fg_sc

        result = ((0 if bg_res is None else bg_res) + (0 if fg_res is None else fg_res)) \
                 * Particles.CONE_INTENSITY_COEF * solar_flux

        # max_r = np.max(result)
        # if max_r > 0:
        #     maxval = ImageProc._img_max_valid(img)
        #     result = (result / max_r) * maxval * np.max(intensities)

        return result

    def flux_density_haze(self, mask, solar_flux):
        result = np.ones(mask.shape, dtype=np.float32) \
                 * Particles.CONE_INTENSITY_COEF * solar_flux * self.haze * 0.5

        if mask is not None:
            result[np.logical_not(mask)] *= 2
        else:
            result *= 2

        return result


class VoxelParticles:
    def __init__(self, voxel_data, cell_size, intensity, lf_ast_v=np.zeros((3,)), lf_ast_q=quaternion.one):
        self.voxel_data = voxel_data
        self.cell_size = cell_size
        self.intensity = intensity
        self.lf_ast_v = lf_ast_v
        self.lf_ast_q = lf_ast_q


class Cone:
    def __init__(self, base_loc, trunc_len, phase_angle, direction, intensity, angular_radius, lf_ast_v, lf_ast_q):
        self.base_loc = base_loc
        self.trunc_len = trunc_len
        self.phase_angle = phase_angle
        self.direction = direction
        self.intensity = intensity
        self.angular_radius = angular_radius
        self.lf_ast_v = lf_ast_v
        self.lf_ast_q = lf_ast_q

    @staticmethod
    def random(n, cam, img, dist, mask, lf_ast_v, lf_ast_q, jet_int_mode, jet_int_conc, trunc_len_m,
               trunc_len_sd=0.2, ang_rad_m=np.pi/30, ang_rad_sd=0.3):

        fg_yx = np.vstack(np.where(np.logical_and(mask, img > np.max(img)*0.2))).T
        bg_yx = np.vstack(np.where(mask)).T

        cones = []
        for i in range(n):
            phase_angle = np.random.uniform(0, np.pi)
            if phase_angle < np.pi / 2 and len(bg_yx) > 0:
                yi, xi = random.choice(bg_yx)
            elif len(fg_yx) > 0:
                yi, xi = random.choice(fg_yx)
            else:
                continue
            z = -dist[yi, xi]
            x, y = cam.calc_xy(xi, yi, z)
            base_loc = np.array((x, y, z))

            alpha = jet_int_mode * (jet_int_conc - 2) + 1
            beta = (1 - jet_int_mode) * (jet_int_conc - 2) + 1
            intensity = np.random.beta(alpha, beta)

            direction = np.random.uniform(-np.pi, np.pi)
            trunc_len = 1e-3 * trunc_len_m * np.random.lognormal(0, trunc_len_sd)
            angular_radius = ang_rad_m * np.random.lognormal(0, ang_rad_sd)
            cones.append(Cone(base_loc, trunc_len, phase_angle, direction, intensity,
                              angular_radius, lf_ast_v, lf_ast_q))

        return cones

