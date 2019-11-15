from functools import lru_cache

import cv2
import math

import numpy as np
import scipy
import scipy.integrate as integrate

from algo import tools
from algo.image import ImageProc
from missions.didymos import DidymosSystemModel


class Sun:
    RADIUS = 695510e3   # in meters
    FLUX_DENSITY_AT_1AU = 1360.8
    AU = 1.496e11   # in meters

    _DIFFRACTION_INTERPOLATION_N = 400
    _diffraction_relative_intensity_fun = {}

    @staticmethod
    def sun_radius_rad(dist):
        return math.atan(Sun.RADIUS/dist)

    @staticmethod
    def _unit_circle_segment_area(r):
        alpha = math.acos(r-1)
        unit_circle_area = np.pi
        if r > 0.5:
            return unit_circle_area - unit_circle_area*alpha/np.pi + (r-0.5)**2*math.tan(alpha)
        else:
            return unit_circle_area * alpha/np.pi - (0.5-r)**2*math.tan(np.pi-alpha)

    @staticmethod
    def flux_density(cam, sun_cf, mask=None):
        """
        calculate flux density from sun hitting the camera, sun_cf in camera frame and in meters
        """

        flux_density = np.zeros((cam.height, cam.width), dtype=np.float32)
        sun_dist = np.linalg.norm(sun_cf)
        sun_cf_n = sun_cf/sun_dist

        visible_ratio, theta = Sun.direct(flux_density, cam, sun_cf, mask)
        Sun.diffraction(flux_density, cam, sun_cf, mask, visible_ratio, theta)

        # TODO: for visible_ratio take into account if cam shadowed by something while sun out of fov
        visible_ratio = 1
        Sun.scattering(flux_density, cam, sun_cf_n, visible_ratio, theta)
        Sun.ghosts(flux_density, cam, sun_cf_n, visible_ratio, theta)

        return (Sun.FLUX_DENSITY_AT_1AU * (Sun.AU / sun_dist)**2) * flux_density

    @staticmethod
    def direct(flux_density, cam, sun_cf, mask, accurate_theta=False):
        sun_dist = np.linalg.norm(sun_cf)
        sun_rad = Sun.sun_radius_rad(sun_dist)
        lat, lon, _ = tools.cartesian2spherical(*sun_cf)
        lon = tools.wrap_rads(lon)
        hfx, hfy = math.radians(cam.x_fov / 2), math.radians(cam.y_fov / 2)

        lats, lons, rs = np.meshgrid(np.linspace(hfy, -hfy, cam.height), np.linspace(hfx, -hfx, cam.width), 1)
        px_v_s = np.stack((lats.squeeze().T, lons.squeeze().T, rs.squeeze().T), axis=2)
        px_v_c = tools.spherical2cartesian_arr(px_v_s.reshape((-1, 3)))
        theta = tools.angle_between_v_mx(sun_cf/sun_dist, px_v_c).reshape((cam.height, cam.width))

        direct = (theta < sun_rad).astype(np.float32)
        full = np.sum(direct)

        if abs(lon)+sun_rad > hfx or abs(lat)+sun_rad > hfy or full == 0:
            # sun out of fov or completely behind asteroid
            return 0, theta

        if mask is not None:
            direct[np.logical_not(mask)] = 0

        # update flux_density array
        flux_density += direct

        x_ratio = 1 if abs(lon)-sun_rad < hfx else Sun._unit_circle_segment_area((hfx-abs(lon)+sun_rad)/(2*sun_rad))
        y_ratio = 1 if abs(lat)-sun_rad < hfy else Sun._unit_circle_segment_area((hfy-abs(lat)+sun_rad)/(2*sun_rad))

        # not strictly true as sun might not be in fov at all in case both x- and y-ratios are very small
        not_obscured = np.sum(direct)
        visible_ratio = not_obscured/full * x_ratio * y_ratio

        if 0 < not_obscured < full and accurate_theta:
            px_sun_v = px_v_c[(direct.reshape((-1, 1)) > 0).squeeze(), :]
            theta = np.min(tools.angle_between_mx(px_sun_v, px_v_c), axis=1).reshape((cam.height, cam.width))

        return visible_ratio, theta

    @staticmethod
    def diffraction(flux_density, cam, sun_cf, mask, visible_ratio, theta):
        # DIFFRACTION
        # from https://en.wikipedia.org/wiki/Airy_disk
        if visible_ratio == 0:
            # sun behind asteroid or out of fov, forget diffraction
            return

        sun_dist = np.linalg.norm(sun_cf)
        sun_rad = Sun.sun_radius_rad(sun_dist)
        diffraction = Sun.diffraction_relative_intensity(cam, sun_rad, theta)
        diffraction[flux_density > 0] = 0   # dont add diffraction on top of direct observation

#        if mask is not None:
#            diffraction[np.logical_not(mask)] = 0

        # update flux_denstity
        flux_density += diffraction * visible_ratio

    @staticmethod
    def scattering(flux_density, cam, sun_cf_n, visible_ratio, theta):
        # Using Rayleigh scattering, https://en.wikipedia.org/wiki/Rayleigh_scattering

        lat, lon, _ = tools.cartesian2spherical(*sun_cf_n)
        if abs(tools.wrap_rads(lon)) > math.radians(cam.exclusion_angle_x) or abs(lat) > math.radians(cam.exclusion_angle_y):
            # baffle protected from scattering effects
            return

        # ~ 1+cos(theta)**2 * "some coef"
        scattering = cam.scattering_coef * (1 + np.cos(theta)**2) * visible_ratio
        flux_density += scattering

    @staticmethod
    def ghosts(flux_density, cam, sun_cf_n, visible_ratio, theta):
        # TODO
        pass

    @staticmethod
    def _diffraction(lam, aperture, theta):
        x = 2*np.pi/lam * aperture/2 * 1e-3 * np.sin(theta)
        return (2*scipy.special.j1(x)/x)**2

    @staticmethod
    def diffraction_relative_intensity(cam, sun_angular_radius, theta):
        shape = theta.shape
        key = hash((cam.x_fov, cam.y_fov, cam.aperture, cam.quantum_eff, cam.lambda_min, cam.lambda_max, sun_angular_radius))
        if key not in Sun._diffraction_relative_intensity_fun:
            lim = math.radians(np.linalg.norm((cam.x_fov, cam.y_fov))) + sun_angular_radius*2
            examples = np.linspace(-lim, lim, Sun._DIFFRACTION_INTERPOLATION_N)
            values = np.array([Sun.diffraction_relative_intensity_single(cam.aperture, cam.quantum_eff,
                                                                         cam.lambda_min, cam.lambda_max, th)
                               for th in examples])
            Sun._diffraction_relative_intensity_fun[key] = scipy.interpolate.interp1d(examples, values)
        return Sun._diffraction_relative_intensity_fun[key](theta.reshape((-1,))).reshape(shape)

    @staticmethod
    @lru_cache(maxsize=_DIFFRACTION_INTERPOLATION_N + 1)
    def diffraction_relative_intensity_single(aperture, quantum_eff, lambda_min, lambda_max, theta):
        """
        Returns relative intensity for diffraction integrated over the spectra of the sun and the sensor
        """

        if theta <= 0:
            return 1

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

        def total(f):
            return qeff(f) * phi(f) * Sun._diffraction(c/f, aperture, theta)

        tphi = integrate.quad(phi, c/1e-2, c/1e-8)
        tint = integrate.quad(total, c/lambda_max, c/lambda_min, limit=50)
        return tint[0]/tphi[0]



if __name__ == '__main__':
    #    cam = RosettaSystemModel(focused_attenuated=False).cam
    cam = DidymosSystemModel(use_narrow_cam=False).cam

    v = tools.q_times_v(tools.ypr_to_q(0, math.radians(10), 0), [1, 0, 0])
    print('v: %s' % (v,))
    flux_density = Sun.flux_density(cam, v * Sun.AU * 1.2)  # [0.31622777, 0.9486833, 0]
    img = cam.sense(flux_density, exposure=0.001, gain=1)

    img = np.clip(img * 255, 0, 255).astype('uint8')
    # img = ImageProc.adjust_gamma(img, 1.8)

    sc = min(768 / cam.width, 768 / cam.height)
    cv2.imshow('sun', cv2.resize(img, None, fx=sc, fy=sc))
    cv2.waitKey()
