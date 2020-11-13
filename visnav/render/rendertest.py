import math
import sys

import numpy as np
import quaternion
import cv2

from visnav.algo.image import ImageProc
from visnav.algo.model import Camera, SystemModel
from visnav.missions.rosetta import RosettaSystemModel
from visnav.render.particles import VoxelParticles, Particles
from visnav.testloop import TestLoop

try:
    from visnav.render.render import RenderEngine
except:
    from visnav.render import RenderEngine
from visnav.algo import tools
from visnav.missions.didymos import DidymosSystemModel
from visnav.settings import *


def render_didymos(show=False):
    sm = DidymosSystemModel(use_narrow_cam=False, target_primary=False, hi_res_shape_model=False)

    re = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=0)
    re.set_frustum(sm.cam.x_fov, sm.cam.y_fov, 0.05, 2)
    obj_idx = re.load_object(sm.asteroid.real_shape_model)

    pos = np.array([0, 0, -sm.min_med_distance * 1])
    q = tools.angleaxis_to_q((math.radians(20), 0, 1, 0))
    light_v = np.array([1, 0, -1]) / math.sqrt(2)

    img = re.render(obj_idx, pos, q, light_v, gamma=1.8, get_depth=False, shadows=True, textures=True,
                    reflection=RenderEngine.REFLMOD_HAPKE)

    output(img, show)


def render_itokawa(show=False):
    cam = Camera(4096, 4096, aperture=1.5, focal_length=120.8, sensor_size=[12.288]*2, px_saturation_e=30e3)
    shape_model_file = [
        r'C:\projects\sispo\data\models\itokawa_16k.obj',
        r'C:\projects\sispo\data\models\itokawa_f3145728.obj',
    ][1]

    RenderEngine.REFLMOD_PARAMS[RenderEngine.REFLMOD_HAPKE] = [
        553.38,          # J         0
        26,         # th_p      26
        0.31,       # w         0.31
        -0.35,      # b         -0.35
        0,          # c         0

        0.86,       # B_SH0     0.86
        0.00021,    # hs        0.00021
        0,          # B_CB0     0
        0.005,      # hc        0.005
        1,          # K         1
    ]

    re = RenderEngine(cam.width, cam.height, antialias_samples=0)
    re.set_frustum(cam.x_fov, cam.y_fov, 0.05, 12)
    obj_idx = re.load_object(shape_model_file)

    if 1:
        g_sc_q = tools.angleaxis_to_q([1.847799, -0.929873, 0.266931, -0.253146])
        #x, y, z = -13.182141, -64.694813, 116.263134
        x, y, z = 93.818, -8.695, 1.263
        g_ast_q = get_ast_q(x, y, z)
        g_sol_ast_v = np.array([146226194732.0, -68812326932.0, -477863381.0]) * 1e-3
        g_sol_sc_v = np.array([146226188132.0, -68812322442.0, -477863711.0]) * 1e-3
        g_sc_ast_v = g_sol_ast_v - g_sol_sc_v
        l_ast_sc_v = tools.q_times_v(SystemModel.sc2gl_q.conj() * g_sc_q.conj(), g_sc_ast_v)
        l_ast_q = SystemModel.sc2gl_q.conj() * g_sc_q.conj() * g_ast_q * SystemModel.sc2gl_q
        l_light_v = tools.q_times_v(SystemModel.sc2gl_q.conj() * g_sc_q.conj(), g_sol_ast_v/np.linalg.norm(g_sol_ast_v))
    else:
        l_ast_sc_v = np.array([0, 0, -7.990 * 1])
        l_ast_q = tools.angleaxis_to_q((math.radians(20), 0, 1, 0))
        l_light_v = np.array([1, 0, -1]) / math.sqrt(2)

    sc_mode = 0
    a, b, c = [0]*3
    ds, dq = 0.135, tools.ypr_to_q(math.radians(1.154), 0, math.radians(-5.643))

    while True:
        img = re.render(obj_idx, tools.q_times_v(dq, l_ast_sc_v*ds), l_ast_q, l_light_v, flux_density=1.0,
                        gamma=1.0, get_depth=False, shadows=True, textures=True,
                        reflection=RenderEngine.REFLMOD_HAPKE)
        k = output(img, show, maxval=0.90)

        if k is None or k == 27:
            break

        tmp = 1 if k in (ord('a'), ord('s'), ord('q')) else -1
        if k in (ord('a'), ord('d')):
            if sc_mode:
                dq = tools.ypr_to_q(math.radians(tmp*0.033), 0, 0) * dq
            else:
                b += tmp
        if k in (ord('w'), ord('s')):
            if sc_mode:
                dq = tools.ypr_to_q(0, 0, math.radians(tmp*0.033)) * dq
            else:
                a += tmp
        if k in (ord('q'), ord('e')):
            if sc_mode:
                ds *= 0.9 ** tmp
            else:
                c += tmp
        if k == ord('i'):
            y, p, r = tools.q_to_ypr(dq)
            print('+c: %.3f, %.3f, %.3f' % (x+a, y+b, z+c))
            print('ds, h, v: %.3f, %.3f, %.3f' % (ds, math.degrees(y), math.degrees(r)))

        g_ast_q = get_ast_q(x+a, y+b, z+c)
        l_ast_q = SystemModel.sc2gl_q.conj() * g_sc_q.conj() * g_ast_q * SystemModel.sc2gl_q


def render_67p(show=False):
    sm = RosettaSystemModel(hi_res_shape_model=False, res_mult=1.0)
    re = RenderEngine(sm.cam.width, sm.cam.height, antialias_samples=0)
    re.set_frustum(sm.cam.x_fov, sm.cam.y_fov, 0.05, sm.max_distance)
    obj_idx = re.load_object(sm.asteroid.real_shape_model)

    RenderEngine.REFLMOD_PARAMS[RenderEngine.REFLMOD_HAPKE] = [
        553.38,             # J         0
        27,                 # th_p      27
        0.034,              # w         0.034
        -0.08,              # b         -0.078577
        0,                  # c         0
        2.25,               # B_SH0     2.25
        math.radians(0.061),# hs        math.radians(0.061)
        0,                  # B_CB0     0
        0.005,              # hc        0.005
        1,                  # K         1
    ]

    g_sc_q = tools.angleaxis_to_q([1.892926, 0.781228, -0.540109, -0.312995])
    # x, y, z = 69.54, 64.11, 162.976134
    x, y, z = 146.540, 167.110, 154.976     # asteroid
    g_ast_q = get_ast_q(x, y, z)
    g_sol_ast_v = np.array([163613595198.0, 101637176823.0, 36457220690.0]) * 1e-3
    g_sol_sc_v = np.array([163613518304.0, 101637309778.0, 36457190373.0]) * 1e-3
    g_sc_ast_v = g_sol_ast_v - g_sol_sc_v
    l_ast_sc_v = tools.q_times_v(SystemModel.sc2gl_q.conj() * g_sc_q.conj(), g_sc_ast_v)
    l_ast_q = SystemModel.sc2gl_q.conj() * g_sc_q.conj() * g_ast_q * SystemModel.sc2gl_q
    l_light_v = tools.q_times_v(SystemModel.sc2gl_q.conj() * g_sc_q.conj(), g_sol_ast_v / np.linalg.norm(g_sol_ast_v))

    l_vx_ast_q = SystemModel.sc2gl_q.conj() * quaternion.one.conj() * g_ast_q * SystemModel.sc2gl_q
    print(str(l_vx_ast_q))
    particles = load_particles(r'C:\projects\sispo\data\models\Jets--ROS_CAM1_20150710T074301.exr', lf_ast_q=l_vx_ast_q, cell_size=0.066667)

    a, b, c = [0] * 3
    w = 10
    while True:
        print('%.3f, %.3f, %.3f' % (x + a, y + b, z + c))

        if particles is None and 0:
            img = re.render(obj_idx, l_ast_sc_v, l_ast_q, l_light_v, flux_density=1.0,
                            gamma=1.0, get_depth=False, shadows=True, textures=False,
                            reflection=RenderEngine.REFLMOD_HAPKE)
        else:
            img = TestLoop.render_navcam_image_static(
                None, re, [obj_idx], rel_pos_v=l_ast_sc_v, rel_rot_q=l_ast_q, light_v=l_light_v, sc_q=g_sc_q,
                sun_distance=np.linalg.norm(g_sol_ast_v)*1e3, exposure=None, gain=None, gamma=1.8, auto_gain=True,
                reflmod_params=RenderEngine.REFLMOD_PARAMS[RenderEngine.REFLMOD_HAPKE],
                use_shadows=True, use_textures=False, cam=sm.cam, fluxes_only=True,
                stars=True, lens_effects=False, particles=particles, return_depth=False)

        k = output(img, show, maxval=0.50, gamma=1.8)
        #k = output(img, show, maxval=0.70, gamma=1.0)

        if k is None or k == 27:
            break
        if k in (ord('a'), ord('d')):
            b += (1 if k == ord('a') else -1) * w
        if k in (ord('w'), ord('s')):
            a += (1 if k == ord('s') else -1) * w
        if k in (ord('q'), ord('e')):
            c += (1 if k == ord('q') else -1) * w

        if 0:
            l_light_v = tools.q_times_v(SystemModel.sc2gl_q.conj()
                                    * tools.ypr_to_q(math.radians(a), math.radians(b), math.radians(c))
                                    * g_sc_q.conj(), g_sol_ast_v / np.linalg.norm(g_sol_ast_v))
        elif 1:
            g_ast_q = get_ast_q(x + a, y + b, z + c)
            l_ast_q = SystemModel.sc2gl_q.conj() * g_sc_q.conj() * g_ast_q * SystemModel.sc2gl_q


def get_ast_q(axis_ra, axis_dec, rotation_zlra):
    # ZYZ, self.rot_conv
    axis_ra, axis_dec, rotation_zlra = list(map(math.radians, (axis_ra, axis_dec, rotation_zlra)))
    return tools.eul_to_q((axis_ra, np.pi / 2 - axis_dec, rotation_zlra), 'zyz', False)


def output(img, show, maxval=1.0, gamma=1.0):
    img = ImageProc.adjust_gamma(maxval * img / np.max(img) * 255, gamma=gamma) / 255
    cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else 'test.png', (255*img).astype('uint8'))
    if show:
        img_sc = cv2.resize(img, (700, 700))
        cv2.imshow('test.png', img_sc)
        return cv2.waitKey()


def load_particles(filename, cell_size, lf_ast_q=quaternion.one):
    import OpenEXR
    import Imath

    image = OpenEXR.InputFile(filename)
    header = image.header()
    mono = 'Y' in header['channels']
    size = header["displayWindow"]
    shape = (size.max.x - size.min.x + 1, size.max.y - size.min.y + 1)

    if mono:
        data2d = np.frombuffer(image.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT)), np.float32)
    else:
        r, g, b = 1.0, 1.0, 0.3   # corresponds to <not use>, gas? (~jets), particles? (~haze)
        #data2d = r * np.frombuffer(image.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)), np.float32)
        data2d = g * np.frombuffer(image.channel('G', Imath.PixelType(Imath.PixelType.FLOAT)), np.float32)
        data2d = data2d + b * np.frombuffer(image.channel('B', Imath.PixelType(Imath.PixelType.FLOAT)), np.float32)

    data2d = data2d.reshape(shape)
    n = int(np.prod(shape) ** (1 / 3)/10)*10
    k = math.ceil(n ** (1 / 2))
    voxel_data = np.zeros((n, n, n), dtype=np.float32)

    for i in range(n):
        x0, y0 = (i % k) * n, (i // k) * n
        voxel_data[:, :, i] = data2d[y0:y0 + n, x0:x0 + n]

    voxel_data = np.transpose(np.flip(voxel_data, axis=2), axes=(1, 0, 2))

    return Particles(None, None, None,
        voxels=VoxelParticles(voxel_data=voxel_data, cell_size=cell_size, intensity=0.1, lf_ast_q=lf_ast_q),
        cones=None, haze=0.0)


def perm(dat, idx):
    # (fx, fy, fz), (x', y', z')
    import itertools as it
    perms = list(it.product(list((it.product(range(2), repeat=3))), list(it.permutations(range(3)))))
    for i, f in enumerate(perms[idx][0]):
        if f:
            dat = np.flip(dat, axis=i)
    return np.transpose(dat, axes=perms[idx][1])


if __name__ == '__main__':
    if 0:
        render_didymos(0)
    elif 1:
        render_67p(1)
    else:
        render_itokawa(1)
