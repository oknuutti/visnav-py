import math
import pickle

from pympler import tracker
import numpy as np
import cv2

from algo import tools
from algo.base import AlgorithmBase
from algo.keypoint import KeypointAlgo
from algo.tools import PositioningException, Stopwatch
from iotools import lblloader
from missions.didymos import DidymosSystemModel
from missions.rosetta import RosettaSystemModel

from settings import *
from render.render import RenderEngine


class InvalidSceneException(Exception):
    pass


class FeatureDatabaseGenerator(AlgorithmBase):
    def __init__(self, system_model, render_engine, obj_idx, render_z=None):
        super(FeatureDatabaseGenerator, self).__init__(system_model, render_engine, obj_idx)

        self.render_z = render_z or -self.system_model.min_med_distance
        self._ransac_err = KeypointAlgo.DEF_RANSAC_ERROR
        self.MAX_FEATURES = 10000

        # for overriding rendering
        self._sc_ast_lat = None
        self._sc_ast_lon = None
        self._light_lat = None
        self._light_lon = None

        # so that no need to always pass these in function call
        self._ref_img_sc = self._cam.width / system_model.view_width
        self._fdb_sc_ast_perms = None
        self._fdb_light_perms = None

    def generate_fdb(self, feat, view_width=None, view_height=None, fdb_tol=KeypointAlgo.FDB_TOL,
                     maxmem=KeypointAlgo.FDB_MAX_MEM, save_progress=False):

        save_file = self.fdb_fname(feat, fdb_tol, maxmem)
        view_width = view_width or self.system_model.view_width
        view_height = view_height or self.system_model.view_height
        assert view_width == self.render_engine.width and view_height == self.render_engine.height,\
            'wrong size render engine: (%d, %d)' % (self.render_engine.width, self.render_engine.height)

        self._ref_img_sc = self._cam.width / view_width
        self._ransac_err = KeypointAlgo.DEF_RANSAC_ERROR * (0.5 if feat == KeypointAlgo.ORB else 1)

        self.set_mesh(*self.calc_mesh(fdb_tol))
        n1 = len(self._fdb_sc_ast_perms)
        n2 = len(self._fdb_light_perms)
        print('%d x %d = %d, <%dMB' % (n1, n2, n1*n2, n1*n2*(maxmem/1024/1024)))

        # initialize fdb array
        # fdb = np.full((n1, n2), None).tolist()
        # fdb = (desc, 2d, 3d, idxs)
        dlen = KeypointAlgo.BYTES_PER_FEATURE[feat]
        n3 = int(maxmem / (dlen + 3*4))

        if save_progress and os.path.exists(save_file):
            # load existing fdb
            status, sc_ast_perms, light_perms, fdb = self.load_fdb(save_file)
            assert len(sc_ast_perms) == n1, \
                'Wrong number of s/c - asteroid relative orientation scenes: %d vs %d'%(len(sc_ast_perms), n1)
            assert len(light_perms) == n2, \
                'Wrong number of light direction scenes: %d vs %d' % (len(light_perms), n2)
            assert fdb[0].shape == (n1, n2, n3, dlen), 'Wrong shape descriptor array: %s vs %s'%(fdb[0].shape, (n1, n2, n3, dlen))
            assert fdb[1].shape == (n1, n2, n3, 2), 'Wrong shape 2d img coord array: %s vs %s'%(fdb[1].shape, (n1, n2, n3, 2))
            assert fdb[2].shape == (n1, n2, n3, 3), 'Wrong shape 3d coord array: %s vs %s'%(fdb[2].shape, (n1, n2, n3, 3))
            assert fdb[3].shape == (n1, n2, n3), 'Wrong shape matched features array: %s vs %s'%(fdb[3].shape, (n1, n2, n3))
            assert fdb[4].shape == (n1, n2), 'Wrong shape feature count array: %s vs %s'%(fdb[4].shape, (n1, n2))
        else:
            # create new fdb
            status = {'stage': 1, 'i1': -1, 'time': 0}
            fdb = [
                np.zeros((n1, n2, n3, dlen), dtype='uint8'),    # descriptors
                np.zeros((n1, n2, n3, 2), dtype='float32'),     # 2d image coords
                np.zeros((n1, n2, n3, 3), dtype='float32'),     # 3d real coords
                np.zeros((n1, n2, n3), dtype='bool'),           # feature has matched other feature
                np.zeros((n1, n2), dtype='uint16'),             # number of features
            ]

        timer = Stopwatch(elapsed=status['time'])
        timer.start()

        # first phase, just generate max amount of features per scene
        print(''.join(['_']*n1), flush=True)

        if status['stage'] == 1:
            for i1, (sc_ast_lat, sc_ast_lon) in enumerate(self._fdb_sc_ast_perms):
                print('.', flush=True, end="")
                if i1 <= status['i1']:
                    continue

                for i2, (light_lat, light_lon) in enumerate(self._fdb_light_perms):
                    # tr = tracker.SummaryTracker()
                    tmp = self.scene_features(feat, maxmem, i1, i2)
                    # tr.print_diff()
                    if tmp is not None:
                        nf = tmp[0].shape[0]
                        fdb[0][i1, i2, 0:nf, :] = tmp[0]
                        fdb[1][i1, i2, 0:nf, :] = tmp[1]
                        fdb[2][i1, i2, 0:nf, :] = tmp[2]
                        fdb[4][i1, i2] = nf

                if save_progress and (i1+1) % 30 == 0:
                    status = {'stage': 1, 'i1': i1, 'time': timer.elapsed}
                    self.save_fdb(status, fdb, save_file)
            print('\n', flush=True, end="")
            status = {'stage': 2, 'i1': -1, 'time': timer.elapsed}
        else:
            self._latest_detector, nfeats = KeypointAlgo.get_detector(feat, 0)
            print(''.join(['.'] * n1), flush=True)

        if False:
            status['stage'] = 2
            status['i1'] = 0

        # second phase, match with neighbours, record matching features
        if True or status['stage'] == 2:
            visited = set()
            for i1 in range(n1):
                print('.', flush=True, end="")
                if i1 <= status['i1']:
                    continue
                for i2 in range(n2):
                    self._update_matched_features(fdb, visited, fdb_tol, i1, i2)
                if save_progress and (i1+1) % 30 == 0:
                    status = {'stage': 2, 'i1': i1, 'time': timer.elapsed}
                    self.save_fdb(status, fdb, save_file)
            print('\n', flush=True, end="")
            # fdb[1] = None
            status = {'stage': 3, 'i1': 0, 'time': timer.elapsed}
        else:
            print(''.join(['.'] * n1), flush=True)

        # third phase, discard features that didn't match with any neighbours
        # for i1 in range(n1):
        #     print('.', flush=True, end="")
        #     for i2 in range(n2):
        #         tmp = fdb[][i1][i2]
        #         if tmp is not None:
        #             a, b, c, idxs = tmp
        #             fdb[i1][i2] = (a[tuple(idxs), :], c[tuple(idxs), :])
        #             #fdb[i1][i2] = list(zip(*[(a[i], b[i], c[i]) for i in idxs]))
        # print('\n', flush=True, end="")

        # finished, save, then exit
        if status['stage'] == 3:
            status = {'stage': 4, 'i1': 0, 'time': timer.elapsed}
            self.save_fdb(status, fdb, save_file)
            timer.stop()
            secs = timer.elapsed
        else:
            secs = status['time']

        print('Total time: %.1fh, per scene: %.3fs'%(secs/3600, secs/n1/n2))
        return fdb

    def set_mesh(self, fdb_sc_ast_perms, fdb_light_perms):
        self._fdb_sc_ast_perms = fdb_sc_ast_perms
        self._fdb_light_perms = fdb_light_perms

    def calc_mesh(self, fdb_tol):
        # s/c-asteroid relative orientation, camera axis rotation zero, in opengl coords
        fdb_sc_ast_perms = np.array(tools.bf2_lat_lon(fdb_tol))
                                                               #, lat_range=(-fdb_tol, fdb_tol)))
        # light direction in opengl coords
        #   z-axis towards cam, x-axis to the right => +90deg lat==sun ahead, -90deg sun behind
        #   0 deg lon => sun on the left
        fdb_light_perms = np.array(
            tools.bf2_lat_lon(fdb_tol, lat_range=(-math.pi/2, math.radians(90 - self.system_model.min_elong)))
        )                              #lat_range=(-fdb_tol, fdb_tol)))
        return fdb_sc_ast_perms, fdb_light_perms

    def scene_features(self, feat, maxmem, i1, i2):
        try:
            ref_img, depth = self.render_scene(i1, i2)
        except InvalidSceneException:
            return None

        # get keypoints and descriptors
        ref_kp, ref_desc, self._latest_detector = KeypointAlgo.detect_features(ref_img, feat, maxmem=maxmem,
                                                                               max_feats=self.MAX_FEATURES, for_ref=True)

        # save only 2d image coordinates, scrap scale, orientation etc
        ref_kp_2d = np.array([p.pt for p in ref_kp], dtype='float32')

        # get 3d coordinates
        ref_kp_3d = KeypointAlgo.inverse_project(self.system_model, ref_kp_2d, depth, self.render_z, self._ref_img_sc)

        if False:
            mm_dist = self.system_model.min_med_distance
            if False:
                pos = (0, 0, -mm_dist)
                qfin = tools.ypr_to_q(sc_ast_lat, 0, sc_ast_lon)
                light_v = tools.spherical2cartesian(light_lat, light_lon, 1)
                reimg = self.render_engine.render(self.obj_idx, pos, qfin, light_v)
                reimg = cv2.cvtColor(reimg, cv2.COLOR_RGB2GRAY)
                img = np.concatenate((cv2.resize(ref_img, (self.system_model.view_width, self.system_model.view_height)), reimg), axis=1)
            else:
                ref_kp = [cv2.KeyPoint(*self._cam.calc_img_xy(x, -y, -z-mm_dist), 1) for x, y, z in ref_kp_3d]
                img = cv2.drawKeypoints(ref_img, ref_kp, ref_img.copy(), (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            cv2.imshow('res', img)
            cv2.waitKey()

        return np.array(ref_desc), ref_kp_2d, ref_kp_3d

    def render_scene(self, i1, i2, get_depth=True):
        self._sc_ast_lat, self._sc_ast_lon = self._fdb_sc_ast_perms[i1]
        self._light_lat, self._light_lon = self._fdb_light_perms[i2]

        depth = None
        img = self.render(depth=get_depth, shadows=True)
        if get_depth:
            img, depth = img

        # scale to match scene image asteroid extent in pixels
        img = cv2.resize(img, None, fx=self._ref_img_sc, fy=self._ref_img_sc, interpolation=cv2.INTER_CUBIC)
        return img, depth

    def _render_params(self, discretize_tol=False, center_model=False):
        # called at self.render, override based on hidden field values

        #qfin = tools.fdb_relrot_to_render_q(self._sc_ast_lat, self._sc_ast_lon)
        qfin = tools.ypr_to_q(self._sc_ast_lat, 0, self._sc_ast_lon)
        #light_v = tools.fdb_light_to_render_light(self._light_lat, self._light_lon)
        light_v = tools.spherical2cartesian(self._light_lat, self._light_lon, 1)

        # if qfin & light_v not reasonable, e.g. because solar elong < 45 deg:
        # seems that not needed, left here in case later notice that useful
        # raise InvalidSceneException()

        return (0, 0, self.render_z), qfin, light_v

    def get_neighbours(self, fdb_tol, i1, i2):
        coef = math.sqrt(2.5)  # math.sqrt(math.sqrt(2)**2 + (math.sqrt(2)/2)**2)
        nearest1 = tools.find_nearest_n(self._fdb_sc_ast_perms, self._fdb_sc_ast_perms[i1], r=fdb_tol*coef, fun=tools.wrap_rads)
        nearest2 = tools.find_nearest_n(self._fdb_light_perms, self._fdb_light_perms[i2], r=fdb_tol*coef, fun=tools.wrap_rads)

        neighbours = {(i1, i2, n1, n2) for n1 in nearest1 for n2 in nearest2} - {(i1, i2, i1, i2)}
        return neighbours

    def _update_matched_features(self, fdb, visited, fdb_tol, i1, i2):
        if fdb[4][i1, i2] == 0:
            return

        visit = self.get_neighbours(fdb_tol, i1, i2) - visited
        for i1, i2, j1, j2 in visit:
            self._update_matched_features_inner(fdb, (i1, i2), (j1, j2))
            visited.add((i1, i2, j1, j2))
            visited.add((j1, j2, i1, i2))

    def _update_matched_features_inner(self, fdb, idxs1, idxs2):
        nf1 = fdb[4][idxs1[0], idxs1[1]]
        nf2 = fdb[4][idxs2[0], idxs2[1]]
        if idxs1 == idxs2 or nf1 == 0 or nf2 == 0:
            return

        sc1_desc = fdb[0][idxs1[0], idxs1[1], 0:nf1, :].reshape((nf1, fdb[0].shape[3]))
        sc1_kp_2d = fdb[1][idxs1[0], idxs1[1], 0:nf1, :].reshape((nf1, fdb[1].shape[3]))

        sc2_desc = fdb[0][idxs2[0], idxs2[1], 0:nf2, :].reshape((nf2, fdb[0].shape[3]))
        sc2_kp_3d = fdb[2][idxs2[0], idxs2[1], 0:nf2, :].reshape((nf2, fdb[2].shape[3]))

        try:
            matches = KeypointAlgo.match_features(sc1_desc, sc2_desc, self._latest_detector.defaultNorm(), method='brute')

            # solve pnp with ransac
            ref_kp_3d = sc2_kp_3d[[m.trainIdx for m in matches], :]
            sce_kp_2d = sc1_kp_2d[[m.queryIdx for m in matches], :]
            rvec, tvec, inliers = KeypointAlgo.solve_pnp_ransac(self.system_model, sce_kp_2d, ref_kp_3d, self._ransac_err)

            # check if solution ok
            ok, err1, err2 = self.calc_err(rvec, tvec, idxs1[0], idxs2[0], warn=len(inliers) > 30)
            if not ok:
                raise PositioningException()

            fdb[3][idxs1[0], idxs1[1], [matches[i[0]].queryIdx for i in inliers]] = True
            fdb[3][idxs2[0], idxs2[1], [matches[i[0]].trainIdx for i in inliers]] = True
        except PositioningException as e:
            # assert inliers is not None, 'at (%s, %s): ransac failed'%(idxs1, idxs2)
            pass

    def calc_err(self, rvec, tvec, i1, j1, warn=False):
        q_res = tools.angleaxis_to_q(rvec)
        lat1, roll1 = self._fdb_sc_ast_perms[i1]
        lat2, roll2 = self._fdb_sc_ast_perms[j1]
        q_src = tools.ypr_to_q(lat1, 0, roll1)
        q_trg = tools.ypr_to_q(lat2, 0, roll2)
        q_rel = q_trg * q_src.conj()

        # q_res.x = -q_res.x
        # np.quaternion(0.707106781186547, 0, -0.707106781186547, 0)
        m = self.system_model
        q_frame = m.frm_conv_q(m.OPENGL_FRAME, m.OPENCV_FRAME)
        q_res = q_frame * q_res.conj() * q_frame.conj()

        err1 = math.degrees(tools.wrap_rads(tools.angle_between_q(q_res, q_rel)))
        err2 = np.linalg.norm(tvec - np.array((0, 0, -self.render_z)).reshape((3, 1)))
        ok = not (abs(err1) > 10 or abs(err2) > 0.10 * abs(self.render_z))

        if not ok and warn:
            print('at (%s, %s), err1: %.1fdeg, err2: %.1fkm\n\tq_real: %s\n\tq_est:  %s' % (
                i1, j1, err1, err2, q_rel, q_res))

        return ok, err1, err2

    def closest_scene(self, sc_ast_q=None, light_v=None):
        """ in opengl frame """

        if sc_ast_q is None:
            sc_ast_q, _ = self.system_model.gl_sc_asteroid_rel_q()
        if light_v is None:
            light_v, _ = self.system_model.gl_light_rel_dir()

        d_sc_ast_q, i1 = tools.discretize_q(sc_ast_q, points=self._fdb_sc_ast_perms)
        err_q = sc_ast_q * d_sc_ast_q.conj()

        c_light_v = tools.q_times_v(err_q.conj(), light_v)
        d_light_v, i2 = tools.discretize_v(c_light_v, points=self._fdb_light_perms)
        err_angle = tools.angle_between_v(light_v, d_light_v)

        return i1, i2, d_sc_ast_q, d_light_v, err_q, err_angle

    @staticmethod
    def calculate_fdb_stats(fdb, feat):
        fcounts = np.sum(fdb[3], axis=2).flatten()
        totmem = 1.0 * np.sum(fcounts) * (KeypointAlgo.BYTES_PER_FEATURE[feat] + 3 * 4)
        n_mean = np.mean(fcounts)
        fails = np.sum(fcounts == 0)

        stats = {
            'min_feat_count': np.min(fcounts),
            'avg_feat_count': n_mean,
            'scene_count': len(fcounts),
            'failed_scenes': fails,
            'weak_scenes': np.sum(fcounts < 100) - fails,
            'total_mem_usage (MB)': totmem/1024/1024,
            'accepted_feature_percent': 100*(n_mean/fdb[3].shape[2]),
        }

        return stats

    def fdb_fname(self, feat, fdb_tol=KeypointAlgo.FDB_TOL, maxmem=KeypointAlgo.FDB_MAX_MEM):
        return os.path.join(CACHE_DIR, self.system_model.mission_id, 'fdb_%s_w%d_m%d_t%d.pickle' % (
            feat,
            self.system_model.view_width,
            maxmem/1024,
            10*math.degrees(fdb_tol)
        ))

    def load_fdb(self, fname):
        with open(fname, 'rb') as fh:
            tmp = pickle.load(fh)
        if len(tmp) == 3:
            # backwards compatibility
            fdb_sc_ast_perms = np.array(tools.bf2_lat_lon(KeypointAlgo.FDB_TOL))
            fdb_light_perms = np.array(tools.bf2_lat_lon(KeypointAlgo.FDB_TOL,
                                       lat_range=(-math.pi / 2, math.radians(90 - self.system_model.min_elong))))
            n1, n2 = len(fdb_sc_ast_perms), len(fdb_light_perms)
            status, scenes, fdb = tmp
            assert len(scenes) == n1 * n2, \
                'Wrong amount of scenes in loaded fdb: %d vs %d' % (len(scenes), n1 * n2)
        else:
            status, fdb_sc_ast_perms, fdb_light_perms, fdb = tmp

        # assert status['stage'] >= 3, 'Incomplete FDB status: %s' % (status,)
        return status, fdb_sc_ast_perms, fdb_light_perms, fdb

    def save_fdb(self, status, fdb, save_file):
        with open(save_file+'.tmp', 'wb') as fh:
            pickle.dump((status, self._fdb_sc_ast_perms, self._fdb_light_perms, fdb), fh, -1)
        if os.path.exists(save_file):
            os.remove(save_file)
        os.rename(save_file+'.tmp', save_file)

    def estimate_mem_usage(self, fdb_tol_deg, sc_mem_kb, acc_ratio=0.5):
        fdb_tol = math.radians(fdb_tol_deg)
        n1 = len(tools.bf2_lat_lon(fdb_tol))
        n2 = len(tools.bf2_lat_lon(fdb_tol, lat_range=(-math.pi/2, math.radians(90 - self.system_model.min_elong))))
        print('%d x %d = %d, <%dMB, ~%dMB'%(n1, n2, n1*n2, n1*n2*(sc_mem_kb/1024), n1*n2*(sc_mem_kb/1024)*acc_ratio))


if __name__ == '__main__':
    # Didw - ORB:
    # * 10 deg, 128kb
    # * 12 deg, 512kb

    # Didw - AKAZE:
    # * 12 deg, 512kb, 0.133, 1186MB

    # Didy - ORB:
    # * 10 deg, 128kb, 0.415, 1984MB
    # * 12 deg, 512kb, ?

    # Didy - AKAZE:
    # * 10 deg, 128kb,
    # * 12 deg, 512kb,

    # Rose - ORB:
    # * 10 deg, 128kb, 0.469, 2246MB
    # * 11 deg, 256kb,
    # * 12 deg, 512kb, 0.267, 2393MB

    # Rose - AKAZE:
    # * 10 deg, 128kb, 0.558, 2670MB
    # * 11 deg, 256kb, 0,291, 1948MB
    # * 12 deg, 512kb, 0.131, 1177MB

    # Rose - SIFT:
    # * 10 deg, 128kb, 0,441, 2111MB
    # * 11 deg, 256kb,
    # * 12 deg, 512kb, 0.217, 1945MB

    # Rose - SURF:
    # \* 10 deg, 128kb,
    # \* 12 deg, 512kb,

    sm = RosettaSystemModel(hi_res_shape_model=True)                          # rose
    # sm = DidymosSystemModel(hi_res_shape_model=True, use_narrow_cam=True)   # didy
    # sm = DidymosSystemModel(hi_res_shape_model=True, use_narrow_cam=False)  # didw

    # sm.view_width = sm.cam.width
    sm.view_width = 512
    feat = KeypointAlgo.ORB
    fdb_tol = math.radians(11)
    maxmem = 256 * 1024

    re = RenderEngine(sm.view_width, sm.view_height, antialias_samples=0)
    obj_idx = re.load_object(sm.asteroid.real_shape_model, smooth=sm.asteroid.render_smooth_faces)
    fdbgen = FeatureDatabaseGenerator(sm, re, obj_idx)

    if True:
        fdb = fdbgen.generate_fdb(feat, fdb_tol=fdb_tol, maxmem=maxmem, save_progress=True)
    else:
        fname = fdbgen.fdb_fname(feat, fdb_tol, maxmem)
        status, sc_ast_perms, light_perms, fdb = fdbgen.load_fdb(fname)
        print('status: %s' % (status,))
        #fdbgen.estimate_mem_usage(12, 512, 0.25)
        #quit()

    stats = FeatureDatabaseGenerator.calculate_fdb_stats(fdb, feat)
    print('FDB stats:\n'+str(stats))
    # print('Total time: %.1fh, per scene: %.3fs' % (status['time'] / 3600, status['time'] / len(scenes)))
    fdb = None

    # feat = KeypointAlgo.ORB
    # fdb_tol = math.radians(12)
    # maxmem = 384 * 1024
    # fname = os.path.join(CACHE_DIR, sm.mission_id, 'fdb_%s_w%d_m%d_t%d.pickle' % (
    #     feat,
    #     sm.view_width,
    #     maxmem / 1024,
    #     10 * math.degrees(fdb_tol)
    # ))
    # scenes, fdb = fdbgen.generate_fdb(feat, fname, fdb_tol=fdb_tol, maxmem=maxmem)
    # stats = fdbgen.calculate_fdb_stats(scenes, fdb, feat)
    # print('fdb stats:\n'+str(stats))
