import math
import pickle

from pympler import tracker
import numpy as np
import cv2

from algo import tools
from algo.keypoint import KeypointAlgo
from algo.model import SystemModel
from algo.tools import PositioningException, Stopwatch
from iotools import lblloader

from settings import *
from render.render import RenderEngine
from testloop import TestLoop



class InvalidSceneException(Exception):
    pass


class FeatureDatabaseGenerator(KeypointAlgo):
    def __init__(self, system_model, render_engine, obj_idx, render_z=-MIN_MED_DISTANCE):
        super(FeatureDatabaseGenerator, self).__init__(system_model, render_engine, obj_idx)

        self.render_z = render_z
        self._ransac_err = self.DEF_RANSAC_ERROR

        # for overriding rendering
        self._sc_ast_lat = None
        self._sc_ast_lon = None
        self._light_lat = None
        self._light_lon = None

        # so that no need to always pass these in function call
        self._ref_img_sc = None
        self._sc_ast_perms = None
        self._light_perms = None

    def generate_fdb(self, feat, save_file, view_width=VIEW_WIDTH, view_height=VIEW_HEIGHT, fdb_tol=KeypointAlgo.FDB_TOL,
                     maxmem=KeypointAlgo.FDB_MAX_MEM, save_progress=False):

        assert view_width == self.render_engine.width and view_height == self.render_engine.height,\
            'wrong size render engine: (%d, %d)' % (self.render_engine.width, self.render_engine.height)

        self._ref_img_sc = CAMERA_WIDTH / view_width

        # s/c-asteroid relative orientation, camera axis rotation zero, in opengl coords
        self._sc_ast_perms = np.array(tools.bf2_lat_lon(fdb_tol))
                                                        #, lat_range=(-fdb_tol, fdb_tol)))

        # light direction in opengl coords
        #   z-axis towards cam, x-axis to the right => +90deg lat==sun ahead, -90deg sun behind
        #   0 deg lon => sun on the left
        self._light_perms = np.array(tools.bf2_lat_lon(fdb_tol,
                                                       lat_range=(-math.pi/2, math.radians(90 - TestLoop.MIN_ELONG))))
                                                       #lat_range=(-fdb_tol, fdb_tol)))

        n1 = len(self._sc_ast_perms)
        n2 = len(self._light_perms)
        print('%d x %d = %d, <%dMB'%(n1, n2, n1*n2, n1*n2*(maxmem/1024/1024)))

        # initialize fdb array
        # fdb = np.full((n1, n2), None).tolist()
        # fdb = (desc, 2d, 3d, idxs)
        dlen = KeypointAlgo.BYTES_PER_FEATURE[feat]
        n3 = int(maxmem / (dlen + 3*4))

        if save_progress and os.path.exists(save_file):
            # load existing fdb
            status, scenes, fdb = self.load_fdb(save_file)
            assert len(scenes) == n1*n2, 'Wrong amount of scenes in loaded fdb: %d vs %d'%(len(scenes), n1*n2)
            assert fdb[0].shape == (n1, n2, n3, dlen), 'Wrong shape descriptor array: %s vs %s'%(fdb[0].shape, (n1, n2, n3, dlen))
            assert fdb[1].shape == (n1, n2, n3, 2), 'Wrong shape 2d img coord array: %s vs %s'%(fdb[1].shape, (n1, n2, n3, 2))
            assert fdb[2].shape == (n1, n2, n3, 3), 'Wrong shape 3d coord array: %s vs %s'%(fdb[2].shape, (n1, n2, n3, 3))
            assert fdb[3].shape == (n1, n2, n3), 'Wrong shape matched features array: %s vs %s'%(fdb[3].shape, (n1, n2, n3))
            assert fdb[4].shape == (n1, n2), 'Wrong shape feature count array: %s vs %s'%(fdb[4].shape, (n1, n2))
        else:
            # create new fdb
            status = {'stage': 1, 'i1': -1, 'time': 0}
            scenes = np.zeros((n1*n2, 6), dtype='float32')
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
            for i1, (sc_ast_lat, sc_ast_lon) in enumerate(self._sc_ast_perms):
                print('.', flush=True, end="")
                if i1 <= status['i1']:
                    continue

                for i2, (light_lat, light_lon) in enumerate(self._light_perms):
                    # tr = tracker.SummaryTracker()
                    tmp = self._scene_features(feat, maxmem, sc_ast_lat, sc_ast_lon, light_lat, light_lon)
                    # tr.print_diff()
                    if tmp is not None:
                        nf = tmp[0].shape[0]
                        fdb[0][i1, i2, 0:nf, :] = tmp[0]
                        fdb[1][i1, i2, 0:nf, :] = tmp[1]
                        fdb[2][i1, i2, 0:nf, :] = tmp[2]
                        fdb[4][i1, i2] = nf
                        scenes[i1*n2+i2, :] = (sc_ast_lat, sc_ast_lon, light_lat, light_lon, i1, i2)

                if save_progress and (i1+1) % 30 == 0:
                    status = {'stage': 1, 'i1': i1, 'time': timer.elapsed}
                    self.save_fdb(status, scenes, fdb, save_file)
            print('\n', flush=True, end="")
            status = {'stage': 2, 'i1': -1, 'time': timer.elapsed}
        else:
            print(''.join(['.'] * n1), flush=True)

        status['stage'] = 2
        # second phase, match with neighbours, record matching features
        if status['stage'] == 2:
            visited = set()
            for i1 in range(n1):
                print('.', flush=True, end="")
                if i1 <= status['i1']:
                    continue
                for i2 in range(n2):
                    self._update_matched_features(fdb_tol, visited, fdb, i1, i2)
                if False and save_progress and (i1+1) % 30 == 0:
                    status = {'stage': 2, 'i1': i1, 'time': timer.elapsed}
                    self.save_fdb(status, scenes, fdb, save_file)
            print('\n', flush=True, end="")
            # fdb[1] = None
            status = {'stage': 3, 'i1': 0, 'time': timer.elapsed}
        else:
            print(''.join(['.'] * n1), flush=True)
        quit()
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
            self.save_fdb(status, scenes, fdb, save_file)
            timer.stop()
            secs = timer.elapsed
        else:
            secs = status['time']

        print('Total time: %.1fh, per scene: %.3fs'%(secs/3600, secs/n1/n2))
        return scenes, fdb

    def _scene_features(self, feat, maxmem, sc_ast_lat, sc_ast_lon, light_lat, light_lon):
        self._sc_ast_lat = sc_ast_lat
        self._sc_ast_lon = sc_ast_lon
        self._light_lat = light_lat
        self._light_lon = light_lon

        # render
        try:
            ref_img, depth = self.render(depth=True, shadows=True)
        except InvalidSceneException:
            return None

        # scale to match scene image asteroid extent in pixels
        ref_img = cv2.resize(ref_img, None, fx=self._ref_img_sc, fy=self._ref_img_sc, interpolation=cv2.INTER_CUBIC)

        # get keypoints and descriptors
        ref_kp, ref_desc = self.detect_features(ref_img, feat, maxmem=maxmem, for_ref=True)

        # save only 2d image coordinates, scrap scale, orientation etc
        ref_kp_2d = np.array([p.pt for p in ref_kp], dtype='float32')

        # get 3d coordinates
        ref_kp_3d = self._inverse_project(ref_kp_2d, depth, self.render_z, self._ref_img_sc, max_dist=30)

        if False:
            if False:
                pos = (0, 0, -MIN_MED_DISTANCE)
                qfin = tools.ypr_to_q(sc_ast_lat, 0, sc_ast_lon)
                light_v = tools.spherical2cartesian(light_lat, light_lon, 1)
                reimg = self.render_engine.render(self.obj_idx, pos, qfin, light_v)
                reimg = cv2.cvtColor(reimg, cv2.COLOR_RGB2GRAY)
                img = np.concatenate((cv2.resize(ref_img, (VIEW_WIDTH, VIEW_HEIGHT)), reimg), axis=1)
            else:
                ref_kp = [cv2.KeyPoint(*tools.calc_img_xy(x, -y, -z-MIN_MED_DISTANCE, CAMERA_WIDTH, CAMERA_HEIGHT), 1) for x, y, z in ref_kp_3d]
                img = cv2.drawKeypoints(ref_img, ref_kp, ref_img.copy(), (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            cv2.imshow('res', img)
            cv2.waitKey()

        return np.array(ref_desc), ref_kp_2d, ref_kp_3d

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

    def _update_matched_features(self, fdb_tol, visited, fdb, i1, i2):
        if fdb[4][i1, i2] == 0:
            return

        coef = math.sqrt(2.5)  # math.sqrt(math.sqrt(2)**2 + (math.sqrt(2)/2)**2)
        nearest1 = tools.find_nearest_n(self._sc_ast_perms, self._sc_ast_perms[i1], r=fdb_tol*coef, fun=tools.wrap_rads)
        nearest2 = tools.find_nearest_n(self._light_perms, self._light_perms[i2], r=fdb_tol*coef, fun=tools.wrap_rads)

        visit = {(i1, i2, n1, n2) for n1 in nearest1 for n2 in nearest2} - {(i1, i2, i1, i2)} - visited
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

        inliers = None
        try:
            matches = self._match_features(sc1_desc, sc2_desc, method='brute')

            # solve pnp with ransac
            ref_kp_3d = sc2_kp_3d[[m.trainIdx for m in matches], :]
            sce_kp_2d = sc1_kp_2d[[m.queryIdx for m in matches], :]
            rvec, tvec, inliers = self._solve_pnp_ransac(sce_kp_2d, ref_kp_3d)

            q_res = tools.angleaxis_to_q(rvec)
            q_src = tools.ypr_to_q(*self._sc_ast_perms[idxs1[0]], 0)
            q_trg = tools.ypr_to_q(*self._sc_ast_perms[idxs2[0]], 0)
            q_rel = q_trg.conj() * q_src
            err1 = math.degrees(tools.angle_between_q(q_res, q_rel))
            err2 = np.linalg.norm(tvec - np.array((0, 0, -self.render_z)).reshape((3, 1)))
            if abs(err1) > 15 or abs(err2) > 0.1*abs(self.render_z):
                assert len(inliers)<20, 'at (%s, %s): q_real: %s, q_est: %s, tvec: %s'%(idxs1, idxs2, q_rel, q_res, tvec)
                raise PositioningException()

            fdb[3][idxs1[0], idxs1[1], [matches[i[0]].queryIdx for i in inliers]] = True
            fdb[3][idxs2[0], idxs2[1], [matches[i[0]].trainIdx for i in inliers]] = True
        except PositioningException as e:
            # assert inliers is not None, 'at (%s, %s): ransac failed'%(idxs1, idxs2)
            pass

    def calculate_fdb_stats(self, scenes, fdb, feat):
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

    def load_fdb(self, save_file):
        with open(fname, 'rb') as fh:
            status, scenes, fdb = pickle.load(fh)
        return status, scenes, fdb

    def save_fdb(self, status, scenes, fdb, save_file):
        with open(save_file+'.tmp', 'wb') as fh:
            pickle.dump((status, scenes, fdb), fh, -1)
        if os.path.exists(save_file):
            os.remove(save_file)
        os.rename(save_file+'.tmp', save_file)

    def estimate_mem_usage(self, fdb_tol_deg, sc_mem_kb, acc_ratio=0.5):
        fdb_tol = math.radians(fdb_tol_deg)
        n1 = len(tools.bf2_lat_lon(fdb_tol))
        n2 = len(tools.bf2_lat_lon(fdb_tol, lat_range=(-math.pi/2, math.radians(90 - TestLoop.MIN_ELONG))))
        print('%d x %d = %d, <%dMB, ~%dMB'%(n1, n2, n1*n2, n1*n2*(sc_mem_kb/1024), n1*n2*(sc_mem_kb/1024)*acc_ratio))


if __name__ == '__main__':
    sm = SystemModel(shape_model=HIRES_TARGET_MODEL_FILE)
    #lblloader.load_image_meta(TARGET_IMAGE_META_FILE, sm)
    re = RenderEngine(VIEW_WIDTH, VIEW_HEIGHT, antialias_samples=0)
    obj_idx = re.load_object(sm.real_shape_model)
    fdbgen = FeatureDatabaseGenerator(sm, re, obj_idx)

    # ORB:
    # - 9 deg, 96kb, 0.327, 1851MB
    # - 10 deg, 192kb, 0.32, 2.2GB
    # - 12 deg, 512kb, 0.126. 1130MB

    feat = KeypointAlgo.ORB
    fdb_tol = math.radians(12)
    maxmem = 512 * 1024
    fname = os.path.join(CACHE_DIR, 'fdb_%s_w%d_m%d_t%d.pickle' % (
        feat,
        VIEW_WIDTH,
        maxmem / 1024,
        10 * math.degrees(fdb_tol)
    ))

    scenes, fdb = fdbgen.generate_fdb(feat, fname, fdb_tol=fdb_tol, maxmem=maxmem, save_progress=True)
    #status, scenes, fdb = fdbgen.load_fdb(fname)
    #fdbgen.estimate_mem_usage(12, 512, 0.25)
    #quit()

    stats = fdbgen.calculate_fdb_stats(scenes, fdb, feat)
    print('FDB stats:\n'+str(stats))
    # print('Total time: %.1fh, per scene: %.3fs' % (status['time'] / 3600, status['time'] / len(scenes)))
    scenes = None
    fdb = None

    # feat = KeypointAlgo.ORB
    # fdb_tol = math.radians(12)
    # maxmem = 384 * 1024
    # fname = os.path.join(CACHE_DIR, 'fdb_%s_w%d_m%d_t%d.pickle' % (
    #     feat,
    #     VIEW_WIDTH,
    #     maxmem / 1024,
    #     10 * math.degrees(fdb_tol)
    # ))
    # scenes, fdb = fdbgen.generate_fdb(feat, fname, fdb_tol=fdb_tol, maxmem=maxmem)
    # stats = fdbgen.calculate_fdb_stats(scenes, fdb, feat)
    # print('fdb stats:\n'+str(stats))
