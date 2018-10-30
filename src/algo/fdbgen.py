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
        self._ransac_err = self.DEF_RANSAC_ERROR/2

        # for overriding rendering
        self._sc_ast_lat = None
        self._sc_ast_lon = None
        self._light_lat = None
        self._light_lon = None

        # so that no need to always pass these in function call
        self._ref_img_sc = None
        self._sc_ast_perms = None
        self._light_perms = None

    def generate_fdb(self, feat, view_width=VIEW_WIDTH, view_height=VIEW_HEIGHT, fdb_tol=KeypointAlgo.FDB_TOL,
                     maxmem=KeypointAlgo.FDB_MAX_MEM, save_file=None):

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
        print('%d x %d = %d, <%dMB'%(n1, n2, n1*n2, n1*n2*(KeypointAlgo.FDB_MAX_MEM/1024/1024)))

        # initialize fdb array
        fdb = np.full((n1, n2), None).tolist()
        scenes = []
        timer = Stopwatch()
        timer.start()

        # first phase, just generate max amount of features per scene
        print(''.join(['_']*n1), flush=True)
        for i2, (light_lat, light_lon) in enumerate(self._light_perms):
            print('.', flush=True, end="")
            for i1, (sc_ast_lat, sc_ast_lon) in enumerate(self._sc_ast_perms):
            # tr = tracker.SummaryTracker()
                tmp = self._scene_features(feat, maxmem, sc_ast_lat, sc_ast_lon, light_lat, light_lon)
                # tr.print_diff()
                if tmp is not None:
                    fdb[i1][i2] = tmp
                    scenes.append((sc_ast_lat, sc_ast_lon, light_lat, light_lon, i1, i2))
        print('\n', flush=True, end="")

        # second phase, match with neighbours, record matching features
        visited = set()
        for i1 in range(n1):
            print('.', flush=True, end="")
            for i2 in range(n2):
                self._update_matched_features(fdb_tol, visited, fdb, i1, i2)
        print('\n', flush=True, end="")

        # third phase, discard features that didn't match with any neighbours
        for i1 in range(n1):
            print('.', flush=True, end="")
            for i2 in range(n2):
                tmp = fdb[i1][i2]
                if tmp is not None:
                    a, b, c, idxs = tmp
                    fdb[i1][i2] = (a[tuple(idxs), :], c[tuple(idxs), :])
                    #fdb[i1][i2] = list(zip(*[(a[i], b[i], c[i]) for i in idxs]))
        print('\n', flush=True, end="")

        # finished, maybe save, then exit
        if save_file is not None:
            self.save_fdb(scenes, fdb, save_file)

        timer.stop()
        secs = timer.elapsed
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

        return np.array(ref_desc), ref_kp_2d, ref_kp_3d, set()

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
        if fdb[i1][i2] is None:
            return

        coef = math.sqrt(2.5)  # math.sqrt(math.sqrt(2)**2 + (math.sqrt(2)/2)**2)
        nearest1 = tools.find_nearest_n(self._sc_ast_perms, self._sc_ast_perms[i1], r=fdb_tol*coef)
        nearest2 = tools.find_nearest_n(self._light_perms, self._light_perms[i2], r=fdb_tol*coef)

        visit = {(i1, i2, n1, n2) for n1 in nearest1 for n2 in nearest2} - {(i1, i2, i1, i2)} - visited
        for i1, i2, j1, j2 in visit:
            self._update_matched_features_inner(fdb, (i1, i2), (j1, j2))
            visited.add((i1, i2, j1, j2))
            visited.add((j1, j2, i1, i2))

    def _update_matched_features_inner(self, fdb, idxs1, idxs2):
        scene1 = fdb[idxs1[0]][idxs1[1]]
        scene2 = fdb[idxs2[0]][idxs2[1]]
        if scene1 is None or scene2 is None or idxs1 == idxs2:
            return

        try:
            matches = self._match_features(scene1[0], scene2[0], method='brute')

            # solve pnp with ransac
            ref_kp_3d = [scene2[2][m.trainIdx] for m in matches]
            sce_kp_2d = np.array([scene1[1][m.queryIdx] for m in matches], dtype='float')
            rvec, tvec, inliers = self._solve_pnp_ransac(sce_kp_2d, ref_kp_3d)

            q_res = tools.angleaxis_to_q(rvec)
            q_src = tools.ypr_to_q(*self._sc_ast_perms[idxs1[0]], 0)
            q_trg = tools.ypr_to_q(*self._sc_ast_perms[idxs2[0]], 0)
            q_rel = q_trg.conj() * q_src
            err1 = math.degrees(tools.angle_between_q(q_res, q_rel))
            err2 = np.linalg.norm(tvec - np.array((0, 0, -self.render_z)).reshape((3, 1)))
            if abs(err1) > 15 or abs(err2) > 0.1*abs(self.render_z):
                raise PositioningException()

            matches1 = [matches[i[0]].queryIdx for i in inliers]
            matches2 = [matches[i[0]].trainIdx for i in inliers]
        except PositioningException as e:
            matches1 = []
            matches2 = []

        scene1[3].update(matches1)
        scene2[3].update(matches2)

    def calculate_fdb_stats(self, scenes, fdb, feat):
        fcounts = np.array([len(b[0]) for a in fdb for b in a if b is not None])
        totmem = np.sum(fcounts) * (KeypointAlgo.BYTES_PER_FEATURE[feat] + 3 * 4)
        stats = {
            'min_feat_count': np.min(fcounts),
            'avg_feat_count': np.mean(fcounts),
            'scene_count': len(fcounts),
            'failed_scenes': np.sum(fcounts == 0),
            'weak_scenes': np.sum(fcounts < 100),
            'total_mem_usage (MB)': totmem/1024/1024,
            'accepted_feature_percent': 100*(totmem/len(fcounts)/KeypointAlgo.FDB_MAX_MEM),
        }

        return stats

    def save_fdb(self, scenes, fdb, save_file):
        with open(save_file, 'wb') as fh:
            pickle.dump((scenes, fdb), fh, -1)

    def estimate_mem_usage(self, fdb_tol_deg, sc_mem_kb, acc_ratio=0.6):
        fdb_tol = math.radians(fdb_tol_deg)
        n1 = len(tools.bf2_lat_lon(fdb_tol))
        n2 = len(tools.bf2_lat_lon(fdb_tol, lat_range=(-math.pi/2, math.radians(90 - TestLoop.MIN_ELONG))))
        print('%d x %d = %d, <%dMB, ~%dMB'%(n1, n2, n1*n2, n1*n2*(sc_mem_kb/1024), n1*n2*(sc_mem_kb/1024)*acc_ratio))


if __name__ == '__main__':
    sm = SystemModel(shape_model=HIRES_TARGET_MODEL_FILE)
    #lblloader.load_image_meta(TARGET_IMAGE_META_FILE, sm)
    re = RenderEngine(VIEW_WIDTH, VIEW_HEIGHT, antialias_samples=0)
    obj_idx = re.load_object(sm.real_shape_model)

    # 9 deg, 64kb
    # 10 deg, 96kb, 0.32 => 192kb?
    # 12 deg, 192kb

    feat = KeypointAlgo.ORB
    fdb_tol = math.radians(10)
    maxmem = 192 * 1024
    fname = os.path.join(CACHE_DIR, 'fdb_%s_w%d_m%d_t%d.pickle' % (
        feat,
        VIEW_WIDTH,
        maxmem / 1024,
        10 * math.degrees(fdb_tol)
    ))

    fdbgen = FeatureDatabaseGenerator(sm, re, obj_idx)
    scenes, fdb = fdbgen.generate_fdb(feat, fdb_tol=fdb_tol, maxmem=maxmem, save_file=fname)
    stats = fdbgen.calculate_fdb_stats(scenes, fdb, feat)
    print('fdb stats:\n'+str(stats))


