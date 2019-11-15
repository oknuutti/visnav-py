import sys

import cv2

from algo.keypoint import KeypointAlgo
from missions.rosetta import RosettaSystemModel
from render.render import RenderEngine

from settings import *

if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print('USAGE: python %s <sift|surf|orb|akaze> <image file>'%(sys.argv[0],))
        quit()

    img = cv2.imread(os.path.join(BASE_DIR, sys.argv[2]), cv2.IMREAD_GRAYSCALE)
    #img = cv2.cvtColor(cv2.resize(img, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)

    method = sys.argv[1].lower()
    if method == 'orb':
        feats = {KeypointAlgo.ORB: method.upper()}
    elif method == 'akaze':
        feats = {KeypointAlgo.AKAZE: method.upper()}
    elif method == 'sift':
        feats = {KeypointAlgo.SIFT: method.upper()}
    elif method == 'surf':
        feats = {KeypointAlgo.SURF: method.upper()}
    elif method == 'all':
        feats = {
            KeypointAlgo.SIFT:'sift',
            KeypointAlgo.SURF:'surf',
            KeypointAlgo.ORB: 'orb',
            KeypointAlgo.AKAZE:'akaze',
        }

    sm = RosettaSystemModel()
    re = RenderEngine(sm.cam.width, sm.cam.height)
    obj_idx = re.load_object(sm.asteroid.target_model_file)
    d = KeypointAlgo(sm, re, obj_idx)

    for feat, method in feats.items():
        if False:
            kp, desc, detector = d.detect_features(img, feat, 0, nfeats=100)
            out = cv2.drawKeypoints(img, kp, img.copy(), (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            d.FEATURE_FILTERING_RELATIVE_GRID_SIZE = 0.01
            d.FEATURE_FILTERING_FALLBACK_GRID_SIZE = 2
            ee = 0
            if len(sys.argv) <= 3 or sys.argv[3] == '2':
                d.FEATURE_FILTERING_SCHEME = d.FFS_SIMPLE_GRID
                ee = sm.pixel_extent(abs(sm.min_med_distance))
            elif sys.argv[3] == '1':
                d.FEATURE_FILTERING_SCHEME = d.FFS_SIMPLE_GRID
            else:
                d.FEATURE_FILTERING_SCHEME = d.FFS_NONE

            kp, desc, detector = d.detect_features(img, feat, 0, max_feats=2000, for_ref=True, expected_pixel_extent=ee)
            out = cv2.drawKeypoints(img, kp, img.copy(), (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        cv2.imshow('%d %s keypoints'%(len(kp), method.upper()), out)
    cv2.waitKey()
