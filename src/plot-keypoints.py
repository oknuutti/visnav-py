import sys

import cv2

from algo.keypoint import KeypointAlgo


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print('USAGE: python %s <sift|surf|orb|akaze> <image file>'%(sys.argv[0],))

    img = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(cv2.resize(img, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)

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

    d = KeypointAlgo(None, None)
    for feat, method in feats.items():
        kp, desc = d.detect_features(img, feat, 0, nfeats=100)
        out = cv2.drawKeypoints(img, kp, img.copy(), (0,0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('%d %s keypoints'%(len(kp), method.upper()), out)
    cv2.waitKey()
