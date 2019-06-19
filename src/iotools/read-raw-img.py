import sys

import numpy as np
import cv2


def main():
    w, h = map(int, (sys.argv[1] if len(sys.argv) > 1 else '2048x1944').split('x'))
    imgfile = sys.argv[2] if len(sys.argv) > 2 else r'D:\Downloads\example-navcam-imgs\navcamTests0619\rubbleML-def-14062019-25.raw'
    imgout = sys.argv[3] if len(sys.argv) > 3 else r'D:\Downloads\example-navcam-imgs\navcamTests0619\rubbleML-def-14062019-25.png'

    with open(imgfile, 'rb') as fh:
        raw_img = np.fromfile(fh, dtype=np.uint16, count=w * h)

    raw_img = raw_img.reshape((h, w))

    # flip rows pairwise
    final_img = raw_img[:, np.array([(i*2+1, i*2) for i in range(0, w//2)]).flatten()]

    cv2.imshow('img', final_img)
    cv2.waitKey()

    cv2.imwrite(imgout, final_img)


if __name__ == '__main__':
    main()