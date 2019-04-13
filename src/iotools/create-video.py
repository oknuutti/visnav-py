import re
import sys

import cv2

from algo import tools
from settings import *

if __name__ == '__main__':
    try:
        folder = os.path.join(LOG_DIR, sys.argv[1])
        assert os.path.isdir(folder), 'invalid folder path given: %s'%(folder,)
        regex = sys.argv[2]
        target_file = sys.argv[3]
        framerate = int(sys.argv[4])
    except Exception as e:
        print(str(e))
        print('\nUSAGE: %s <log-dir relative input-dir> <img_regex> <target_file> <framerate>'%(sys.argv[0],))
        quit()

    test = re.compile(regex)
    img_files = []
    for file in os.listdir(folder):
        if test.match(file):
            img_files.append(file)

    img_files = sorted(img_files)
    assert len(img_files)>3, 'too few images found: %s'%(img_files,)

    img0 = cv2.imread(os.path.join(folder, img_files[0]), cv2.IMREAD_COLOR)
    h, w, c = img0.shape
    writer = cv2.VideoWriter(target_file, cv2.VideoWriter_fourcc(*'DIVX'), framerate, (w, h))
    try:
        for i, f in enumerate(img_files):
            tools.show_progress(len(img_files), i)
            img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_COLOR)
            writer.write(img)
    finally:
        writer.release()
