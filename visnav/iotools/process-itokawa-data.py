
import os
import argparse
import gzip
import shutil

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from osgeo import gdal      # use separate data_io conda env!
import cv2

# read amica images of hayabusa 1 to itokawa, instrument details:
#    - https://arxiv.org/ftp/arxiv/papers/0912/0912.4797.pdf
#    - main info is fov: 5.83° x 5.69°, focal length: 120.80 mm, px size 12 um, active pixels 1024x1000,
#      zero level monitoring with 12px left & 12px right, 12-bits, eff aperture 15 mm, full well 70ke-,
#      gain factor: 17 DN/e-, readout noise 60e-
#
# info on data format: https://sbnarchive.psi.edu/pds3/hayabusa/HAY_A_AMICA_3_AMICAGEOM_V1_0/catalog/dataset.cat
#


def main():
    parser = argparse.ArgumentParser('Process data from Hayabusa about Itokawa')
    parser.add_argument('--src', help="input folder")
    parser.add_argument('--dst', help="output folder")
    parser.add_argument('--min-size', default=256,
                        help="minimum size of an image side, discard image if less")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    for fname in tqdm(os.listdir(args.src)):
        if fname[-4:].lower() == '.lbl':
            path = os.path.join(args.src, fname[:-4])
            extracted = False

            if not os.path.exists(path + '.img'):
                extracted = True
                with gzip.open(path + '.img.gz', 'rb') as fh_in:
                    with open(path + '.img', 'wb') as fh_out:
                        shutil.copyfileobj(fh_in, fh_out)

            img, data = read_img(path + '.lbl')

            if np.min(img.shape[:2]) >= args.min_size:
                write_data(os.path.join(args.dst, fname[:-4]), img, data)

            if extracted:
                os.unlink(path + '.img')


def read_img(path):
    handle = gdal.Open(path, gdal.GA_ReadOnly)
    w, h, n = handle.RasterXSize, handle.RasterYSize, handle.RasterCount

    rawdata = handle.ReadRaster(xoff=0, yoff=0, xsize=w, ysize=h, buf_xsize=w, buf_ysize=h, buf_type=gdal.GDT_Float32)
    data = np.frombuffer(rawdata, dtype=np.float32).reshape((n, h, w))  # TODO: verify that order is h, w; not w, h

    # reorder axes
    data = np.moveaxis(data, (0, 1, 2), (2, 0, 1))

    # scale and reduce depth to 8-bits
    top_v = np.quantile(data[:, :, 0], 0.999)
    img = np.clip((0.95 * 255 / top_v) * data[:, :, 0] + 0.5, 0, 255).astype(np.uint8)

    # select only pixel value and x, y, z; calculate pixel size by taking max of px
    # - for band indexes, see https://sbnarchive.psi.edu/pds3/hayabusa/HAY_A_AMICA_3_AMICAGEOM_V1_0/catalog/dataset.cat
    px_size = np.atleast_3d(np.max(data[:, :, 10:12], axis=2))
    data = np.concatenate((data[:, :, 1:4], px_size), axis=2)
    data[data <= -1e30] = np.nan

    return img, data


def write_data(path, img, data):
    cv2.imwrite(path + '.png', img, (cv2.IMWRITE_PNG_COMPRESSION, 9))
    cv2.imwrite(path + '.xyz.exr', data[:, :, :3], (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT))
    cv2.imwrite(path + '.s.exr', data[:, :, 3:], (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT))


if __name__ == '__main__':
    if 0:
        main()
    else:
#        read_img(r'D:\projects\navex\data\itokawa\st_2393618726_v_ddr.lbl')
#        read_img(r'D:\projects\navex\data\N20150603T065950912ID40F24.IMG')
        read_img(r'D:\projects\navex\data\N20150603T065928709ID50F22 - Copy.IMG')
