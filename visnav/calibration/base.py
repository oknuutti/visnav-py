import configparser
from abc import ABC
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

from datetime import datetime

from visnav.algo.image import ImageProc
from visnav.algo.model import Camera

MISSING_BG_REMOVE_STRIPES = True
RAW_IMG_MAX_VALUE = 567


def merge(old, new):
    for n, v in new.items():
        if n not in old:
            old[n] = []
        old[n].extend(v)


def plot_bgr_qeff(cams, ax=None, color=None, hold=False, **kwargs):
    color = color or ('b', 'g', 'r')
    for col, cam in zip(color, cams):
        cam.plot_qeff_fn(ax=ax, color=col, **kwargs)
    if ax is None and not hold:
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Quantum efficiency [%]')
        plt.tight_layout()
        plt.show()
    # while not plt.waitforbuttonpress():
    #    pass


def nanocam_gain(gainval: int, debug=False):
    gain = gainval >> 6
    if gain <= 32:
        val = gain
    elif gain <= 64:
        val = (gain >> 1) | 0x40
    else:
        val = (gain - 63) >> 3 << 8 | 0x60

    g0 = 1 + (val >> 6 & 0x01)
    g1 = float(val & 0x3f) / 8
    g2 = 1 + float(val >> 8 & 0x7f) / 8
    actual_gain = g0 * g1 * g2

    if debug:
        print('gain value: %d\nregister value: %d (0x%x)\nactual gain: %.3f (%.3f x %.3f x %.3f)' % (
            gainval, val, val, actual_gain, g0, g1, g2
        ))
    return actual_gain


# NANOCAM_GAINS = 65535: x128, 32768: x64, 16384: x32, 2048: x4, 64: x0.125
NANOCAM_GAINS = {g: nanocam_gain(g) for g in (65535, 32768, 16384, 2048, 64)}


class Measure(ABC):
    def __init__(self, frame, cam_i, obj_id, du_count, weight=1):
        self.frame = frame
        self.cam_i = cam_i
        self.obj_id = obj_id
        self.du_count = du_count
        self.weight = weight

        # cached value for expected du
        self.c_expected_du = None

    def expected_du(self, pre_sat_gain=1, post_sat_gain=1, qeff_coefs=None, psf_coef=(1, 1, 1)):
        assert False, 'not implemented'


class Frame:
    _bg_imgs = {}
    CURRENT_ID = 0
    MISSING_BG_REMOVE_STRIPES = MISSING_BG_REMOVE_STRIPES

    def __init__(self, cam, gain, exposure, timestamp, raw_image, background_img, bg_offset=0, bits=8, applied_gamma=1.0,
                 applied_gamma_break=0.0, applied_bgr_mx=None, debug=False):
        self.id = Frame.CURRENT_ID
        Frame.CURRENT_ID += 1

        self.cam = [cam] if isinstance(cam, Camera) else cam
        self.resize_scale = raw_image.shape[1] / self.cam[0].width
        for c in self.cam:
            c.height, c.width = raw_image.shape[:2]
        self.bits = bits = int(bits)
        self.gain = gain
        self.exposure = exposure
        self.timestamp = timestamp
        self.raw_image = raw_image
        self.applied_gamma = applied_gamma
        self.applied_gamma_break = applied_gamma_break
        self.applied_bgr_mx = applied_bgr_mx
        self.debug = debug

        img_bits = int(str(raw_image.dtype)[4:])
        max_val = 2**img_bits-1
        img = raw_image.astype('float')

        # NOTE: NanoCam has this, doesnt make sense in general!
        operation_order = reversed((
            'ex_gamma',
            'depth',
            'color',
            'gamma',
        ))

        for op in operation_order:
            if op == 'depth' and img_bits != bits:
                img = ImageProc.change_color_depth(img, img_bits, bits)
                max_val = 2 ** bits - 1
            if op == 'gamma' and applied_gamma != 1.0:
                img = ImageProc.adjust_gamma(img, applied_gamma, gamma_break=applied_gamma_break, inverse=True, max_val=max_val)
            if op == 'color' and applied_bgr_mx is not None:
                img = ImageProc.color_correct(img, applied_bgr_mx, inverse=True, max_val=max_val)
            # if op == 'ex_gamma' and GAMMA_ADJUSTMENT:
            #     img = ImageProc.adjust_gamma(img, GAMMA_ADJUSTMENT, inverse=True, max_val=max_val)

        self.background_img = background_img
        if background_img is not None:
            self.image = ImageProc.remove_bg(img, background_img, gain=1, offset=bg_offset, max_val=max_val)
        elif self.MISSING_BG_REMOVE_STRIPES:
            for k in range(img.shape[2]):
                img[:, :, k] -= np.percentile(img[:, :, k], 50, axis=0).reshape((1, -1))
                img[:, :, k] -= np.percentile(img[:, :, k], 50, axis=1).reshape((-1, 1))
            img += bg_offset - np.min(img)
            self.image = np.clip(img, 0, max_val)
        else:
            self.image = img

        if bg_offset is not False:
            self.image = np.round(self.image).astype('uint16')

        self.measures = []

    @property
    def max_val(self):
        return 2**self.bits - 1

    @classmethod
    def from_file(cls, cam, img_file, lbl_file, section='main', mapping=None, override=None, bg_offset=0, debug=False):
        assert os.path.exists(lbl_file), 'file %s for metadata is missing' % lbl_file

        meta = configparser.ConfigParser()
        meta.read(lbl_file)

        class Mapping:
            def __init__(self, meta, section, mapping, override):
                self.meta = meta
                self.section = section
                self.mapping = mapping or {}
                self.override = override or {}

            def __getitem__(self, param):
                return self.get(param)

            def get(self, param, default=None):
                v = self.meta.get(self.section, self.mapping.get(param, param), vars=override, fallback=default)
                try:
                    return float(v)
                except:
                    pass
                try:
                    if v[0] == '[' and v[-1] == ']' or v[0] == '(' and v[-1] == ')':
                        a = np.fromstring(v.strip('([])'), sep=',')
                        if len(a) > v.count(','):
                            return a if v[0] == '[' else tuple(a)
                except:
                    pass
                try:
                    return datetime.strptime(v, '%Y-%m-%d %H:%M:%S %z')
                except:
                    pass
                return v

        meta = Mapping(meta, section, mapping, override)

        if meta['bg_image']:
            bg_path = os.path.realpath(os.path.join(os.path.dirname(lbl_file), meta['bg_image']))
            if bg_path not in Frame._bg_imgs:
                bg_img = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
                Frame._bg_imgs[bg_path] = bg_img
            background_img = Frame._bg_imgs[bg_path]
        else:
            background_img = None
        raw_image = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

        bgr_mx = None
        if meta['channels'] == 3 and len(meta.get('ccm_bgr_red', [])) == 3:
            bgr_mx = np.array([meta['ccm_bgr_blue'], meta['ccm_bgr_green'], meta['ccm_bgr_red']])

        frame = cls(cam, meta['gain'], meta['exposure'], meta['timestamp'], raw_image,
                    background_img, bg_offset, meta['bits'], meta['gamma'], meta['gamma_break'], bgr_mx, debug=debug)

        cls.process_metadata(frame, meta)
        return frame

    @classmethod
    def process_metadata(cls, frame, meta):
        # override in subclasses
        pass

    def show_image(self, gain=1, processed=False, compare=False, median_filter=False, zero_bg=False, save_as=None):
        img = self.image.astype('float')
        if processed:
            if zero_bg:
                img = np.clip(img - np.min(img) - (0 if zero_bg is True else zero_bg), 0, np.inf)
            img *= gain
            if median_filter:
                img = cv2.medianBlur(img.astype('uint16'), median_filter)
            img = ImageProc.color_correct(img, self.applied_bgr_mx, max_val=self.max_val)
            img = ImageProc.adjust_gamma(img, self.applied_gamma, self.applied_gamma_break, max_val=self.max_val)
        else:
            img = np.clip(img * gain, 0, 2**self.bits - 1)
        img = ImageProc.change_color_depth(img, self.bits, 8).astype('uint8')

        if save_as is not None:
            cv2.imwrite(save_as, img)

        s = self.image.shape
        if compare:
            img = np.hstack((self.raw_image.astype(img.dtype), np.ones((s[0], 1, s[2]), dtype=img.dtype), img))

        sc = 1
        plt.imshow(np.flip(img, axis=2))
        plt.show()
        return img, sc

