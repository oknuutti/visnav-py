import numpy as np
import matplotlib.pyplot as plt
import cv2

from visnav.algo import tools
from visnav.algo.image import ImageProc
from visnav.algo.model import Camera
from visnav.calibration.base import Frame, RAW_IMG_MAX_VALUE


def analyze_aurora_img(img_file, get_bgr_cam):
    debug = 1
    n = 0
    Frame.MISSING_BG_REMOVE_STRIPES = 0

    bgr_cam = get_bgr_cam(thumbnail=False, estimated=1, final=1)
    f = Frame.from_file(bgr_cam, img_file, img_file[:-4]+'.lbl', bg_offset=False, debug=debug)

    if 0:
        f.show_image(processed=True, save_as='C:/projects/s100imgs/processed-aurora.png')

    img = f.image.astype('float')

    if 0:
        img = img - np.percentile(img, 5, axis=1).reshape((-1, 1, 3))

    bg1 = (830, 1070), (1180, 1400)
#    bg1 = (0, 900), (660, 1280)
#    bg2 = (1560, 1050), (2048, 1350)
    mean_bg = np.mean(img[bg1[0][1]:bg1[1][1], bg1[0][0]:bg1[1][0], :].reshape((-1, 3)), axis=0)
#    mean_bg = np.mean(np.vstack((img[bg1[0][1]:bg1[1][1], bg1[0][0]:bg1[1][0], :].reshape((-1, 3)),
#                                 img[bg2[0][1]:bg2[1][1], bg2[0][0]:bg2[1][0], :].reshape((-1, 3)))), axis=0)
    img = img - mean_bg
    # img = ImageProc.apply_point_spread_fn(img - mean_bg, 0.01)
    # img = np.clip(img, 0, 1023).astype('uint16')
    # img = cv2.medianBlur(img, 31)
    # img = np.clip(img, 0, RAW_IMG_MAX_VALUE) / RAW_IMG_MAX_VALUE

    n += 1
    plt.figure(n)
    imsh = (np.clip(img * 2 + RAW_IMG_MAX_VALUE * 0.3, 0, RAW_IMG_MAX_VALUE) / RAW_IMG_MAX_VALUE * 255).astype('uint8')
    rd_y = (720, 700), (890, 720)
    rd_r1 = (720, 830), (890, 870)
    rd_r2 = (1080, 820), (1250, 860)
    gr_r = (1280, 770), (1450, 795)
    cv2.rectangle(imsh, bg1[0], bg1[1], (255, 0, 0), 2)       # bg1
#    cv2.rectangle(imsh, bg2[0], bg2[1], (255, 0, 0), 2)       # bg2
    cv2.rectangle(imsh, rd_y[0], rd_y[1], (0, 200, 200), 2)       # yellow
    cv2.rectangle(imsh, rd_r1[0], rd_r1[1], (0, 0, 255), 2)     # red1
    cv2.rectangle(imsh, rd_r2[0], rd_r2[1], (0, 0, 255), 2)     # red2
    cv2.rectangle(imsh, gr_r[0], gr_r[1], (0, 255, 0), 2)     # green
    plt.imshow(np.flip(imsh, axis=2))
    plt.show()

    def electrons(lam):
        h = 6.626e-34  # planck constant (m2kg/s)
        c = 3e8  # speed of light
        return h * c / lam  # energy per photon

    blue = 427.8e-9
    green = 557.7e-9
    yellow = 589.3e-9
    red = 630.0e-9
    colors = (blue, green, yellow, red)
    dus_per_rad = dict(zip(colors, ([], [], [], [])))   # DUs per 1 W/m2/sr of radiance
    coef = f.exposure * f.gain * RAW_IMG_MAX_VALUE

    for wl in dus_per_rad.keys():
        for cam in bgr_cam:
            cgain = cam.gain * cam.emp_coef * cam.aperture_area
            fn, _ = Camera.qeff_fn(tuple(cam.qeff_coefs), 350e-9, 1000e-9)

            # W/m2/sr => phot/s/m2/sr => elec/s/m2/sr => DUs/sr
            dus_per_rad[wl].append(1/electrons(wl) * fn(wl) * coef * cgain)
    for wl in dus_per_rad.keys():
        dus_per_rad[wl] = np.array(dus_per_rad[wl])

    class Patch:
        def __init__(self, name, rect, bands, mean=None, rad=None):
            self.name, self.rect, self.bands, self.mean, self.rad = name, rect, bands, mean, rad
    nt = lambda n, r, b: Patch(name=n, rect=r, bands=b)

    patches = [
        nt('Clean Red', rd_r1, (blue, green, red)),
        nt('Strong Red', rd_r2, (blue, green, red)),
        nt('Green', gr_r, (blue, green, red)),
        nt('Sodium', rd_y, (blue, green, yellow)),
    ]

    # pseudo inverse
    for p in patches:
        p.mean = np.mean(img[p.rect[0][1]:p.rect[1][1], p.rect[0][0]:p.rect[1][0], :].reshape((-1, 3)), axis=0)
        px_sr = cam.pixel_solid_angle((p.rect[0][0]+p.rect[1][0])//2, (p.rect[0][1]+p.rect[1][1])//2)
        E = np.hstack((dus_per_rad[p.bands[0]], dus_per_rad[p.bands[1]], dus_per_rad[p.bands[2]])) * px_sr
        invE = np.linalg.inv(E.T.dot(E)).dot(E.T)
        rad = invE.dot(p.mean)    # radiance in W/m2/sr
        # e = E.dot(rad)
        # diff = (p.mean - e) * 100 / np.linalg.norm(p.mean)
        p.rad = [''] * len(colors)
        for i, b in enumerate(p.bands):
            idx = colors.index(b)
            p.rad[idx] = rad[i]

    sep = '\t' if 1 else ' & '
    le = '\n' if 1 else ' \\\\\n'
    if 0:
        print(sep.join(('Patch', 'Emission at', '', '', 'Red', 'Green', 'Blue')), end=le)
        for name, irr, mean, model, diff in patches:
            print(sep.join((name, '428 nm', ('%.3e' % irr[0]) if irr[0] else 'n/a', 'Mean',
                            *('%.1f' % m for m in np.flip(mean.flatten())))), end=le)
            print(sep.join(('', '557.7 nm', ('%.3e' % irr[1]) if irr[1] else 'n/a', 'Modeled',
                            *('%.1f' % m for m in np.flip(model.flatten())))), end=le)
            print(sep.join(('', '589 nm', ('%.3e' % irr[2]) if irr[2] else 'n/a', 'Diff. [%]',
                            *('%.1f' % m for m in np.flip(diff.flatten())))), end=le)
            print(sep.join(('', '630 nm', ('%.3e' % irr[3]) if irr[3] else 'n/a', *(['']*4))), end=le)
    else:
        print(sep.join(('Patch', 'Red', 'Green', 'Blue', '428 nm', '557.7 nm', '589 nm', '630 nm')), end=le)
        for p in patches:
            # in kilo Rayleigh (kR) == 6330*1e9*lambda * W/m2/sr    or  4*pi*10^(-10)*10^(-3) * photon flux
            print(sep.join((p.name, *('%.1f' % m for m in np.flip(p.mean.flatten())),
                                    *(tools.fixed_precision(r*4*np.pi*1e-13/electrons(colors[i]), 3, True) if r else ''  # r*1e-13*4*np.pi
                                      for i, r in enumerate(p.rad)))), end=le)
    quit()

    aurora = np.zeros_like(img)
    for i, color in enumerate(colors):
        # d/dx[(r-aw)'*(r-aw)] == 0
        #  => w == (r'*a)/(a'*a)
        a = emission[color]
        w = np.sum(img.reshape((-1, 3)) * a.T, axis=1) / sum(a**2)
        e = (w*a).T
        r = img.reshape((-1, 3)) - e
        x = w / np.linalg.norm(r, axis=1)

        # plt.figure(2)
        # plt.imshow(w.reshape(img.shape[:2])/np.max(w))
        # plt.title('weight (max=%f)' % np.max(w))

        # plt.figure(3)
        # plt.imshow(x.reshape(img.shape[:2])/np.max(x))
        # plt.title('x (max=%f)' % np.max(x))

        n += 1
        plt.figure(n)
        x[x < {red: 10, green: 6, yellow: 16}[color]] = 0
        x[w < 100] = 0
        xf = ImageProc.apply_point_spread_fn(x.reshape(img.shape[:2]), 0.03)
        xf = cv2.medianBlur(xf.astype('uint16'), 11)
        plt.imshow(xf / np.max(xf))
        plt.title('Emission detection @ %.1fnm' % (color * 1e9))

        e[xf.flatten() == 0, :] = (0, 0, 0)
        aurora += e.reshape(img.shape)

        # plt.figure(6)
        # plt.imshow(np.flip(e.reshape(img.shape) / np.max(e), axis=2))
        # plt.title('modeled aurora')

        # plt.figure(7)
        # plt.imshow(np.flip(r.reshape(img.shape)/np.max(r), axis=2))
        # plt.title('residual')
        # plt.show()

    plt.figure(8)
    plt.imshow(np.flip(aurora / np.max(aurora), axis=2))
    plt.title('modeled aurora')
    plt.show()

    # TODO: translate rgb values to aurora (ir)radiance
    #  - following uses W/m2/sr for "in-band radiance"
    #  - https://www.osapublishing.org/DirectPDFAccess/A2F3D832-975A-1850-088634AAFCF21258_186134/ETOP-2009-ESB4.pdf?da=1&id=186134&uri=ETOP-2009-ESB4&seq=0&mobile=no
    #  - use pixel sr?

    print('done')
