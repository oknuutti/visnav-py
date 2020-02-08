import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from visnav.algo import tools
from visnav.algo.base import AlgorithmBase
from visnav.algo.image import ImageProc
from visnav.algo.tools import Stopwatch, PositioningException
from poseilluminet import PoseIllumiNet, PoseIllumiDataset
from visnav.settings import *


class AbsoluteNavigationNN(AlgorithmBase):
#    DEF_MODEL_NAME = 'model_best_rose-mob-gradloss-x1.pth.tar'
#    DEF_MODEL_NAME = 'model_best_rose-mob-v6.pth.tar'
#    DEF_MODEL_NAME = 'model_best_rose-mob-v4.pth.tar'
#    DEF_MODEL_NAME = 'rose-mob-adv-v2.pth.tar'
    DEF_MODEL_NAME = 'model_best_rose-mob-gradloss-x03.pth.tar'

    DEF_LUMINOSITY_THRESHOLD = 65
    DEF_CROP_MARGIN = 10
    DEF_MIN_PIXELS = int(np.pi * 50 ** 2 * 0.3)
    DEF_ESTIMATE_THRESHOLD = False

    def __init__(self, system_model, render_engine, obj_idx, model_name=None, use_cuda=True, verbose=True):
        super(AbsoluteNavigationNN, self).__init__(system_model, render_engine, obj_idx)
        
        self.model_name = model_name or AbsoluteNavigationNN.DEF_MODEL_NAME
        self.model = None    # lazy load
        self.verbose = verbose

        if use_cuda:
            torch.cuda.current_device()
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

    def load_model(self, path=None):
        path = path or os.path.join(DATA_DIR, self.model_name)
        data = torch.load(path)

        name = data.get('name', '')
        if len(name) > 0:
            assert name.split('-')[0] == self.system_model.mission_id, \
                    "wrong model loaded (%s) for current mission (%s)" % (name.split('-')[0], self.system_model.mission_id)

        if self.verbose:
            print("model '%s' loaded (%s: %s, nf=%d, do=%.1f), trained for %d epochs, validation loss %.3f" % (
                path, name, data['arch'], data.get('features', 2048), data.get('dropout', 0.5),
                data['epoch'], data.get('loss', np.nan),))

        # referred from densepose-project
        self.model = PoseIllumiNet(arch=data['arch'],
                                   width_mult=data.get('width_mult', 1.0),
                                   num_features=data.get('features', 2048),
                                   dropout=data.get('dropout', 0.5))

        for k in ('cost_fn.gamma', 'cost_fn.beta'):
            data['model'].pop(k)
        self.model.load_state_dict(data['model'])
        # optimizer.load_state_dict(data['optimizer'])

        self.model.to(self.device)
        self.model.eval()

    def process(self, orig_sce_img, outfile, rotate_sc=False, **kwargs):
        # maybe load torch model
        if self.model is None:
            self.load_model()

        if outfile is not None:
            self.debug_filebase = outfile + ('n' if isinstance(orig_sce_img, str) else '')

        # maybe load scene image
        if isinstance(orig_sce_img, str):
            orig_sce_img = self.load_target_image(orig_sce_img)

        self.timer = Stopwatch()
        self.timer.start()

        if self.DEF_ESTIMATE_THRESHOLD:
            threshold = ImageProc.optimal_threshold(None, orig_sce_img)
        else:
            threshold = self.DEF_LUMINOSITY_THRESHOLD

        # detect target, get bounds
        x, y, w, h = ImageProc.single_object_bounds(orig_sce_img, threshold=threshold,
                                                    crop_marg=self.DEF_CROP_MARGIN,
                                                    min_px=self.DEF_MIN_PIXELS, debug=DEBUG)
        if x is None:
            raise PositioningException('asteroid not detected in image')
        
        # crop image
        img_bw = ImageProc.crop_and_zoom_image(orig_sce_img, x, y, w, h, None, (224, 224))

        # save cropped image in log archive
        if BATCH_MODE and self.debug_filebase:
            self.timer.stop()
            cv2.imwrite(self.debug_filebase+'a.png', img_bw)
            self.timer.start()

        # massage input
        input = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2BGR)
        input = Image.fromarray(input)
        input = PoseIllumiDataset.eval_transform(input)[None, :, :, :].to(self.device, non_blocking=True)

        # run model
        with torch.no_grad():
            output = self.model(input)
        
        # massage output
        output = output[0] if isinstance(output, (list, tuple)) else output
        output = output.detach().cpu().numpy()

        # check if estimated illumination direction is close or not
        ill_est = self.model.illumination(output)[0]
        r_ini, q_ini, ill_ini = self.system_model.get_cropped_system_scf(x, y, w, h)
        if tools.angle_between_v(ill_est, ill_ini) > 10:    # max 10 degree discrepancy accepted
            print('bad illumination direction estimated, initial=%s, estimated=%s' % (ill_ini, ill_est))

        # apply result
        r_est = self.model.position(output)[0]
        q_est = np.quaternion(*self.model.rotation(output)[0])
        self.system_model.set_cropped_system_scf(x, y, w, h, r_est, q_est, rotate_sc=rotate_sc)
        self.timer.stop()

        if False:
            r_est2, q_est2, ill_est2 = self.system_model.get_cropped_system_scf(x, y, w, h)
            self.system_model.swap_values_with_real_vals()
            r_real, q_real, ill_real = self.system_model.get_cropped_system_scf(x, y, w, h)
            self.system_model.swap_values_with_real_vals()
            print('compare q_est vs q_est2, q_real vs q_est, q_real vs q_est2')

        # save result image
        if BATCH_MODE and self.debug_filebase:
            # save result in log archive
            res_img = self.render(textures=False)
            sce_img = cv2.resize(orig_sce_img, tuple(np.flipud(res_img.shape)))
            cv2.imwrite(self.debug_filebase+'b.png', np.concatenate((sce_img, res_img), axis=1))
            if DEBUG:
                cv2.imshow('compare', np.concatenate((sce_img, res_img), axis=1))
                cv2.waitKey()
