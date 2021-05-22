import os
from functools import lru_cache

import numpy as np
from scipy.interpolate import interp1d
import pysynphot as S
from pysynphot.spectrum import CompositeSourceSpectrum

from visnav.algo.model import Camera
from visnav.render.stars import Stars


def photon_E(lam):
    h = 6.626e-34  # planck constant m2kg/s
    c = 3e8  # speed of light m/s
    return h * c / lam  # energy per photon (J/ph)


@lru_cache(maxsize=1000)
def sensed_electron_flux_star_spectrum(path, bayer, mag_v, Teff, log_g, fe_h,
                                       lam_min, lam_max, qeff_coefs, gomos_mag_v=None):
    spectrum_fn = get_star_spectrum(path, bayer, mag_v, Teff, log_g, fe_h, lam_min, lam_max, gomos_mag_v)
    electrons, _ = Camera.electron_flux_in_sensed_spectrum_fn(qeff_coefs, spectrum_fn, lam_min, lam_max)
    return electrons


@lru_cache(maxsize=1000)
def get_star_spectrum(path, bayer, mag_v, Teff, log_g, fe_h, lam_min, lam_max, gomos_mag_v=None):
    lam_min, lam_max = lam_min - 10e-9, lam_max + 10e-9

    sp = Stars.uncached_synthetic_radiation_fn(Teff, fe_h, log_g, mag_v=mag_v, model='ck04models',
                                               return_sp=True, lam_min=lam_min, lam_max=lam_max)

    g_file, spec_g = os.path.join(path, 'gomos-' + bayer + '.npz'), None
    if os.path.exists(g_file):
        spec_dat_g = np.load(g_file)
        lam_g = spec_dat_g['lam']
        spec_g = spec_dat_g['q50'] * photon_E(lam_g) * 1e4 * 1e9 * 1e9  # was in 1e-9 ph/s/cm2/nm, turn into W/m3

        sp_g = Stars.synthetic_radiation_fn(None, None, None, mag_v=gomos_mag_v,
                                            model=(tuple(lam_g * 1e-9), tuple(spec_g)),
                                            return_sp=True, lam_min=lam_min, lam_max=lam_max)

        sp = MixedSourceSpectrum(sp_g, sp, ((lam_min*1e10, 6900), (7550, 7750), (9260, 9540)))

    # for performance reasons (caching) (?)
    from scipy.interpolate import interp1d
    I = np.logical_and(sp.wave >= lam_min*1e10, sp.wave <= lam_max*1e10)
    sample_fn = interp1d(sp.wave[I], sp.flux[I], kind='linear', assume_sorted=True)

    def phi(lam):
        r = sample_fn(lam*1e10)    # wavelength in Å, result in "flam" (erg/s/cm2/Å)
        return r * 1e-7 / 1e-4 / 1e-10  # result in W/m3

    return phi


class MixedSourceSpectrum(CompositeSourceSpectrum):
    def __init__(self, source1, source2, source1_valid_intervals):
        super(MixedSourceSpectrum, self).__init__(source1, source2, 'add')
        self.operation = 'mix'

        n = len(source1_valid_intervals)
        bounds = np.concatenate((np.array([0]),
                                 np.array(source1_valid_intervals).flatten(),
                                 np.array([np.inf])))
        valid = np.concatenate((np.array([0]),
                                np.stack((np.ones((n,)), np.zeros((n,))), axis=1).flatten(),
                                np.array([0])))
        self.mask_fn = interp1d(bounds, valid, kind='previous')

    def __str__(self):
        return "%s fallback on %s" % (str(self.component1), str(self.component2))

    def __call__(self, wavelength):
        """
        If source1 not valid as defined by source1_valid_intervals, return component from source2
        """
        mask = self.mask_fn(wavelength)
        return self.component1(wavelength) * mask + self.component2(wavelength) * (1 - mask)
