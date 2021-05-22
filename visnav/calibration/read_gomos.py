import os
import argparse
import re

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# use separate conda environment with
#   - conda create -n coda -c stcorp -c conda-forge python=3.8 coda scipy
#   - need to download definition files manually to from https://github.com/stcorp/codadef-envisat-gomos
#     to ...\envs\coda\Library\share\coda\codadef
#   - then download script codadef.sh from https://github.com/stcorp/coda/blob/master/codadef.sh
#   - then run `codadef.sh envisat-gomos`
from visnav.calibration.spectrum import get_star_spectrum

os.putenv('CODA_DEFINITION', r'C:\ProgramData\Anaconda3\envs\coda\Library\share\coda\codadef')
try:
    import coda
except Exception as e:
    print("Can't import coda: %s" % e)


def main():
    parser = argparse.ArgumentParser('Read star reference spectra from a folder,'
                                     ' calculate mean, std, and quantiles (0.05, 0.50, 0.95),'
                                     ' save into a file in numpy format')
    parser.add_argument('--path', help="input folder")
    parser.add_argument('--out', help="output file")
    parser.add_argument('--use-measures', action="store_true",
                        help="use measurement data instead of the reference star spectrum")
    parser.add_argument('--plot-all', action="store_true", help="plot all spectra and also the result")
    parser.add_argument('--plot', action="store_true", help="plot the result")
    args = parser.parse_args()

    spectra = []
    for file in os.listdir(args.path):
        if file[-3:] != '.N1':
            continue

        #   - see https://github.com/stcorp/codadef-documentation for product_class, product_type, version
        #   - to list field names: coda.get_field_names(h,""): [
        #       'mph', 'sph', 'dsd', 'tra_summary_quality', 'tra_occultation_data',
        #       'tra_nom_wav_assignment', 'tra_ref_star_spectrum', 'tra_ref_atm_dens_profile',
        #       'tra_transmission', 'tra_satu_and_sfa_data', 'tra_auxiliary_data',
        #       'tra_geolocation'
        #     ]
        #   - meanings of the fields: http://envisat.esa.int/handbooks/gomos/CNTR2-2-3.html

        # file, product_class, product_type, version
        h = coda.open_as(os.path.join(args.path, file), 'ENVISAT_GOMOS', 'GOM_TRA_1P', 1)

        v = coda.fetch(h, 'sph')
        m = re.match(r'\d*(.*)', v[12].strip())
        bayer = m[1] if m else v[12].strip()

        v = coda.fetch(h, 'tra_ref_star_spectrum')
        spe_ref_raw = np.array(v[0][1]).reshape((1, -1))  # in electrons/pixel/0.5s

        if args.use_measures:
            # the actual measurement data:
            n = 10
            v = coda.fetch(h, 'tra_transmission')
            spe_ref_raw = np.stack([np.array(r[2]) for r in v], axis=0)[:n, :] * spe_ref_raw

        v = coda.fetch(h, 'tra_nom_wav_assignment')
        lam = np.array(v[0][0])                         # in nm

        v = coda.fetch(h, 'tra_occultation_data')
        sens_len = v[0][10]                          # how many elements the curve has
        sens_lam = np.array(v[0][11])[0:sens_len]    # in nm
        sens_val = np.array(v[0][12])[0:sens_len]    # in (photons/s/cm2/nm) / (electrons/pixel/0.5s)

        coda.close(h)

        # hack to alleviate a problem arising from interpolation and a sharp fall in sensitivity around 389nm
        dl = np.array([-0.3, 0, 0.3])
        sens_lam = np.concatenate((sens_lam[:12], sens_lam[12:13] + dl, sens_lam[13:]))
        sens_val = np.concatenate((sens_val[:12], sens_val[11:14], sens_val[13:]))

        interp = interp1d(sens_lam, sens_val, kind='linear')
        spe_ref = spe_ref_raw * interp(lam).reshape((1, -1))

        spectra.append(spe_ref)

        if args.plot_all:
            plt.plot(lam.reshape((1, -1)).repeat(len(spe_ref), axis=0), spe_ref)
            # plt.plot(lam, spe_ref_raw)
            # plt.plot(sens_lam, sens_val * 0.8*np.max(spe_ref)/np.max(sens_val), 'x-')
            plt.title(file)
            plt.xlabel('Wavelenth [nm]')
            plt.ylabel('Photon flux density [ph/s/cm2/nm]')
            plt.show()

    if len(spectra) == 0:
        raise Exception('no spectra found in folder %s' % args.path)

    spectra = np.concatenate(spectra, axis=0)
    quantiles = np.quantile(spectra, (0.05, 0.50, 0.95), axis=0)
    mean = np.mean(spectra, axis=0)
    std = np.std(spectra, axis=0)

    outfile = args.out.replace('{{bayer}}', bayer.lower().replace(' ', '_'))
    if outfile[-4:] != '.npz':
        outfile = outfile + '.npz'
    np.savez(outfile, bayer=bayer, lam=lam, mean=mean, std=std,
             q05=quantiles[0, :], q50=quantiles[1, :], q95=quantiles[2, :])

    if args.plot or args.plot_all:
        plt.plot(lam, quantiles[1, :], 'C0-')
        plt.plot(lam, quantiles[0, :], 'C1:')
        plt.plot(lam, quantiles[2, :], 'C1:')
        plt.title('Median spectrum of star %s' % (bayer,))
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Photon flux density [ph/s/cm2/nm]')
        plt.show()


def costfn(p, lam_g, spec_g):
    from visnav.algo import tools

    mag_v, Teff, log_g, fe_h = p
    Teff, log_g = abs(Teff), abs(log_g)

    try:
        spec_fn = Stars.synthetic_radiation_fn(Teff, fe_h, log_g, mag_v=mag_v, model='ck04models')
    except AssertionError:
        return np.ones_like(spec_g)

    spec_m = spec_fn(lam_g * 1e-9)
    diff = tools.pseudo_huber_loss(0.1, np.log(spec_m) - np.log(spec_g))
    if np.any(np.isnan(diff)):
        diff = np.ones_like(spec_g)
    return diff


def photon_E(lam):
    h = 6.626e-34  # planck constant m2kg/s
    c = 3e8  # speed of light m/s
    return h * c / lam  # energy per photon (J/ph)


if __name__ == '__main__':
    if 0:
        main()
    else:
        from scipy.optimize import leastsq
        from visnav.render.stars import Stars

        all, fit_model = True, False
        NORMALIZE_ALL = True
        stellar_atm_model, model_label = [('ck04models', 'Castelli & Kurucz 2004'),
                                          ('k93models', 'Kurucz 1993')][0]

        path = 'C:\projects\s100imgs\spectra'
        full_star_data = (
#            ('Sirius', 'alp_cma', -1.46, 9847, 4.3, 0.49),
#            ('Rigel', 'bet_ori',  0.13, 9379, 0.863, 0.0),  # median Pastel & Simbad: 0.13, var star db: 0.17-0.22
            ('Procyon', 'alp_cmi', 0.37, 6586.5, 4.0, -0.015),   # 0.40
            ('Betelgeuse', 'alp_ori', (0.35, 0.708), 3540, 0.0, 0.05),  # 0.45, 3654, 3520, 3450   o: 3626, 0.13, 0.09
            ('Bellatrix', 'gam_ori', 1.64, 22339, 3.84, -0.07),
            ('Alnilam', 'eps_ori', 1.69, 15339, 1.574, 0.0),
            ('Alnitak', 'zet_ori', 1.74, 20788, 2.409, 0.0),
            ('Mirzam', 'bet_cma', 1.97, 24953, 3.624, 0.0),
            ('Saiph', 'kap_ori', 2.06, 15257, 2.662, 0.0),
            ('Mintaka', 'del_ori', 2.25, 21298, 3.21, 0.0),
            ('Cursa', 'bet_eri', 2.79, 8002, 3.78, -0.2),
            ('Gomeisa', 'bet_cmi', 2.89, 8197, 3.12, 0.0),
        )   # missing: Cursa (bet eri), Gomeisa (bet cmi), [and Alzirr (ksi gem), Meissa (lam ori)]
        #full_star_data = (full_star_data[1],)
        full_star_data = sorted(full_star_data, key=lambda x: x[2][1] if isinstance(x[2], tuple) else x[2])  # sort by vmag

        for k in range(1 if all else len(full_star_data)):
            if not all:
                star_data = (full_star_data[k],)
            else:
                star_data = full_star_data

            if all:
                fig, axs = plt.subplots(3, 4, sharex=True, sharey=False, figsize=[11.0, 5.5])
            else:
                fig, axs = plt.subplots(1, 1, sharex=True, sharey=False, figsize=[11.0, 5.5])
                axs = np.array([axs])
            axs = axs.flatten()

            jump = 0
            for i, (name, bayer, mag_v, Teff, log_g, fe_h) in enumerate(star_data):
                g_file, spec_g = os.path.join(path, 'gomos-' + bayer + '.npz'), None
                if os.path.exists(g_file):
                    spec_dat_g = np.load(g_file)
                    lam_g = spec_dat_g['lam']
                    spec_g = spec_dat_g['q50'] * photon_E(lam_g) * 1e4 * 1e9 * 1e9   # was in 1e-9 ph/s/cm2/nm, turn into W/m3

                    src_mag_v = None
                    if isinstance(mag_v, tuple):
                        src_mag_v, mag_v = mag_v

                    if NORMALIZE_ALL or src_mag_v is not None:
                        spec_g_fn = Stars.synthetic_radiation_fn(None, None, None,
                                                                 mag_v=mag_v, model=(tuple(lam_g*1e-9), tuple(spec_g)))
                        spec_g = spec_g_fn(lam_g*1e-9)

                    if not all and fit_model:
                        I = np.logical_and.reduce((lam_g > 340, lam_g < 1000, np.logical_not(np.isnan(spec_g))))
                        res = leastsq(costfn, (mag_v, Teff, log_g, fe_h), args=(lam_g[I], spec_g[I]),
                                      full_output=True, maxfev=100)
                        mag_v, Teff, log_g, fe_h = res[0]
                        Teff, log_g = abs(Teff), abs(log_g)

                spec_fn = Stars.synthetic_radiation_fn(Teff, fe_h, log_g, mag_v=mag_v, model=stellar_atm_model)
                lam_s = np.linspace(350, 1000, 3000)
                spec_s = spec_fn(lam_s * 1e-9)

                if i >= 2:
                    i += 2

                if spec_g is not None:
                    spec_g[1415] = np.nan   # so that line wont continue over the gaps
                    spec_g[1835] = np.nan
                    I = lam_g >= 340
                    axs[i].plot(lam_g[I], spec_g[I] * 1e-9, 'C0', linewidth=0.6, label='GOMOS reference spectrum')   # to W/m2/nm
                axs[i].plot(lam_s, spec_s * 1e-9, 'C1', linewidth=0.6, label=model_label)  # to W/m2/nm

                if i // 4 == 2 or not all:
                    axs[i].set_xlabel('Wavelength [nm]')

                if i % 4 == 0 or not all:
                    axs[i].set_ylabel('Irradiance [W/m2/nm]')

                if not all:
                    axs[i].set_title('%s - GOMOS reference spectrum vs \nKurucz-93 model with Vmag=%.2f, Teff=%d, log_g=%.3f, fe_h=%.2f' % (
                    name, mag_v, Teff, log_g, fe_h))
                else:
                    axs[i].set_title(name)

            fig.tight_layout()

            if all:
                axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axs[2].axis('off')
                axs[3].axis('off')
            else:
                axs[len(star_data) - 1].legend()

            plt.show()
