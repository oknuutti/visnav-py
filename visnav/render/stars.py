from datetime import datetime

import cv2
import math
import os
import sqlite3
import re

import numpy as np
import quaternion

from visnav.algo import tools
from visnav.algo.image import ImageProc
from visnav.algo.model import SystemModel
from visnav.missions.didymos import DidymosSystemModel
from visnav.missions.rosetta import RosettaSystemModel
from visnav.settings import *


# TODO: investigate the usage of UCAC4 catalog (113M stars up to mag 16) through astropy


class Stars:
    STARDB = os.path.join(DATA_DIR, 'deep_space_objects.sqlite')
    MAG_CUTOFF = 10
    MAG_V_LAM0 = 545e-9
    SUN_MAG_V = -26.74
    SUN_MAG_B = 0.6222 + SUN_MAG_V

    # from sc cam frame (axis: +x, up: +z) to equatorial frame (axis: +y, up: +z)
    sc2ec_q = np.quaternion(1, 0, 0, 1).normalized().conj()

    @staticmethod
    def black_body_radiation(T, lam):
        return Stars.black_body_radiation_fn(T)(lam)

    @staticmethod
    def black_body_radiation_fn(T):
        def phi(lam):
            # planck's law of black body radiation [W/m3/sr]
            h = 6.626e-34  # planck constant (m2kg/s)
            c = 3e8  # speed of light
            k = 1.380649e-23  # Boltzmann constant
            r = 2*h*c**2/lam**5/(np.exp(h*c/lam/k/T) - 1)
            return r
        return phi

    @staticmethod
    def magnitude_to_spectral_flux_density(mag):
        # spectral flux density for standard magnitude for V-band (at 545nm)
        #   from "Model atmospheres broad-band colors, bolometric corrections and temperature calibrations for O - M stars"
        #   Bessel M.S. et al, Astronomy and Astrophysics, 1998, table A2
        #   Also at http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/magsystems.pdf
        #   363.1e-11 erg/cm2/s/Å  (erg=1e-7J, 1cm2=1e-4m2, Å=1e-10m)
        phi0 = 363.1e-11 * 1e-7 / 1e-4 / 1e-10    # W/m3
        return np.power(10., -0.4 * mag) * phi0

    @staticmethod
    def tycho_to_johnson(mag_bt, mag_vt):
        v = mag_vt - 0.09 * (mag_bt - mag_vt)
        b = 0.85 * (mag_bt - mag_vt) + v
        return b, v

    @staticmethod
    def effective_temp(b_v, metal=0, log_g=0):
        """ magnitudes in johnson system """
        # calculate star effective temperatures, from:
        #   - http://help.agi.com/stk/index.htm#stk/starConstruction.htm
        #   - Sekiguchi, M. and Fukugita, M., 2000. A Study of the B−V Color-Temperature Relation. The Astronomical Journal, 120(2), p.1072.
        #   - metallicity (Fe/H) and log surface gravity can be set to zero without big impact
        c0 = 3.939654
        c1 = -0.395361
        c2 = 0.2082113
        c3 = -0.0604097
        f1 = 0.027153
        f2 = 0.005036
        g1 = 0.007367
        h1 = -0.01069
        return 10**(c0+c1*(b_v)+c2*(b_v)**2+c3*(b_v)**3 + f1*metal + f2*metal**2 + g1*log_g + h1*(b_v)*log_g)

    @staticmethod
    def flux_density(cam_q, cam, mask=None, mag_cutoff=MAG_CUTOFF, array=False, undistorted=False):
        """
        plots stars based on Tycho-2 database, gives out photon count per unit area given exposure time in seconds,
        cam_q is a quaternion in ICRS coord frame, x_fov and y_fov in degrees
        """

        # calculate query conditions for star ra and dec
        cam_dec, cam_ra, _ = tools.q_to_ypr(cam_q)   # camera boresight in ICRS coords
        d = np.linalg.norm((cam.x_fov, cam.y_fov))/2

        min_dec, max_dec = math.degrees(cam_dec) - d, math.degrees(cam_dec) + d
        dec_cond = '(dec BETWEEN %s AND %s)' % (min_dec, max_dec)

        # goes over the pole to the other side of the sphere, easy solution => ignore limit on ra
        skip_ra_cond = min_dec < -90 or max_dec > 90

        if skip_ra_cond:
            ra_cond = '1'
        else:
            min_ra, max_ra = math.degrees(cam_ra) - d, math.degrees(cam_ra) + d
            if min_ra < 0:
                ra_cond = '(ra < %s OR ra > %s)' % (max_ra, (min_ra + 360) % 360)
            elif max_ra > 360:
                ra_cond = '(ra > %s OR ra < %s)' % (min_ra, max_ra % 360)
            else:
                ra_cond = '(ra BETWEEN %s AND %s)' % (min_ra, max_ra)

        conn = sqlite3.connect(Stars.STARDB)
        cursor = conn.cursor()
        results = cursor.execute("""
            SELECT x, y, z, mag_v""" + (", mag_b, t_eff, dec, ra, id" if array else "") + """
            FROM deep_sky_objects
            WHERE tycho like '%-1' AND mag_v < """ + str(mag_cutoff) + " AND " + dec_cond + " AND " + ra_cond)
        stars = np.array(results.fetchall())
        conn.close()

        flux_density = ([], None) if array else np.zeros((cam.height, cam.width), dtype=np.float32)
        if len(stars) == 0:
            return flux_density

        stars[:, 0:3] = tools.q_times_mx(SystemModel.sc2gl_q.conj() * cam_q.conj(), stars[:, 0:3])
        stars_ixy_ = cam.calc_img_R(stars[:, 0:3], undistorted=undistorted)
        stars_ixy = np.round(stars_ixy_.astype(np.float)).astype(np.int)
        I = np.logical_and.reduce((np.all(stars_ixy >= 0, axis=1),
                                   stars_ixy[:, 0] <= cam.width-1,
                                   stars_ixy[:, 1] <= cam.height-1))
        if array:
            cols = ('ix', 'iy', 'x', 'y', 'z', 'mag_v', 'mag_b', 't_eff', 'dec', 'ra', 'id')
            return (
                np.hstack((stars_ixy_[I, :], stars[I, :])),
                dict(zip(cols, range(len(cols))))
            )

        stars_ixy = stars_ixy[I, :]
        flux_density_per_star = Stars.magnitude_to_spectral_flux_density(stars[I, 3])
        for i, f in enumerate(flux_density_per_star):
            flux_density[stars_ixy[i, 1], stars_ixy[i, 0]] += f

        if mask is not None:
            flux_density[np.logical_not(mask)] = 0

        if True:
            # assume every star is like our sun, convert to total flux density [W/m2]
            solar_constant = 1360.8
            # sun magnitude from http://mips.as.arizona.edu/~cnaw/sun.html
            sun_flux_density = Stars.magnitude_to_spectral_flux_density(Stars.SUN_MAG_V)
            flux_density = flux_density * (solar_constant / sun_flux_density)

        return flux_density

    @staticmethod
    def get_tycho_id(id):
        if Stars._query_conn is None:
            Stars._conn = sqlite3.connect(Stars.STARDB)
            Stars._query_cursor = Stars._conn.cursor()
        results = Stars._query_cursor.execute("select tycho from deep_sky_objects where id = ?", (id,))
        res = results.fetchone()[0]
        return res
    _query_conn, _query_cursor = None, None

    @staticmethod
    def import_stars():
        conn = sqlite3.connect(Stars.STARDB)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS deep_sky_objects")
        cursor.execute("""
            CREATE TABLE deep_sky_objects (
                id INTEGER PRIMARY KEY ASC NOT NULL,
                tycho CHAR(12),
                ra REAL NOT NULL,
                dec REAL NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                mag_b REAL NOT NULL,
                mag_v REAL NOT NULL,
                t_eff REAL DEFAULT NULL
            )""")
        cursor.execute("CREATE INDEX ra_idx ON deep_sky_objects (ra)")
        cursor.execute("CREATE INDEX dec_idx ON deep_sky_objects (dec)")
        cursor.execute("CREATE INDEX mag_idx ON deep_sky_objects (mag_v)")
        cursor.execute("CREATE UNIQUE INDEX tycho ON deep_sky_objects (tycho)")
        conn.commit()

        # Tycho-2 catalogue, from http://archive.eso.org/ASTROM/TYC-2/data/
        for file in ('catalog.dat', 'suppl_1.dat'):
            with open(os.path.join(DATA_DIR, file), 'r') as fh:
                line = fh.readline()
                while line:
                    c = line
                    line = fh.readline()

                    # mean position, ICRS, at epoch 2000.0
                    # proper motion milliarcsecond/year
                    # apparent magnitude
                    if file == 'catalog.dat':
                        # main catalog
                        epoch = 2000.0
                        tycho, ra, dec, pmra, pmdec, mag_bt, mag_vt = c[0:12], c[15:27], c[28:40], c[41:48], c[49:56], c[110:116], c[123:129]
                        mag_b, mag_v = Stars.tycho_to_johnson(float(mag_bt), float(mag_vt))
                    else:
                        # supplement-1 has the brightest stars, from hipparcos and tycho-1
                        epoch = 1991.25
                        tycho, ra, dec, pmra, pmdec, mag_bt, mag_vt, flag, hip = \
                            c[0:12], c[15:27], c[28:40], c[41:48], c[49:56], c[83:89], c[96:102], c[81:82], c[115:120]
                        if flag in ('H', 'V', 'B'):
                            if len(hip.strip()) > 0:
                                mag_b, mag_v = Stars.get_hip_mag_bv(hip)
                            else:
                                continue
                        else:
                            mag_b, mag_v = Stars.tycho_to_johnson(float(mag_bt), float(mag_vt))

                    tycho = tycho.replace(' ', '-')
                    if np.all(list(map(tools.numeric, (ra, dec)))):
                        ra, dec = list(map(float, (ra, dec)))
                        if -10 < mag_v < Stars.MAG_CUTOFF:
                            curr_epoch = datetime.now().year + \
                                         (datetime.now().timestamp()
                                            - datetime.strptime(str(datetime.now().year),'%Y').timestamp()
                                          )/365.25/24/3600
                            years = curr_epoch - epoch

                            # TODO: (1) adjust to current epoch using proper motion and years since epoch

                            x, y, z = tools.spherical2cartesian(math.radians(dec), math.radians(ra), 1)
                            cursor.execute("INSERT INTO deep_sky_objects (tycho,ra,dec,x,y,z,mag_b,mag_v) VALUES (?,?,?,?,?,?,?,?)", (
                                tycho, (ra+360)%360, dec, x, y, z, mag_b, mag_v
                            ))
        conn.commit()
        conn.close()

    @staticmethod
    def query_t_eff():
        from astroquery.vizier import Vizier
        v = Vizier(columns=["Tycho", "Teff"], catalog="V/136/tycall")
        vhd = Vizier(columns=["TYC1", "TYC2", "TYC3", "HD"], catalog="IV/25/tyc2_hd")
        v2 = Vizier(catalog="B/pastel/pastel", columns=["ID", "Teff", "RAJ2000", "DEJ2000"])
        #v.query_constraints(ID='2MASS J06450887-1642566') # ID=HIP000085, TYC0002-01155-1, HD000005, BD+000444

        #v = Vizier(catalog='J/A+A/450/735/table2', columns=['HIP', 'Teff'])    # cant find sirius
        #v = Vizier(catalog='J/BaltA/20/89', columns=["Name", "Teff"])  # no sirius

        conn = sqlite3.connect(Stars.STARDB)
        cursor_r = conn.cursor()
        cursor_w = conn.cursor()
        N_tot = cursor_r.execute("SELECT max(id) FROM deep_sky_objects WHERE t_eff is null").fetchone()[0]
        skip = 355062

        results = cursor_r.execute("""
            SELECT id, tycho, ra, dec
            FROM deep_sky_objects
            WHERE t_eff is null and id >= ?
            ORDER BY id ASC
            """, (skip,))

        N = 35
        while True:
            rows = results.fetchmany(N)
            if rows is None or len(rows) == 0:
                break

            tools.show_progress(N_tot, rows[0][0])

            ids = {row[0]: i for i, row in enumerate(rows)}
            r = v.query_constraints(Tycho='=,'+','.join([row[1] for row in rows]))
            insert = []
            if len(r):
                r = r[0]
                r.add_index('Tycho')
                for row in rows:
                    try:
                        t_eff = r.loc[row[1]]['Teff']
                        if not np.ma.is_masked(t_eff):
                            insert.append("(%d, %.1f, 0,0,0,0,0,0,0)" % (row[0], float(t_eff)))
                            ids.pop(row[0])
                    except:
                        pass

            if len(ids) > 0:
                # try using other catalog
                for id, i in ids.items():
                    r = v2.query_constraints(ID='TYC' + Stars.get_tycho_id(id))
                    if len(r) == 0:
                        t1, t2, t3 = rows[i][1].split('-')
                        hdr = vhd.query_constraints(TYC1=t1, TYC2=t2, TYC3=t3)
                        if len(hdr) > 0 and not np.ma.is_masked(hdr[0]['HD'][0]):
                            hd = 'HD%06d' % hdr[0]['HD'][0]
                            r = v2.query_constraints(ID=hd)
#                    if len(r) == 0:
#                        r = v2.query_constraints(RAJ2000=rows[i][2], DEJ2000=rows[i][3])
                    if len(r) > 0:
                        t_eff = np.mean(r[0]['Teff'])
                        insert.append("(%d, %.1f, 0,0,0,0,0,0,0)" % (rows[i][0], float(t_eff)))

            if len(insert) > 0:
                cursor_w.execute("""
                INSERT INTO deep_sky_objects (id, t_eff, ra, dec, x, y, z, mag_b, mag_v) VALUES """ + ','.join(insert) + """
                ON CONFLICT(id) DO UPDATE SET t_eff = excluded.t_eff
                """)
                conn.commit()

    @staticmethod
    def correct_supplement_data():
        conn = sqlite3.connect(Stars.STARDB)
        cursor = conn.cursor()

        def insert_mags(hips):
            res = Stars.get_hip_mag_bv([h[0] for h in hips.values()])
            insert = ["('%s', %f, %f, %f, %f, %f, %f, %f)" %
                      (t, h[1], h[2], h[3], h[4], h[5], res[h[0]][0], res[h[0]][1])
                      for t, h in hips.items() if h[0] in res and -10 < res[h[0]][1] < Stars.MAG_CUTOFF]
            if len(insert) > 0:
                cursor.execute("""
                    INSERT INTO deep_sky_objects (tycho, ra, dec, x, y, z, mag_b, mag_v) VALUES
                     """ + ','.join(insert) + """
                    ON CONFLICT(tycho) DO UPDATE SET mag_b = excluded.mag_b, mag_v = excluded.mag_v """)
                conn.commit()

        file = 'suppl_1.dat'
        N = 30
        rx = re.compile(r'0*(\d+)')
        with open(os.path.join(DATA_DIR, file), 'r') as fh:
            hips = {}
            line = fh.readline()
            while line:
                c = line
                line = fh.readline()
                tycho, ra, dec, mag_bt, mag_vt, flag, hip = c[0:12], c[15:27], c[28:40], c[83:89], c[96:102], c[81:82], c[115:123]
                tycho = tycho.replace(' ', '-')
                hip = rx.findall(hip)[0] if len(hip.strip()) > 0 else False

                if flag in ('H', 'V', 'B') and hip:
                    ra, dec = float(ra), float(dec)
                    x, y, z = tools.spherical2cartesian(math.radians(dec), math.radians(ra), 1)
                    hips[tycho] = (hip, ra, dec, x, y, z)
                    if len(hips) >= N:
                        insert_mags(hips)
                        hips.clear()
                    else:
                        continue

        if len(hips) > 0:
            insert_mags(hips)

    @staticmethod
    def get_hip_mag_bv(hip, v=None):
        from astroquery.vizier import Vizier
        hips = [hip] if isinstance(hip, str) else hip

        v = Vizier(columns=["HIP", "Vmag", "B-V"], catalog="I/239/hip_main")
        r = v.query_constraints(HIP='=,'+','.join(hips))

        results = {}
        if len(r):
            r = r[0]
            r.add_index('HIP')
            for h in hips:
                try:
                    if not np.ma.is_masked(r.loc[int(h)]['Vmag']) and not np.ma.is_masked(r.loc[int(h)]['B-V']):
                        mag_v, b_v = float(r.loc[int(h)]['Vmag']), float(r.loc[int(h)]['B-V'])
                        results[h] = (mag_v + b_v, mag_v)
                except:
                    continue

        return results.get(hip, (None, None)) if isinstance(hip, str) else results

    @staticmethod
    def override_betelgeuse():
        conn = sqlite3.connect(Stars.STARDB)
        cursor = conn.cursor()

        # from "The Advanced Spectral Library (ASTRAL): Reference Spectra for Evolved M Stars",
        # The Astrophysical Journal, 2018, https://iopscience.iop.org/article/10.3847/1538-4357/aaf164/pdf
        #t_eff = 3650        # based on query_t_eff was 3562
        #mag_v = 0.42        # based on tycho2 suppl2 was 0.58

        # from CTOA observations on 2018-12-07 and 18-12-22, accessed through https://www.aavso.org database
        mag_v = 0.8680
        mag_b = 2.6745  # based on tycho2 suppl2 was 2.3498
        t_eff = None  #  Stars.effective_temp(mag_b - mag_v, metal=0.006, log_g=-0.26) gives 3565K vs 3538K without log_g & metal

        cursor.execute("UPDATE deep_sky_objects SET t_eff=?, mag_v=?, mag_b=? where tycho='0129-01873-1'", (t_eff, mag_v, mag_b))
        conn.commit()
        conn.close()


if __name__ == '__main__':
    if 0:
        Stars.import_stars()
        quit()
    elif 1:
        Stars.override_betelgeuse()
        quit()
    elif 0:
        Stars.query_t_eff()
        quit()
    elif 0:
        img = np.zeros((1024, 1024), dtype=np.uint8)
        for i in range(1000):
            Stars.plot_stars(img, tools.rand_q(math.radians(180)), cam, exposure=5, gain=1)
        quit()

#    cam = RosettaSystemModel(focused_attenuated=False).cam
    cam = DidymosSystemModel(use_narrow_cam=True).cam

#    cam_q = tools.rand_q(math.radians(180))
    cam_q = quaternion.one
    for i in range(100):
        cam_q = tools.ypr_to_q(0, np.radians(1), 0) * cam_q
        flux_density = Stars.flux_density(cam_q, cam)
        img = cam.sense(flux_density, exposure=2, gain=2)

        img = np.clip(img*255, 0, 255).astype('uint8')
        img = ImageProc.adjust_gamma(img, 1.8)

        sc = min(768/cam.width, 768/cam.height)
        cv2.imshow('stars', cv2.resize(img, None, fx=sc, fy=sc))
        cv2.waitKey()
    print('done')