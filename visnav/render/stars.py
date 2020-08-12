from datetime import datetime
from functools import lru_cache

import cv2
import math
import os
import sqlite3
import re
import time

import numpy as np
import quaternion

from visnav.algo import tools
from visnav.algo.image import ImageProc
from visnav.algo.model import SystemModel
from visnav.missions.didymos import DidymosSystemModel
from visnav.missions.rosetta import RosettaSystemModel
from visnav.settings import *

#  https://pysynphot.readthedocs.io/en/latest/index.html#pysynphot-installation-setup
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import importlib
    mod = importlib.util.find_spec('pysynphot')
    if mod is not None:
        root = mod.submodule_search_locations[0]
        os.environ['PYSYN_CDBS'] = os.path.join(root, 'data', 'cdbs')  # http://ssb.stsci.edu/cdbs/tarfiles/synphot1.tar.gz
        import pysynphot as S                                          # http://ssb.stsci.edu/cdbs/tarfiles/synphot2.tar.gz
                                                                       # http://ssb.stsci.edu/cdbs/tarfiles/synphot3.tar.gz
    else:
        print('warning: module pysynphot not found')


class Stars:
                                # from VizieR catalogs:
    SOURCE_HIPPARCHOS = 'H'     #   I/239/hip_main
    SOURCE_PASTEL = 'P'         #   B/pastel/pastel
    SOURCE_WU = 'W'             #   J/A+A/525/A71/table2
    SOURCE_GAIA1 = 'G'          #   J/MNRAS/471/770/table2

    STARDB_TYC = os.path.join(DATA_DIR, 'deep_space_objects_tyc.sqlite')
    STARDB_HIP = os.path.join(DATA_DIR, 'deep_space_objects_hip.sqlite')
    STARDB = STARDB_HIP
    MAG_CUTOFF = 10
    MAG_V_LAM0 = 545e-9
    SUN_MAG_V = -26.74
    SUN_MAG_B = 0.6222 + SUN_MAG_V

    # from sc cam frame (axis: +x, up: +z) to equatorial frame (axis: +y, up: +z)
    sc2ec_q = np.quaternion(1, 0, 0, 1).normalized().conj()

    @staticmethod
    def black_body_radiation(Teff, lam):
        return Stars.black_body_radiation_fn(Teff)(lam)

    @staticmethod
    def black_body_radiation_fn(Teff):
        def phi(lam):
            # planck's law of black body radiation [W/m3/sr]
            h = 6.626e-34  # planck constant (m2kg/s)
            c = 3e8  # speed of light
            k = 1.380649e-23  # Boltzmann constant
            r = 2*h*c**2/lam**5/(np.exp(h*c/lam/k/Teff) - 1)
            return r
        return phi

    @staticmethod
    def synthetic_radiation(Teff, fe_h, log_g, lam, mag_v=None):
        return Stars.synthetic_radiation_fn(Teff, fe_h, log_g, mag_v=mag_v)(lam)

    @staticmethod
    @lru_cache(maxsize=1000)
    def synthetic_radiation_fn(Teff, fe_h, log_g, mag_v=None):
        sp = None
        orig_log_g = log_g
        if 1:
            first_try = True

            if Teff < 3500:
                print('could not init spectral model with given t_eff=%s, using t_eff=3500K instead' % Teff)
                Teff = 3500

            for i in range(15):
                try:
                    sp = S.Icat('k93models', Teff, fe_h, log_g)    # 'ck04models' or 'k93models'
                    break
                except:
                    first_try = False
                    log_g = log_g + (0.2 if Teff > 6000 else -0.2)
            assert sp is not None, 'could not init spectral model with given params: t_eff=%s, log_g=%s, fe_h=%s' % (Teff, orig_log_g, fe_h)
            if not first_try:
                print('could not init spectral model with given params (t_eff=%s, log_g=%s, fe_h=%s), changed log_g to %s' %
                      (Teff, orig_log_g, fe_h, log_g))
        else:
            sp = S.Icat('ck04models', Teff, fe_h, log_g)

        if mag_v is not None:
            sp = sp.renorm(mag_v, 'vegamag', S.ObsBandpass('johnson,v'))

        from scipy.interpolate import interp1d
        I = np.logical_and(sp.wave >= 3000, sp.wave <= 11000)  # TODO: make configurable
        sample_fn = interp1d(sp.wave[I], sp.flux[I], kind='linear', assume_sorted=True)

        def phi(lam):
            r = sample_fn(lam*1e10)    # wavelength in Å, result in "flam" (erg/s/cm2/Å)
            return r * 1e-7 / 1e-4 / 1e-10

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
    def flux_density(cam_q, cam, mask=None, mag_cutoff=MAG_CUTOFF, array=False, undistorted=False, order_by=None):
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
        # the magnitudes for tycho id xxxx-xxxxx-2 entries are bad as they are most likely taken from hip catalog that bundles all .*-(\d)
        results = cursor.execute("""
            SELECT x, y, z, mag_v""" + (", mag_b, t_eff, fe_h, log_g, dec, ra, id" if array else "") + """
            FROM deep_sky_objects
            WHERE """ + ("tycho like '%-1' AND " if Stars.STARDB == Stars.STARDB_TYC else "") +
             "mag_v < " + str(mag_cutoff) + " AND " + dec_cond + " AND " + ra_cond +
            ((" ORDER BY %s ASC" % order_by) if order_by is not None else ''))
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
            cols = ('ix', 'iy', 'x', 'y', 'z', 'mag_v', 'mag_b', 't_eff', 'fe_h', 'log_g', 'dec', 'ra', 'id')
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
    def get_catalog_id(id, field=None):
        try:
            is_arr = False
            id = int(id)
        except:
            is_arr = True
        if Stars._query_conn is None:
            Stars._conn = sqlite3.connect(Stars.STARDB)
            Stars._query_cursor = Stars._conn.cursor()
        field = field or ("tycho" if Stars.STARDB == Stars.STARDB_TYC else "hip")
        if is_arr:
            res = Stars._query_cursor.execute(
                "select id, %s from deep_sky_objects where id IN (%s)" % (
                field, ','.join(str(i) for i in id))).fetchall()
            return {r[0]: str(r[1]) for r in res}
        else:
            res = Stars._query_cursor.execute(
                "select %s from deep_sky_objects where id = %s" % (
                field, id)).fetchone()[0]
            return str(res)
    _query_conn, _query_cursor = None, None

    @staticmethod
    def _create_stardb(fname):
        conn = sqlite3.connect(fname)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS deep_sky_objects")
        cursor.execute("""
            CREATE TABLE deep_sky_objects (
                id INTEGER PRIMARY KEY ASC NOT NULL,
                hip INT,
                hd INT DEFAULT NULL,
                simbad CHAR(20) DEFAULT NULL,
                ra REAL NOT NULL,           /* src[0] */
                dec REAL NOT NULL,          /* src[0] */
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                mag_v REAL NOT NULL,        /* src[1] */
                mag_b REAL DEFAULT NULL,    /* src[2] */
                t_eff REAL DEFAULT NULL,    /* src[3] */
                log_g REAL DEFAULT NULL,    /* src[4] */
                fe_h REAL DEFAULT NULL,     /* src[5] */
                src CHAR(6) DEFAULT 'HHHPPP'
            )""")
        cursor.execute("DROP INDEX IF EXISTS ra_idx")
        cursor.execute("CREATE INDEX ra_idx ON deep_sky_objects (ra)")

        cursor.execute("DROP INDEX IF EXISTS dec_idx")
        cursor.execute("CREATE INDEX dec_idx ON deep_sky_objects (dec)")

        cursor.execute("DROP INDEX IF EXISTS mag_idx")
        cursor.execute("CREATE INDEX mag_idx ON deep_sky_objects (mag_v)")

        cursor.execute("DROP INDEX IF EXISTS hd")
        cursor.execute("CREATE INDEX hd ON deep_sky_objects (hd)")

        cursor.execute("DROP INDEX IF EXISTS simbad")
        cursor.execute("CREATE INDEX simbad ON deep_sky_objects (simbad)")

        cursor.execute("DROP INDEX IF EXISTS hip")
        cursor.execute("CREATE UNIQUE INDEX hip ON deep_sky_objects (hip)")
        conn.commit()

    @staticmethod
    def import_stars_hip():
        # I/239/hip_main
        Stars._create_stardb(Stars.STARDB_HIP)
        conn = sqlite3.connect(Stars.STARDB_HIP)
        cursor = conn.cursor()

        from astroquery.vizier import Vizier
        Vizier.ROW_LIMIT = -1

        cols = ["HIP", "HD", "_RA.icrs", "_DE.icrs", "Vmag", "B-V"]
        r = Vizier(catalog="I/239/hip_main", columns=cols, row_limit=-1).query_constraints()[0]

        for i, row in enumerate(r):
            hip, hd, ra, dec, mag_v, b_v = [row[f] for f in cols]
            if np.any(list(map(np.ma.is_masked, (ra, dec, mag_v)))):
                continue
            hd = 'null' if np.ma.is_masked(hd) else hd
            mag_b = 'null' if np.ma.is_masked(b_v) or np.isnan(b_v) else b_v + mag_v
            x, y, z = tools.spherical2cartesian(math.radians(dec), math.radians(ra), 1)
            cursor.execute("""
                INSERT INTO deep_sky_objects (hip, hd, ra, dec, x, y, z, mag_v, mag_b)
                VALUES (%s, %s, %f, %f, %f, %f, %f, %f, %s)"""
                   % (hip, hd, ra, dec, x, y, z, mag_v, mag_b))
            if i % 100 == 0:
                conn.commit()
                tools.show_progress(len(r), i)

        conn.commit()
        conn.close()


    @staticmethod
    def import_stars_tyc():
        assert False, 'not supported anymore'
        Stars._create_stardb(Stars.STARDB_TYC, 12)
        conn = sqlite3.connect(Stars.STARDB_TYC)
        cursor = conn.cursor()

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
    def add_simbad_col():
        conn = sqlite3.connect(Stars.STARDB)
        cursor_r = conn.cursor()
        cursor_w = conn.cursor()

        # cursor_w.execute("alter table deep_sky_objects add column simbad char(20) default null")
        # conn.commit()

        N_tot = cursor_r.execute("SELECT max(id) FROM deep_sky_objects WHERE 1").fetchone()[0]

        skip = 0
        result = cursor_r.execute("select id, hip from deep_sky_objects where id >= %d" % skip)

        import time
        from astroquery.simbad import Simbad
        Simbad.add_votable_fields('typed_id')
        while 1:
            rows = result.fetchmany(1000)
            if rows is None or len(rows) == 0:
                break
            tools.show_progress(N_tot, rows[0][0]-1)

            s = Simbad.query_objects(['HIP %d' % int(row[1]) for row in rows])
            time.sleep(2)

            values = []
            if s is not None:
                s.add_index('TYPED_ID')
                for row in rows:
                    sr = get(s, ('HIP %d' % int(row[1])).encode('utf-8'))
                    if sr is not None:
                        k = sr['MAIN_ID'].decode('utf-8')
                        values.append("(%d, '%s', 0,0,0,0,0,0)" % (row[0], k.replace("'", "''")))
            if len(values) > 0:
                cursor_w.execute("""
                    INSERT INTO deep_sky_objects (id, simbad, ra, dec, x, y, z, mag_v) VALUES """ + ','.join(values) + """
                    ON CONFLICT(id) DO UPDATE SET simbad = excluded.simbad""")
                conn.commit()
        conn.close()


    @staticmethod
    def query_t_eff():
        from astroquery.vizier import Vizier
        v = Vizier(catalog="B/pastel/pastel", columns=["ID", "Teff", "logg", "[Fe/H]"], row_limit=-1)
        v2 = Vizier(catalog="J/A+A/525/A71/table2", columns=["Name", "Teff", "log(g)", "[Fe/H]"], row_limit=-1)
        v3 = Vizier(catalog="J/MNRAS/471/770/table2", columns=["HIP", "Teff", "log(g)"], row_limit=-1)

        conn = sqlite3.connect(Stars.STARDB)
        cursor_r = conn.cursor()
        cursor_w = conn.cursor()

        cond = "(t_eff is null OR log_g is null OR 1)"
        N_tot = cursor_r.execute("""
            SELECT max(id) FROM deep_sky_objects 
            WHERE %s
            """ % cond).fetchone()[0]

        skip = 37601
        f_id, f_hip, f_hd, f_sim, f_ra, f_dec, f_t, f_g, f_m, f_src = range(10)
        results = cursor_r.execute("""
            SELECT id, hip, hd, simbad, ra, dec, t_eff, log_g, fe_h, src
            FROM deep_sky_objects
            WHERE %s AND id >= ?
            ORDER BY id ASC
            """ % cond, (skip,))

        r = v.query_constraints()[0]
        r.add_index('ID')

        N = 40
        while True:
            rows = results.fetchmany(N)
            if rows is None or len(rows) == 0:
                break
            tools.show_progress(N_tot, rows[0][f_id]-1)

            ids = {row[f_id]: [i, row[f_src][:3] + '___'] for i, row in enumerate(rows)}
            insert = {}
            for i, row in enumerate(rows):
                k = 'HIP %6d' % int(row[f_hip])
                if get(r, k) is None and row[f_hd]:
                    k = 'HD %6d' % int(row[f_hd])
                if get(r, k) is None and row[f_sim]:
                    k = row[f_sim]
                if get(r, k) is None and row[f_sim]:
                    k = row[f_sim] + ' A'
                dr = get(r, k)
                if dr is not None:
                    t_eff, log_g, fe_h = median(dr, ('Teff', 'logg', '__Fe_H_'), null='null')
                    src = row[f_src][0:3] + ''.join([('_' if v == 'null' else Stars.SOURCE_PASTEL) for v in (t_eff, log_g, fe_h)])
                    insert[row[f_id]] = [t_eff, log_g, fe_h, src]
                    if '_' not in src[3:5]:
                        ids.pop(row[f_id])
                    else:
                        ids[row[f_id]][1] = src

            if len(ids) > 0:
                # try using other catalog
                r = v2.query_constraints(Name='=,' + ','.join([
                    ('HD%06d' % int(rows[i][f_hd])) for i, s in ids.values() if rows[i][f_hd] is not None
                ]))
                time.sleep(2)
                if len(r) > 0:
                    r = r[0]
                    r.add_index('Name')
                    for id, (i, src) in ids.copy().items():
                        dr = get(r, 'HD%06d' % int(rows[i][f_hd])) if rows[i][f_hd] else None
                        if dr is not None:
                            t_eff, log_g, fe_h = median(dr, ('Teff', 'log_g_', '__Fe_H_'), null='null')
                            src = src[0:3] + ''.join([('_' if v == 'null' else Stars.SOURCE_WU) for v in (t_eff, log_g, fe_h)])
                            insert[id] = [t_eff, log_g, fe_h, src]
                            if '_' not in src[3:5]:
                                ids.pop(rows[i][f_id])
                            else:
                                ids[rows[i][f_id]][1] = src

            if len(ids) > 0:
                # try using other catalog
                r = v3.query_constraints(HIP='=,' + ','.join([str(rows[i][f_hip]) for i, s in ids.values()]))[0]
                r.add_index('HIP')
                for id, (i, src) in ids.copy().items():
                    dr = get(r, int(rows[i][f_hip]))
                    if dr is not None:
                        t_eff, log_g = median(dr, ('Teff', 'log_g_'), null='null')
                        src = src[0:3] + ''.join([('_' if v == 'null' else Stars.SOURCE_GAIA1) for v in (t_eff, log_g)]) + src[5]
                        insert[id] = [t_eff, log_g, insert[id][2] if id in insert else 'null', src]
                        # if '_' not in src[3:5]:
                        #     ids.pop(rows[i][f_id])
                        # else:
                        #     ids[rows[i][f_id]][1] = src

            if len(insert) > 0:
                values = ["(%d, %s, %s, %s, '%s', 0,0,0,0,0,0)" % (
                                id, t_eff, log_g, fe_h, src)
                            for id, (t_eff, log_g, fe_h, src) in insert.items()]
                cursor_w.execute("""
                    INSERT INTO deep_sky_objects (id, t_eff, log_g, fe_h, src, ra, dec, x, y, z, mag_v) VALUES """ + ','.join(values) + """
                    ON CONFLICT(id) DO UPDATE SET 
                        t_eff = excluded.t_eff, 
                        log_g = excluded.log_g, 
                        fe_h = excluded.fe_h,
                        src = excluded.src
                """)
                conn.commit()
        conn.close()

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
        Vizier.ROW_LIMIT = -1

        hips = [hip] if isinstance(hip, str) else hip

        v = Vizier(columns=["HIP", "Vmag", "B-V"], catalog="I/239/hip_main", row_limit=-1)
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


def get(r, k, d=None):
    if k is None or r is None:
        return d
    try:
        return r.loc[k]
    except:
        return d

def median(dr, fields, null='null'):
    try:
        values = [np.ma.median(dr[f]) for f in fields]
        values = [(null if np.ma.is_masked(v) else v) for v in values]
    except:
        values = [null if np.ma.is_masked(dr[f]) or np.isnan(dr[f]) else dr[f] for f in fields]
    return values


if __name__ == '__main__':
    if 0:
        Stars.import_stars_hip()
        quit()
    elif 0:
        Stars.add_simbad_col()
        #Stars.override_rho_ori_b()
        #Stars.override_delta_ori_b()
        quit()
    elif 0:
        Stars.query_t_eff()
        quit()
    elif 0:
        img = np.zeros((1024, 1024), dtype=np.uint8)
        for i in range(1000):
            Stars.plot_stars(img, tools.rand_q(math.radians(180)), cam, exposure=5, gain=1)
        quit()
    elif 1:
        conn = sqlite3.connect(Stars.STARDB)
        cursor = conn.cursor()
        f_id, f_hip, f_sim, f_hd, f_magv, f_magb, f_teff, f_logg, f_feh, f_src = range(10)
        r = cursor.execute("""
            SELECT id, hip, simbad, hd, mag_v, mag_b, t_eff, log_g, fe_h, src
            FROM deep_sky_objects
            WHERE hd in (48915,34085,61421,39801,35468,37128,37742,37743,44743,38771,36486,48737,36861)
            ORDER BY mag_v
        """)
        rows = r.fetchall()
        stars = {}
        print('id\thip\tsim\thd\tmag_v\tmag_b\tt_eff\tlog_g\tfe_h\tsrc')
        for row in rows:
            stars[row[f_hd]] = row
            print('\t'.join([str(c) for c in row]))
        conn.close()

        from astropy.io import fits
        import matplotlib.pyplot as plt

        def testf(fdat, teff, logg, feh):
            sp = S.Icat('k93models', float(teff), float(feh), float(logg))\
                  .renorm(0, 'vegamag', S.ObsBandpass('johnson,v'))
            sp_real = S.ArraySpectrum(wave=fdat[0][0], flux=fdat[0][1], fluxunits='flam')\
                       .renorm(0, 'vegamag', S.ObsBandpass('johnson,v'))
            plt.plot(sp_real.wave, sp_real.flux)
            plt.plot(sp.wave, sp.flux)
            plt.xlim(3000, 10000)
            plt.show()

        for hd in (48737, 35468, 39801):  # Lambda Orionis (HD36861) Teff too high for model (37689K)
            fname = r'C:\projects\s100imgs\spectra\%s.fits' % hd
            fdat = fits.getdata(fname)
            teff, logg, feh = [stars[hd][f] for f in (f_teff, f_logg, f_feh)]
            if teff > 30000:
                logg = max(logg, 4.0)
            testf(fdat, teff, logg, feh or 0)

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