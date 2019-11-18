import cv2
import math
import os
import sqlite3

import numpy as np
import quaternion

from visnav.algo import tools
from visnav.algo.image import ImageProc
from visnav.algo.model import Camera, SystemModel
from visnav.missions.didymos import DidymosSystemModel
from visnav.missions.rosetta import RosettaSystemModel
from visnav.settings import *


class Stars:
    STARDB = os.path.join(DATA_DIR, 'deep_space_objects.sqlite')
    MAG_CUTOFF = 10
    SUN_MAG_V = -26.74
    SUN_MAG_B = 0.6222 + SUN_MAG_V

    # from sc cam frame (axis: +x, up: +z) to equatorial frame (axis: +y, up: +z)
    sc2ec_q = np.quaternion(1, 0, 0, 1).normalized().conj()

    @staticmethod
    def magnitude_to_flux_density(mag):
        # flux density for standard magnitude for V-band
        #   from https://en.wikipedia.org/wiki/Apparent_magnitude  (erg=1e-7J, 1cm2=1e-4m2)
        phi0 = 3.636e-20 * 1e-7 / 1e-4    # J/(s*m2*Hz)
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
    def flux_density(cam_q, cam, mask=None):
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
            SELECT x, y, z, mag_v
            FROM deep_sky_objects
            WHERE """ + dec_cond + " AND " + ra_cond)
        stars = np.array(results.fetchall())
        conn.close()

        stars[:, 0:3] = tools.q_times_mx(SystemModel.sc2gl_q.conj() * cam_q.conj(), stars[:, 0:3])
        stars_ixy = np.round(cam.calc_img_R(stars[:, 0:3])).astype(np.int)
        I = np.logical_and.reduce((np.all(stars_ixy >= 0, axis=1),
                                   stars_ixy[:, 0] <= cam.width-1,
                                   stars_ixy[:, 1] <= cam.height-1))
        stars_ixy = stars_ixy[I, :]

        flux_density_per_star = Stars.magnitude_to_flux_density(stars[I, 3])
        flux_density = np.zeros((cam.height, cam.width), dtype=np.float32)
        for i, f in enumerate(flux_density_per_star):
            flux_density[stars_ixy[i, 1], stars_ixy[i, 0]] += f

        if mask is not None:
            flux_density[np.logical_not(mask)] = 0

        if True:
            # assume every star is like our sun, convert to total flux density [W/m2]
            solar_constant = 1360.8
            # sun magnitude from http://mips.as.arizona.edu/~cnaw/sun.html
            sun_flux_density = Stars.magnitude_to_flux_density(Stars.SUN_MAG_V)
            flux_density = flux_density * (solar_constant / sun_flux_density)

        return flux_density


    @staticmethod
    def import_stars():
        conn = sqlite3.connect(Stars.STARDB)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS deep_sky_objects")
        cursor.execute("""
            CREATE TABLE deep_sky_objects (
                id INTEGER PRIMARY KEY ASC NOT NULL,
                ra REAL NOT NULL,
                dec REAL NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                mag_b REAL NOT NULL,
                mag_v REAL NOT NULL
            )""")
        cursor.execute("CREATE INDEX ra_idx ON deep_sky_objects (ra)")
        cursor.execute("CREATE INDEX dec_idx ON deep_sky_objects (dec)")
        cursor.execute("CREATE INDEX mag_idx ON deep_sky_objects (mag_v)")
        conn.commit()

        # Tycho-2 catalogue, from http://archive.eso.org/ASTROM/TYC-2/data/
        with open(os.path.join(DATA_DIR, 'catalog.dat'), 'r') as fh:
            line = fh.readline()
            while line:
                c = line
                line = fh.readline()

                # mean position, ICRS, at epoch 2000.0
                # proper motion milliarcsecond/year
                # apparent magnitude
                ra, dec, pmra, pmdec, mag_b, mag_v = c[15:27], c[28:40], c[41:48], c[49:56], c[110:116], c[123:129]

                if np.all(list(map(tools.numeric, (ra, dec, mag_b, mag_v)))):
                    ra, dec, mag_b, mag_v = list(map(float, (ra, dec, mag_b, mag_v)))
                    mag_b, mag_v = Stars.tycho_to_johnson(mag_b, mag_v)
                    if -10 < mag_v < Stars.MAG_CUTOFF:
                        # TODO: adjust for proper motion
                        x, y, z = tools.spherical2cartesian(math.radians(dec), math.radians(ra), 1)
                        cursor.execute("INSERT INTO deep_sky_objects (ra,dec,x,y,z,mag_b,mag_v) VALUES (?,?,?,?,?,?,?)", (
                            (ra+360)%360, dec, x, y, z, mag_b, mag_v
                        ))
        conn.commit()

        # TODO: (3) import also supplement1.dat (stars with not so high quality estimates as the rest of tycho-2)
        # TODO: (1) import brightest stars (Vt (or V?) < 1.9) from hipparcos-2 & -1, supposedly missing from tycho-2

        conn.close()


if __name__ == '__main__':
    if False:
        Stars.import_stars()
        quit()
    elif False:
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