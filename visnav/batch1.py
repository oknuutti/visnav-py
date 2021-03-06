import re
import sys

from visnav.missions.bennu import BennuSystemModel

sys.tracebacklimit = 10

from scipy import optimize

from visnav.settings import *
from visnav.missions.didymos import DidymosSystemModel
from visnav.missions.rosetta import RosettaSystemModel
from visnav.testloop import TestLoop


def get_system_model(mission, hi_res_shape_model=False, res_mult=1.0):
    rose = re.match(r'^rose(\d{3})?$', mission)

    if rose:
        mission = 'rose'
        batch = rose[1] if rose[1] else '006'
        sm = RosettaSystemModel(hi_res_shape_model=hi_res_shape_model, rosetta_batch='mtp'+batch, res_mult=res_mult)
    elif mission == 'orex':
        sm = BennuSystemModel(hi_res_shape_model=hi_res_shape_model, res_mult=res_mult)
    elif mission == 'didy1n':
        sm = DidymosSystemModel(target_primary=True, hi_res_shape_model=hi_res_shape_model, res_mult=res_mult)
    elif mission == 'didy1w':
        sm = DidymosSystemModel(target_primary=True, hi_res_shape_model=hi_res_shape_model, use_narrow_cam=False, res_mult=res_mult)
    elif mission == 'didy2n':
        sm = DidymosSystemModel(target_primary=False, hi_res_shape_model=hi_res_shape_model, res_mult=res_mult)
    elif mission == 'didy2w':
        sm = DidymosSystemModel(target_primary=False, hi_res_shape_model=hi_res_shape_model, use_narrow_cam=False, res_mult=res_mult)
    else:
        assert False, 'Unknown mission given as argument: %s' % mission
    assert mission == sm.mission_id, 'wrong system model mission id'

    return sm


def run_batch(mission, full_method, count, est_real_ast_orient=False):
    m = full_method.split('+')
    sg = None
    method=m[0]
    
    if False:
        count = 10
        kwargs = {
            'method':'phasecorr', 'min_options':{
            'method':'anneal',
            'niter':50, 'T':0.15, 'stepsize':0.03,
            'minimizer_kwargs':{
                'method':'COBYLA',
                'options':{'maxiter':6, 'rhobeg': 0.02},
        }}}
    elif False:
        count = 10
        kwargs = {
            'method':'phasecorr', 'min_options':{
            'method':'anneal',
            'niter':20, 'T':0.15, 'stepsize':0.05,
            'minimizer_kwargs':{
                'method':'Nelder-Mead',
                'options':{'maxiter':8, 'xtol':1e-3, 'ftol':1e-3}
        }}}
    elif False:
        def finfun(f, x0, *args, **kwargs):
            res = optimize.minimize(f, x0,
#                    method='Nelder-Mead',
#                    options={'maxiter':20, 'xtol':1e-3, 'ftol':1e-3},
               method='COBYLA',
               options={'maxiter':10, 'rhobeg': 0.01},
            )
            return res.x, res.fun, res.status

        count = 1000
        kwargs = {
            'method':'phasecorr', 'centroid_init':True,
            'min_options':{
                'method':'brute',
                'max_iter':50,
                'finish':None, #finfun,
            }
        }
    elif method=='phasecorr':
        kwargs = {
            'method':'phasecorr', 'centroid_init':False,
            'min_options':{
                'method':'two-step-brute',
                'first':{
                    'max_iter':60,
                },
                'second':{
                    'margin':20,            # in original image pixels 
                    'distance_margin':0.05, # distance search space centered around first round result
                    'max_iter':20,
                },
            }
        }
    elif method=='centroid':
        kwargs = {'method':'centroid'}
        assert not set(m).intersection({'keypoint', 'orb', 'akaze', 'sift', 'surf'}), \
            'for args: first keypoint method, then centroid'
    elif method=='keypoint':
        kwargs = {'method':'keypoint'}
    elif method=='orb':
        kwargs = {'method':'keypoint', 'feat':0}
    elif method=='akaze':
        kwargs = {'method':'keypoint', 'feat':1}
    elif method=='sift':
        kwargs = {'method':'keypoint', 'feat':2}
    elif method=='surf':
        kwargs = {'method':'keypoint', 'feat':3}
    elif method == 'absnet':
        kwargs = {'method': 'absnet'}
    elif method=='vo':
        # sg = OrbitAroundPoint(point=(0, 0, 0.18), vel=, sm_axis=0.02, incl=, asc_node=, corotating=True, target=(0, 0, 0))
        kwargs = {}
    else:
        assert False, 'Invalid method "%s"'%method

    if kwargs['method'] == 'keypoint' and 'centroid' in m:
        kwargs['method'] = 'keypoint+'
        kwargs['centroid_fallback'] = True

    kwargs0 = {'min_options':{}}
    kwargs0.update(kwargs)
    kwargs = kwargs0
    
    # shape model noise
    smn_cache_id = ''
    sm_noise = 0
    if 'smn_' in m:
        smn_cache_id = 'hi'
        sm_noise = SHAPE_MODEL_NOISE_LV[smn_cache_id]
    elif 'smn' in m:
        smn_cache_id = 'lo'
        sm_noise = SHAPE_MODEL_NOISE_LV[smn_cache_id]

    # feature db
    hi_res_shape_model = False
    if 'fdb' in m:
        kwargs['use_feature_db'] = True
        hi_res_shape_model = True
        if smn_cache_id:
            kwargs['add_noise'] = smn_cache_id

    sm = get_system_model(mission, hi_res_shape_model=hi_res_shape_model)

    if method == 'centroid':
        # target always fits fov: [min_med_distance, max_distance]
        sm.min_distance = sm.min_med_distance
        file_prefix_mod = 'far_'
    elif method == 'keypoint+':
        # full distance range: [min_distance, max_distance]
        file_prefix_mod = 'ful_'
    else:
        # default is [min_distance, max_med_distnace]
        sm.max_distance = sm.max_med_distance
        file_prefix_mod = ''

    if 'fdb' in m:
        #sm.view_width = sm.cam.width
        sm.view_width = VIEW_WIDTH

    if 'real' in m:
        kwargs['state_db_path'] = sm.asteroid.image_db_path
        #kwargs['scale_cam_img'] = True
        #kwargs['rotation_noise'] = False

    if 'real_' in m:
        # TODO: continue here!
        #   - load real lbl files, render and use synthetic images based on them
        kwargs['state_db_path'] = sm.asteroid.image_db_path
        kwargs['resynth_cam_image'] = True

    # noise settings
    ini_kwargs = {
        # gaussian sd in seconds
        'noise_time': 0,  # disabled as _noise_ast_phase_shift does same thing, was 95% within +-30s,

        # uniform, max dev in deg
        'noise_ast_rot_axis': 10,  # 0 - 10 deg uniform
        'noise_ast_phase_shift': 10 / 2,  # 95% within 10 deg,

        # s/c orientation noise, gaussian sd in deg
        'noise_sco_lat': 2 / 2,  # 95% within 2 deg
        'noise_sco_lon': 2 / 2,  # 95% within 2 deg
        'noise_sco_rot': 2 / 2,  # 95% within 2 deg,

        # s/c position noise, gaussian sd in km per km of distance
        'noise_lateral': 0.3,  # 0.298 calculated using centroid algo AND 5 deg fov
        'noise_altitude': 0.10,  # 0.131 when calculated using centroid algo AND 5 deg fov
    }

    if 'didy' in mission:
        ini_kwargs.update({
            # operation zone only (i.e. less noise),
            # uniform, max dev in deg
            'noise_ast_rot_axis': 5,  # 0 - 5 deg uniform
            'noise_ast_phase_shift': 5 / 2,  # 95% within 5 deg,

            # s/c orientation noise, gaussian sd in deg
            'noise_sco_lat': 1 / 2,  # 95% within 1 deg
            'noise_sco_lon': 1 / 2,  # 95% within 1 deg
            'noise_sco_rot': 1 / 2,  # 95% within 1 deg
        })

    tl = TestLoop(sm, file_prefix_mod=file_prefix_mod,
                  est_real_ast_orient=est_real_ast_orient, operation_zone_only=('didy' in mission),
                  sm_noise=sm_noise, sm_noise_len_sc=SHAPE_MODEL_NOISE_LEN_SC,
                  state_generator=sg, **ini_kwargs)

    if sm.mission_id == 'rose':
        tl.enable_initial_location = False
        ka = tl.keypoint.__class__
        ka.FEATURE_FILTERING_RELATIVE_GRID_SIZE = 0.01
        ka.FEATURE_FILTERING_FALLBACK_GRID_SIZE = 0
        ka.FEATURE_FILTERING_SCHEME = ka.FFS_NONE  # ka.FFS_SIMPLE_GRID
        ka.MAX_FEATURES = 2000
        if kwargs.get('feat', -1) == ka.ORB:
            ka.DEF_RANSAC_ERROR = 10
            #ka.LOWE_METHOD_COEF = 0.825

    tl.run(count, log_prefix=mission+'-'+full_method+'-', smn_cache_id=smn_cache_id, **kwargs)


if __name__ == '__main__':
    mission = sys.argv[1]
    full_method = sys.argv[2]
    count = sys.argv[3] if len(sys.argv) > 3 else 10
    est_ast_rot = sys.argv[4] == 'ear' if len(sys.argv) > 4 else False

    run_batch(mission, full_method, count, est_real_ast_orient=est_ast_rot)
