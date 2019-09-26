import re
import sys
sys.tracebacklimit = 10

from scipy import optimize

from settings import *
from missions.didymos import DidymosSystemModel
from missions.rosetta import RosettaSystemModel
from testloop import TestLoop


def get_system_model(mission, hi_res_shape_model=False):
    rose = re.match(r'^rose(\d{3})?$', mission)

    if rose:
        mission = 'rose'
        batch = rose[1] if rose[1] else '006'
        sm = RosettaSystemModel(hi_res_shape_model=hi_res_shape_model, rosetta_batch='mtp'+batch)
    elif mission == 'didy1n':
        sm = DidymosSystemModel(target_primary=True, hi_res_shape_model=hi_res_shape_model)
    elif mission == 'didy1w':
        sm = DidymosSystemModel(target_primary=True, hi_res_shape_model=hi_res_shape_model, use_narrow_cam=False)
    elif mission == 'didy2n':
        sm = DidymosSystemModel(target_primary=False, hi_res_shape_model=hi_res_shape_model)
    elif mission == 'didy2w':
        sm = DidymosSystemModel(target_primary=False, hi_res_shape_model=hi_res_shape_model, use_narrow_cam=False)
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
    if 'smn_' in m:
        kwargs['smn_type'] = 'hi'
    elif 'smn' in m:
        kwargs['smn_type'] = 'lo'

    # feature db
    hi_res_shape_model = False
    if 'fdb' in m:
        kwargs['use_feature_db'] = True
        hi_res_shape_model = True
        noise = kwargs.pop('smn_type', False)
        if noise:
            kwargs['add_noise'] = noise

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

    tl = TestLoop(sm, file_prefix_mod=file_prefix_mod,
                  est_real_ast_orient=est_real_ast_orient, operation_zone_only=('didy' in mission),
                  state_generator=sg)

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

    tl.run(count, log_prefix=mission+'-'+full_method+'-', **kwargs)


if __name__ == '__main__':
    mission = sys.argv[1]
    full_method = sys.argv[2]
    count = sys.argv[3] if len(sys.argv) > 3 else 10
    est_ast_rot = sys.argv[4] == 'ear' if len(sys.argv) > 4 else False

    run_batch(mission, full_method, count, est_real_ast_orient=est_ast_rot)
