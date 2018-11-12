import math

import settings
from missions.didymos import DidymosSystemModel
from missions.rosetta import RosettaSystemModel

settings.BATCH_MODE = True

import sys
import os
sys.tracebacklimit = 10

from scipy import optimize


if __name__ == '__main__':
    mission = sys.argv[1]
    full_method = sys.argv[2]
    count = sys.argv[3] if len(sys.argv)>3 else 10
    
    m = full_method.split('+')
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
        assert not set(m).union({'keypoint', 'orb', 'akaze', 'sift', 'surf'}), 'for args: first keypoint method, then centroid'
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

    if mission == 'rose':
        sm = RosettaSystemModel(hi_res_shape_model=hi_res_shape_model)
    elif mission == 'didy':
        sm = DidymosSystemModel(hi_res_shape_model=hi_res_shape_model)
    elif mission == 'didw':
        sm = DidymosSystemModel(hi_res_shape_model=hi_res_shape_model, use_narrow_cam=False)
    else:
        assert False, 'Unknown mission given as argument: %s' % mission
    assert mission == sm.mission_id, 'wrong system model mission id'

    if 'fdb' in m:
        #sm.view_width = sm.cam.width
        sm.view_width = 512

    if 'real' in m:
        kwargs['state_db_path'] = sm.asteroid.image_db_path
        #kwargs['scale_cam_img'] = True
        #kwargs['rotation_noise'] = False

    from settings import *
    from testloop import TestLoop

    tl = TestLoop(sm, far=(kwargs['method'] in ('centroid', 'keypoint+')))
    tl.run(count, log_prefix=mission+'-'+full_method+'-', **kwargs)