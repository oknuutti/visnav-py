import math

import settings
settings.BATCH_MODE = True

import sys
import os
sys.tracebacklimit = 10

from scipy import optimize


if __name__ == '__main__':
    full_method = sys.argv[1] if len(sys.argv)>1 else 'keypoint'
    count = sys.argv[2] if len(sys.argv)>2 else 10
    
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

    kwargs0 = {'min_options':{}}
    kwargs0.update(kwargs)
    kwargs = kwargs0
    
    # shape model noise
    settings.ADD_SHAPE_MODEL_NOISE = False
    if 'smn_' in m:
        settings.ADD_SHAPE_MODEL_NOISE = True
        settings.SHAPE_MODEL_NOISE_LV = 0.04
        kwargs['smn_type'] = 'hi'
    elif 'smn' in m:
        settings.ADD_SHAPE_MODEL_NOISE = True
        settings.SHAPE_MODEL_NOISE_LV = 0.01
        kwargs['smn_type'] = 'lo'

    # feature db
    if 'fdb' in m:
        kwargs['use_feature_db'] = True
        settings.TARGET_MODEL_FILE = os.path.join(settings.SCRIPT_DIR, '../data/CSHP_DV_130_01_LORES_00200.obj') # _XLRES_, _LORES_        
        settings.VIEW_WIDTH = 512
        settings.VIEW_HEIGHT = 512
        noise = kwargs.pop('smn_type',False)
        settings.ADD_SHAPE_MODEL_NOISE = False
        if noise:
            kwargs['add_noise'] = True
    
    if 'real' in m:
        kwargs['state_db_path'] = settings.IMAGE_DB_PATH
        #kwargs['scale_cam_img'] = True
        #kwargs['rotation_noise'] = False
        

    from settings import *
    from testloop import TestLoop

    if 'fdb' in m:
        from algo.keypoint import KeypointAlgo
        from algo import tools
        lats, lons = tools.bf_lat_lon(KeypointAlgo.FDB_TOL)
        elongs = math.ceil((360 - 2 * TestLoop.MIN_ELONG) / math.degrees(KeypointAlgo.FDB_TOL) / 2) - 1
        max_feat_mem = KeypointAlgo.FDB_MAX_MEM * len(lats) * len(lons) * elongs
        print('Using feature DB not bigger than %.1fMB (%d x %d x %d x %.0fkB)'%(
                max_feat_mem/1024/1024,
                len(lats), len(lons), elongs,
                KeypointAlgo.FDB_MAX_MEM/1024))

    tl = TestLoop(far=(method == 'centroid'))
    tl.run(count, log_prefix=full_method+'-', **kwargs)