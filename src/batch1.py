import settings
settings.BATCH_MODE = True

import math
import sys
import os
sys.tracebacklimit = 10

from scipy import optimize


if __name__ == '__main__':
    method = sys.argv[1] if len(sys.argv)>1 else 'keypoint'
    count = int(sys.argv[2]) if len(sys.argv)>2 else 100
    
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
    elif method=='keypoint+':
        kwargs = {'method':'keypoint+'}
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
    elif method=='orb+fdb':
        kwargs = {'method':'keypoint', 'feat':0, 'use_feature_db':True}
    elif method=='akaze+fdb':
        kwargs = {'method':'keypoint', 'feat':1, 'use_feature_db':True}
    elif method=='sift+fdb':
        kwargs = {'method':'keypoint', 'feat':2, 'use_feature_db':True}
    elif method=='surf+fdb':
        kwargs = {'method':'keypoint', 'feat':3, 'use_feature_db':True}
    else:
        assert False, 'Invalid method "%s"'%method

    kwargs0 = {'cleanup':False, 'min_options':{}}
    kwargs0.update(kwargs)
    kwargs = kwargs0
    
    if kwargs.get('use_feature_db', False):
        settings.VIEW_WIDTH = 768
        settings.VIEW_HEIGHT = 768
        settings.TARGET_MODEL_FILE = os.path.join(settings.SCRIPT_DIR, '../data/CSHP_DV_130_01_LORES_00200.obj') # _XLRES_, _LORES_        
#        settings.VIEW_WIDTH = 512
#        settings.VIEW_HEIGHT = 512
    
    from settings import *
    from visnav import MainThread
    from testloop import TestLoop

    th1 = MainThread(1)
    th1.start()
    th1.wait_until_ready()

    tl = TestLoop(th1.window)
    tl.run(count, **kwargs)
    
    th1.app.quit()