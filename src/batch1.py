import settings
settings.BATCH_MODE = True
from settings import *

import sys
sys.tracebacklimit = 10

from scipy import optimize

from visnav import MainThread
from testloop import TestLoop

## profile by e.g. "D:\Program Files\Anaconda3\python" -m cProfile -o profile.out src\batch1.py keypoint 10
## then snakeviz profile.out

if __name__ == '__main__':
    th1 = MainThread(1)
    th1.start()
    th1.wait_until_ready()

    tl = TestLoop(th1.window)
    
    #tl.run(1000)
    
    method = sys.argv[1] if len(sys.argv)>1 else 'keypoint'
    count = int(sys.argv[2]) if len(sys.argv)>2 else 100
    
    if False:
        tl.run(10, cleanup=False, method='phasecorr', min_options={
            'method':'anneal',
            'niter':50, 'T':0.15, 'stepsize':0.03,
            'minimizer_kwargs':{
                'method':'COBYLA',
                'options':{'maxiter':6, 'rhobeg': 0.02},
        }})
    elif False:
        tl.run(10, cleanup=False, method='phasecorr', min_options={
            'method':'anneal',
            'niter':20, 'T':0.15, 'stepsize':0.05,
            'minimizer_kwargs':{
                'method':'Nelder-Mead',
                'options':{'maxiter':8, 'xtol':1e-3, 'ftol':1e-3}
        }})
    elif False:
        def finfun(f, x0, *args, **kwargs):
            res = optimize.minimize(f, x0,
#                    method='Nelder-Mead',
#                    options={'maxiter':20, 'xtol':1e-3, 'ftol':1e-3},
               method='COBYLA',
               options={'maxiter':10, 'rhobeg': 0.01},
            )
            return res.x, res.fun, res.status

        tl.run(1000, cleanup=False, method='phasecorr', centroid_init=True,
                min_options={
                    'method':'brute',
                    'max_iter':50,
                    'finish':None, #finfun,
                })
    elif method=='phasecorr':
        tl.run(count, cleanup=False, method='phasecorr', centroid_init=False,
                min_options={
                    'method':'two-step-brute',
                    'first':{
                        'max_iter':60,
                    },
                    'second':{
                        'margin':20,            # in original image pixels 
                        'distance_margin':0.05, # distance search space centered around first round result
                        'max_iter':20,
                    },
                })
    elif method=='keypoint+':
        tl.run(count, cleanup=False, method='keypoint+', min_options={})
    elif method=='keypoint':
        tl.run(count, cleanup=False, method='keypoint', min_options={})
    elif method=='centroid':
        tl.run(count, cleanup=False, method='centroid', min_options={})
    else:
        assert False, 'Invalid method "%s"'%method

    th1.app.quit()