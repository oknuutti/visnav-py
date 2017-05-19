import settings
settings.BATCH_MODE = True
from settings import *

import sys
sys.tracebacklimit = 10

from scipy import optimize

from visnav import MainThread
from testloop import TestLoop

if __name__ == '__main__':
    th1 = MainThread(1)
    th1.start()
    th1.wait_until_ready()

    tl = TestLoop(th1.window)
    
    #tl.run(1000)
    
    if False:
        tl.run(10, cleanup=False, method='anneal', min_options={
            'niter':50, 'T':0.15, 'stepsize':0.03,
            'minimizer_kwargs':{
                'method':'COBYLA',
                'options':{'maxiter':6, 'rhobeg': 0.02},
        }})
    elif False:
        tl.run(10, cleanup=False, method='anneal', min_options={
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

        tl.run(1000, cleanup=False, method='brute', centroid_init=True,
                min_options={
                    'max_iter':50,
                    'finish':None, #finfun,
                })
    elif True:
        tl.run(10, cleanup=False, method='two-step-brute', centroid_init=True,
                min_options={
                    'first':{
                        'max_iter':50,
                    },
                    'second':{
                        'margin':50,           # in original image pixels 
                        'distance_margin':0.2, # distance search space centered around first round result
                        'max_iter':20,
                    },
                })

    th1.app.quit()