import settings
settings.BATCH_MODE = True
from settings import *

from visnav import MainThread
from testloop import TestLoop

if __name__ == '__main__':
    th1 = MainThread(1)
    th1.start()
    th1.wait_until_ready()

    tl = TestLoop(th1.window)
    
    tl.run(1000)
    
    if False:
        tl.run(1000, cleanup=False, method='anneal', min_options={
            'niter':30, 'T':0.01, 'stepsize':0.05,
            'minimizer_kwargs':{
    #           'method':'Nelder-Mead',
    #           'options':{'maxiter':20, 'xtol':2e-2, 'ftol':1e-1}
                'method':'COBYLA',
                'options':{'maxiter':20, 'rhobeg': 0.1},
        }})
    
    th1.app.quit()