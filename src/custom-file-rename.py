
import os
import re

from settings import CACHE_DIR

if False:
    for fn in os.listdir(CACHE_DIR):
        m = re.match('^(.*?)(hi|lo)(\d{4})\.nsm$', fn)
        nfn = None
        if m:
            nfn = m[1]+m[3]+'_'+m[2]+'.nsm'
        elif re.match('.*?\d\.nsm$', fn):
            nfn = fn[:-4]+'_lo.nsm'

        if nfn is not None:
            os.rename(os.path.join(CACHE_DIR, fn), os.path.join(CACHE_DIR, nfn))
else:
    # ^iteration_ => rose_
    # ^far => rose_far_
    # ^shapemodel_ => rose_
    for fn in os.listdir(CACHE_DIR):
        m = re.match('^(iteration_|shapemodel_|far)(.*?)$', fn)
        nfn = None
        if m:
            nfn = 'rose_' + ('far_' if m[1]=='far' else '') + m[2]
        if nfn is not None:
            #print('%s => %s'%(fn, nfn))
            os.rename(os.path.join(CACHE_DIR, fn), os.path.join(CACHE_DIR, nfn))

