
import os
import re

cache = os.path.join(os.path.dirname(__file__), '../cache/')
for fn in os.listdir(cache):
    m = re.match('^(.*?)(hi|lo)(\d{4})\.nsm$', fn)
    nfn = None
    if m:
        nfn = m[1]+m[3]+'_'+m[2]+'.nsm'
    elif re.match('.*?\d\.nsm$', fn):
        nfn = fn[:-4]+'_lo.nsm'

    if nfn is not None:
        os.rename(os.path.join(cache, fn), os.path.join(cache, nfn))
