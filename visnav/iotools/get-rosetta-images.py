import os
import re
import time
import re

import requests
import urllib.request

import sys
from bs4 import BeautifulSoup

def get_file(url, path):
    max_retries = 6
    sleep_time = 3
    ok = False
    last_err = None
    for i in range(max_retries):
        try:
            urllib.request.urlretrieve(url, path)
            ok = True
            break
        except urllib.error.ContentTooShortError as e:
            last_err = e
            time.sleep(sleep_time)
    
    if not ok:
        raise Exception('Error: %s'%last_err)


if __name__ == '__main__':
    try:
        m = re.match(r'mtp\d{3}', sys.argv[1])
        batch = m[0]
        skip = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    except:
        print('USAGE: python %s <mtpxxx> [skip-count]' % sys.argv[0])

    base = 'https://imagearchives.esac.esa.int'
    script_dir = os.path.dirname(__file__)
    save_dir = os.path.join(script_dir, '../../data/rosetta-'+batch)

    # cid (class id), pid_s (starting img id), pn (image count)
    ids = {
        ## PRELANDING
        'mtp003': (30, 2141, 146),    # done
        'mtp006': (63, 6971, 770),    # done
        'mtp007': (62, 7561, 527),    #

        ## COMET ESCORT 2
        'mtp015': (119, 28703, 438),
        'mtp016': (140, 31996, 277),
        'mtp017': (139, 32273, 404),  # done

        ## COMET ESCORT 3
        'mtp018': (167, 38173, 320),

        ## COMET ESCORT 4
        'mtp023': (275, 66085, 397),
        'mtp024': (236, 55132, 452),  #

        ## ROSETTA EXTENSION 1
        'mtp025': (237, 55584, 532),  # done
        'mtp026': (238, 56116, 819),  #
    }
    cid, pid_s, pn = ids[batch]

    skip = 0
    pid_s += skip
    pn -= skip

    print('000', end='', flush=True)
    for pid in range(pid_s, pid_s+pn):
        page = requests.get(base+'/picture.php?/'+str(pid)+'/category/'+str(cid), verify=False)
        soup = BeautifulSoup(page.content, 'html.parser')
        imgele = soup.find(id="theMainImage")
        imgurl = base+imgele['visnav'][7:-7]+'.png'
        imgname = imgele['alt'].replace('F._P.', '_P.')
        get_file(imgurl, os.path.join(save_dir, imgname))

        lblurl = soup.find(class_="download_link", href=re.compile("LBL$"))['href']
        get_file(lblurl, os.path.join(save_dir, lblurl.split('/')[-1]))
        
        print('\b\b\b%03d'%(pid-pid_s+1), end='', flush=True)
