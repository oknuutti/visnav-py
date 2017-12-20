import os
import re
import time
import requests
import urllib.request
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
    base = 'https://imagearchives.esac.esa.int'
    script_dir = os.path.dirname(__file__)
    save_dir = os.path.join(script_dir, '../../data/rosetta-mtp025')
    ## PRELANDING MTP006: cid=63, pid_s=6971, pn=770
    # COMET ESCORT 2 MTP015: cid=119, pid_s=28703, pn=438
    # COMET ESCORT 2 MTP016: cid=140, pid_s=31996, pn=277
    ## COMET ESCORT 2 MTP017: cid=139, pid_s=32273, pn=404
    # COMET ESCORT 4 MTP023: cid=275, pid_s=66085, pn=397
    ## ROSETTA EXTENSION 1 MTP025: cid=237, pid_s=55584, pn=532
    
    cid = 237
    skip = 198
    pid_s = 55584+skip
    pn = 532-skip

    print('000', end='', flush=True)
    for pid in range(pid_s, pid_s+pn):
        page = requests.get(base+'/picture.php?/'+str(pid)+'/category/'+str(cid))
        soup = BeautifulSoup(page.content, 'html.parser')
        imgele = soup.find(id="theMainImage")
        imgurl = base+imgele['src'][7:-7]+'.png'
        imgname = imgele['alt']
        get_file(imgurl, os.path.join(save_dir, imgname))

        lblurl = soup.find(class_="download_link", href=re.compile("LBL$"))['href']
        get_file(lblurl, os.path.join(save_dir, lblurl.split('/')[-1]))
        
        print('\b\b\b%03d'%(pid-pid_s+1), end='', flush=True)
