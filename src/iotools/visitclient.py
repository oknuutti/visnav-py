
import os
import time
import socket
import json
import threading
from subprocess import call
#from multiprocessing import cpu_count

class VisitClient():
    DEFAULT_VISIT_PORT = 8787
    
    def __init__(self, port=DEFAULT_VISIT_PORT):
        self._port = port
        self._sock = None
        self._visit_th = None
        
    def render(self, params):
        for i in range(3):
            self._send(json.dumps(params))
            imgfile = self._receive()
            if len(imgfile)>0:
                break
        
        if len(imgfile)==0:
            raise RuntimeError("Cant connect to VISIT")
        
        return imgfile
        
        
    def _send(self, msg):
        bmsg = msg.encode('utf-8')
        totalsent = 0
        while totalsent < len(bmsg):
            sent = self._sock.send(bmsg[totalsent:]) \
                    if self._sock is not None else 0
            totalsent = totalsent + sent
            if sent == 0:
                self._connect()
                totalsent = 0
                
        self._sock.shutdown(1)
        
    def _receive(self):
        imgfile = self._sock.recv(256)
        self._sock.close()
        self._sock = None
        return imgfile.decode('utf-8')    

    def _connect(self):
        if self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            self._sock.connect(('127.0.0.1', self._port))
        except ConnectionRefusedError:
            self._visit_th = VisitThread()
            self._visit_th.start()
            
            for i in range(12):
                time.sleep(5)
                try:
                    self._sock.connect(('127.0.0.1', self._port))
                    ok = True
                except ConnectionRefusedError:
                    ok = False
                if ok:
                    break
                
            if not ok:
                raise RuntimeError("Cant connect to VISIT")
    

class VisitThread(threading.Thread):
    VISIT_SCRIPT_PY_FILE = os.path.join(os.path.dirname(__file__), 'visit-py-script.py')
    
    def __init__(self):
        super(VisitThread, self).__init__()
        self.threadID = 2
        self.name = 'visit-thread'
        self.counter = 2
        self.window = None
        
    def run(self):
        # -nowin crashes VISIT
        call(['visit', '-cli', '-l', 'srun', '-np', '1',
                '-s', VisitThread.VISIT_SCRIPT_PY_FILE])
                
#        call(['visit', '-cli', '-l', 'srun', '-np', '%d'%int(cpu_count()/2),
#               '-s', VisitThread.VISIT_SCRIPT_PY_FILE]) # multiple threads seemed slower