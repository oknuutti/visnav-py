import socket
import json
#from multiprocessing import cpu_count

def _render(params):
    DisableRedraw()

    lit = GetLight(0)
    lit.type = lit.Object
    lit.direction = tuple(params['light_direction'])
    SetLight(0,lit)

    v = GetView3D()
    v.viewNormal = tuple(params['view_normal'])
    v.focus = tuple(params['focus'])
    v.viewUp = tuple(params['up_vector'])
    v.viewAngle = params['view_angle']
    v.parallelScale = 1
    v.nearPlane = -params['max_distance']
    v.farPlane = params['max_distance']
    SetView3D(v)

    s = SaveWindowAttributes()
    s.fileName = params['out_file']
    s.outputDirectory = params['out_dir']
    s.format = s.PNG
    s.resConstraint = s.NoConstraint
    s.width = params['out_width']
    s.height = params['out_height']
    s.quality = 100
    s.compression = s.None
    SetSaveWindowAttributes(s)
    
    RedrawWindow()
    return SaveWindow()

def _receive(sock):
    chunks = []
    while True:
        chunk = sock.recv(2048)
        chunks.append(chunk)
        if chunk == b'':
            break
    
    return (b''.join(chunks)).decode('utf-8')
    
RestoreSession("data/default-visit.session",0)
ssock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ssock.bind(('127.0.0.1', 8787))
ssock.listen(1)

def _reset():
    print('Restarting computing engine to avoid memory leak')
    CloseComputeEngine("localhost", "")
    OpenComputeEngine("localhost", ("-l", "srun", "-np", "1"))
    RestoreSession("data/default-visit.session",0)

since_reset = 0

# main loop
while True:
    (csock, addr) = ssock.accept()
    msg = _receive(csock)
    if msg == 'quit':
        csock.close()
        break
    if len(msg)>0:
        params = json.loads(msg)
        for i in range(3):
            try:
                fname = _render(params)
                ok = True
            except Exception as e:
                print('Trying to open compute engine again because of: %s'%e)
                _reset()
                ok = False
            if ok:
                break
        
        csock.send(fname.encode('utf-8'))
    csock.close()
    since_reset += 1
    if since_reset >= 200:
        _reset()
        since_reset = 0

ssock.close()
quit()