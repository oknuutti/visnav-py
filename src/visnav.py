
# TODO:
#- nothing
#   "test framework for vision based localization algorithms targeting an asteroid"
#   + nothing to implement
#   - scarce results
#
#- better projection (rendering, fft?)
#   "mid-range vision based localization using phase correlation for a cubesat asteroid mission"
#	+ fast to implement
#	+ some science
#
#- landmark based (actual science)
#   "vision based localization for a cubesat asteroid mission"
#	+ all current methods use => good for comparision
#	+ science & cv
#	+ good accuracy
#	- slow to implement
#
# - better centroid (gp-regression?)
#   "cubesat-asteroid distance estimation from navcam image using gaussian process regression"
#	+ can use previous knowledge
#	+ fast to run
#	+ fast to implement
#	- bad accuracy
#	- no science
#
#- nn based (wheee?!)
#   "vision based localization using deep learning for a cubesat asteroid mission"
#	+ new, hyped stuff
#	+ science & cv
#	- slow to implement
#

import sys
import math
import threading

import numpy as np
import cv2

from PyQt5.QtGui import QColor, QSurfaceFormat, QOpenGLVersionProfile
from PyQt5.QtCore import (pyqtSignal, QPoint, QSize, Qt, QBuffer, QIODevice,
        QCoreApplication)
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout,
        QOpenGLWidget, QSlider, QPushButton, QWidget)


from settings import *
from model import SystemModel
from algorithm import ImageProc, PhaseCorrelationAlgo, CentroidAlgo
import obj_loader
import lbl_loader
import tools

class MainThread(threading.Thread):
    def __init__(self, counter):
        super(MainThread, self).__init__()
        self.threadID = counter
        self.name = 'main-thread-%s'%counter
        self.counter = counter
        self.ready = threading.Event()
        self.window = None
        
    def run(self):
        self.app = QApplication(sys.argv)
        self.window = Window()
        self.window.show()
        self.ready.set()
        self.app.exec_()
        
    def wait_until_ready(self):
        self.ready.wait()



class Window(QWidget):
    tsRun = pyqtSignal(tuple)
    
    def __init__(self):
        super(Window, self).__init__()
        
        self.systemModel = SystemModel()
        self.glWidget = GLWidget(self.systemModel, parent=self)
        self.closing = []
        
        # so that can run algorithms as a batch from different thread
        def tsRunHandler(f):
            fun, args, kwargs = f[0], f[1] or [], f[2] or {}
            self.tsRunResult = fun(*args, **kwargs)
        self.tsRun.connect(tsRunHandler)
        self.tsRunResult = None
        
        self.sliders = dict(
            (n, self.slider(p))
            for n, p in self.systemModel.get_params(all=True)
        )
        
        topLayout = QHBoxLayout()
        topLayout.addWidget(self.glWidget)
        topLayout.addWidget(self.sliders['x_off'])
        topLayout.addWidget(self.sliders['y_off'])
        topLayout.addWidget(self.sliders['z_off'])
        topLayout.addWidget(self.sliders['x_rot'])
        topLayout.addWidget(self.sliders['y_rot'])
        topLayout.addWidget(self.sliders['z_rot'])
        topLayout.addWidget(self.sliders['ast_x_rot'])
        topLayout.addWidget(self.sliders['ast_y_rot'])
        topLayout.addWidget(self.sliders['ast_z_rot'])
        topLayout.addWidget(self.sliders['time'])

        bottomLayout = QHBoxLayout()
        self.optim = PhaseCorrelationAlgo(self.systemModel, self.glWidget)
        
        self.buttons = dict(
            (m.lower(), self.optbutton(m, bottomLayout))
            for m in ('Simplex', 'Powell', 'COBYLA', 'CG',
                      'BFGS', 'Anneal', 'Brute'))
        
        self.infobtn = QPushButton('Info', self)
        self.infobtn.clicked.connect(lambda: self.printInfo())
        bottomLayout.addWidget(self.infobtn)
        
        mainLayout = QVBoxLayout()
        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)
        self.setWindowTitle("Hello 67P/C-G")

    def slider(self, param):
        if param.is_gl_z:
            slider = QSliderF(Qt.Vertical, reverse=True, inverse=True)
        else:
            slider = QSliderF(Qt.Vertical)
        slider.setRange(*param.range)
        slider.setTickPosition(QSlider.TicksRight)
        slider.setValue(param.value)
        
        def setter(val):
            value = slider.getValue()
            if value != param.value:
                param.value = value
                self.glWidget.update()
        slider.valueChanged.connect(setter)
        
        def change_callback(val, vmin=None, vmax=None):
            if vmin is not None:
                slider.setRange(vmin, vmax)
            slider.setValue(val)
        param.change_callback = change_callback
        
        return slider
    
    def optbutton(self, m, layout):
        btn = QPushButton(m, self)

        def profhandler():
            import cProfile
            ls = locals()
            ls.update({'self':self, 'm':m})
            cProfile.runctx('self.optim.findstate(method=m.lower())', 
                    globals(), ls, PROFILE_OUT_FILE)
        def handler():
            self.optim.findstate(self.glWidget.image, method=m.lower())
        
        btn.clicked.connect(profhandler if PROFILE else handler)
        layout.addWidget(btn)
        return btn
    
    def closeEvent(self, evnt):
        for f in self.closing:
            f()
        super(Window, self).closeEvent(evnt) 
        
    def printInfo(self):
        print('\n%r\n' % self.systemModel)
        

class GLWidget(QOpenGLWidget):
    def __init__(self, systemModel, parent=None):
        super(GLWidget, self).__init__(parent)
        
        self.systemModel = systemModel
        self.min_method = None
        self.min_options = None
        self.errval0 = None
        self.errval1 = None
        self.errval  = None
        self.iter_count = 0
        self.image = None

        self.debug_c = 0
        self._paint_entered = False

        self._render = True
        self._algo_render = False
        
        self._rawimage = None
        self._object = 0
        self._lastPos = QPoint()
        self._imgColor = QColor.fromRgbF(1, 1, 1, 0.4)
        self._fgColor = QColor.fromRgbF(0.6, 0.6, 0.6, 1)
        self._bgColor = QColor.fromCmykF(0.0, 0.0, 0.0, 1.0)
        self._persp = {
            'fov': CAMERA_FIELD_OF_VIEW,
            'aspect': CAMERA_ASPECT_RATIO,
            'near': 0.1,
            'far': MAX_DISTANCE,
        }
        # calculate frustum based on fov, aspect & near
        top = self._persp['near'] * math.tan(math.radians(self._persp['fov']/2))
        right = top*self._persp['aspect']
        self._frustum = {
            'left':-right,
            'right':right,
            'bottom':-top,
            'top':top, 
        }
        
    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(CAMERA_WIDTH, CAMERA_HEIGHT)

    def initializeGL(self):
        f = QSurfaceFormat()
        p = QOpenGLVersionProfile(f)
        self.gl = self.context().versionFunctions(p)
        self.gl.initializeOpenGLFunctions()

        self.setClearColor(self._bgColor)
        self._object = self.loadObject()

        self.gl.glEnable(self.gl.GL_CULL_FACE)

        # for transparent asteroid image on top of model
        self.gl.glBlendFunc(
                self.gl.GL_SRC_ALPHA, self.gl.GL_ONE_MINUS_SRC_ALPHA)

        if self._render:
            self._rendOpts()
        else:
            self._projOpts()
        
    def _projOpts(self):
        # reset from potentially set rendering options
        self.gl.glDisable(self.gl.GL_LIGHTING)
        self.gl.glDisable(self.gl.GL_DEPTH_TEST);
        self.gl.glShadeModel(self.gl.GL_FLAT)
    
    def _rendOpts(self):
        # rendering options
        self.gl.glEnable(self.gl.GL_LIGHTING)
        self.gl.glEnable(self.gl.GL_DEPTH_TEST)
        self.gl.glShadeModel(self.gl.GL_SMOOTH) # TODO: try with flat
        self.gl.glEnable(self.gl.GL_LIGHT0)
        
    def resizeGL(self, width, height):
        side = min(width, height)
        if side < 0:
            return

        if not BATCH_MODE:
            self.loadTargetImage(TARGET_IMAGE_FILE, side)
            self.loadTargetImageMeta(TARGET_IMAGE_META_FILE)
            if not USE_IMG_LABEL_FOR_SC_POS:
                CentroidAlgo.update_sc_pos(self.systemModel, self.thr_image)
        
        self.gl.glViewport((width-side) // 2, (height-side) // 2, side, side)

        self.gl.glMatrixMode(self.gl.GL_PROJECTION)
        self.gl.glLoadIdentity()
        self.gl.glFrustum(
                self._frustum['left'], self._frustum['right'],
                self._frustum['bottom'], self._frustum['top'],
                self._persp['near'], self._persp['far'])
        self.gl.glMatrixMode(self.gl.GL_MODELVIEW)

    def paintGL(self):
        if self._algo_render:
            self.gl.glDisable(self.gl.GL_BLEND)
        else:
            self.gl.glEnable(self.gl.GL_BLEND)
        
        self.gl.glClear(
                self.gl.GL_COLOR_BUFFER_BIT | self.gl.GL_DEPTH_BUFFER_BIT)

        self.gl.glLoadIdentity()
        m = self.systemModel
        
        # sunlight config
        if self._render or self._algo_render:
            light = m.light_rel_dir()
            self.gl.glLightfv(self.gl.GL_LIGHT0, self.gl.GL_POSITION, tuple(-light)+(0,))
            self.gl.glLightfv(self.gl.GL_LIGHT0, self.gl.GL_SPOT_DIRECTION, tuple(light))
        
        self.gl.glTranslated(
                0 if self._algo_render else m.x_off.value,
                0 if self._algo_render else m.y_off.value,
                m.z_off.value)
        self.gl.glRotated(*m.sc_asteroid_rel_rot())
        self.gl.glCallList(self._object)
        
        if not self._algo_render and self._rawimage is not None:
            self.gl.glLoadIdentity()
            self.setColor(self._imgColor)
            self.gl.glRasterPos3f(self._frustum['left'],
                                  self._frustum['bottom'], -self._persp['near'])
            self.gl.glDrawPixels(self._image_w, self._image_h, self.gl.GL_RGBA,
                                 self.gl.GL_UNSIGNED_BYTE, self._rawimage)

        self.gl.glFinish()
        
    def saveView(self):
        fbo = self.grabFramebuffer() # calls paintGL
        buffer = QBuffer()
        buffer.open(QIODevice.ReadWrite)
        fbo.save(buffer, "PNG", quality=100)
        
        # TODO: find a more efficient way than
        # bytes => buffer => PNG => nparray => cvimage
        # - maybe here's some ideas:
        #       https://vec.io/posts/faster-alternatives-to-glreadpixels\
        #                                       -and-glteximage2d-in-opengl-es
        
        imdata = np.frombuffer(buffer.data(), dtype='int8')
        view = cv2.imdecode(imdata, cv2.IMREAD_GRAYSCALE)
        return view
        
    def saveViewTest(self):
        if False:
            # this takes around double the time than current way
            fbo = self.grabFramebuffer()
            arr = fbo.constBits().asarray(512*512*4)
            view = [0] * 256
            for i in range(1,512*512*4,4):
                view[arr[i]] += 1        
        elif False:
            # glReadPixels doesnt exist
            arr = self.gl.glReadPixels(0, 0, 512, 512,
                    self.gl.GL_RGB, self.gl.GL_UNSIGNED_BYTE)
            view = [0] * 256
            for i in range(1,512*512*3,3):
                view[arr[i]] += 1
        
        return view
        
    def saveViewToFile(self, imgfile):
        cv2.imwrite(imgfile, self.saveView())
 
    def render(self):
        if not self._render:
            self._rendOpts()
        self._algo_render = True
        rr = self.saveView()
        self._algo_render = False
        if not self._render:
            self._projOpts()
        return rr
    
    def mousePressEvent(self, event):
        self._lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self._lastPos.x()
        dy = event.y() - self._lastPos.y()

        if event.buttons() & (Qt.LeftButton | Qt.RightButton):
            self.systemModel.x_rot.value = \
                    (self.systemModel.x_rot.value + dy/2 + 90) % 180 - 90
            
            param = 'y_rot' if event.buttons() & Qt.LeftButton else 'z_rot'
            getattr(self.systemModel, param).value = \
                    (getattr(self.systemModel, param).value + dx/2) % 360
            
            self.update()

        self._lastPos = event.pos()

    def loadTargetImage(self, src, side):
        tmp = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if tmp is None:
            raise Exception('Cant load image from file %s'%(src,))
        
        self._orig_image_w = len(tmp[0])
        self._orig_image_h = len(tmp)
        
        self.image = cv2.resize(tmp, None,
                                 fx=side/self._orig_image_w,
                                 fy=side/self._orig_image_h,
                                 interpolation=cv2.INTER_CUBIC)
        self._image_w = len(self.image[0])
        self._image_h = len(self.image)
        self.thr_image, h, th = ImageProc.process_target_image(self.image)
        self._rawimage = np.flipud(self.thr_image).tobytes()

    def loadTargetImageMeta(self, src):
        lbl_loader.load_image_meta(src, self.systemModel)
        self.update()

    def loadObject(self):
        genList = self.gl.glGenLists(1)
        self.gl.glNewList(genList, self.gl.GL_COMPILE)
        self.gl.glBegin(self.gl.GL_TRIANGLES) # GL_POLYGON?
        self.setColor(self._fgColor)
        #self.gl.glEnable(self.gl.GL_COLOR_MATERIAL);
        #self.gl.glMaterialfv(self.gl.GL_FRONT, self.gl.GL_SPECULAR, (0,0,0,1));
        #self.gl.glMaterialfv(self.gl.GL_FRONT, self.gl.GL_SHININESS, (0,));
        
        model = obj_loader.OBJ(TARGET_MODEL_FILE)
        print('x[%s, %s] y[%s, %s] z[%s, %s]'%(
            min(v[0] for v in model.vertices),
            max(v[0] for v in model.vertices),
            min(v[1] for v in model.vertices),
            max(v[1] for v in model.vertices),
            min(v[2] for v in model.vertices),
            max(v[2] for v in model.vertices),
        ))

        for triangle in model.triangles:
            self.triangle(triangle[0], triangle[1], triangle[2])
        
        self.gl.glEnd()
        self.gl.glEndList()

        return genList

    def triangle(self, x1, x2, x3):
        self.gl.glNormal3f(*tools.surf_normal(x1, x2, x3))
        self.gl.glVertex3f(x1[0], x1[1], x1[2])
        self.gl.glVertex3f(x2[0], x2[1], x2[2])
        self.gl.glVertex3f(x3[0], x3[1], x3[2])

    def setClearColor(self, c):
        self.gl.glClearColor(c.redF(), c.greenF(), c.blueF(), c.alphaF())

    def setColor(self, c):
        self.gl.glColor4f(c.redF(), c.greenF(), c.blueF(), c.alphaF())


class QSliderF(QSlider):
    def __init__(self, *args, **kwargs):
        self.inverse = kwargs.pop('inverse', False)
        self.reverse = kwargs.pop('reverse', False)
        super(QSliderF, self).__init__(*args, **kwargs)
    
    def getValue(self, *args, **kwargs):
        val = self.value()/self.scale
        val = 1/val if self.inverse else val
        val = -val if self.reverse else val
        return val

    def setValue(self, val, *args, **kwargs):
        val = -val if self.reverse else val
        val = 1/val if self.inverse else val
        val = val * self.scale
        if self.value()!= val:
            super(QSliderF, self).setValue(val, *args, **kwargs)
    
    def setRange(self, minv, maxv, *args, **kwargs):
        minv, maxv = (-maxv, -minv) if self.reverse else (minv, maxv)
        minv, maxv = (1/minv, 1/maxv) if self.inverse else (minv, maxv)
        self.scale = 5000 / (maxv - minv)
        super(QSliderF, self).setRange(minv*self.scale, maxv*self.scale,
                                       *args, **kwargs)
        self.setSingleStep(5000/100)
        self.setPageStep(5000/20)
        self.setTickInterval(5000/20)


if __name__ == '__main__':
    if START_IN_THREAD:
        th1 = MainThread(1)
        th1.start()
    else:
        app = QApplication(sys.argv)
        window = Window()
        window.show()
        app.exec_()

# ideas for 3d model rendering if have to do manually:
# - discard quads with wrong rotation, exit if quad outside image plane limits
# - select quad points with nearly zero area (or vertices nearly on same line) => limb quads
#    * plot a line between farthest vertices of these quads
