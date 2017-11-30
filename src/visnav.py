
import sys
import math
import threading

import numpy as np
import cv2

from OpenGL.GL.images import glReadPixels
from PyQt5.QtGui import QColor, QSurfaceFormat, QOpenGLVersionProfile
from PyQt5.QtCore import (pyqtSignal, QPoint, QSize, Qt, QBuffer, QIODevice,
        QCoreApplication)
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout,
        QOpenGLWidget, QSlider, QPushButton, QWidget)

from settings import *
from algo import tools
from algo.model import SystemModel
from algo.image import ImageProc
from algo.phasecorr import PhaseCorrelationAlgo
from algo.centroid import CentroidAlgo
from algo.keypoint import KeypointAlgo
from algo.mixed import MixedAlgo
from algo.tools import PositioningException

from iotools import objloader
from iotools import lblloader

class MainThread(threading.Thread):
    def __init__(self, counter):
        super(MainThread, self).__init__()
        self.threadID = counter
        self.name = 'main-thread-%s'%counter
        self.counter = counter
        self.ready = threading.Event()
        self.window = None
        
    def run(self):
        sys.tracebacklimit = 10
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
        self.tsRunResult = None, float('nan')
        
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
        self.phasecorr = PhaseCorrelationAlgo(self.systemModel, self.glWidget)
        self.keypoint = KeypointAlgo(self.systemModel, self.glWidget)
        self.centroid = CentroidAlgo(self.systemModel, self.glWidget,
                                     bg_threshold=self.glWidget.image_bg_threshold)
        self.mixed = MixedAlgo(self.centroid, self.keypoint)
        
        self.buttons = dict(
            (m.lower(), self.optbutton(m, bottomLayout))
            for m in (
                'Simplex',
#                'Powell',
                'COBYLA',
#                'CG',
#                'BFGS',
                'Anneal',
                'Brute',
            )
        )
        
        self.infobtn = QPushButton('Info', self)
        self.infobtn.clicked.connect(lambda: self.printInfo())
        bottomLayout.addWidget(self.infobtn)
        
#        self.zoombtn = QPushButton('+', self)
#        self.zoombtn.clicked.connect(lambda: self.glWidget.setImageZoomAndResolution(
#                im_xoff=300, im_yoff=180, 
#                im_width=512, im_height=512, im_scale=1))
#        bottomLayout.addWidget(self.zoombtn)
#        
#        self.defviewbtn = QPushButton('=', self)
#        self.defviewbtn.clicked.connect(lambda: self.glWidget.setImageZoomAndResolution(
#                im_xoff=0, im_yoff=0, 
#                im_width=1024, im_height=1024, im_scale=0.5))
#        bottomLayout.addWidget(self.defviewbtn)
        
        def testfun1():
            try:
                self.centroid.adjust_iteratively(self.glWidget.image_file)
            except PositioningException as e:
                print('algorithm failed: %s' % e)
        self.test1 = QPushButton('T1', self)
        self.test1.clicked.connect(testfun1)
        bottomLayout.addWidget(self.test1)
        
        def testfun2():
            #self.glWidget.saveViewToFile('testimg.png')
            try:
                init_z = self.systemModel.z_off.value
                self.keypoint.solve_pnp(self.glWidget.image_file, init_z=init_z)
            except PositioningException as e:
                print('algorithm failed: %s' % e)
        self.test2 = QPushButton('T2', self)
        self.test2.clicked.connect(testfun2)
        bottomLayout.addWidget(self.test2)
        
        mainLayout = QVBoxLayout()
        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)
        self.setWindowTitle("Hello 67P/C-G")
        self.adjustSize()

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
            #if not np.isclose(value, param.value):
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
            cProfile.runctx('self.phasecorr.findstate(TARGET_IMAGE_FILE, method=m.lower())', 
                    globals(), ls, PROFILE_OUT_FILE)
        def handler():
            self.phasecorr.findstate(TARGET_IMAGE_FILE, method=m.lower())
        
        btn.clicked.connect(profhandler if PROFILE else handler)
        layout.addWidget(btn)
        return btn
    
    def closeEvent(self, evnt):
        for f in self.closing:
            f()
        super(Window, self).closeEvent(evnt) 
        
    def printInfo(self):
        print('solar elong, dir: %s'%(tuple(map(math.degrees, self.systemModel.solar_elongation())),))
        print('')
        self.systemModel.save_state('no-matter', printout=True)
        print('')
        print('shift-err: %.1fm'%(self.systemModel.calc_shift_err()*1000))
        

class GLWidget(QOpenGLWidget):
    def __init__(self, systemModel, parent=None):
        super(GLWidget, self).__init__(parent)
        
        self.setFixedSize(VIEW_WIDTH, VIEW_HEIGHT)
        
        self.systemModel = systemModel
        self.min_method = None
        self.min_options = None
        self.errval0 = None
        self.errval1 = None
        self.errval  = None
        self.iter_count = 0
        self.image = None
        self.image_file = None
        
        self.full_image = None
        self.image_bg_threshold = None
        self.latest_rendered_image = None

        self.im_def_scale = min(VIEW_WIDTH/CAMERA_WIDTH, VIEW_HEIGHT/CAMERA_HEIGHT)
        self.im_scale = self.im_def_scale
        self.im_xoff = 0
        self.im_yoff = 0
        self.im_width = CAMERA_WIDTH
        self.im_height = CAMERA_HEIGHT
        
        self.debug_c = 0
        self.gl = None
        self._paint_entered = False

        self._render = True
        self._algo_render = False
        self._center_model = False
        self._discretize_tol = False
        self.latest_discretization_err_q = False
        
        self._noise_image = os.path.join(SCRIPT_DIR, '../data/noise-fg.png')
        
        self._width = None
        self._height = None
        self._side = None
        self._gl_image = None
        self._object = None
        self._lastPos = QPoint()
        self._imgColor = QColor.fromRgbF(1, 1, 1, 0.4)
        self._fgColor = QColor.fromRgbF(0.6, 0.6, 0.6, 1)
        self._bgColor = QColor.fromRgbF(0, 0, 0, 1)
        #self._bgColor = QColor.fromCmykF(0.0, 0.0, 0.0, 1.0)
        self._frustum_near = 0.1
        self._frustum_far = MAX_DISTANCE
        self._expire=0

        if not BATCH_MODE:
            self.loadTargetImageMeta(TARGET_IMAGE_META_FILE)


    def minimumSizeHint(self):
        return QSize(VIEW_WIDTH, VIEW_HEIGHT)

    def maximumSizeHint(self):
        return QSize(VIEW_WIDTH, VIEW_HEIGHT)

    def sizeHint(self):
        return QSize(VIEW_WIDTH, VIEW_HEIGHT)

    def initializeGL(self):
        f = QSurfaceFormat()
        #f.setVersion(2, 2)
        f.setDepthBufferSize(32)
        p = QOpenGLVersionProfile(f)
        self.gl = self.context().versionFunctions(p)
        self.gl.initializeOpenGLFunctions()

        self.setClearColor(self._bgColor)
        self.loadObject()

        self.gl.glEnable(self.gl.GL_CULL_FACE)

        # for transparent asteroid image on top of model
        self.gl.glBlendFunc(
                self.gl.GL_SRC_ALPHA, self.gl.GL_ONE_MINUS_SRC_ALPHA)

        if self._render:
            self._rendOpts()
        else:
            self._projOpts()

        if not BATCH_MODE:
            self.loadTargetImage(TARGET_IMAGE_FILE)
#            if not USE_IMG_LABEL_FOR_SC_POS:
#                CentroidAlgo.update_sc_pos(self.systemModel, self.full_image)
        
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
        rel_scale = self.im_scale / self.im_def_scale
        self.im_def_scale = min(width/CAMERA_WIDTH, height/CAMERA_HEIGHT)
        self.im_scale = self.im_def_scale * rel_scale
        self._width = width
        self._height = height
        self._side = min(width, height)
        self.updateFrustum()
        

    def updateFrustum(self):
        # calculate frustum based on fov, aspect & near
        # NOTE: with wide angle camera, would need to take into account
        #       im_xoff, im_yoff, im_width and im_height
        x_fov = CAMERA_X_FOV * self.im_def_scale / self.im_scale
        y_fov = CAMERA_Y_FOV * self.im_def_scale / self.im_scale
        
        # calculate frustum based on fov, aspect & near
        right = self._frustum_near * math.tan(math.radians(x_fov/2))
        top = self._frustum_near * math.tan(math.radians(y_fov/2))
        self._frustum = {
            'left':-right,
            'right':right,
            'bottom':-top,
            'top':top, 
        }
        
        if self._width is not None:
            self.gl.glViewport(
                    (self._width-self._side)//2, (self._height-self._side)//2,
                    self._side, self._side)

        self.gl.glMatrixMode(self.gl.GL_PROJECTION)
        self.gl.glLoadIdentity()
        self.gl.glFrustum(
                self._frustum['left'], self._frustum['right'],
                self._frustum['bottom'], self._frustum['top'],
                self._frustum_near, self._frustum_far)
        self.gl.glMatrixMode(self.gl.GL_MODELVIEW)

    def renderParams(self):
        m = self.systemModel
        
        # NOTE: with wide angle camera, would need to take into account
        #       im_xoff, im_yoff, im_width and im_height
        xc_off = (self.im_xoff+self.im_width/2 - CAMERA_WIDTH/2)
        xc_angle = xc_off/CAMERA_WIDTH * math.radians(CAMERA_X_FOV)
        
        yc_off = (self.im_yoff+self.im_height/2 - CAMERA_HEIGHT/2)
        yc_angle = yc_off/CAMERA_HEIGHT * math.radians(CAMERA_Y_FOV)
        
        # first rotate around x-axis, then y-axis,
        # note that diff angle in image y direction corresponds to rotation
        # around x-axis and vise versa
        q_crop = (
            np.quaternion(math.cos(-yc_angle/2), math.sin(-yc_angle/2), 0, 0)
          * np.quaternion(math.cos(-xc_angle/2), 0, math.sin(-xc_angle/2), 0)
        )
        
        x = m.x_off.value
        y = m.y_off.value
        z = m.z_off.value

        # rotate offsets using q_crop
        x, y, z = tools.q_times_v(q_crop.conj(), np.array([x, y, z]))
        
        # maybe put object in center of view
        if self._center_model:
            x, y = 0, 0
        
        # get object rotation and turn it a bit based on cropping effect
        q, err_q = m.gl_sc_asteroid_rel_q(self._discretize_tol)
        if self._discretize_tol:
            self.latest_discretization_err_q = err_q
        
        qfin = (q * q_crop.conj())
        rv = tools.q_to_angleaxis(qfin)

        # light direction
        light = m.light_rel_dir(err_q)
        
        res = (light, (x, y, z), (math.degrees(rv[0]),)+tuple(rv[1:]))
        return res
    
    
    def paintGL(self):
        if self._algo_render:
            self.gl.glDisable(self.gl.GL_BLEND)
        else:
            self.gl.glEnable(self.gl.GL_BLEND)
        
        self.gl.glClear(
                self.gl.GL_COLOR_BUFFER_BIT | self.gl.GL_DEPTH_BUFFER_BIT)

        light, transl, rot = self.renderParams()
        
        self.gl.glLoadIdentity()
        
        # sunlight config
        if self._render or self._algo_render:
            self.gl.glLightfv(self.gl.GL_LIGHT0, self.gl.GL_POSITION, tuple(-light)+(0,))
            self.gl.glLightfv(self.gl.GL_LIGHT0, self.gl.GL_SPOT_DIRECTION, tuple(light))
            self.gl.glLightfv(self.gl.GL_LIGHT0, self.gl.GL_DIFFUSE, (.4,.4,.4,1))
            self.gl.glLightfv(self.gl.GL_LIGHT0, self.gl.GL_AMBIENT, (0,0,0,1))
            self.gl.glLightModelfv(self.gl.GL_LIGHT_MODEL_AMBIENT, (0,0,0,1))
        
        self.gl.glTranslated(*transl)
        self.gl.glRotated(*rot)
        if self._object is not None:
            self.gl.glCallList(self._object)
        
        if not self._algo_render and self._gl_image is not None:
            self.gl.glLoadIdentity()
            self.setColor(self._imgColor)
#            self.gl.glRasterPos3f(self._frustum['left'],
#                                  self._frustum['bottom'], -self._frustum_near)
            self.gl.glWindowPos2f(0,0)
            self.gl.glDrawPixels(self._image_w, self._image_h, self.gl.GL_RGBA,
                                 self.gl.GL_UNSIGNED_BYTE, self._gl_image)

        # dont need? commented out glFinish as it takes around 4s to run
        # after win10 "creators update"
        #self.gl.glFinish()
        
    def saveView(self, grayscale=True, depth=False):
        self.makeCurrent()

        # for some reason following gives 24 and not 32 bits
        # f = self.format() 
        # print('dbs: %s'%f.depthBufferSize())
        
        pixels = glReadPixels(0, 0, VIEW_WIDTH, VIEW_HEIGHT,
                self.gl.GL_DEPTH_COMPONENT if depth else self.gl.GL_LUMINANCE if grayscale else self.gl.GL_RGBA,
                self.gl.GL_FLOAT if depth else self.gl.GL_UNSIGNED_BYTE)

        data = np.frombuffer(pixels, dtype=('float32' if depth else 'uint8'))
        
        if depth:
            near = self._frustum_near
            far = self._frustum_far
            a = -(far - near) / (2.0 * far * near)
            b =  (far + near) / (2.0 * far * near)
            data = np.divide(1.0,(2.0*a)*data -(a-b)) # 1/((2*X-1)*a+b)
        
        data = np.flipud(data.reshape([VIEW_HEIGHT, VIEW_WIDTH] + ([] if depth or grayscale else [4])))
        
        # print('data: %s'%(data,))
        # cv2.imshow('target_image', data)
        # cv2.waitKey()
        
        return data

    def saveViewOld(self):
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
        cv2.imwrite(imgfile, self.render(center=False))
 
    def render(self, center=True, depth=False, discretize_tol=False):
        if not self._render:
            self._rendOpts()
        self._algo_render = True
        self._discretize_tol = discretize_tol
        tmp = self._center_model
        self._center_model = center
        
        fbo = self.grabFramebuffer() # calls paintGL
        self.latest_rendered_image = rr = self.saveView(depth=False)
        if depth:
            dr = self.saveView(depth=True)
        
        self._center_model = tmp
        self._discretize_tol = False
        self._algo_render = False
        if not self._render:
            self._projOpts()
        return (rr, dr) if depth else rr
    
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

    def setImageZoomAndResolution(self, im_xoff=0, im_yoff=0,
                    im_width=CAMERA_WIDTH, im_height=CAMERA_HEIGHT, im_scale=1):
        
        self.im_xoff = im_xoff
        self.im_yoff = im_yoff
        self.im_width = im_width
        self.im_height = im_height
        self.im_scale = im_scale
        
        self.image = ImageProc.crop_and_zoom_image(self.full_image, im_xoff, im_yoff,
                                                   im_width, im_height, im_scale)
        self._image_h = self.image.shape[0]
        self._image_w = self.image.shape[1]
        
        # form _gl_image that is used for rendering
        # black => 0 alpha, non-black => white => .5 alpha
        im = self.image.copy()
        alpha = np.zeros(im.shape, im.dtype)
        #im[im > 0] = 255
        alpha[im > 0] = 128
        self._gl_image = np.flipud(cv2.merge((im, im, im, alpha))).tobytes()
        self.updateFrustum()
        
        # WORK-AROUND: for some reason wont use new frustum if window not resized
        s = self.parent().size()
        self.parent().resize(s.width()+1, s.height())
        self.parent().resize(s.width(), s.height())
        self.update()
        QCoreApplication.processEvents()
        
    def loadTargetImage(self, src, remove_bg=True):
        tmp = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if tmp is None:
            raise Exception('Cant load image from file %s'%(src,))

        if tmp.shape != (CAMERA_HEIGHT, CAMERA_WIDTH):
            # visit fails to generate 1024 high images
            tmp = cv2.resize(tmp, None,
                            fx=CAMERA_WIDTH/tmp.shape[1],
                            fy=CAMERA_HEIGHT/tmp.shape[0],
                            interpolation=cv2.INTER_CUBIC)

        if BATCH_MODE and self._noise_image:
            tmp = ImageProc.add_noise_to_image(tmp, self._noise_image)
            
        self.image_file = src
        if remove_bg:
            self.full_image, h, th = ImageProc.process_target_image(tmp)
            self.image_bg_threshold = th
            self.parent().centroid.bg_threshold = th
        else:
            self.full_image = tmp
            self.image_bg_threshold = None
            self.parent().centroid.bg_threshold = None
        
        self.setImageZoomAndResolution(im_scale=self.im_def_scale)
        
    def loadTargetImageMeta(self, src):
        if True:
            # FIX: currently doesnt correspond with what is shown
            lblloader.load_image_meta(src, self.systemModel)
        else:
            self.systemModel.ast_x_rot.value = -83.88
            self.systemModel.ast_y_rot.value = 74.38
            self.systemModel.ast_z_rot.value = -77.98
            self.systemModel.time.range = (1437391848.27*0.99, 1437391848.27*1.01)
            self.systemModel.time.value = 1437391848.27
            self.systemModel.x_off.value = -0.42
            self.systemModel.x_rot.value = -49.39
            self.systemModel.y_off.value = 2.47
            self.systemModel.y_rot.value = 123.26
            self.systemModel.z_off.value = -158.39
            self.systemModel.z_rot.value = -96.62   

        self.update()

    def loadObject(self, noisy_model=None):
        genList = self.gl.glGenLists(1)
        self.gl.glNewList(genList, self.gl.GL_COMPILE)
        self.gl.glBegin(self.gl.GL_TRIANGLES) # GL_POLYGON?
        self.setColor(self._fgColor)
        #self.gl.glEnable(self.gl.GL_COLOR_MATERIAL);
        #self.gl.glMaterialfv(self.gl.GL_FRONT, self.gl.GL_SPECULAR, (0,0,0,1));
        #self.gl.glMaterialfv(self.gl.GL_FRONT, self.gl.GL_SHININESS, (0,));
        
        if self.systemModel.real_shape_model is None:
            rsm = self.systemModel.real_shape_model = objloader.ShapeModel(fname=TARGET_MODEL_FILE)
        else:
            rsm = self.systemModel.real_shape_model
        
        if noisy_model is not None:
            sm = noisy_model
        elif ADD_SHAPE_MODEL_NOISE and not BATCH_MODE:
            sup = objloader.ShapeModel(fname=SHAPE_MODEL_NOISE_SUPPORT)
            sm, noise = tools.apply_noise(rsm, support=np.array(sup.vertices))
        else:
            sm = rsm
        
        for triangle, norm in sm.faces:
            self.triangle(sm.vertices[triangle[0]],
                          sm.vertices[triangle[1]],
                          sm.vertices[triangle[2]],
                          norm)
        
        self.gl.glEnd()
        self.gl.glEndList()
        
        if DEBUG:
            # assume all 32bit (4B) variables, no reuse of vertices
            # => triangle count x (3 vertices + 1 normal) x 3d vectors x bytes per variable
            mem_needed = len(sm.faces) * 4 * 3 * 4
            print('3D model mem use: %.0fx %.0fB => %.1fMB'%(len(sm.faces), 4*3*4, mem_needed/1024/1024))

        self._object = genList

    def triangle(self, x1, x2, x3, n):
        self.gl.glNormal3f(*n)
        self.gl.glVertex3f(*x1)
        self.gl.glVertex3f(*x2)
        self.gl.glVertex3f(*x3)

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
