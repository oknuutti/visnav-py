import numpy as np
import cv2

img = cv2.imread('C:\\projects\\visnav\\data\\targetimgs\\ROS_CAM1_20150720T064939_P.png', cv2.IMREAD_GRAYSCALE)
imgf = np.fft.fft2(img)

if True:
    imgfr = np.fft.fftshift(imgf)
    #imgfr2 = np.zeros(imgfr.shape)

    o = imgfr.shape[0]
    d = int(o/4)
    s = int(o/2) - d
    e = int(o/2) + d
    if True:
        imgfr2 = cv2.resize(np.real(imgfr[s:e, s:e]), (o, o), interpolation=cv2.INTER_CUBIC).astype('complex128')
        imgfr2 += cv2.resize(np.imag(imgfr[s:e, s:e]), (o, o), interpolation=cv2.INTER_CUBIC)*1j
    else:
        imgfr2_r = cv2.resize(np.abs(imgfr[s:e, s:e]), (o, o), interpolation=cv2.INTER_CUBIC)
        imgfr2_a = cv2.resize(np.angle(imgfr[s:e, s:e]), (o, o), interpolation=cv2.INTER_CUBIC)
        imgfr2 = imgfr2_r * np.exp(1j * imgfr2_a)

    #from scipy.misc import imresize as imresize
    #imgfr2 = imresize(imgfr[s:e, s:e], (o, o))
    #from scipy.ndimage import zoom
    #imgfr2 = zoom(imgfr[s:e, s:e], 2)

    #imgfr2[256:768, 256:768] = imgfr[256:768, 256:768]
    #imgfr2 = imgfr

    imgf2 = np.fft.ifftshift(imgfr2)
elif False:
    imgf2 = np.zeros(imgf.shape)
    imgf2[256:768, 256:768] = imgf[256:768, 256:768]

if False:
    imgf = imgf2
    imgfn = np.log(np.abs(np.fft.fftshift(imgf)))
    cv2.imshow('fs', imgfn/np.max(imgfn.flatten()))

    imgh_ = cv2.resize(img, (512, 512))
    imgh = np.zeros(img.shape)
    imgh[256:768, 256:768] = imgh_
    imghf = np.fft.fft2(imgh)
    imghfn = np.log(np.abs(np.fft.fftshift(imghf)))
    cv2.imshow('fsh', imghfn/np.max(imghfn.flatten()))

    cv2.waitKey()
    quit()


img2 = np.real(np.fft.ifft2(imgf2))

cv2.imshow('fs', img2.astype('uint8'))
cv2.waitKey()