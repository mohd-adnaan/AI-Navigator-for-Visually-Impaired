import jetson.inference
import jetson.utils
import time
import cv2
import numpy as np 

timeStamp=time.time()
fpsFilt=0
net=jetson.inference.detectNet('ssd-mobilenet-v2',threshold=.5)
dispW=1280
dispH=720



cam=jetson.utils.gstCamera(dispW,dispH,'/dev/video0')
display=jetson.utils.glDisplay()
while display.IsOpen():
    img, width, height= cam.CaptureRGBA()
    detections=net.Detect(frame, width, height)
    display.RenderOnce(img,width,height)
    dt=time.time()-timeStamp
    timeStamp=time.time()
    fps=1/dt
    fpsFilt=.9*fpsFilt + .1*fps
    print(str(round(fps,1))+' fps')

