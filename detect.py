import jetson.inference
import jetson.utils
import numpy as np 
import time
import os
import pyttsx3
import threading

speak=True
commands = False
command = "Move to start"
LastLoc = "zone"
item='Your device has started!'
confidence=0

itemOld=''
itemList = ["person","hat","backpack","umbrella","shoe","eye glasses","handbag","tie","suitcase","bottle","cup","fork","knife","spoon","bowl","apple","banana","orange","carrot","chair","couch","bed","mirror","desk","door","laptop","mouse","remote","keyboard","cell phone","microwave","oven","refrigerator","clock","vase","scissors","toothbrush"]

import cv2
print(cv2.__version__)
width=1280
height=720
flip=2
#Uncomment These next Two Line for Pi Camera
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#cam= cv2.VideoCapture(camSet)
def isItem():
	global item
	
	if item in itemList:
		print(item)
		return True
	else:
		return False
def sayItem():
    global speak
    global item
    global command
    global commands	
    while True:
        if speak ==True:
            engine = pyttsx3.init()
            engine.setProperty('rate',150)
            engine.say(item)
            engine.runAndWait()
            speak=False
        if commands == True:
            engine = pyttsx3.init()
            engine.setProperty('rate',150)
            engine.say(command)
            engine.runAndWait()
            commands=False

x=threading.Thread(target=sayItem, daemon=True)
x.start()

#Or, if you have a WEB cam, uncomment the next line
#(If it does not work, try setting to '1' instead of '0')
cam=cv2.VideoCapture('/dev/video0')
w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)+0.5)
h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)+0.5)
size  = (w,h)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out =cv2.VideoWriter('chair1.avi',fourcc,20.0,size)

cam.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
net=jetson.inference.detectNet('ssd-mobilenet-v2')
font=cv2.FONT_HERSHEY_SIMPLEX
timeMark=time.time()
fpsFilter=0

while True:
    ret, frame = cam.read()
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
    img=jetson.utils.cudaFromNumpy(img)
    if speak==False:
        detect = net.Detect(img,width,height)
        for detections in detect:
            classID = detections.ClassID
            confidence = detections.Confidence
            top = detections.Top
            left = detections.Left
            bottom = detections.Bottom
            right = detections.Right
            center = detections.Center
	
            if confidence>=.5:
                item=net.GetClassDesc(classID)
                if item == "chair":
                    commands = True	
                    speak = False
                    if center[0] <220 :
                        LastLoc = "Left"
                        command = "Move Left"
                        print("Move Left")
                        cv2.putText(frame,"Move Left",(0,400),font,1,(0,0,255),2)
                    elif center[0] > 440:
                        LastLoc = "Right"
                        command = "Move Right"
                        print("Move Right")
                        cv2.putText(frame,"Move Right",(0,400),font,1,(0,0,255),2)
                    else:
                        LastLoc = "Left"
                        command = "You are in the zone"
                        print("You are in the zone")
                        cv2.putText(frame,"You are in the zone",(0,400),font,1,(0,0,255),2)
                else:
                    if LastLoc == "Left":
                        commands = True
                        speak = False
                        print("Move Left, Object exited")
                        command = "Move Left, Object exited"
                        cv2.putText(frame,"Move Left, Object exited",(0,400),font,1,(0,0,255),2)
                    elif LastLoc == "Right":
                        commands = True
                        speak = False
                        command = "Move Right, Object exited"
                        print("Move Right, Object exited")
                        cv2.putText(frame,"Move Right, Object exited",(0,400),font,1,(0,0,255),2)
                if item!=itemOld and item in itemList:
                    speak=True
            if confidence<.5:
                item=''
            itemOld=item
    dt=time.time()-timeMark
    timeMark=time.time()
    fps=1/dt
    fpsFilter=.95*fpsFilter + .05 *fps
    cv2.line(frame, (220,0), (220,500),(0,255,0),10)
    cv2.line(frame, (440,0), (440,500),(255,0,0),10)
    if item in itemList:
        cv2.putText(frame,str(round(fpsFilter,1))+'  fps  '+item+'   '+str(round(confidence,2)),(0,30),font,1,(0,0,255),2)
    cv2.imshow('nanoCam',frame)
    out.write(frame)
    cv2.moveWindow('nanoCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
out.release()
cv2.destroyAllWindows()
