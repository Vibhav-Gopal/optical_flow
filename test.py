import cv2
import numpy as np
import time
import math
import mouse

def generateMouseEvent(angle,magnitude,sens=0.2):
    
    magnitudeThreshold = 60
    if magnitude<magnitudeThreshold:return
    #up movement = -90
    # down movement = 90
    # right movement = -180
    # left movement =0
    # cos = horiz comp
    # sin = vert comp
    vertMag = magnitude * math.sin(angle) * sens
    horizMag = magnitude * math.cos(angle) * sens*-1
    mouse.move(horizMag,vertMag,absolute=False)


downSampleLevel = 4
cam = cv2.VideoCapture(0)
ret,frm = cam.read()
frm = frm[::downSampleLevel,::downSampleLevel,:]
frm = cv2.GaussianBlur(frm,(3,3),1)

fpsCount=0
prevTime = time.time()
prev_gray = cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(frm)
nmask = np.zeros_like(frm)
mask[...,1] = 255

while(cam.isOpened()):
    ret,frm = cam.read()
    
    # frm =frm[y:y+h,x:x+w,:]
    frm = frm[::downSampleLevel,::downSampleLevel,:]
    frm = cv2.GaussianBlur(frm,(3,3),1)
    # frm = cv2.GaussianBlur(frm,(5,5),2)
    cv2.imshow("Input",frm)
    gray = cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,None,
                                        0.5,3,15,3,5,1.2,0)
    
    xflows= flow[:,:,0].flatten()
    yflows = flow[:,:,1].flatten()
    xAvg = np.sum(xflows)/np.shape(flow.flatten())
    yAvg = np.sum(yflows)/np.shape(flow.flatten())

    avgAngle = math.atan2(yAvg,xAvg)
    avgAngle =  math.degrees(avgAngle)
    avgMag = math.sqrt(xAvg**2 + yAvg**2)
    opAvgMag = avgMag*100
    mag, angle = cv2.cartToPolar(flow[...,0],flow[...,1])


    mask[...,0] = angle*180/np.pi/2
    mask[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # generateMouseEvent(avgAngle,opAvgMag)
    if opAvgMag>50: print((avgAngle,255,opAvgMag))
    nmask[:] = ((avgAngle,255,opAvgMag))

    rgb = cv2.cvtColor(mask,cv2.COLOR_HSV2RGB)
    rgb = cv2.GaussianBlur(rgb,(3,3),1)
    avgRgb = cv2.cvtColor(nmask,cv2.COLOR_HSV2RGB)
    avgRgb = cv2.GaussianBlur(avgRgb,(3,3),1)
    cv2.imshow("Optical Flow",rgb)
    cv2.imshow("Optical Flow Avg",avgRgb)
    prev_gray = gray
    if cv2.waitKey(1) == ord('q') : break
    fpsCount+=1
    if time.time() - prevTime >=1:
        print(f"\nFrameRate is {fpsCount}")
        fpsCount=0
        prevTime = time.time()

cam.release()

