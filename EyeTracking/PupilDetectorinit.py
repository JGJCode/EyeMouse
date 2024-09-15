import cv2
import numpy as np
import time
import mediapipe as mp
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import keyboard
import pyautogui

#Creates face mesh tessellation for user and detects initial horizontal and vertical speeds of eye movement to cursor movement
def captureEyes(ac, cap):
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)
    mp_drawing_styles = mp.solutions.drawing_styles
    screen_w,screen_h = pyautogui.size()
    with mp_face_mesh.FaceMesh(refine_landmarks = True, min_detection_confidence = 0.5) as face_mesh:
        while True:
            ret,frame = cap.read()
            if not ret: break
            res = face_mesh.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).multi_face_landmarks
            
            if res:
                lm = res[0].landmark
                landmarks = [lm[468],lm[473]]
                height,width,p = frame.shape
                for face_landmarks in res:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                for landmark in landmarks:
                    x = int(landmark.x*width)
                    y = int(landmark.y*height)
                    cv2.circle(frame,(x,y),1,(0,0,255),thickness=2)
            flp = cv2.flip(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),1)
            fmt = QImage(flp.data,flp.shape[1],flp.shape[0],QImage.Format_RGB888)
            pic = fmt.scaled(screen_w,screen_h,Qt.KeepAspectRatio)
            ac.ImageUpdate.emit(pic)
            #cv2.imshow('Frame',frame)
            if keyboard.is_pressed('q'):
                break
        def captureRegion(x,y):
            currTime = time.time()
            mask, nFrame = None, None
            masked=False
            lastLeftX, lastLeftY, lastRightX, lastRightY = -1,-1,-1,-1
            while time.time()-currTime<=3:
                ret, frame  = cap.read()
                if not ret: break
                if not masked:
                    mask = np.zeros_like(frame)
                    cv2.circle(mask,(y,x),20,(255,255,255),-1)
                    nFrame = cv2.bitwise_or(mask,np.zeros_like(frame))
                    masked=True
                res = face_mesh.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).multi_face_landmarks
                if res:
                    lm = res[0].landmark
                    landmarks = [lm[468],lm[473]]
                    height,width,p = frame.shape
                    lastLeftX = int(landmarks[0].x*width)
                    lastLeftY = int(landmarks[0].y*height)
                    lastRightX = int(landmarks[1].x*width)
                    lastRightY = int(landmarks[1].y*height)
                flp = cv2.flip(cv2.cvtColor(nFrame,cv2.COLOR_BGR2RGB),1)
                fmt = QImage(flp.data,flp.shape[1],flp.shape[0],QImage.Format_RGB888)
                pic = fmt.scaled(screen_w,screen_h,Qt.KeepAspectRatio)
                ac.ImageUpdate.emit(pic)
                
            return (lastLeftX, lastLeftY), (lastRightX, lastRightY)
        ret,frame = cap.read()
        flp = cv2.flip(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),1)
        fmt = QImage(flp.data,flp.shape[1],flp.shape[0],QImage.Format_RGB888)
        pic = fmt.scaled(screen_w,screen_h,Qt.KeepAspectRatio)
        ac.ImageUpdate.emit(pic)
        print(frame.shape[0],frame.shape[1])
        rows,cols = frame.shape[0],frame.shape[1]
        ROIS = [(rows//2+200,int(frame.shape[1]*0.9)),(rows//2-200,int(frame.shape[1]*.10)),(int(rows-frame.shape[0]*.10),cols//2-200),(int(frame.shape[0]*.10),cols//2+200)]
        lastLeft, lastRight = captureRegion(rows//2,cols//2)
        lastPos = (rows//2,cols//2)
        runningX = runningY = 0
        ct = 0
        for x,y in ROIS:
            leftPos,rightPos = captureRegion(x,y)
            ct+=1
            if x!=lastPos[0]:
                xAvg = -1*((leftPos[0]-lastLeft[0])/(x-lastPos[0])+(rightPos[0]-lastRight[0])/(x-lastPos[0]))/2
                runningX = (runningX*(ct-1)+xAvg)/ct
            if y!=lastPos[1]:
                yAvg = ((leftPos[1]-lastLeft[1])/(y-lastPos[1])+(rightPos[1]-lastRight[1])/(y-lastPos[1]))/2
                runningY = (runningY*(ct-1)+yAvg)/ct
            lastLeft=leftPos
            lastRight=rightPos
            lastPos = (x,y)
    return 1/runningX,1/runningY