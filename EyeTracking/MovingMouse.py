import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from .PupilDetectorinit import captureEyes
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import keyboard
import time

#Handles eye-to-cursor tracking and frame processing events
def move(ac, cap, keybindTracker, xDPI = -1, yDPI = -1):
    print(xDPI,yDPI)
    if xDPI==-1:
        xDPI, yDPI = captureEyes(ac,cap)
        xDPI/=2
        ac.xSpeed = xDPI
        ac.ySpeed = yDPI
    
    screen_w, screen_h = pyautogui.size()
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    pyautogui.FAILSAFE = False

    #Temporarily pauses the eye tracking mouse. During this time, the normal mouse may be used.
    def pause():
        while True:
            ret, frame = cap.read()
            if not ret: break
            flp = cv2.flip(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),1)
            fmt = QImage(flp.data,flp.shape[1],flp.shape[0],QImage.Format_RGB888)
            pic = fmt.scaled(screen_w,screen_h,Qt.KeepAspectRatio)
            ac.ImageUpdate.emit(pic)
            if keyboard.is_pressed('x'):
                return
            
    with mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        ret, frame = cap.read()
        currX = frame.shape[1] // 2
        currY = frame.shape[0] // 2
        pyautogui.moveTo(currX, currY)
        lastLeftX, lastLeftY, lastRightX, lastRightY = None, None, None, None
        
        while ac.active:
            ret, frame = cap.read()
            if not ret:
                break
            keybindTracker.process(frame)()
            flp = cv2.flip(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),1)
            fmt = QImage(flp.data,flp.shape[1],flp.shape[0],QImage.Format_RGB888)
            pic = fmt.scaled(screen_w,screen_h,Qt.KeepAspectRatio)
            ac.ImageUpdate.emit(pic)
            res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks
            if res:
                lm = res[0].landmark
                landmarks = [lm[468], lm[473]]
                height, width, _ = frame.shape
                xDiff = 0
                if lastLeftX is not None:
                    xDiff = ac.xSpeed * ((int(landmarks[0].x * width) - lastLeftX) + (int(landmarks[1].x * width) - lastRightX)) // 2
                yDiff = 0
                if lastLeftY is not None:
                    yDiff = ac.ySpeed * ((int(landmarks[0].y * height) - lastLeftY) + (int(landmarks[1].y * height) - lastRightY)) // 2
                
                pyautogui.moveTo((currX + xDiff), (currY + yDiff))
                currX += xDiff
                currY += yDiff
                lastLeftX = int(landmarks[0].x * width)
                lastLeftY = int(landmarks[0].y * height)
                lastRightX = int(landmarks[1].x * width)
                lastRightY = int(landmarks[1].y * height)
            if keyboard.is_pressed('q'):
                pause()
                currX = frame.shape[1] // 2
                currY = frame.shape[0] // 2
                pyautogui.moveTo(currX, currY)
    cap.release()
    cv2.destroyAllWindows()
