from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import time
import pyautogui
import cv2
'''
Graphical interface for capturing frames to send to the machine learning model
'''
class CaptureScreen(QWidget):
    def __init__(self, cap, settings):
        super().__init__()
        self.setWindowTitle("Capturing Keybind")
        self.screen_w, self.screen_h = pyautogui.size()
        self.layout = QVBoxLayout()
        self.fl = QLabel()
        self.fl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fl.setScaledContents(True)
        self.label = QLabel("Time until Capture: 3")
        self.layout.addWidget(self.label, alignment = Qt.AlignRight|Qt.AlignTop)
        self.layout.addWidget(self.fl,alignment = Qt.AlignLeft)
        self.cap = cap
        self.settings_screen = settings
        self.setLayout(self.layout)
        
    #Captures frame after exactly 3 seconds, automatically returning to the settings screen
    def getFrame(self):
        self.settings_screen.hide()
        QApplication.processEvents()
        self.showFullScreen()
        QApplication.processEvents()
        curr = time.time()
        while time.time()-curr<=3:
            ret,frame = self.cap.read()
            if not ret: break
            self.label.setText(f'Time until Capture: {max(0,3-(time.time()-curr)):.2f} seconds')
            print(self.label.text())
            flp = cv2.flip(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),1)
            fmt = QImage(flp.data,flp.shape[1],flp.shape[0],QImage.Format_RGB888)
            pic = fmt.scaled(self.screen_w,self.screen_h,Qt.KeepAspectRatio)
            self.fl.setPixmap(QPixmap.fromImage(pic))
            QApplication.processEvents()
        return frame