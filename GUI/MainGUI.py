'''
MAKE SURE YOU HAVE READ THE README.md FILE BEFORE USE
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import pyautogui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from EyeTracking.MovingMouse import move
from GUI.SettingsGUI import SettingsScreen
'''
Main graphical interface where user can move cursor using their eyes.
 - Sidebars to adjust horizontal and vertical speed
 - Initialize mouse, face tessellation
 - Redirect to settings or home
 - Helper thread and signals to efficiently display cursor movements and webcam
'''
class QtVideo(QWidget):
    def __init__(self):
        super(QtVideo, self).__init__()
        self.setWindowTitle("Eye Tracking Mouse")
        self.VBL = QHBoxLayout()
        self.feedLabel = QLabel()
        screen_w, screen_h = pyautogui.size()
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        self.settings = SettingsScreen(self,frame,self.cap)
        self.kb = self.settings.keybinds
        self.feedLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.feedLabel.setScaledContents(True)
        self.worker = ThreadWorker(self.cap, self.kb)
        self.worker.start()
        self.controlLayout = QGridLayout()
        self.xInput = QLineEdit(self)
        self.xInput.setPlaceholderText(f"x DPI {self.worker.xSpeed}")
        self.xInput.setFixedWidth(100)
        self.xInput.textChanged.connect(self.updateX)
        self.yInput = QLineEdit(self)
        self.yInput.setPlaceholderText(f"y DPI {self.worker.ySpeed}")
        self.yInput.setFixedWidth(100)
        self.topSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.bottomSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.VBL.addWidget(self.feedLabel, alignment = Qt.AlignLeft)
        self.VBL.addItem(self.topSpacer)
        self.VBL.addItem(self.bottomSpacer)
        self.yInput.textChanged.connect(self.updateY)
        self.controlLayout.addWidget(self.xInput,0,0,alignment=Qt.AlignRight|Qt.AlignTop)
        self.controlLayout.addWidget(self.yInput,1,0,alignment=Qt.AlignRight|Qt.AlignTop)
        self.setbut = QPushButton("Settings", self)
        self.setbut.clicked.connect(self.open_settings)
        self.controlLayout.addWidget(self.setbut,2,0, alignment = Qt.AlignRight|Qt.AlignTop)
        self.cancel = QPushButton("Cancel")
        self.cancel.clicked.connect(self.can)
        self.controlLayout.addWidget(self.cancel, 3,0,alignment = Qt.AlignRight|Qt.AlignTop)
        self.VBL.addLayout(self.controlLayout)
        self.showFullScreen()
        self.worker.ImageUpdate.connect(self.upd)
        self.setLayout(self.VBL)
    
    #Updates displayed frame by changing the pixel map of the feed label
    def upd(self,image):
        self.feedLabel.setPixmap(QPixmap.fromImage(image))
    
    #Connected to cancel button, exits application
    def can(self):
        self.worker.stop()
        self.close()
    
    #Updates horizontal speed based on value in sidebar
    def updateX(self, text):
        try:
            self.worker.xSpeed = -1*float(text)
            self.xInput.setPlaceholderText(f"x DPI {self.worker.xSpeed}")
        except:
            pass

    #Updates vertical speed based on value in sidebar
    def updateY(self, text):
        try:
            self.worker.ySpeed = float(text)
            self.yInput.setPlaceholderText(f"y DPI {self.worker.ySpeed}")
        except:
            pass

    #Redirects to settings 
    def open_settings(self):
        self.close()
        self.settings.show()

class ThreadWorker(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def __init__(self, cap, keybinds):
        super().__init__()
        self.xSpeed = -1
        self.ySpeed = -1
        self.active = True
        self.kb = keybinds
        self.capture = cap

    #Initiates eye tracking and webcam feed capture
    def run(self):
        move(self,self.capture, self.kb, self.xSpeed, self.ySpeed)

    #Ends current thread
    def stop(self):
        self.active = False
        self.quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    root = QtVideo()
    root.show()
    sys.exit(app.exec())