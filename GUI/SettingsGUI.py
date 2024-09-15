import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import time
from EyeTracking.KeybindClassification import Keybinds
from CaptureGUI import CaptureScreen
import pyautogui
'''
Graphical interface where user can assign keybinds such as left click, right click, and screenshot to specific actions.
'''
class SettingsScreen(QWidget):
    def __init__(self, main_screen, frame, capture):
        super().__init__()
        self.setWindowTitle("Settings")
        self.main_screen = main_screen
        layout = QGridLayout()
        layout.addWidget(QLabel("Left Click"), 0, 0)
        self.left_click_keybind = QPushButton("Create Left Click Binding")
        self.left_click_keybind.clicked.connect(self.process_left_click)
        layout.addWidget(self.left_click_keybind, 0, 1)
        layout.addWidget(QLabel("Right Click"), 1, 0)
        self.right_click_keybind = QPushButton("Create Right Click Binding")
        self.right_click_keybind.clicked.connect(self.process_right_click)
        layout.addWidget(self.right_click_keybind, 1, 1)
        layout.addWidget(QLabel("Screenshot"), 2, 0)
        self.screenshot_keybind = QPushButton("Create Screenshot Binding")
        self.screenshot_keybind.clicked.connect(self.process_screenshot)
        layout.addWidget(self.screenshot_keybind, 2, 1)
        self.back_button = QPushButton('Go Back to Main Screen', self)
        self.back_button.clicked.connect(self.back)
        layout.addWidget(self.back_button, 3, 0, 1, 2)
        self.setLayout(layout)
        self.cap = capture
        self.capGUI = CaptureScreen(self.cap, self)
        self.keybinds = Keybinds(frame)

    #Sends frame to machine learning model to be identified as a left click
    def process_left_click(self):
        fr = self.capGUI.getFrame()
        self.capGUI.close()
        self.keybinds.update(fr,pyautogui.leftClick)
        self.main_screen.show()
        QApplication.processEvents()

    #Sends frame to machine learning model to be identified as a right click
    def process_right_click(self):
        fr = self.capGUI.getFrame()
        self.capGUI.close()
        self.keybinds.update(fr,pyautogui.rightClick)
        self.main_screen.show()
        QApplication.processEvents()
    #Saves screenshot with random name
    def save_ss(self):
        screenshot = pyautogui.screenshot()
        screenshot.save(f'{str(time.time())[:11]}.png')
    #Sends frame to machine learning model to be identified as a screenshot
    def process_screenshot(self):
        fr = self.capGUI.getFrame()
        self.capGUI.close()
        self.keybinds.update(fr, self.save_ss)
        self.main_screen.show()
        QApplication.processEvents()
    #Returns to main screen
    def back(self):
        self.main_screen.show()
        self.main_screen.worker().run()
        self.close()
