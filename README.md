# **Eye Controlled Mouse**
Move the cursor on your screen with your eyes. This project currently includes features such as left click, right click, and screenshotting a selected portion of the screen through custom facial actions (ex. blinking, head orientation, etc).

# Brief Technical Overview
- This project uses pretrained machine learning models from Mediapipe to track the pupils and over 460 other facial landmarks.
- The customized keybinds were implemented by creating a retrainable Convolutional Neural Network Model using image augmentation techniques, which helped create a large dataset through just a few frames.
- The graphical user interface was implemented using PyQt5, where the main window continously updates the webcam feed based on its connected signal. A threaded worker was used for video capture and processing, ensuring smooth and efficient feed updates.

# How to Run the Program
Clone the repository into a local code editor
```bash
git clone https://github.com/JGJCode/EyeMouse.git
```
Run the MainGUI.py file located in the GUI folder

# Eye Mouse Initialization Directions and Controls
- When you are ready to initialize the mouse, press q on the keyboard. For best results, follow the white dots on the screen without moving your head and keeping the dot at the center of your vision at all times.
- After you have followed the dots, the initial horizontal or vertical velocity may not be appropriate. To change the velocity, experiment with the x DPI and y DPI boxes on the side of the screen. A suggested x DPI is 20 and a suggested y DPI is 30. To change the DPI, enter your desired DPI into the respective box and leave that number in the box.
- To pause the eye tracking software and return to ordinary mouse function, press q on the keyboard.
- To unpause the eye tracking software, press x on the keyboard.

# Custom Keybinds Directions
To redirect to the settings menu, press the settings button on the right side of the screen.
First, click which keybind you would like to set (left click, right click, or screenshot). The frame which will be used to recognize that keybind will be captured in approximately 3 seconds. The action which is used to recognize that keybind should be visible within a single frame, rather than a series of frames. This action should also be unique to that keybind and easily distinguishable from other set actions. Be aware that a horizontal flip is applied as part of the image augmentation process, so do not use actions that mirror each other for different keybinds. A few examples of actions that were successfully used are putting a colored object in the camera view, raising a hand, and drastically changing head orientation. After the action frame is captured, wait a few minutes to finish training the model. During this time, be sure not to interact with the computer.

# Possible Improvements
- Though the x and y DPI can be adjusted by the user, smoother initial eye tracking may be possible by tracking other facial landmarks.
- Instead of fully retraining the machine learning model every time a new keybind is assigned, dynamic machine learning algorithms could be used to improve performance.