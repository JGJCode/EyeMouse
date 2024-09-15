import os
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
import cv2
import keras
import time
import shutil

'''
Processes each frame of a live video stream, and performs the action that frame was mapped to.
There are currently 4 types of actions that may be returned: None, left click, right click, or screenshot.
These mappings are produced using a retrainable model.
'''
class Keybinds:
    #Initializes Keybind class with a frame mapping to no action. Creates a folder for data which trains the model
    def __init__(self, frame, action = lambda: None, path = r"/data"):
        gpus=tf.config.experimental.list_physical_devices('GPU')
        self.sz = 96
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
        self.num_classes = 0
        self.action_map = []
        self.model = keras.models.Sequential()
        self.gen = keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip = True,
            rescale = 1./(self.sz-1),
        )
        self.path = path
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        
        os.makedirs(path,exist_ok=True)
        self.update(frame, action)

    #Returns index of action if it has already been mapped to a frame before
    def find_action(self, action):
        for i, ac in enumerate(self.action_map):
            if ac==action:
                return i
        return -1
    
    #Maps frame to action in the ML model. If the action was previously mapped to before, this frame will override the old frame.
    def update(self, frame, action):
        idx = self.find_action(action)
        frame = np.expand_dims(cv2.resize(frame, (self.sz,self.sz)), axis = 0)
        pt = os.path.join(self.path,f'class_{self.num_classes if idx==-1 else idx}')
        os.makedirs(pt,exist_ok=True)
        t_image_path = os.path.join(pt,f'c{self.num_classes if idx==-1 else idx}.jpg')
        cv2.imwrite(t_image_path,np.squeeze(frame))
        if idx==-1:
            self.action_map.append(action)
        ct = 0
        for batch in self.gen.flow(frame, batch_size = 1):
            ct+=1
            pt1 = os.path.join(pt,f'aug{ct}.jpg')
            cv2.imwrite(pt1,(batch[0]*(self.sz-1)).astype(np.uint8))
            if ct==1000: break
        if idx==-1:
            self.num_classes+=1
        if self.num_classes>1:
            '''
            ALTERNATE MODEL

            self.model = keras.models.Sequential([
                keras.layers.Rescaling(1./(self.sz-1)),
                keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = "same", activation="relu", input_shape = (self.sz,self.sz,3)),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size = (2,2), strides = 2),
                keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size = (2,2), strides = 2),
                keras.layers.Flatten(),
                keras.layers.Dense(units = 128, activation = "relu", kernel_regularizer = keras.regularizers.l2(0.01)),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(units = self.num_classes, activation = "softmax")
            ]
            )
            self.model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
            train = keras.utils.image_dataset_from_directory(
                self.path,
                validation_split = 0.2,
                subset = "training",
                seed = 123,
                image_size = (96,96),
                batch_size = 32)
            val = keras.utils.image_dataset_from_directory(
                self.path,
                validation_split = 0.2,
                subset = "validation",
                seed = 123,
                image_size = (96,96),
                batch_size = 32
            )
            AUTOTUNE = tf.data.AUTOTUNE
            train = train.cache().prefetch(buffer_size=AUTOTUNE)
            val = val.cache().prefetch(buffer_size=AUTOTUNE)
            early_stop = EarlyStopping(monitor = "val_loss", patience = 2, restore_best_weights = True)
            def train_model():
                try:
                    self.model.fit(train,validation_data= val,epochs = 20, callbacks = [early_stop], verbose = 1)
                except Exception as e:
                    print(f'Error {e}')
            self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

            train = keras.utils.image_dataset_from_directory(
                self.path,
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(96, 96),
                batch_size=16)
            val = keras.utils.image_dataset_from_directory(
                self.path,
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(96, 96),
                batch_size=16
            )
            AUTOTUNE = tf.data.AUTOTUNE
            train = train.cache().prefetch(buffer_size=AUTOTUNE)
            val = val.cache().prefetch(buffer_size=AUTOTUNE)
            early_stop = EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=2,
                                        restore_best_weights=True)

            try:
                self.model.fit(train, validation_data=val, epochs=6, callbacks=[early_stop], verbose=1)
            except Exception as e:
                print(f'Error {e}')
            '''
            data = tf.keras.utils.image_dataset_from_directory(self.path, seed = 123, image_size = (self.sz, self.sz))
            data = data.shuffle(buffer_size = 1000)
            data_iterator = data.as_numpy_iterator()
            data = data.map(lambda x,y : (x/(self.sz-1),y))
            data_length = len(data)
            train_size = int(data_length*.7)
            val_size = int(data_length*.2)
            test_size = int(data_length*.1)
            train = data.take(train_size)
            val = data.skip(train_size).take(val_size)
            test = data.skip(train_size+val_size).take(test_size)
            self.model = keras.models.Sequential([
                keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(self.sz,self.sz,3)),
                keras.layers.MaxPooling2D(),
                keras.layers.Conv2D(32,(3,3),activation='relu'),
                keras.layers.MaxPooling2D(),
                keras.layers.Flatten(),
                keras.layers.Dense(self.sz,activation='relu'),
                keras.layers.Dense(units = self.num_classes,activation='softmax')
            ])
            self.model.compile("adam", loss = tf.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
            logdir='logs'
            early_stop = EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=2,
                                        restore_best_weights=True)
            tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logdir)
            self.model.fit(train, epochs = 20, validation_data = val, callbacks = [early_stop, tensorboard_callback])

    #Sends frame to machine learning model to identify which action must be performed
    def process(self, frame):
        if self.num_classes <= 1:
            return lambda: None
        else:
            curr = time.time()
            dat = np.expand_dims(cv2.resize(frame,(self.sz,self.sz)),axis = 0)
            output = self.model.predict(dat)
            return self.action_map[np.argmax(output)]
        