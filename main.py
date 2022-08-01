import tensorflow as tf
from sklearn.datasets import load_files 
from keras.utils import np_utils
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt 
import cv2
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
from helper_func import load_dataset ,path_to_tensor, paths_to_tensor

gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

# load the data and split the train_set and valid_set
train_files, train_targets = load_dataset('./Data/Data_train')
test_files, test_targets = load_dataset('./Data/Data_test')
train_tensors = paths_to_tensor(train_files)/255
test_tensors = paths_to_tensor(test_files)/255
train_tensors, valid_tensors, train_targets, valid_targets = train_test_split(train_tensors, train_targets, test_size=0.33, random_state=42)
print('The shape of train_tensors:',train_tensors.shape)
print('The shape of train_target:',train_targets.shape)

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint  

### TODO: Define your architecture.
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=train_tensors.shape[1:]))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(GlobalAveragePooling2D())
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='./save_model/weights.best.fruit.hdf5', monitor='accuracy',verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=20, batch_size=32, callbacks=[checkpointer], verbose=1)

fruit_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(fruit_predictions)==np.argmax(test_targets, axis=1))/len(fruit_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
