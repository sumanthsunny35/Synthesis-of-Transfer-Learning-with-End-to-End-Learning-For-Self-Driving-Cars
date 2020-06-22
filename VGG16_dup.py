#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import os

## Keras
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Lambda, Conv2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import multi_gpu_model
import keras.backend.tensorflow_backend as tfback
import tensorflow as tf

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


import cv2
import pandas as pd
import random
import ntpath

## Sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# In[5]:


columns = ["img","steering"]
data = pd.read_csv(r'/home/u43783/driving_dataset/data.txt', sep = " ", names = columns)


# In[10]:


datadir = '/home/u43783/driving_dataset'


# In[11]:


def load_img_steering(datadir, df):
  """Get img and steering data into arrays"""
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center = indexed_data[0]
    image_path.append(os.path.join(datadir, center.strip()))
    steering.append(float(indexed_data[1]))
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings

image_paths, steerings = load_img_steering(datadir , data)


# In[16]:


X_train, X_valid, Y_train, Y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=0)


# In[19]:


def img_preprocess(img):
  """Take in path of img, returns preprocessed image"""
  img = npimg.imread(img)
  # Cropping the image
  img = img[60:-25, :, :]
  # Resizing the image
  img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
  # Converting the image to YUV
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  return img


# In[ ]:


X_train = np.array(list(map(img_preprocess, X_train)))
X_valid = np.array(list(map(img_preprocess, X_valid)))

# In[ ]:


from keras.applications import VGG16
#Load the ResNet50 model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(66, 200, 3))


# In[ ]:

for layer in vgg16.layers[:-4]:
    layer.trainable = False
 
for layer in vgg16.layers:
    print(layer, layer.trainable)

def my_model():
  model = Sequential()
  model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
  model.add(vgg16)
  #model.add(Dropout(0.5))
  model.add(Flatten())

  model.add(Dense(512, activation='relu'))
  #model.add(Dropout(0.5))
  
  model.add(Dense(256, activation='relu'))
  #model.add(Dropout(0.5))
  
  model.add(Dense(64, activation='relu'))
  #model.add(Dropout(0.5))
  
  model.add(Dense(1))
  # model.summary()
  optimizer = Adam(lr=1e-4)
  model.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_error'])

  return model


# In[ ]:


parallel_model = my_model()
print(parallel_model.summary())


# In[ ]:

filepath="VGG16_final_dup.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = parallel_model.fit(X_train, Y_train, epochs=40, validation_data=(X_valid, Y_valid), batch_size=128, verbose=1, shuffle=1, callbacks=callbacks_list)
#parallel_model.save("VGG16.h5")
#print("VGG16 saved")

from matplotlib.pyplot import figure
figure(num=None, figsize=(3, 3), dpi=300, facecolor='w', edgecolor='k')

plt.plot(history.history['loss'], color = 'red')
plt.plot(history.history['val_loss'], color = 'green')

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('VGG16 transfer learning model')
plt.ylabel('Mean Squared error')
plt.xlabel('Epoch')
plt.legend(['training_loss', 'validation_loss'], loc='upper right')
plt.savefig('vgg16_Final_dup.png', dpi=300, bbox_inches='tight')
plt.show()

