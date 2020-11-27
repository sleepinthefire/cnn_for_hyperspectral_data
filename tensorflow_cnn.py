# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:43:28 2020

@author: mtl98
"""
import numpy as np
import pandas as pd
from spectrals import spectrals
from matplotlib import pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras import layers, models

hs = spectrals()
band = hs.bands
pixel = 512
bands = 204


path_cod = {filepath:codinate,}

#load the spectral file, and extract ROI, put the label
#リターンは(データの数、バンド数、１)バンドの最後尾にラベル情報を付加　
def mk_data(path, label, coordinate):
    hs = spectrals()
    x1, y1, x2, y2 = coordinate
    w = x2 - x1
    h = y2 - y1
    
    array = hs.load_specim(path)
    array = array[y1:y1+h, x1:x1+w, :]
    data = hs.mk_dataset(array, glid_length=16, slide=8)
    iteration = len(data.columns.values.tolist())
    label = pd.DataFrame([label for i in range(iteration)], index=data.columns.values.tolist()).T

    data = pd.concat([data, label], ignore_index=True)
    reshape = data.values.T.reshape(iteration, 205, 1)
    
    return reshape

data = []    
for path, cod in path_cod.items():
    data.append(mk_data(path, 0, cod))
    
data = np.concatenate([data[0],data[2]])
train = data[:,:204,:]
label = data[:,204,:]

#data.dump('ser_gre_data.dat')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = \
    train_test_split(train, label, test_size=0.2)

model = models.Sequential()
model.add(layers.Conv1D(64, 3, padding='same', activation='relu', input_shape=(204, 1)))
model.add(layers.MaxPooling1D())
model.add(layers.Conv1D(128, 3, activation='relu'))
model.add(layers.MaxPooling1D())
model.add(layers.Conv1D(256, 3, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

epoch = 200

history = model.fit(x_train, y_train, validation_split=0.1, shuffle=True,epochs=epoch)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epoch)


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('serpentine_model.h5')

learned_model = models.load_model('serpentine_model.h5')
result = learned_model.predict(x_test)
print(result.argmax(axis=1))

test_loss, test_acc = learned_model.evaluate(x_test, y_test, verbose=0)