import datetime
from re import L

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))

plt.show()

pt_x_train = []
pt_y_train = []
pt_x_test = []
pt_y_test = []

tl_x_train = []
tl_y_train = []
tl_x_test = []
tl_y_test = []

for i in range(len(x_train)):
    if y_train[i] < 5:
        pt_x_train.append(x_train[i]/255)
        pt_y_train.append(y_train[i])
    else:
        tl_x_train.append(x_train[i]/255)
        tl_y_train.append(y_train[i])

for i in range(len(x_test)):
    if y_test[i] < 5:
        pt_x_test.append(x_test[i]/255)
        pt_y_test.append(y_test[i])
    else:
        tl_x_test.append(x_test[i]/255)
        tl_y_test.append(y_test[i])

pt_x_train = np.asarray(pt_x_train).reshape(-1, 28, 28, 1)
pt_x_test = np.asarray(pt_x_train).reshape(-1, 28, 28, 1)
pt_y_train = np_utils.to_categorical(np.asarray(pt_y_train))
pt_y_test = np_utils.to_categorical(np.asarray(pt_y_test))

tl_x_train = np.asarray(tl_x_train).reshape(-1, 28, 28, 1)
tl_x_test = np.asarray(tl_x_train).reshape(-1, 28, 28, 1)
tl_y_train = np_utils.to_categorical(np.asarray(tl_y_train))
tl_y_test = np_utils.to_categorical(np.asarray(tl_y_test))

print("pre Training [Train and Test data]")
print(pt_x_train.shape, pt_y_train.shape)
print(pt_x_test.shape, pt_y_test.shape)

print("\nTransfer Learning [Train and Test data]")
print(tl_x_train.shape, tl_y_train.shape)
print(tl_x_test.shape, tl_y_test.shape)

model = Sequential()

model.add(Conv2D(32, 5, input_shape = (28, 28, 1), activation='relu'))
model.add(Conv2D(16, 5, activation='relu'))
model.add(MaxPool2D(pool_size= (2, 2)))
model.add(Conv2D(8, 3, activation='relu'))

model.add(Flatten())
model.add(Dropout(0.4))

model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

model.summary()

start = datetime.datetime.now()

model.fit(pt_x_train, pt_y_train,
        validation_data=(pt_x_test, pt_y_test),
        epochs=10,
        shuffle=True,
        batch_size=100,
        verbose=2)
end = datetime.datetime.now()
print('\n Time taken for pre-train model: ',end - start)

model.layers 

for layer in model.layers[:6]:
    layer.trainable = False

for layer in model.layers:
    print(layer.trainable)

tl_model = Sequential(model.layers[:6])

tl_model.add(Dense(128, activation='relu'))
tl_model.add(Dense(10, activation='softmax'))

tl_model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

tl_model.summary()

start = datetime.datetime.now()

history = tl_model.fit(tl_x_train, tl_y_train,
                        validation_data=(tl_x_test, tl_y_test),
                        epochs=10,
                        shuffle=True,
                        batch_size=100,
                        verbose=2)

end = datetime.datetime.now()
print('\n Time taken for transfer kearning model: ',end - start)

def display_training_curves(training, validation, title, subplot):
    if subplot%10==1:
        plt.subplot(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_xlabel('epoch')
    ax.lefend(['train','valid.'])

display_training_curves(history.history['loss'],history.history['val_loss'], 'loss', 211)