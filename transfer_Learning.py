
import numpy as np
from tensorflow import keras
from keras.utils import np_utils
import os
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Activation, MaxPool2D, Dropout, Flatten

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_label),(test_images,test_label)=fashion_mnist.load_data()

pt_x_train = []
pt_y_train = []
pt_x_test = []
pt_y_test = []

tl_x_train = []
tl_y_train = []
tl_x_test = []
tl_y_test = []

m = 60000

for i in range(m):
    if train_label[i] < 5:
        pt_x_train.append(train_images[i]/255)
        pt_y_train.append(train_label[i])
    else:
        tl_x_train.append(train_images[i]/255)
        tl_y_train.append(train_label[i])

m2 = 10000

for i in range(m2):
    if test_label[i] < 5:
        pt_x_test.append(test_images[i]/255)
        pt_y_test.append(test_label[i])
    else:
        tl_x_test.append(test_images[i]/255)
        tl_y_test.append(test_images[i])

pt_x_train = np.asarray(pt_x_train).reshape(-1,28,28,1)
pt_x_test = np.asarray(pt_x_test).reshape(-1,28,28,1)
pt_y_train = np_utils.to_categorical(np.asarray(pt_y_train))
pt_y_test = np_utils.to_categorical(np.asarray(pt_y_test))

tl_x_train = np.asarray(tl_x_train).reshape(-1,28,28,1)
tl_x_test = np.asarray(tl_x_test).reshape(-1,28,28,1)
tl_y_train = np_utils.to_categorical(np.asarray(tl_y_train))
tl_y_test = np_utils.to_categorical(np.asarray(tl_y_test))

print(pt_x_train.shape,pt_y_train.shape)
print(pt_x_test.shape,pt_y_test.shape)

print(tl_x_train.shape,tl_y_train.shape)
print(tl_x_train.shape,tl_y_test.shape)

model = Sequential()

model.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation='relu'))
model.add(Conv2D(16,(5,5),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(8,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(5,activation='softmax'))
model.summary()

model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
model.fit(pt_x_train, pt_y_train,
        validation_data=(pt_x_test, pt_y_test),
        epochs=10,
        batch_size=100,
        verbose=2,
        shuffle=True)

for layer in model.layers[:5]:
    layer.trainable = False
x = model.layers[4].output

x = Dropout(0.5)(x)
x = Dense(32,activation='relu')(x)
x = Dense(16,activation='relu')(x)

predictions = Dense(10,activation='softmax')(x)

tl_model = Model(model.input,predictions)

tl_model.summary()

tl_model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

tl_model.fit(tl_x_train,tl_y_train,
            validation_data=(tl_x_test,tl_y_test),
            batch_size=100,
            epochs=10,
            verbose=2,
            shuffle=True)

