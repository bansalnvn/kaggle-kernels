import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob  # path manipulation.
import cv2
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.utils import shuffle

from common import *



def model(num_classes, input_shape):
    print(input_shape)
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(8, 8), strides=(2, 2),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    # model.add(Conv2D(8, (16, 16), strides=(4, 4), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    # # model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


if __name__ == "__main__":
    x_train, y_train, n_classes, labels_dictionary = read_input_data()
    x_train, y_train = shuffle(x_train, y_train)
    print(labels_dictionary)
    keras_model = model(n_classes, x_train[0].shape)
    keras_model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
    print(keras_model.summary())
    keras_model.fit(np.asarray(x_train), np.asarray(y_train), epochs=10, batch_size=64)
