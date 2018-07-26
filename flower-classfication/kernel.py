import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob  # path manipulation.
import cv2
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from common import *


def model(num_classes, input_shape):
    print(input_shape)
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(8, 8), strides=(2, 2),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model


if __name__ == "__main__":
    x_train, y_train, n_classes, labels_dictionary, _ = read_input_data()
    x_train, y_train = shuffle(x_train, y_train, random_state=2)
    print(labels_dictionary)
    keras_model = model(n_classes, x_train[0].shape)
    keras_model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

    datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

    datagen.fit(x_train)
    print(keras_model.summary())
    # history = keras_model.fit_generator(datagen.flow(np.asarray(x_train), y_train, batch_size=64),
    #                                     samples_per_epoch=len(x_train), epochs=50, validation_data=(np.asarray(
    #             x_train), y_train))
    history = keras_model.fit(np.asarray(x_train), np.asarray(y_train), epochs=10, validation_split=0.2)
    # plot the history
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
