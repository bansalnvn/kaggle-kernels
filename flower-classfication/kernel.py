import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob  # path manipulation.
import cv2
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from sklearn.utils import shuffle
import os

def read_input_data():
    label_dictionary = {}
    labels_all = os.listdir("./data/flowers/")
    n_classes = len(labels_all)
    print(n_classes)
    print(labels_all)
    for label in labels_all:
        label_dictionary[label] = len(label_dictionary)
    print(label_dictionary)
    imgs = glob.glob("./data/flowers/*/*.jpg")
    print(len(imgs))
    x_train = []
    y_train = []
    for img in imgs:
        y_train.append(label_dictionary[img.split('/')[-2]])
        img_array = np.asarray(cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), (128, 128)))
        img_array = img_array[..., np.newaxis]
        x_train.append(img_array)
    s = pd.Series(y_train)  # one hot encoding.
    one_hot_encoded_y = pd.get_dummies(s)
    return x_train, one_hot_encoded_y, n_classes, label_dictionary


def model(num_classes, input_shape):
    print(input_shape)
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
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
    print(keras_model)
    keras_model.fit(np.asarray(x_train), np.asarray(y_train), epochs=10, batch_size=32)
