from common import read_and_sanitize_data
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, Flatten
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

_BATCH_SIZE = 10132


def model(input_shape):
    print(input_shape)
    model = Sequential()
    model.add(Dense(32, input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


if __name__ == '__main__':
    input_data = read_and_sanitize_data()
    print(input_data.shape)
    train_x = input_data.drop(labels='Class', axis=1)
    print(train_x.shape)
    train_y = input_data[['Class']]
    print(train_y.shape)
    train_x.drop(train_x.head(1).index, inplace=True)
    train_y.drop(train_y.head(1).index, inplace=True)
    model = model(train_x.shape[1:])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    train_x, train_y = shuffle(train_x, train_y, random_state=2)
    history = model.fit(np.asarray(train_x), np.asarray(train_y), epochs=100, batch_size=train_x.shape[0])
    print(history.history.keys())