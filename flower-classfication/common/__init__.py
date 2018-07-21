import os
import numpy as np
import pandas as pd
import cv2
import glob


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
    x_train_rgb = []
    for img in imgs:
        y_train.append(label_dictionary[img.split('/')[-2]])
        img_array = np.asarray(cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), (128, 128)))
        img_array = img_array[..., np.newaxis]
        x_train.append(img_array)
        x_train_rgb.append(np.asarray(cv2.resize(cv2.imread(img), (128, 128))))
    s = pd.Series(y_train)  # one hot encoding.
    one_hot_encoded_y = pd.get_dummies(s)
    # cv2.imshow("test_image", x_train[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return x_train, one_hot_encoded_y, n_classes, label_dictionary, np.asarray(x_train_rgb)
