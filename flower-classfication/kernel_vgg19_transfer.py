from common import *
from sklearn.utils import shuffle
from keras import applications


def model(num_classes, input_shape):
    model = applications.VGG19(weights="imagenet", include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
    print(model.summary())

if __name__ == "__main__":
    x_train, y_train, n_classes, labels_dictionary = read_input_data()
    x_train, y_train = shuffle(x_train, y_train)
    print(labels_dictionary)
    model(len(labels_dictionary), x_train[0].shape)