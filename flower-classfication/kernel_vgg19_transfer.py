from common import *
from sklearn.utils import shuffle
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras import applications
from keras import optimizers


def model(num_classes, input_shape):
    model = applications.VGG19(weights="imagenet", include_top=False, input_shape=input_shape)
    print(model.summary())
    for layer in model.layers[:20]:
        layer.trainable = False
    print(model.summary())
    x = model.output
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    # creating the final model
    model_final = Model(input=model.input, output=predictions)
    model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                        metrics=["accuracy"])
    print(model_final.summary())
    return model_final


if __name__ == "__main__":
    _, y_train, n_classes, labels_dictionary, x_train = read_input_data()
    x_train, y_train = shuffle(x_train, y_train)
    print(labels_dictionary)
    print(x_train.shape)
    final_model = model(len(labels_dictionary), x_train[0].shape)
    final_model.fit(x_train, np.asarray(y_train), epochs=10, batch_size=64)