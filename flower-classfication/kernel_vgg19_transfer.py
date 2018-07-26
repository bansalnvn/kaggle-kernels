from common import *
from sklearn.utils import shuffle
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras import applications
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
# from keras import backend as KerasBackend
# from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt


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
    model_final = Model(inputs=model.input, outputs=predictions)
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
    # config = KerasBackend.tf.ConfigProto(intra_op_parallelism_threads=6, inter_op_parallelism_threads=6)
    # session = KerasBackend.tf.Session(config=config)
    # session.run(KerasBackend.tf.global_variables_initializer())
    # KerasBackend.set_session(session)
    # print(device_lib.list_local_devices())

    datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

    datagen.fit(x_train)
    print(final_model.summary())
    # history = keras_model.fit_generator(datagen.flow(np.asarray(x_train), y_train, batch_size=64),
    #                                     samples_per_epoch=len(x_train), epochs=50, validation_data=(np.asarray(
    #             x_train), y_train))
    # history = final_model.fit(x_train, np.asarray(y_train), epochs=10, batch_size=64, validation_split=0.1)
    x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(x_train, y_train, test_size=323)
    history = final_model.fit_generator(datagen.flow(x_train_split, np.asarray(y_train_split), batch_size=100),
                                        validation_data=(x_test_split, y_test_split), epochs=50)

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
