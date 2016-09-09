#!/usr/bin/env python

# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
from keras.models import Sequential
from keras.layers import (Convolution2D, MaxPooling2D, Dense,
                          Activation, Dropout, Flatten)
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
import pydot
from numpy import savetxt

# (256, 256) didnt work when trying to fit the model
TARGET_SIZE = (150, 150)


def load_data(img_path, **kwargs):
    data_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    gen = data_gen.flow_from_directory(
        img_path,
        batch_size=100,
        target_size=TARGET_SIZE,
        class_mode='binary',
        **kwargs)
    return gen


def new_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, 150, 150)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

# need to compile
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    print('******returning model')
    return model


def visual():
    model = new_model()
    plot(model, to_file='model.png',
         show_shapes=True, show_layer_names=True)


def main():
    train_gen = load_data("../input/train")

    model = new_model()

    model.fit_generator(train_gen, 2000, 50)

    # save the model
    with open('model.json', 'w') as f:
        f.write(model.to_json())

    test_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_data = test_gen.flow_from_directory(
        '../input/test',
        target_size=TARGET_SIZE,
        class_mode='binary')
    predictions = model.predict_generator(test_data, test_data.N)
    savetxt('predictions.csv', predictions)


if __name__ == '__main__':
    main()
    # visual()
