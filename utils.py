from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np
import argparse

def setup_generator(train_path, test_path, batch_size, dimentions):
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_path,  # this is the target directory
        target_size=dimentions,
        batch_size=batch_size)

    validation_generator = test_datagen.flow_from_directory(
        test_path, # this is the target directory
        target_size=dimentions,
        batch_size=batch_size)

    return train_generator, validation_generator

def load_image(img_path, dimentions, rescale=1. / 255):
    img = load_img(img_path, target_size=dimentions)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x *= rescale # rescale the same as when trained

    return x
