import cv2
import os
import random
import keras
import numpy as np
from keras.preprocessing.image import img_to_array

img_height, img_width = 128, 128

def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img_aux = cv2.resize(img, (img_height, img_width))
            img_aux = img_to_array(img_aux) / 255.0
            images.append(img_aux.astype('float32'))

    return images

def load_image_sets(shuffled_images, samples):
    images = np.empty((samples, img_height, img_width, 3))

    for i in range(samples):
        img = shuffled_images[i]
        images[i] = img

    return images

def prepare_sets(imgs_normal, imgs_pneumonia):
    x = np.vstack((imgs_normal,imgs_pneumonia))

    y_normal = [0 for i in range(len(imgs_normal))]
    y_pneumonia = [1 for i in range(len(imgs_pneumonia))]
    y = y_normal + y_pneumonia
    y = keras.utils.to_categorical(y)

    aux_list = list(zip(x, y))
    random.shuffle(aux_list)
    x, y = zip(*aux_list)

    x = load_image_sets(x, len(x))

    return x, y

def load_images():
    print("-- Loading datasets --")
    imgs_train_normal = load_images_from_folder("data/train/NORMAL/")
    print("- Loaded data/train/NORMAL/")
    imgs_train_pneumonia = load_images_from_folder("data/train/PNEUMONIA/")
    print("- Loaded data/train/PNEUMONIA/")
    imgs_test_normal = load_images_from_folder("data/test/NORMAL/")
    print("- Loaded data/test/NORMAL/")
    imgs_test_pneumonia = load_images_from_folder("data/test/PNEUMONIA/")
    print("- Loaded data/test/PNEUMONIA/")

    x_train, y_train = prepare_sets(imgs_train_normal, imgs_train_pneumonia)
    x_test, y_test = prepare_sets(imgs_test_normal, imgs_test_pneumonia)

    print("- x_train size: {}".format(x_train.shape))
    print("- y_train size: {}".format(len(y_train)))
    print("- x_test size: {}".format(x_test.shape))
    print("- y_test size: {}".format(len(y_test)))
    print("-- Finished loading --")

    return x_train, y_train, x_test, y_test