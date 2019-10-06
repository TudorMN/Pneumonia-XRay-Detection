import ts_ignore
import img_utils
import networks
import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D, Input, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras.models import load_model
from keras.models import clone_model
from keras.callbacks import EarlyStopping
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.optimizers import SGD
from keras import optimizers

import numpy as np


def training():
    print("----------------------")
    x_train, y_train, x_test, y_test = img_utils.load_images()
    input_shape = (img_utils.img_height, img_utils.img_width, 3)

    print("-- Training --")
    model = networks.cnn_model(input_shape)
    optimzer = optimizers.Adadelta(lr=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimzer, metrics=['accuracy'])
    model.fit(np.array(x_train), np.array(y_train), batch_size=64, epochs=60, verbose=1, shuffle=True)

    metric_loss, metric_acc = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
    print('- Test set loss: ', metric_loss)
    print('- Test set accuracy: ', metric_acc)
    print("----------------------")

training()