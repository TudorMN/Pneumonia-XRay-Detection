from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D, Input, GlobalAveragePooling2D, BatchNormalization
from keras.layers.convolutional import Conv2D

def cnn_model(input_shape):
    input = Input(shape=input_shape)
    model = Conv2D(32,(3,3),padding='same', activation='relu')(input)
    model = Conv2D(32, (3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.2)(model)
    
    model = Conv2D(32, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(64, (3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.25)(model)

    model = Conv2D(64, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(128, (3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.3)(model)

    model = Flatten()(model)
    model = Dense(512, activation='relu')(model)
    model = Dropout(0.4)(model)
    output = Dense(2, activation='softmax')(model)
    
    model = Model(inputs=input, outputs=output)

    return model