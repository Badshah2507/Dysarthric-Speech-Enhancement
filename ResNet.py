from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import *

def UNet(input_size):
    inputs = Input(shape=input_size)

    conv1 = Conv2D(8, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(16, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2) 

    conv3 = Conv2D(32, 3, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3) 

    conv4 = Conv2D(64, 3, activation='relu', padding='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

    conv5 = Conv2D(128, 3, activation='relu', padding='same')(pool4)
    pool5 = MaxPooling2D(pool_size=(2,2))(conv5)

    conv6 = Conv2D(256, 3, activation='relu', padding='same')(pool5)
    pool6 = MaxPooling2D(pool_size=(2,2))(conv6)

    up7 = Conv2DTranspose(128, 3, strides=(2,2), activation='relu', padding='valid')(pool6) 
    sc7 = Concatenate()([up7,pool5])

    up8 = Conv2DTranspose(64, 3, strides=(2,2), activation='relu', padding='valid')(sc7)
    sc8 = Concatenate()([up8,pool4])

    up9 = Conv2DTranspose(32, 3, strides=(2,2), activation='relu', padding='valid')(sc8) 
    sc9 = Concatenate()([up9,pool3])

    up10 = Conv2DTranspose(16, 3, strides=(2,2), activation='relu', padding='valid')(sc9) 
    sc10 = Concatenate()([up10,pool2])

    up11 = Conv2DTranspose(8, 3, strides=(2,2), activation='relu', padding='valid')(sc10) 
    sc11 = Concatenate()([up11,pool1])

    up12 = Conv2DTranspose(1, 3, strides=(2,2), activation='relu', padding='valid')(sc11)

    model = Model(inputs, up12)

    model.compile(optimizer='adam', metrics=['accuracy'], loss=tf.keras.losses.MeanSquaredError())

    return model