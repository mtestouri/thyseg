from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

import numpy as np
import cv2
import os

class Segmenter:
    def __init__(self, input_shape=(512, 512, 3)):
        self.input_shape = input_shape
        self.model = unet(input_shape)

    def train(self, x_train, y_train, weights_file=None):
        self.model.fit(x_train, y_train, epochs=5, batch_size=1)
        if(weights_file):
            self.model.save_weights(weights_file)

    def segment(self, x_test, y_test=None, weights_file=None):
        if(weights_file):
            self.model.load_weights(weights_file)
        y_predicts = self.model.predict(x_test)

        if not os.path.exists('segmentations'):
            os.makedirs('segmentations')
        for i in range(len(y_predicts)):
            img = np.concatenate((x_test[i], y_test[i], y_predicts[i]), axis=1)
            cv2.imwrite("segmentations/seg" + str(i) + ".jpg", img)
            
def unet(input_shape):
    nb_filters = 16

    inputs = Input(input_shape)
    conv1 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    nb_filters = nb_filters * 2
    conv2 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    nb_filters = nb_filters * 2
    conv3 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    nb_filters = nb_filters * 2
    conv4 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    nb_filters = nb_filters * 2
    conv5 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    nb_filters = int(nb_filters / 2)
    up6 = Conv2D(nb_filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    #crop6 = Cropping2D(cropping=((1, 0), (1, 0)))(drop4)
    #merge6 = concatenate([crop6,up6], axis = 3)
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    nb_filters = int(nb_filters / 2)
    up7 = Conv2D(nb_filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    #crop7 = Cropping2D(cropping=((2, 1), (2, 1)))(conv3)
    #merge7 = concatenate([conv3,up7], axis = 3)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    nb_filters = int(nb_filters / 2)
    up8 = Conv2D(nb_filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    #crop8 = Cropping2D(cropping=((3, 3), (3, 3)))(conv2)
    #merge8 = concatenate([crop8,up8], axis = 3)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    nb_filters = int(nb_filters / 2)
    up9 = Conv2D(nb_filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    #crop9 = Cropping2D(cropping=((6, 6), (6, 6)))(conv1)
    #merge9 = concatenate([crop9,up9], axis = 3)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(nb_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model