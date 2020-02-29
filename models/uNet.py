import keras


def unet(lInputSize  = (256,256,3), iNumOutputCh = 1, bPadding = True):
    """ 
    Keras implementation of https://arxiv.org/abs/1505.04597. 
    
    Inputs:
            lInputSize: List. Size of the input image
            iNumOutputCh: integer. Number of channels in the output segmentation. This may
                        be useful if you need a 3 channel segmentation where each colour channel is a different class.
                        Multiple class segmentation can also be achieved by using 1 colour channel but setting the values in the mask
                        to be 1: class 1, 2: class 2, 3: class 3... etc. 
            bPadding: Boolean. Indicates if padding will or will not be used in the convolutional layers. By
                        adding padding, the resulting segmentation map will be the same dimensions (X and Y) as the input map
    

    
    """
    # determine if padding will or will not be used
    if bPadding == True:
        sPadding = 'same' # indicates input will be the same dimensions of the output for a conv2d layer
    else:
        sPadding = 'valid' # padding = 'valid' in conv2d layer means padding is not used
    
    inputs = keras.layers.Input(lInputSize)
    conv1 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(inputs)
    conv1 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(pool1)
    conv2 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(pool2)
    conv3 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(pool3)
    conv4 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(pool4)
    conv5 = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(conv5)
    # drop out layer was said to be done at the end of the contracting path
    drop5 = keras.layers.Dropout(0.2)(conv5)

    # up sampling 
    up6 = keras.layers.UpSampling2D(size = (2,2))(drop5)
    merge6 = keras.layers.concatenate([conv4, up6], axis = 3)
    conv6 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(merge6)
    conv6 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(conv6)

    up7 = keras.layers.UpSampling2D(size = (2,2))(conv6)
    merge7 = keras.layers.concatenate([conv3, up7], axis = 3)    
    conv7 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(merge7)
    conv7 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(conv7)

    up8 = keras.layers.UpSampling2D(size = (2,2))(conv7)
    merge8 = keras.layers.concatenate([conv2, up8], axis = 3)
    conv8 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(merge8)
    conv8 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(conv8)

    up9 = keras.layers.UpSampling2D(size = (2,2))(conv8)
    merge9 = keras.layers.concatenate([conv1, up9], axis = 3)
    conv9 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(merge9)
    conv9 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(conv9)
    conv9 = keras.layers.Conv2D(2, 2, activation = 'relu', padding = sPadding, kernel_initializer = 'he_normal')(conv9)
    conv10 = keras.layers.Conv2D(iNumOutputCh, 1, activation = 'sigmoid', padding = sPadding)(conv9)

    model = keras.models.Model(input = inputs, output = conv10)

    return model

