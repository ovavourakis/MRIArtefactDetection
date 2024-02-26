from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv3D, MaxPooling3D, Input

def getConvNet(out_classes=2,input_shape=(256,256,64,1)):
    inp = Input(input_shape)

    # 256 x 256 x 64 x 1
    x = Conv3D(8, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(inp)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = Conv3D(8, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
   
    # 128 x 128 x 32 x 8
    x = Conv3D(16, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = Conv3D(16, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # 64 x 64 x 16 x 16
    x = Conv3D(32, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = Conv3D(32, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # 32 x 32 x 8 x 32
    x = Conv3D(64, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = Conv3D(64, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # 16 x 16 x 4 x 64
    x = Conv3D(128, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = Conv3D(128, (3,3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)

    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # 8 x 8 x 2 x 128
    # x = Dropout(0.25)(x,training=True) # = always apply dropout (permanent 'training mode')
    x = Flatten()(x)
    
    # 16384 x 1
    x = Dense(128,name='dense_pre')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    # 128 x 1
    x = Dropout(0.5)(x, training=True) # = always apply dropout (permanent 'training mode')
    out = Dense(out_classes, name='dense_post', activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    return model