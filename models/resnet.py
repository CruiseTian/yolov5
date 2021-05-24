from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, ZeroPadding2D, Add
from keras.layers import Input, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.initializers import glorot_uniform


def identity_block(X, C):
    # p1 = ZeroPadding2D((2,2))
    conv1 = Conv2D(C, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, kernel_regularizer=l2(0.01))
    bn1 = BatchNormalization(axis=3)
    relu1 = Activation('relu')

    # p2 = ZeroPadding2D((1,1))
    conv2 = Conv2D(C, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.01))
    bn2 = BatchNormalization(axis=3)
    relu2 = Activation('relu')

    short_cut_conv = Conv2D(C, (1, 1), strides=(1, 1), kernel_regularizer=l2(0.01))
    short_cut_bn = BatchNormalization(axis=3)

    # p1 = p1(X)
    c1 = conv1(X)
    b1 = bn1(c1)
    a1 = relu1(b1)

    # p2 = p2(a1)
    c2 = conv2(a1)
    b2 = bn2(c2)
    a2 = relu2(b2)

    short_cut = short_cut_conv(X)
    s_c = short_cut_bn(short_cut)

    X = Add()([b2, s_c])

    X = Activation('relu')(X)
    return X

# This net structure is simplified!
# You may add X = identity_block(X, 64 * 4)  and X = identity_block(X, 64 * 8)  for better performance
def resnet_model(input_shape):
    initializer = glorot_uniform()
    X_input = Input(input_shape)

    X = Conv2D(64, (7, 7), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.01))(X_input)

    X = identity_block(X, 64)
    X = identity_block(X, 64 * 2)
    # For better performance, apply the next two lines (but can be significantly slower!)
    #X = identity_block(X, 64 * 4)
    #X = identity_block(X, 64 * 8)
    X = Flatten()(X)

    # X = Dropout(0.4)(X)
    # X = Dense(64,activation = 'relu', kernel_regularizer = l2(0.01))(X)
    # X = Dropout(0.4)(X)
    # X = Dense(14)(X)

    # X = Dropout(0.4)(X)
    # X = Dense(64, activation = 'relu',kernel_regularizer=l2(0.01))(X)
    X = Dense(14)(X)

    model = Model(inputs=X_input, outputs=X, name='resnet')

    return model