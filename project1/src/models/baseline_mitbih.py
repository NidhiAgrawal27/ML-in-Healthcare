from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate


def get_model():
    nclass = 5
    inp = Input(shape=(187, 1))
    img_1 = Convolution1D(16,
                          kernel_size=5,
                          activation=activations.relu,
                          padding="valid")(inp)
    img_1 = Convolution1D(16,
                          kernel_size=5,
                          activation=activations.relu,
                          padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32,
                          kernel_size=3,
                          activation=activations.relu,
                          padding="valid")(img_1)
    img_1 = Convolution1D(32,
                          kernel_size=3,
                          activation=activations.relu,
                          padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32,
                          kernel_size=3,
                          activation=activations.relu,
                          padding="valid")(img_1)
    img_1 = Convolution1D(32,
                          kernel_size=3,
                          activation=activations.relu,
                          padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256,
                          kernel_size=3,
                          activation=activations.relu,
                          padding="valid")(img_1)
    img_1 = Convolution1D(256,
                          kernel_size=3,
                          activation=activations.relu,
                          padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass,
                    activation=activations.softmax,
                    name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt,
                  loss=losses.sparse_categorical_crossentropy,
                  metrics=['acc'])
    model.summary()
    return model
