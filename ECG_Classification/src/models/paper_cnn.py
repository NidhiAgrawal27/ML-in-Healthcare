# CNN architecture following https://arxiv.org/pdf/1805.00794.pdf
# with Dropout (was not in paper)

from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import (
    Dense,
    Input,
    Convolution1D,
    MaxPool1D,
    ReLU,
    Flatten,
    Dropout,
)

PARAMS = {"lr": 0.001, "dropout_cnn": 0.1}


def block(x, params):
    x = Dropout(rate=params["dropout_cnn"])(x)
    x_new = Convolution1D(32, kernel_size=5, activation=None, padding="same")(x)
    x_new = ReLU()(x_new)
    x_new = Convolution1D(32, kernel_size=5, activation=None, padding="same")(x_new)
    x = x + x_new  # skip connection
    x = ReLU()(x)
    x = MaxPool1D(pool_size=5, strides=2)(x)
    return x


def get_model(n_classes, params=PARAMS):

    if n_classes == 2:
        final_activation = activations.sigmoid
        loss = losses.binary_crossentropy
        n_logits = 1
    elif n_classes > 2:
        final_activation = activations.softmax
        loss = losses.sparse_categorical_crossentropy
        n_logits = n_classes
    else:
        raise ValueError("Need to predict at least 2 classes")

    inp = Input(shape=(187, 1))

    # first conv layer (no activation mentioned in paper here)
    x = Convolution1D(32, kernel_size=5, activation=None, padding="same")(inp)

    # 5 residual blocks
    x = block(x, params=params)
    x = block(x, params=params)
    x = block(x, params=params)
    x = block(x, params=params)
    x = block(x, params=params)

    # flatten
    x = Flatten()(x)

    # linear layers for prediction
    x = Dense(32, activation=activations.relu, name="dense_1")(x)
    outputs = Dense(n_logits, activation=final_activation, name="dense_2")(x)

    model = models.Model(inputs=inp, outputs=outputs)
    opt = optimizers.Adam(params["lr"])

    model.compile(optimizer=opt, loss=loss, metrics=["acc"])
    model.summary()
    return model
