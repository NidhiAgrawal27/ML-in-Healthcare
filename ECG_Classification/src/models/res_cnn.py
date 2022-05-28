from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import (
    Dense,
    Input,
    Dropout,
    Convolution1D,
    MaxPool1D,
    GlobalMaxPool1D,
    ReLU,
    BatchNormalization,
)

PARAMS = {"dropout_cnn": 0.1, "dropout_linear": 0.2, "batch_norm": False, "lr": 0.001}

# TODO search lr 0.01, 0.005, 0.001, 0.0005


def res_block(x, kernel_size, n_channels, params):
    x_new = Dropout(rate=params["dropout_cnn"])(x)
    x_new = Convolution1D(
        n_channels, kernel_size=kernel_size, activation=None, padding="same"
    )(x_new)
    if params["batch_norm"]:
        x_new = BatchNormalization()(x_new)
    x_new = ReLU()(x_new)
    x_new = Convolution1D(
        n_channels, kernel_size=kernel_size, activation=None, padding="same"
    )(x)
    if params["batch_norm"]:
        x_new = BatchNormalization()(x_new)

    # use 1x1 conv to match channels
    x_skip = Convolution1D(n_channels, kernel_size=1, activation=None, padding="same")(
        x
    )

    x_out = x_skip + x_new  # skip connection
    x_out = ReLU()(x_out)

    return x_out


def res_cnn(x, params):

    x = res_block(x, kernel_size=5, n_channels=16, params=params)
    x = res_block(x, kernel_size=5, n_channels=16, params=params)
    x = MaxPool1D(pool_size=2)(x)

    x = res_block(x, kernel_size=3, n_channels=32, params=params)
    x = res_block(x, kernel_size=3, n_channels=32, params=params)
    x = res_block(x, kernel_size=3, n_channels=32, params=params)
    x = res_block(x, kernel_size=3, n_channels=32, params=params)
    x = MaxPool1D(pool_size=2)(x)

    x = res_block(x, kernel_size=3, n_channels=256, params=params)
    x = res_block(x, kernel_size=3, n_channels=256, params=params)
    x = res_block(x, kernel_size=3, n_channels=256, params=params)
    x = res_block(x, kernel_size=3, n_channels=256, params=params)

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

    x = res_cnn(inp, params=params)

    flattened = GlobalMaxPool1D()(x)
    flattened = Dropout(rate=params["dropout_linear"])(flattened)
    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(flattened)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(n_logits, activation=final_activation, name="dense_3")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(params["lr"])

    model.compile(optimizer=opt, loss=loss, metrics=["acc"])
    model.summary()
    return model
