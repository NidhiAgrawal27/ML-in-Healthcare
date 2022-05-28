from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import (
    Dense,
    Input,
    Dropout,
    MaxPool1D,
    GlobalMaxPool1D,
    MultiHeadAttention,
    LayerNormalization,
)
from models.res_cnn import res_block

PARAMS = {
    # CNN
    "dropout_cnn": 0.1,
    "dropout_linear": 0.2,
    "batch_norm": False,
    # Attention
    "num_heads": 4,
    "key_dim": 32,
    "dropout": 0.1,
    # Training
    "lr": 0.001,
}


def attention_block(x, params):
    x_att = LayerNormalization()(x)
    att_out = MultiHeadAttention(
        num_heads=params["num_heads"],
        key_dim=params["key_dim"],
        value_dim=x.shape[-1],
        dropout=params["dropout"],
    )(x_att, x_att)
    return x + att_out


def attention_res_cnn(x, params):
    x = res_block(x, kernel_size=5, n_channels=16, params=params)
    x = attention_block(x, params=params)
    x = MaxPool1D(pool_size=2)(x)

    x = res_block(x, kernel_size=3, n_channels=32, params=params)
    x = attention_block(x, params=params)
    x = MaxPool1D(pool_size=2)(x)

    x = res_block(x, kernel_size=3, n_channels=64, params=params)
    x = attention_block(x, params=params)
    x = MaxPool1D(pool_size=2)(x)

    x = res_block(x, kernel_size=3, n_channels=256, params=params)
    x = attention_block(x, params=params)

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

    x = attention_res_cnn(inp, params=params)

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
