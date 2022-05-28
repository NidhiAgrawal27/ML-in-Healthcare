from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    Flatten,
    Dropout,
    Dense,
    MaxPool1D,
)
from tensorflow.keras import optimizers, losses, activations, models

PARAMS = {"dropout_linear": 0.1, "lr": 0.001}


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

    model = Sequential()

    model.add(
        Conv1D(
            128, kernel_size=7, activation="relu", input_shape=(187, 1), padding="valid"
        )
    )
    model.add(Conv1D(128, kernel_size=9, activation="relu", padding="valid"))
    model.add(MaxPool1D(pool_size=3))
    model.add(Conv1D(256, kernel_size=11, activation="relu", padding="valid"))
    model.add(Conv1D(256, kernel_size=11, activation="relu", padding="valid"))
    model.add(MaxPool1D(pool_size=3))
    model.add(Flatten())
    model.add(Dropout(params["dropout_linear"]))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(params["dropout_linear"]))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(params["dropout_linear"]))
    model.add(Dense(n_logits, activation=final_activation))
    model.build()

    opt = optimizers.Adam(params["lr"])
    model.compile(optimizer=opt, loss=loss, metrics=["acc"])
    model.summary()
    return model
