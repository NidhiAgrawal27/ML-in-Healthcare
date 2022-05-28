from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    SimpleRNN,
    AveragePooling1D,
    Flatten,
    Dropout,
    Dense,
    GRU,
)
from tensorflow.keras import optimizers
from utils.utils import get_act_loss_logits

PARAMS = {"lr": 0.001, "dropout_linear": 0, "rnn_dim": 512}


def get_model(
    n_classes,
    rnn_type,
    activation="tanh",
    clip_grad=False,
    params=PARAMS,
):
    """
    rnn_type should be one of "rnn", "gru"
    Based on empirical results we use tanh for GRU but relu for RNN
    """
    final_activation, loss, n_logits = get_act_loss_logits(n_classes)
    if rnn_type == "rnn":
        rnn_constr = SimpleRNN
    elif rnn_type == "gru":
        rnn_constr = GRU
    else:
        raise ValueError("Unknown RNN type")

    model = Sequential()
    model.add(
        rnn_constr(
            512, input_shape=(187, 1), activation=activation, return_sequences=True
        )
    )
    model.add(rnn_constr(512, activation=activation, return_sequences=True))
    model.add(rnn_constr(512, activation=activation, return_sequences=True))
    model.add(rnn_constr(512, activation=activation, return_sequences=True))
    model.add(AveragePooling1D(pool_size=187))
    model.add(Flatten())
    model.add(Dropout(params["dropout_linear"]))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(n_logits, activation=final_activation))
    model.build()

    if clip_grad:
        opt = optimizers.Adam(params["lr"], clipvalue=0.5)
    else:
        opt = optimizers.Adam(params["lr"])

    model.compile(optimizer=opt, loss=loss, metrics=["acc"])
    model.summary()
    return model
