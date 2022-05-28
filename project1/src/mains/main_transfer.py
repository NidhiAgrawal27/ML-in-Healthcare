import argparse
import os
from models import our_cnn, rnn
from utils import utils, train_tf


params = {
    "log_dir": "../logs/",
    "epochs": 1000,
    "es_patience": 7,
    "lr_patience": 3,
    "load_from_checkpoint": False,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--model_name", type=str, help="name of model")
    parser.add_argument(
        "--load_model", type=str, help="whether to load the model from checkpoint [y/n]"
    )
    args = parser.parse_args()

    utils.set_seed(args.seed)

    if args.model_name == "our_cnn":
        mitbih_model = our_cnn.get_model(n_classes=5)
        mitbih_model_name = "our_cnn_mitbih"

        ptbdb_model = our_cnn.get_model(n_classes=2)
        ptbdb_model_name = "transfer_our_cnn_ptbdb"

        layer_condition = lambda l: "conv" in l.name
    elif args.model_name == "gru_clip":
        mitbih_model = rnn.get_model(n_classes=5, rnn_type="gru")
        mitbih_model_name = "gru_clip_mitbih"

        ptbdb_model = rnn.get_model(n_classes=2, rnn_type="gru")
        ptbdb_model_name = "transfer_gru_clip_ptbdb"

        layer_condition = lambda l: "gru" in l.name
    else:
        raise ValueError("Unknown model")

    # Train model on bigger mitbih dataset
    if args.load_model == "y":
        experiment_dir = os.path.join(
            params["log_dir"], mitbih_model_name, str(args.seed)
        )
        if not (os.path.exists(experiment_dir) and len(os.listdir(experiment_dir)) > 0):
            raise FileNotFoundError("Could not find checkpoint file")

        # load trained model from existing checkpoint
        print("Loading mitbih model weights from checkpoint")
        mitbih_checkpoint = os.path.join(
            experiment_dir, os.listdir(experiment_dir)[0], "model.h5"
        )
    elif args.load_model == "n":
        # train model on mitbih dataset
        print("Training mitbih model")
        mitbih_checkpoint = train_tf.train_and_evaluate(
            dataset="mitbih",
            model=mitbih_model,
            params={**params, "model_name": mitbih_model_name},
            seed=args.seed,
        )
    else:
        raise ValueError("Unknown load model option")

    mitbih_model.load_weights(mitbih_checkpoint)  # make sure to use final weights

    print("\n\nTransferring weights to ptbdb model")
    # load weights and freeze for all conv layers
    for l, l_trained in zip(ptbdb_model.layers, mitbih_model.layers):
        if layer_condition(l):
            assert l.name.split("_")[0] == l_trained.name.split("_")[0]
            l.set_weights(l_trained.get_weights())
            l.trainable = False

    print("\n\nTraining last layers of ptbdb model")
    # Train final layers of ptbdb model
    ptbdb_checkpoint = train_tf.train_and_evaluate(
        dataset="ptbdb",
        model=ptbdb_model,
        params={**params, "model_name": ptbdb_model_name},
        seed=args.seed,
    )
    ptbdb_model.load_weights(ptbdb_checkpoint)  # make sure to use final weights


if __name__ == "__main__":
    main()
