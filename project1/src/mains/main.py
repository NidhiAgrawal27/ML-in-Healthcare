import argparse
from models import (
    attention_cnn,
    baseline_mitbih,
    baseline_ptbdb,
    ensemble_model_averaging,
    ensemble_stacking,
    extra_trees_mitbih,
    extra_trees_ptbdb,
    our_cnn,
    paper_cnn,
    res_cnn,
    rnn,
)
from utils import (
    utils,
    train_tf,
    train_ensemble_model_averaging,
    train_ensemble_stacking,
    train_sklearn,
)

PARAMS = {
    "log_dir": "../logs/",
    "epochs": 1000,
    "es_patience": 7,
    "lr_patience": 3,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--dataset", type=str, help="ptbdb or mitbih")
    parser.add_argument("--model_name", type=str, help="name of model")
    parser.add_argument(
        "--loss_weight",
        default=False,
        action="store_true",
        help="whether to weigh loss to combat imbalance",
    )
    args = parser.parse_args()

    utils.set_seed(args.seed)

    if args.dataset == "ptbdb":
        n_classes = 2
    elif args.dataset == "mitbih":
        n_classes = 5
    else:
        raise ValueError("Unknown dataset")

    if args.loss_weight:
        model_log_name = args.model_name + "_weightedloss_" + args.dataset
    else:
        model_log_name = args.model_name + "_" + args.dataset
    params = {**PARAMS, "model_name": model_log_name}

    # ensemble and tree models
    if args.model_name == "ensemble_model_averaging":
        assert not args.loss_weight, "Only available for tf models"
        model = ensemble_model_averaging.get_model(
            log_dir=PARAMS["log_dir"], dataset=args.dataset
        )
        train_ensemble_model_averaging.train_and_evaluate(
            dataset=args.dataset, model=model, params=params, seed=args.seed
        )
        return
    elif args.model_name == "ensemble_stacking":
        assert not args.loss_weight, "Only available for tf models"
        model = ensemble_stacking.get_model(
            log_dir=PARAMS["log_dir"], dataset=args.dataset
        )
        train_ensemble_stacking.train_and_evaluate(
            dataset=args.dataset, model=model, params=params, seed=args.seed
        )
        return
    elif args.model_name == "extra_trees":
        assert not args.loss_weight, "Only available for tf models"
        if args.dataset == "mitbih":
            model = extra_trees_mitbih.get_model()
        else:
            model = extra_trees_ptbdb.get_model()
        train_sklearn.train_and_evaluate(
            dataset=args.dataset, model=model, params=params, seed=args.seed
        )
        return

    # rnn models
    if args.model_name[:3] in ["rnn", "gru"]:
        # have option relu and clip. Default is tanh and no clipping
        # e.g. rnn_relu_clip, gru_clip
        if "relu" in args.model_name:
            activation = "relu"
        else:
            activation = "tanh"
        clip = "clip" in args.model_name
        model = rnn.get_model(
            n_classes=n_classes,
            rnn_type=args.model_name[:3],
            activation=activation,
            clip_grad=clip,
        )
        train_tf.train_and_evaluate(
            dataset=args.dataset,
            model=model,
            params=params,
            seed=args.seed,
            fit_params={"batch_size": 256},
            weighted_loss=args.loss_weight,
        )
        return

    # standard deep learning models
    if args.model_name == "res_cnn":
        model = res_cnn.get_model(n_classes=n_classes)
    elif args.model_name == "our_cnn":
        model = our_cnn.get_model(n_classes=n_classes)
    elif args.model_name == "paper_cnn":
        model = paper_cnn.get_model(n_classes=n_classes)
    elif args.model_name == "baseline":
        params = {**params, "es_patience": 5}  # baseline uses different patience
        if args.dataset == "mitbih":
            model = baseline_mitbih.get_model()
        else:
            model = baseline_ptbdb.get_model()
    elif args.model_name == "attention_cnn":
        model = attention_cnn.get_model(n_classes=n_classes)
    else:
        raise ValueError("Unknown model")

    train_tf.train_and_evaluate(
        dataset=args.dataset,
        model=model,
        params=params,
        seed=args.seed,
        weighted_loss=args.loss_weight,
    )


if __name__ == "__main__":
    main()
