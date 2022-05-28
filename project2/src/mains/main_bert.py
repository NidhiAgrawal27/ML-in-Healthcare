import argparse
import os

import transformers

from bert.dataset import PubMedBERTDataset
from bert.model import BERTClassifier
from bert.train import *

CONFIG = {
    "seed": 0,
    # Model
    "model_name": "emilyalsentzer/Bio_ClinicalBERT",
    "dropout": 0.1,
    "freeze_bert": True,
    # Learning Params
    "n_epochs": 20,
    "batch_size": 32,
    # Patience
    "es_patience": 5,
    "lr_patience": 3,
    # Paths
    "data_path": "../data/pubmed-rct/PubMed_200k_RCT/",
    "log_path": "../logs",
}


def main():
    config = CONFIG

    # parse freeze bert option
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze_bert",
        default=False,
        action="store_true",
        help="whether to freeze bert model",
    )
    parser.add_argument(
        "--extra_layers",
        default=False,
        action="store_true",
        help="whether to add extra layers after bert model",
    )
    parser.add_argument(
        "--include_index",
        default=False,
        action="store_true",
        help="whether to include the index of the sentence as feature",
    )
    parser.add_argument(
        "--test_only",
        default=False,
        action="store_true",
        help="whether to only load last model and run evaluation",
    )
    parser.add_argument(
        "--build_only",
        default=False,
        action="store_true",
        help="whether to only build model to download weights etc. from huggingface",
    )
    args = parser.parse_args()
    config["freeze_bert"] = args.freeze_bert
    config["extra_layers"] = args.extra_layers
    config["include_index"] = args.include_index

    # Use GPU if available
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # set lr, dependent on whether bert trained or not
    if config["freeze_bert"]:
        config["lr"] = 0.001  # standard Adam lr
    else:
        config["lr"] = 3e-5  # bert finetuning lr

    # Build model
    model = BERTClassifier(
        config["model_name"],
        dropout=config["dropout"],
        freeze_bert=config["freeze_bert"],
        extra_layers=config["extra_layers"],
        include_index=config["include_index"],
    )

    # return if only building model
    if args.build_only:
        # make sure to download tokenizer
        tokenizer = transformers.BertTokenizer.from_pretrained(config["model_name"])
        return

    # Setup log files
    if args.test_only:
        model_dir = os.path.join(
            config["log_path"],
            construct_model_name(config),
            str(config["seed"]),
        )
        run_dirs = os.listdir(model_dir)
        # take latest run
        assert len(run_dirs) > 0, "No runs found"
        run_dir = sorted(run_dirs)[-1]

        # get log files
        config["log_files"] = {
            "checkpoint": os.path.join(model_dir, run_dir, "model.pt"),
            "history": os.path.join(model_dir, run_dir, "history.csv"),
            "pred": os.path.join(model_dir, run_dir, "pred.npy"),
            "pred_proba": os.path.join(model_dir, run_dir, "pred_proba.npy"),
            "metrics": os.path.join(model_dir, run_dir, "metrics.txt"),
        }
    else:
        log_files = get_logs(
            config["log_path"],
            model_name=construct_model_name(config),
            seed=config["seed"],
            torch_model=True,
        )
        config["log_files"] = {
            k: v
            for k, v in zip(
                ["checkpoint", "history", "pred", "pred_proba", "metrics"],
                log_files,
            )
        }

    # Load data
    if not args.test_only:
        train_set = PubMedBERTDataset(
            config["model_name"],
            config["data_path"] + "train.txt",
            include_index=config["include_index"],
        )
        val_set = PubMedBERTDataset(
            config["model_name"],
            config["data_path"] + "dev.txt",
            include_index=config["include_index"],
        )
    test_set = PubMedBERTDataset(
        config["model_name"],
        config["data_path"] + "test.txt",
        include_index=config["include_index"],
    )

    # Train and test
    model.to(config["device"])
    if not args.test_only:
        train(
            model,
            train_set,
            val_set,
            **config,
        )
    else:
        model.load_state_dict(torch.load(config["log_files"]["checkpoint"]))

    test(
        model,
        test_set=test_set,
        batch_size=config["batch_size"],
        device=config["device"],
        log_files=config["log_files"],
    )


if __name__ == "__main__":
    main()
