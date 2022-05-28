from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from utils.utils import get_logs
from utils.evaluation import evaluate as write_evaluation


def get_class_weights(dataset):
    counts = [0 for _ in range(5)]
    for sample in dataset:
        counts[sample[-1]] += 1
    return torch.tensor([max(counts) / c for c in counts])


def to_device(x, y, device):
    x = {k: v.to(device) for k, v in x.items()}
    y = y.to(device)
    return x, y


def log(log_files, df, epoch, step, losses, preds, labels, lr, val_metrics=None):
    df = df.append(
        {
            "epoch": epoch,
            "step": step,
            "acc": accuracy_score(labels, preds),
            "loss": np.mean(losses),
            "lr": lr,
            "val_acc": None if val_metrics is None else val_metrics["accuracy"],
            "val_loss": None if val_metrics is None else val_metrics["loss"],
        },
        ignore_index=True,
    )
    df.to_csv(log_files["history"])
    return df


def train(
    model,
    train_set,
    val_set,
    n_epochs,
    lr,
    batch_size,
    es_patience,
    lr_patience,
    log_files,
    device,
    log_steps=1000,
    **kwargs,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=lr_patience, verbose=True
    )
    dataloader = DataLoader(
        train_set,
        collate_fn=train_set.collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_losses = []
    class_weights = get_class_weights(train_set)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)

    # Setup logging
    all_losses = []
    all_preds = []
    all_labels = []
    df_log = pd.DataFrame(
        columns=["epoch", "step", "acc", "loss", "lr", "val_acc", "val_loss"]
    )

    step = 0
    for epoch in range(n_epochs):
        print("Epoch {}".format(epoch))
        model.train()

        for x, y in tqdm(dataloader):
            x, y = to_device(x, y, device)
            preds = model(x)
            loss = loss_fn(preds, y)

            # Logging
            all_losses.append(float(loss))
            all_preds += torch.argmax(preds, dim=1).cpu().numpy().tolist()
            all_labels += y.cpu().numpy().tolist()
            if step % log_steps == 0:
                df_log = log(
                    log_files,
                    df_log,
                    epoch,
                    step,
                    losses=all_losses,
                    preds=all_preds,
                    labels=all_labels,
                    lr=optimizer.param_groups[0]["lr"],
                )

            # Step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

        val_metrics = validate(
            model, val_set, batch_size=batch_size, loss_fn=loss_fn, device=device
        )
        scheduler.step(val_metrics["loss"])
        print(
            "Validation loss: {loss}, accuracy: {accuracy}, f1: {f1}".format(
                **val_metrics
            )
        )
        df_log = log(
            log_files,
            df_log,
            epoch,
            step,
            losses=all_losses,
            preds=all_preds,
            labels=all_labels,
            lr=optimizer.param_groups[0]["lr"],
            val_metrics=val_metrics,
        )
        val_losses.append(val_metrics["loss"])

        # Save model if has best val loss so far
        if epoch == 0 or val_losses[-1] < min(val_losses[:-1]):
            torch.save(model.state_dict(), log_files["checkpoint"])

        # Early stopping
        if epoch >= es_patience and min(val_losses[-es_patience:]) == val_losses[-1]:
            print("Early stopping")
            break


def test(model, test_set, batch_size, device, log_files):
    metrics = validate(
        model, test_set, batch_size=batch_size, device=device, return_preds=True
    )

    write_evaluation(
        metrics["labels"], metrics["pred"], metrics["pred_proba"], log_files["metrics"]
    )

    # log metrics, preds, pres_proba
    np.save(log_files["pred_proba"], metrics["pred_proba"])
    np.save(log_files["pred"], metrics["pred"])


def validate(model, val_set, batch_size, device, loss_fn=None, return_preds=False):
    """
    Runs validation and returns dict with loss, accuracy, and f1 score (macro).
    """
    model.eval()
    dataloader = DataLoader(
        val_set,
        collate_fn=val_set.collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    total_loss = 0
    all_preds = []
    all_pred_probas = []
    all_labels = []
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = to_device(x, y, device)
            preds = model(x)
            if loss_fn is not None:
                loss = loss_fn(preds, y)
                total_loss += float(loss)
            all_preds += torch.argmax(preds, dim=1).cpu().numpy().tolist()
            all_pred_probas += preds.cpu().numpy().tolist()
            all_labels += y.cpu().numpy().tolist()

    metrics = {
        "loss": total_loss / len(dataloader) if loss_fn is not None else None,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted"),
    }

    if return_preds:
        metrics["pred"] = np.array(all_preds)
        metrics["pred_proba"] = np.array(all_pred_probas)
        metrics["labels"] = np.array(all_labels)

    return metrics


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def construct_model_name(config):
    # construct model_config_name containing relevant info
    name_params = [
        "model_name",
        "freeze_bert",
        "n_epochs",
        "dropout",
        "lr",
        "batch_size",
        "es_patience",
        "lr_patience",
        "extra_layers",
        "include_index",
    ]
    return "__".join(
        [str(config[p]).replace(".", "-").replace("/", "-") for p in name_params]
    )
