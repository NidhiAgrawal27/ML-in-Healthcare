import pathlib
import pickle
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorflow.keras.models import load_model

from models import transfer_learning
from utilities import utils, evaluation, data
from visualization import shap_func


CONFIG = {"log_dir": "../logs/", 
            "figure_dir": "../logs/figures/shap_transfer_learning/",
            "model_name": "transfer_learning"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seed", required=True)
    args = parser.parse_args()

    utils.set_seed(args.seed)

    (checkpoint_filename, _, pred_filename, pred_proba_filename,
     metrics_filename) = (utils.get_logs(log_dir=CONFIG["log_dir"],
                                         model_name=CONFIG["model_name"],
                                         seed=args.seed,
                                         model_ext=".pkl"))

    # Load data
    train_dataset, val_dataset, test_dataset = data.get_img_dataset(transform=[
                                                transforms.RandomHorizontalFlip(0.2),
                                                transforms.RandomVerticalFlip(0.2),
                                                transforms.RandomAutocontrast(0.5),
                                                transforms.RandomAdjustSharpness(0.5)
                                            ])

    train_dataloader = DataLoader(train_dataset, batch_size = len(train_dataset), shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = len(val_dataset), shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = len(test_dataset), shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    val_features, val_labels = next(iter(val_dataloader))
    test_features, test_labels = next(iter(test_dataloader))

    if torch.cuda.is_available():
        for n in [train_features, train_labels, val_features, val_labels, test_features, test_labels]: n = n.cuda()

    train_x = torch.swapaxes(train_features,1,-1)
    train_x = train_x.cpu().numpy()
    train_y = train_labels.cpu().numpy()

    X_val = torch.swapaxes(val_features,1,-1)
    X_val = X_val.cpu().numpy()
    y_val = val_labels.cpu().numpy()

    X_train = np.concatenate([train_x, X_val], axis=0)
    y_train = np.concatenate([train_y, y_val], axis=0)

    X_test = torch.swapaxes(torch.swapaxes(test_features,1,-1), 1,2)
    X_test = X_test.cpu().numpy()
    y_test = test_labels.cpu().numpy()

    input_shape = X_train.shape[1:4]

    # Retrieve model.
    model = transfer_learning.get_transfer_model(input_shape)

    # Train final model on both trainset and devset.
    start_time = time.time()
    model = transfer_learning.model_fit(model, X_train, y_train)
    elapsed_time = time.time() - start_time

    # Predict and evaluate on testset.
    pred_proba = model.predict(X_test)
    pred = np.asarray([(v>0.5).astype(np.int8) for val in pred_proba for v in val])
    np.save(pred_proba_filename, pred_proba)
    np.save(pred_filename, pred)

    model.save('transfer_learning.h5')
    model = load_model('transfer_learning.h5')

    utils.log_training_time(elapsed_time, metrics_filename)
    evaluation.evaluate(y_test, pred, pred_proba, metrics_filename)

    # SHAP values and visualization
    
    # Create figure directory if it does not exist.
    pathlib.Path(CONFIG["figure_dir"]).mkdir(parents=True, exist_ok=True)

    classes = {'0': 'No Tumour', '1': 'Tumour'}
    shap_func.shap_numpy(model, X_test, y_test, pred.reshape(-1,1), classes, CONFIG["figure_dir"]+'transfer_learning_')
    

if __name__ == "__main__":
    main()

