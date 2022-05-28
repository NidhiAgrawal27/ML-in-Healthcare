#!/bin/bash

# Run simple models
for seed in {42..46}
do
    python -m mains.main_random_forest --seed $seed
    python -m mains.main_decision_tree --seed $seed
    python -m mains.main_logistic_regression --seed $seed
done

# Run deep learning models (only once)
python -m mains.main_baseline_cnn_shap --seed 0
python -m mains.main_transfer_learning_shap --seed 0
python -m mains.main_baseline_cnn_lime --seed 0

# generate visualizations
python -m visualization.visualize
