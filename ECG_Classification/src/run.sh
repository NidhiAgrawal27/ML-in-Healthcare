#!/bin/bash

for seed in {42..46}
do
    for dataset in "ptbdb" "mitbih"
    do
        # standard models, rnn variations
        for model in "extra_trees" "res_cnn" "our_cnn" "attention_cnn" "paper_cnn" "baseline" "rnn" "gru" "rnn_relu_clip" "gru_relu_clip" "rnn_clip" "gru_clip"
        do
            python -m mains.main --seed $seed --dataset $dataset --model_name $model
        done
        # loss weighting
        for model in "our_cnn"
        do
            python -m mains.main --seed $seed --dataset $dataset --model_name $model --loss_weight
        done
    done
done

# Ensemble and transfer learning after all previous runs have completed
for seed in {42..46}
do
    for dataset in "ptbdb" "mitbih"
    do
        for model in "ensemble_model_averaging" "ensemble_stacking"
        do
            python -m mains.main --seed $seed --dataset $dataset --model_name $model
        done
    done
    python -m mains.main_transfer --seed $seed --model_name our_cnn --load_model y
    python -m mains.main_transfer --seed $seed --model_name gru_clip --load_model y
done

python -m visualization.visualize
