#!/bin/bash

# make sure to run the following commands for loading the proper python version before executing
# (have to be run in shell thus not in script):
#   env2lmod
#   module load gcc/6.3.0 python_gpu/3.8.5 hdf5/1.10.1

for seed in {42..46}
do
    for dataset in "ptbdb" "mitbih"
    do
        # standard models
        for model in "extra_trees" "res_cnn" "our_cnn" "attention_cnn" "paper_cnn" "baseline"
        do
            bsub -R "rusage[mem=8000,ngpus_excl_p=1]" python -m mains.main --seed $seed --dataset $dataset --model_name $model
        done
        # rnn variations, especially GRU runs can in rare cases take longer than standard 4h limit
        for model in "rnn" "gru" "rnn_relu_clip" "gru_relu_clip" "rnn_clip" "gru_clip"
        do
            bsub -R "rusage[mem=8000,ngpus_excl_p=1]"  -W 6:00 python -m mains.main --seed $seed --dataset $dataset --model_name $model
        done
        # loss weighting
        for model in "our_cnn"
        do
            bsub -R "rusage[mem=8000,ngpus_excl_p=1]" python -m mains.main --seed $seed --dataset $dataset --model_name $model --loss_weight
        done
    done
done
