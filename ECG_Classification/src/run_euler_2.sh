#!/bin/bash

# make sure to run the following commands for loading the proper python version before executing
# (have to be run in shell thus not in script):
#   env2lmod
#   module load gcc/6.3.0 python_gpu/3.8.5 hdf5/1.10.1

for seed in {42..46}
do
    for dataset in "ptbdb" "mitbih"
    do
        for model in "ensemble_model_averaging" "ensemble_stacking"
        do
            bsub -R "rusage[mem=8000,ngpus_excl_p=1]" python -m mains.main --seed $seed --dataset $dataset --model_name $model
        done
    done
    bsub -R "rusage[mem=8000,ngpus_excl_p=1]" python -m mains.main_transfer --seed $seed --model_name our_cnn --load_model y
    bsub -R "rusage[mem=8000,ngpus_excl_p=1]" python -m mains.main_transfer --seed $seed --model_name gru_clip --load_model y
done
