#!/bin/bash

seed=42

# TFIDF baseline
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 4:00 python -m mains.main_baseline --seed $seed --lemmatize
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 4:00 python -m mains.main_baseline --seed $seed --no-lemmatize

# word2vec MLP
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 4:00 python -m mains.main_word2vec_mlp --seed $seed --lemmatize
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 4:00 python -m mains.main_word2vec_mlp --seed $seed --no-lemmatize

# word2vec LR
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 4:00 python -m mains.main_word2vec_lr --seed $seed --lemmatize
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 4:00 python -m mains.main_word2vec_lr --seed $seed --no-lemmatize

# GloVe MLP
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 4:00 python -m mains.main_glove_mlp --seed $seed


# one-hot encoding
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 4:00 python -m mains.main_one_hot_encoding --seed $seed --lemmatize
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 4:00 python -m mains.main_one_hot_encoding --seed $seed --no-lemmatize


bsub -R "rusage[mem=32000]" -W 2:00 python -m mains.main_tsne_visualization --seed $seed --lemmatize

# BERT eval
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 2:00 python -m mains.main_bert --test_only
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 2:00 python -m mains.main_bert --freeze_bert --test_only
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 2:00 python -m mains.main_bert --freeze_bert --extra_layers --test_only
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 2:00 python -m mains.main_bert --include_index --test_only
