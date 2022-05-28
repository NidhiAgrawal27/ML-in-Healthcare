#!/bin/bash

for seed in {42..46}
do
    python -m mains.main_baseline --seed $seed --lemmatize
    python -m mains.main_baseline --seed $seed --no-lemmatize

    python -m mains.main_word2vec_mlp --seed $seed --lemmatize
    python -m mains.main_word2vec_mlp --seed $seed --no-lemmatize

    python -m mains.main_word2vec_lr --seed $seed --lemmatize
    python -m mains.main_word2vec_lr --seed $seed --no-lemmatize

    python -m mains.main_one_hot_encoding --seed $seed --lemmatize
    python -m mains.main_one_hot_encoding --seed $seed --no-lemmatize

    python -m mains.main_glove_mlp --seed $seed


done

python -m mains.main_tsne_visualization --seed 42 --lemmatize

python -m mains.main_bert
python -m mains.main_bert --freeze_bert
python -m mains.main_bert --freeze_bert --extra_layers
python -m mains.main_bert --include_index

python -m visualization.visualize
