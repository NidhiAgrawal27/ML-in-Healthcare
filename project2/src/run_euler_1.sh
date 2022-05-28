#!/bin/bash

# tokenization
bsub -R "rusage[mem=32000]" -W 4:00 python -m mains.main_tokenize --lemmatize
bsub -R "rusage[mem=32000]" -W 4:00 python -m mains.main_tokenize --no-lemmatize


# BERT
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 12:00 python -m mains.main_bert
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 12:00 python -m mains.main_bert --freeze_bert
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 12:00 python -m mains.main_bert --freeze_bert --extra_layers
bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 12:00 python -m mains.main_bert --include_index

