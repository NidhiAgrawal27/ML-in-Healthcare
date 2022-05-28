# Project 2: Natural Language Processing

<p align="center">
  <a href="#about">About</a> •
  <a href="#setup-and-usage">Setup and Usage</a> •
  <a href="#requirements">Requirements</a> •
  <a href="#reproducibility">Reproducibility</a>
</p>

## About
This repository contains the code and report for sequential sentence classification as part of project 2 for the course [Machine Learning for Healthcare 2022](http://www.vvz.ethz.ch/lerneinheitPre.do?semkez=2022S&lerneinheitId=158957&lang=en). The following is a brief overview of the repository:
- `data`: contains all data files (not included in the repository).
- `logs`: contains all log files generated during execution (not included in the repository).
- `report`: contains all files used to generate the report in LaTeX.
- `src`: contains all source code files.

## Setup and usage
### Local
1. Install conda environment from [project2.yml](project2.yml) (note: this environment was created on an M1 MacBook running macOS and may not work on other systems, see [Requirements](#requirements) below for a list of main dependencies).
2. Load the installed environment.
3. Download the spacy model:
```bash
python -m spacy download en_core_web_sm
```
4. Copy data files into the `data` directory by cloning [pubmed-rct](https://github.com/Franck-Dernoncourt/pubmed-rct):
```bash
cd data
git clone https://github.com/Franck-Dernoncourt/pubmed-rct.git
```
5. Extract the .7z data files (here shown using [the Unarchiver](https://theunarchiver.com/command-line)):
```bash
cd pubmed-rct/PubMed_200k_RCT/
unar train.7z
```
6. Download and extract glove embeddings
```bash
cd ../../glove
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```
7. Execute:
```bash
cd ../../src
./run.sh
```
(Note that this will train the BERT model for 20 epochs while on Euler they run for 12h only. Results in the report correspond to 12h Euler runs.)

8. Results are by default saved in the `logs` directory. All figures and tables used in the report are further generated and saved in `logs/figures`.

### ETH Euler cluster
1. Connect to the ETH Euler cluster.
2. Copy project files to the cluster.
3. Execute in shell to load environment:
```bash
env2lmod
module load gcc/6.3.0 python_gpu/3.8.5 hdf5/1.10.1
pip install --upgrade pip
pip install transformers
pip install --upgrade matplotlib
pip install py7zr spacy gensim Levenshtein
```
4. Copy data files into the `data` directory by cloning [pubmed-rct](https://github.com/Franck-Dernoncourt/pubmed-rct):
```bash
cd data
git clone https://github.com/Franck-Dernoncourt/pubmed-rct.git
```
5. Extract the .7z data files (using python as euler has no suitable installed programs):
```bash
cd pubmed-rct/PubMed_200k_RCT/
python -c "from py7zr import unpack_7zarchive; import shutil; shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive); shutil.unpack_archive('train.7z', '.')"
```
6. Download and extract glove embeddings
```bash
cd ../../glove
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```
7. Download the spacy model and BERT weights + tokenizer:
```bash
cd ../../src
python -m spacy download en_core_web_sm
python -m mains.main_bert --build_only
```
8. Execute:
```bash
./run_euler_1.sh
```
9. Wait for submitted jobs to complete then run the bert evaluation
```
./run_euler_2.sh
```
10. Wait for submitted jobs to complete then finally produce the visualizations via
```
python -m visualization.visualize
```
11. Results are by default saved in the `logs` directory. All figures and tables used in the report are further generated and saved in `logs/figures`.

## Requirements
* `tensorflow`
* `torch`
* `transformers`
* `scikit-learn`
* `matplotlib`
* `numpy`
* `pandas`
* `pytorch`
* `seaborn`
* `python-Levenshtein`
* `spacy`
* `gensim`

## Reproducibility
All experiments are run with a fixed global seed (determined by argument `--seed`) to ensure reproducibility. Unfortunately, actual results may differ slightly due to some unknown stochastic behaviour. Rerunning the experiments may therefore produce slightly different figures and tables of the results.
