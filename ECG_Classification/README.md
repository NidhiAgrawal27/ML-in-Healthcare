# Project 1: Timeseries

<p align="center">
  <a href="#description">Description</a> •
  <a href="#setup-and-usage">Setup and Usage</a> •
  <a href="#requirements">Requirements</a>
</p>

## About
This repository contains the code and report for ECG classification as part of project 1 for the course [Machine Learning for Healthcare 2022](http://www.vvz.ethz.ch/lerneinheitPre.do?semkez=2022S&lerneinheitId=158957&lang=en). The following is a brief overview of the repository:
- `data`: contains all data files (not included in the repository).
- `logs`: contains all log files generated during execution (not included in the repository).
- `report`: contains all files used to generate the report in LaTeX.
- `src`: contains all source code files.

## Setup and usage
### Local
1. Install conda environment from [project1.yml](project1.yml) (note: this environment was created on an M1 MacBook running macOS and may not work on other systems, see [Requirements](#requirements) below for a list of main dependencies).
2. Load the installed environment.
3. Copy data files into the `data` directory.
4. Execute:
```bash
cd src
./run.sh
```
5. Results are by default saved in the `logs` directory. All figures and tables used in the report are further generated and saved in `logs/figures`.

### ETH Euler cluster
1. Connect to the ETH Euler cluster.
2. Copy project files to the cluster.
3. Copy data files into the `data` directory.
4. Execute in shell to load environment:
```bash
env2lmod
module load gcc/6.3.0 python_gpu/3.8.5 hdf5/1.10.1
pip install --upgrade pip
pip install --upgrade matplotlib
```
4. Execute:
```bash
cd src
./run_euler.sh
```
5. Wait for the submitted jobs to complete, then run the transfer learning and ensemble methods that depend on these results via
```
./run_euler_2.sh
```
6. Wait for submitted jobs to complete then finally produce the visualizations via submitting a job running
```
python -m visualization.visualize
```
7. Results are by default saved in the `logs` directory. All figures and tables used in the report are further generated and saved in `logs/figures`.

## Requirements
* `tensorflow`
* `scikit-learn`
* `matplotlib`
* `numpy`
* `pandas`

## Reproducibility
All experiments are run with a fixed global seed (determined by argument `--seed`) to ensure reproducibility. Unfortunately, actual results may differ slightly due to some unknown stochastic behaviour. Rerunning the experiments may therefore produce slightly different figures and tables of the results.
