[![Accepted at KR 2026](https://img.shields.io/badge/Accepted%20at-KR%202026-blue)](https://kr.org/)
[![Paper](http://img.shields.io/badge/paper-arxiv.2509.21663-B31B1B.svg)](https://arxiv.org/abs/2509.21663)
# Logic of Hypotheses

This repository contains an implementation of the *Logic of Hypotheses (LoH)* framework. 

`models.py` contains the class for zero-knowledge, pure rule-learning LoH neural networks. Instead, `general_models.py` contains a parser directly compiling LoH formulas.

## Dependencies
* ML core: `torch`, `torchvision`, `numpy`, `scikit-learn`, `pandas`
* Experiment tooling: `optuna`, `wandb`
* Logic: `sympy`, `pyparsing`
* Baselines: `xgboost`, `difflogic`, `tensorflow`
* Data: `ucimlrepo`

## Reproducing the main experiments

Each script accepts `--help` for additional flags. 

### Tabular benchmarks (from UCI‑ML)

To train LoH models and MLLP baselines on the *adult* dataset and replicate the experiment:

```bash
# LoH (ours)
python LoH/experiments/uciml_experiment.py --dataset=adult --epochs=200 --repetitions=10 --nohypertuning

# MLLP baseline
python LoH/experiments/uciml_baseline_mllp.py -d adult -e 200 --nohypertuning

# DLN baseline
python LoH/experiments/uciml_baseline_difflogic.py -d adult -e 200 --nohypertuning

# Other baselines
python  LoH/experiments/uciml_baselines.py --dataset=adult --epochs=200
```

Replace `adult` with any dataset in `{bank_marketing, banknote, blogger, …}` to run the experiments with the other datasets. 
Results and Hyperparameters are stored in `./experiments/results/uciml_results.json` (and similar files for the baselines).


### Visual Tic‑Tac‑Toe

To train (as an example) the dnf LoH model and the cnf baselines:

```bash
# LoH – DNF
python LoH/experiments/MNISTttt.py --cnf_or_dnf=dnf --nohypertuning

# MLLP baseline – CNF 
python LoH/experiments/MNISTttt_mllp.py --cnf_or_dnf=cnf --nohypertuning

# DLN baseline
python LoH/experiments/MNISTttt_difflogic.py --nohypertuning

# NN baseline
python  LoH/experiments/MNISTttt.py --nn_baseline --tune_n_bits --nohypertuning
```

Results and Hyperparameters are stored in `./experiments/results/MNISTttt-results.json` (and similar files for the baselines).

### Other experiments

See the notebooks.

## Project layout

```
LoH/
├── experiments/                    # scripts for the experiments
│   ├── data/                          # datasets
│   ├── DILP/                          # code for running dILP on the wildfire risk task
│   ├── results/                       # json files containing the hyperparams, and storing the results
│   ├── notebooks/                     # notebooks producing the plots of the other experiments
│   ├── mllp/                          # code for preprocessing, and mllp baseline
│   ├── generate_data.py               # functions for producing artificial datasets
│   ├── uciml_experiment.py            # LoH experiment on tabular data
│   ├── uciml_baseline_difflogic.py    # DLN baseline on tabular data
│   ├── uciml_baseline_mllp.py         # MLLP baseline on tabular data
│   ├── uciml_baselines.py             # Other baselines on tabular data
│   ├── MNISTttt.py                    # LoH (and NN baseline) on Visual tic-tac-toe
│   ├── MNISTttt_difflogic.py          # DLN baseline on Visual tic-tac-toe 
│   └── MNISTttt_mllp.py               # MLLP baseline on Visual tic-tac-toe
├── general_models.py               # functions for compiling general LoH formulas
├── models.py                       # class for the rule-learning LoH neural network
├── layers.py                       # classes for the rule-learning LoH layers
└── utils.py                        # utility functions
```

## Citation
```
@article{bizzaro2025logic,
  title={Logic of Hypotheses: from Zero to Full Knowledge in Neurosymbolic Integration},
  author={Bizzaro, Davide and Daniele, Alessandro},
  journal={arXiv preprint arXiv:2509.21663},
  year={2025}
}
```