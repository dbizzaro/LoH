import argparse
import numpy as np
import torch
import optuna
import json
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import time
import wandb

import warnings
warnings.filterwarnings("ignore")
import os, sys
sys.path.append('LoH')
DATA_DIR = 'LoH/experiments/data'
RESULTS_FILE = 'LoH/experiments/results/uciml_results_mllp_dnf.json'

from experiments.mllp.mllp_utils import read_csv, DBEncoder
from experiments.mllp.mllp_models import MLLP
from utils import *


if torch.cuda.is_available():
    device = torch.device('cuda')
#elif torch.backends.mps.is_available():
#    device = torch.device('mps')
else:
    device = torch.device('cpu')
print('Device: ', device)


def get_mllp_formula(mllp_model, prop_names, verbose=True):
    starting = mllp_model.starting
    rules = mllp_model.get_rules()
    if verbose:
        print('raw', rules)
        #print(prop_names)
    formulas = []
    for output_class, list_nodes in rules[0].items():
        clauses = []
        for node, value in rules[1].items():
            if node in list_nodes:
                if starting == 'and':
                    clause = " & ".join([prop_names[node] for node in value])
                    if not clause:
                        clause = "True"
                else:
                    clause = " | ".join([prop_names[node] for node in value])
                    if not clause:
                        clause = "False"
                clauses.append("(" + clause + ")")
        if starting == 'and':
            formula = " | ".join(clauses) if clauses else 'False'
        else:
            formula = " & ".join(clauses) if clauses else 'True'
        if verbose:
            print(formula)
        formulas.append(formula)
    return formulas


def train(model, train_loader, val_loader, device, epochs=50, 
          lr=0.01, lr_decay_epoch=100, lr_decay_rate=0.75, weight_decay=0.0, 
          epochs_between_eval=10, seed=42, criterion=nn.MSELoss(), use_wandb=False):
    
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs."""
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer
    
    n_steps = epochs // epochs_between_eval
    alpha = np.exp(-5. / n_steps)
    ewma_f1_b = 0.
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        optimizer = exp_lr_scheduler(optimizer, epoch, init_lr=lr, lr_decay_rate=lr_decay_rate, lr_decay_epoch=lr_decay_epoch)
        running_loss = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model.forward(X, randomly_binarize=True)
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.clip()
        
        # Change the set of weights to be binarized every epoch.
        model.randomly_binarize_layer_refresh()
        
        # Evaluate the model
        if epoch % epochs_between_eval == 0:
            with torch.no_grad():
                outputs_train, predicted_train, labels_train = predict(model, train_loader, device)
                acc_train, f1_train = eval(predicted_train, labels_train)
                outputs_val, predicted_val, labels_val = predict(model, val_loader, device)
                acc_val, f1_val = eval(predicted_val, labels_val)

                outputs_train_b, predicted_train_b, labels_train_b = predict(model, train_loader, device, binarized_forward=True)
                acc_train_b, f1_train_b = eval(predicted_train_b, labels_train_b)
                outputs_val_b, predicted_val_b, labels_val_b = predict(model, val_loader, device, binarized_forward=True)
                acc_val_b, f1_val_b = eval(predicted_val_b, labels_val_b)

            step = epoch // epochs_between_eval
            ewma_f1_b = alpha * ewma_f1_b + (1. - alpha) * f1_val_b
            if use_wandb:
                wandb.log(data={'loss': running_loss,
                            "acc_train_b": acc_train_b, "f1_train_b": f1_train_b,
                            "acc_val_b": acc_val_b, "f1_val_b": f1_val_b,
                            "acc_train": acc_train, "f1_train": f1_train,
                            "acc_val": acc_val, "f1_val": f1_val,
                            "ewma_f1": ewma_f1_b}, step=step)
                   
    return acc_val, acc_val_b, f1_val, f1_val_b, ewma_f1_b, acc_train, acc_train_b, f1_train, f1_train_b



def experiment(args, best=False):
    net_structure = [args.input_units] + list(map(int, args.structure.split('_'))) + [args.output_units]

    net = MLLP(net_structure,
               device=device,
               random_binarization_rate=args.random_binarization_rate,
               use_not=args.use_not)
    net.to(device)

    if best:
        train_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_torch, y_val_torch), batch_size=args.batch_size, shuffle=False)

    acc_val, acc_val_b, f1_val, f1_val_b, acc_train, acc_train_b, f1_train, f1_train_b, ewma_f1_b = train(
        net,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.learning_rate,
        lr_decay_epoch=args.lr_decay_epoch,
        lr_decay_rate=args.lr_decay_rate,
        weight_decay=args.weight_decay,
        epochs_between_eval=args.epochsbetweeneval,
        seed=args.seed,
        use_wandb=args.wandb
    )

    return acc_val, acc_val_b, f1_val, f1_val_b, ewma_f1_b, net


def objective(trial, best=False, verbose=False):
    global args, col_names, X_train_val, X_fname, y_fname

    n_layers = 1 if args.dnf else trial.suggest_categorical('n_layers', [1, 3])
    layers_sizes = []
    for i in range(n_layers):
        layers_sizes.append(trial.suggest_int(f'n_units_{i}', 16, 256))
    args.structure = '_'.join(map(str, layers_sizes))
    args.weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
    args.learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    args.random_binarization_rate = trial.suggest_float('p', 0.0, 0.99)
    if args.use_not:
        args.use_not = trial.suggest_categorical('use_not', [True, False])
    
    repetitions = args.repetitions

    net_structure = [args.input_units] + list(map(int, args.structure.split('_'))) + [args.output_units]
    parameter_count = 0
    for i in range(len(net_structure) - 1):
        parameter_count += net_structure[i] * net_structure[i + 1]
    print("Architecture: ", net_structure)

    id_list = []
    acc_list = []
    f1_list = []
    acc_b_list = []
    f1_b_list = []
    ewf1_b_list = []
    runtimes = []
    active_formulas_list = []
    clauses_lengths_list = []
    clauses_numbers_list = []
    pruned = False

    for repetition in range(repetitions):
        
        config = dict(trial.params)
        config["trial.number"] = trial.number
        config["repeat"] = repetition
        group = args.group
        job_type = 'Trials' if not best else 'Best'
        name_trial = f'trial_{trial.number}_{repetition}' if not best else f'best_{repetition}'
        if args.wandb:
            run = wandb.init(
                project=experiment_name,
                config=config,
                group=group,
                name=name_trial,
                job_type=job_type,
                reinit=True
            )
            id_list.append(run.id)

        args.learning_rate = trial.params['learning_rate']

        start_time = time.time()
        acc_val, acc_val_b, f1_val, f1_val_b, ewma_f1_b, net = experiment(args, best)
        time_elapsed = time.time() - start_time
        
        acc_list.append(acc_val)
        f1_list.append(f1_val)
        acc_b_list.append(acc_val_b)
        f1_b_list.append(f1_val_b)
        ewf1_b_list.append(ewma_f1_b)
        runtimes.append(time_elapsed)

        trial.report(f1_val, repetition)
        if trial.should_prune():
            pruned = True
            if args.wandb:
                run.finish()
            raise optuna.TrialPruned()

        if args.formulas and len(net.dim_list) == 3:
            formulas = get_mllp_formula(net, col_names)
            formulas_simplified = [non_redundant_clauses(sympy.parse_expr(formula), starting=net.starting) for formula in formulas]
            clause_activation_info = []
            formulas_active = []
            for f_simp in formulas_simplified:
                info, filtered_expr = clause_activation_status(f_simp, X_train_val, col_names, starting=net.starting, return_expression=True)
                clause_activation_info.append(info)
                formulas_active.append(non_redundant_clauses(filtered_expr, starting=net.starting))
            num_clauses_list, avg_clause_length_list = [], []
            for f_active in formulas_active:
                num_clauses, avg_clause_length = clause_stats(f_active, starting=net.starting)
                num_clauses_list.append(num_clauses)
                avg_clause_length_list.append(avg_clause_length)
            formulas_simplified_str = [str(f_simp) for f_simp in formulas_simplified]
            formulas_active_str = [str(f_active) for f_active in formulas_active]
            active_formulas_list.append(formulas_active_str)
            clauses_lengths_list.append(avg_clause_length_list)
            clauses_numbers_list.append(num_clauses_list)
            if args.wandb:
                wandb.summary["formulas"] = formulas
                wandb.summary["formulas_simplified"] = formulas_simplified_str
                wandb.summary["formulas_active"] = formulas_active_str
                wandb.summary["num_clauses"] = num_clauses_list
                wandb.summary["avg_clause_length"] = avg_clause_length_list
                #wandb.summary["clause_activation"] = clause_activation_info
            if verbose:
                print()
                print("Formula: ", formulas)
                print("Formula simplified: ", formulas_simplified_str)
                print("Formula active-only: ", formulas_active_str)
                print(f'Num clauses (active): {num_clauses_list}, avg clause length: {avg_clause_length_list}')
                print()

        if args.wandb:
            wandb.summary["finished"] = True
            run.finish()
    
    
    print(f'Finished trial {trial.number}')
    mean_acc = np.mean(acc_list)
    mean_f1 = np.mean(f1_list)
    mean_ewf1_b = np.mean(ewf1_b_list)
    mean_runtime = np.mean(runtimes)
    std_acc = np.std(acc_list)
    std_f1 = np.std(f1_list)
    std_ewf1_b = np.std(ewf1_b_list)
    mean_acc_b = np.mean(acc_b_list)
    mean_f1_b = np.mean(f1_b_list)
    std_acc_b = np.std(acc_b_list)
    std_f1_b = np.std(f1_b_list)
    std_runtime = np.std(runtimes)
    if args.formulas:
        mean_num_clauses = np.mean(clauses_numbers_list)
        mean_avg_clause_length = np.mean(clauses_lengths_list)
        std_num_clauses = np.std(clauses_numbers_list)
        std_avg_clause_length = np.std(clauses_lengths_list)

    if args.wandb:
        for run_id in id_list:
            run = wandb.init(id=run_id, project=experiment_name, resume="must")
            wandb.summary["mean_acc"] = mean_acc
            wandb.summary["mean_f1"] = mean_f1
            wandb.summary["mean_ewf1_b"] = mean_ewf1_b
            wandb.summary["std_acc"] = std_acc
            wandb.summary["std_f1"] = std_f1
            wandb.summary["std_ewf1"] = std_ewf1_b
            wandb.summary["mean_acc_b"] = mean_acc_b
            wandb.summary["mean_f1_b"] = mean_f1_b
            wandb.summary["std_acc_b"] = std_acc_b
            wandb.summary["std_f1_b"] = std_f1_b
            wandb.summary["pruned"] = pruned
            wandb.summary["repetitions"] = len(acc_list)
            wandb.summary["runtime"] = mean_runtime
            wandb.summary["std_runtime"] = std_runtime
            if args.formulas:
                wandb.summary["mean_num_clauses"] = mean_num_clauses
                wandb.summary["mean_avg_clause_length"] = mean_avg_clause_length
                wandb.summary["std_num_clauses"] = std_num_clauses
                wandb.summary["std_avg_clause_length"] = std_avg_clause_length
            run.finish()

    if best:
        summary = {'mean_acc': mean_acc,
                    'mean_f1': mean_f1,
                    'mean_ewf1': mean_ewf1_b,
                    'mean_runtime': mean_runtime,
                    'std_acc': std_acc,
                    'std_f1': std_f1,
                    'std_ewf1': std_ewf1_b,
                    'std_runtime': std_runtime,
                    'mean_acc_b': mean_acc_b,
                    'mean_f1_b': mean_f1_b,
                    'std_acc_b': std_acc_b,
                    'std_f1_b': std_f1_b,
                }
        hyper_values = dict(trial.params)
        whole_dict = {'experiment': experiment_name, 
                      'group': args.group, 
                      'data_set': args.data_set,
                      'epochs': args.epochs,
                      'repetitions': repetitions,
                      'seed': args.seed,
                      'batch_size': args.batch_size,
                      'lr_decay_rate': args.lr_decay_rate,
                      'lr_decay_epoch': args.lr_decay_epoch,
                      'use_not': args.use_not,
                      'epochs_between_eval': args.epochsbetweeneval,
                      'hyperparameters': hyper_values, 
                      'results': summary,
                      'parameter_count': parameter_count
                      }
        if args.formulas:
            whole_dict["mean_num_clauses"] = mean_num_clauses
            whole_dict["mean_avg_clause_length"] = mean_avg_clause_length
            whole_dict["std_num_clauses"] = std_num_clauses
            whole_dict["std_avg_clause_length"] = std_avg_clause_length
            whole_dict['formulas'] = active_formulas_list

        write_results_to_file(whole_dict, RESULTS_FILE)

    return mean_ewf1_b



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_set', type=str, default='tic-tac-toe',
                        help='Set the data set for training. All the data sets in the dataset folder are available.')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='Set the total epoch.')
    parser.add_argument('-bs', '--batch_size', type=int, default=256, help='Set the batch size.')
    parser.add_argument('-lrdr', '--lr_decay_rate', type=float, default=0.75, help='Set the learning rate decay rate.')
    parser.add_argument('-lrde', '--lr_decay_epoch', type=int, default=100, help='Set the learning rate decay epoch.')
    parser.add_argument('--use_not', action="store_true",
                        help='Use the NOT (~) operator in logical rules. '
                             'It will enhance model capability but make the CRS more complex.')
    parser.add_argument('--trials', type=int, default=80, help='Number of trials')
    parser.add_argument('--pruningafter', type=int, default=10, help='Number of trials before pruning')
    parser.add_argument('--pruningwarmup', type=int, default=10, help='Number of warmup steps before pruning')
    parser.add_argument('--repetitions', type=int, default=2, help='Number of repetitions')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--group', type=str, default='Trials', help='Group name for wandb')
    parser.add_argument('--epochsbetweeneval', type=int, default=10, help='Number of epochs between evaluations')
    parser.add_argument('--experimentname', type=str, default='MLLP', help='Name of the experiment')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True, help='Use weights and biases for logging')
    parser.add_argument('--nohypertuning', action='store_true', help='Use fixed hyperparameters instead of tuning')
    parser.add_argument('--formulas', action='store_true', help='Compute and store formulas')
    parser.add_argument('--dnf', action='store_true', help='Restrict model to dnf format')
    parser.add_argument('--binaryclassification', action='store_true', help='Use only one target value for binary classification')


    args = parser.parse_args()
    dataset = args.data_set

    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    print('Discretizing and binarizing data. Please wait ...')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)
    db_enc = DBEncoder(f_df, discrete=True)
    X_fname = db_enc.X_fname
    y_fname = db_enc.y_fname
    db_enc.fit(X_df, y_df)
    X, y = db_enc.transform(X_df, y_df)
    if args.binaryclassification and y.ndim == 2:
        y = y[:, 1:2]
    print('Data discretization and binarization are done.')

    col_names = ['P_' + str(i) for i in range(X.shape[1])]
    print("Propositions: ", col_names)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=56)       

    args.input_units = X_train.shape[1]
    args.output_units = y_train.shape[1]

    X_train_val_torch = torch.tensor(X_train_val, dtype=torch.float)
    X_train_torch = torch.tensor(X_train, dtype=torch.float)
    X_val_torch = torch.tensor(X_val, dtype=torch.float)
    X_test_torch = torch.tensor(X_test, dtype=torch.float)
    y_train_val_torch = torch.tensor(y_train_val, dtype=torch.float)
    y_train_torch = torch.tensor(y_train, dtype=torch.float)
    y_test_torch = torch.tensor(y_test, dtype=torch.float)
    y_val_torch = torch.tensor(y_val, dtype=torch.float)

    n_trials = args.trials
    experiment_name = args.data_set + '_' + args.experimentname
    
    
    if not args.nohypertuning:
        print('Starting hyperparameter tuning.')
        seed = set_seed(args.seed)
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, n_ei_candidates=10, n_startup_trials=n_trials//4, seed=12)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=args.pruningwarmup, n_startup_trials=args.pruningafter)
        study = optuna.create_study(direction='maximize', study_name=experiment_name, sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        print()
        print("Best hyperparameters: ", study.best_params)
        print("Best score: ", study.best_value)
        print("--------------------------------------")

        args.repetitions = 10
        seed = set_seed(args.seed)
        objective(optuna.trial.FixedTrial(study.best_params), best=True)
    else:
        print('Starting training with fixed hyperparameters.')
        seed = set_seed(args.seed)
        hyperparams = get_hyperparameters(RESULTS_FILE, experiment_name, args.group)
        args.group = f'{args.group}_new'
        experiment_name = args.data_set + '_' + args.experimentname + '_new'
        objective(optuna.trial.FixedTrial(hyperparams), best=True, verbose=True)
    


