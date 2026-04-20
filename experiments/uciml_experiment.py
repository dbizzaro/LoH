import argparse
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna
import os
import time

import sys
sys.path.append('LoH')
DATA_DIR = 'LoH/experiments/data'
RESULTS_FILE = 'LoH/experiments/results/uciml_results_dnf.json'

from models import *
from utils import *
from layers import *
from experiments.mllp.mllp_utils import read_csv, DBEncoder


parser = argparse.ArgumentParser(description='UCIML Experiment')
parser.add_argument('--dataset', type=str, default='tic-tac-toe', help='Name of the dataset')
parser.add_argument('--experimentname', type=str, default='AndOrModel', help='Name of the experiment')
parser.add_argument('--repetitions', type=int, default=2, help='Number of repetitions')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--epochsbetweeneval', type=int, default=10, help='Number of epochs between evaluations')
parser.add_argument('--batchsize', type=int, default=256, help='Batch size')
parser.add_argument('--trials', type=int, default=80, help='Number of trials')
parser.add_argument('--pruningafter', type=int, default=10, help='Number of trials before pruning')
parser.add_argument('--pruningwarmup', type=int, default=10, help='Number of warmup steps before pruning')
parser.add_argument('--nohypertuning', action='store_true', help='Use fixed hyperparameters instead of tuning')
parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
parser.add_argument('--formulas', action='store_true', help='Compute and store formulas')
parser.add_argument('--dnf', action='store_true', help='Restrict model to dnf format')
parser.add_argument('--binaryclassification', action='store_true', help='Use only one target value for binary classification')
parser.add_argument('--negations', action='store_true', help='Use layers with potential negations')
parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True, help='Use weights and biases for logging')
parser.add_argument('--group', type=str, default='Trials', help='Group name for wandb')

args = parser.parse_args()

experiment_name = args.dataset + '_' + args.experimentname
n_trials = args.trials
pruning_after = args.pruningafter
pruning_warmup = args.pruningwarmup
repeats = args.repetitions
use_wandb = args.wandb
compute_formulas = args.formulas
batch_size = args.batchsize
seed = set_seed(args.seed)
dnf = args.dnf

whole_dict = {'experiment': experiment_name, 
                'dataset': args.dataset,
                'group': args.group,  
                'epochs': args.epochs,
                'seed': args.seed,
                'batch_size': args.batchsize,
                'epochs_between_eval': args.epochsbetweeneval,
                'negations': args.negations,
                'binary_classification': args.binaryclassification,
                'repetitions': args.repetitions,
             }


print('Discretizing and binarizing data. Please wait ...')
data_path = os.path.join(DATA_DIR, args.dataset + '.data')
info_path = os.path.join(DATA_DIR, args.dataset + '.info')

X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)
db_enc = DBEncoder(f_df, discrete=True)
db_enc.fit(X_df, y_df)
X, y = db_enc.transform(X_df, y_df)

if args.binaryclassification and y.ndim == 2:
    y = y[:, 1:2]

#def invert_parts(s):
#    if re.match(r'^\d', s):
#        before, after = s.split('_', 1)  # Split at the first underscore
#        return after + '_' + before
#    return s
#col_names = [invert_parts(name).capitalize() for name in db_enc.X_fname]
col_names = ['P_' + str(i) for i in range(X.shape[1])]
print("Propositions: ", col_names)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=56)
input_units = X_train.shape[1]
output_units = y_train.shape[1]
print('Data discretization and binarization are done.')


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print('Device: ', device)

X_train_val_torch = torch.tensor(X_train_val, dtype=torch.float)
X_train_torch = torch.tensor(X_train, dtype=torch.float)
X_val_torch = torch.tensor(X_val, dtype=torch.float)
X_test_torch = torch.tensor(X_test, dtype=torch.float)
y_train_val_torch = torch.tensor(y_train_val, dtype=torch.float)
y_train_torch = torch.tensor(y_train, dtype=torch.float)
y_test_torch = torch.tensor(y_test, dtype=torch.float)
y_val_torch = torch.tensor(y_val, dtype=torch.float)

train_val_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=batch_size, shuffle=True)
train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_torch, y_val_torch), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size, shuffle=False)


# Initialize W&B API
if use_wandb:
    import wandb
    api = wandb.Api()


def objective(trial, verbose=False, best=False):
    global use_wandb, repeats, compute_formulas, device, whole_dict, col_names, X_train_val, dnf

    layers_type = trial.suggest_categorical('layer_type', ['with ks', 'without ks'])
    ks = None if layers_type == 'without ks' else []
    n_layers = 2 if dnf else trial.suggest_int('n_layers', 2, 3) #considering also the output layer
    n_units = [input_units]
    for i in range(n_layers):
        if i == n_layers - 1:
            n_units.append(output_units)
        else:
            n_units.append(trial.suggest_int(f'n_units_{i}', 16, 256))
        if ks is not None:
            ks.append(trial.suggest_int(f'k_{i}', 4, 10))
    starting = 'and' if dnf else trial.suggest_categorical('starting', ['and', 'or']) 
    lr = trial.suggest_float('lr', 0.01, 0.2)
    noise_w = trial.suggest_float('noise_w', 0.4, 1.2)
    temp = trial.suggest_float('temp', 0.4, 1.2)
    multiplicative_coef_temp = trial.suggest_float('multiplicative_coef_temp', 0.9925, 1.)
    min_temp = trial.suggest_float('min_temp', 0.1, 0.4)

    n_steps = whole_dict['epochs'] // whole_dict['epochs_between_eval']
    alpha = np.exp(-6. / n_steps) # decay factor for smoothing of the f1 on the validation set
    compute_formulas = False if n_layers > 2 else compute_formulas
    pruned = False
    id_list = []
    acc_list = []
    f1_list = []
    ewf1_list = []
    runtimes = []
    active_formulas_list = []
    clauses_lengths_list = []
    clauses_numbers_list = []

    for repeat in range(repeats):
        
        config = dict(trial.params)
        config["trial.number"] = trial.number
        config["repeat"] = repeat
        job_type = 'Trials' if not best else 'Best'
        name_trial = f'trial_{trial.number}_{repeat}' if not best else f'best_{repeat}'

        if use_wandb:
            run = wandb.init(
                project=whole_dict['experiment'],
                config=config,
                group=whole_dict['group'],
                name=name_trial,
                job_type=job_type,
                reinit=True
            )
            id_list.append(run.id)

        
        model = AndOrModel(n_units, ks, device, starting, whole_dict['negations']).to(device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        ewma_f1 = 0.0
        temp = trial.params['temp']
        model.train()
        start_time = time.time()

        for epoch in range(whole_dict['epochs']):
            running_loss = 0.0
            for i, (features, labels) in enumerate(train_loader):
                features = features.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(features, noise_w, temp)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            if epoch % whole_dict['epochs_between_eval'] == 0:
                model.eval()
                with torch.no_grad():
                    outputs_train, predicted_train, labels_train = predict(model, train_loader, device)
                    outputs_val, predicted_val, labels_val = predict(model, val_loader, device)
                    acc_train, f1_train = eval(predicted_train, labels_train)
                    acc_val, f1_val = eval(predicted_val, labels_val)

                
                step = epoch // whole_dict['epochs_between_eval']
                ewma_f1 = alpha * ewma_f1 + (1. - alpha) * f1_val
                if verbose: 
                    print(f'Epoch [{epoch}/{whole_dict["epochs"]}], Loss: {loss.item():.5f}')
                    print(f'train acc = {acc_train:.3f} %, val acc = {acc_val:.3f} %')
                    print(f'f1 train = {f1_train:.3f}, f1 val = {f1_val:.3f}')
                if use_wandb:
                    data_log = {"acc_val": acc_val, "f1_val": f1_val, "ewma_f1": ewma_f1, 
                                "loss": running_loss, "acc_train": acc_train, "f1_train": f1_train}
                    wandb.log(data=data_log, step=step)
                
                model.train()

                trial.report(ewma_f1, n_steps * repeat + step)
                if trial.should_prune():
                    pruned = True
                    if use_wandb:
                        run.finish()
                    raise optuna.TrialPruned()
        
            if temp > min_temp:
                temp *= multiplicative_coef_temp

        time_elapsed = time.time() - start_time
        runtimes.append(time_elapsed)
        acc_list.append(acc_val)
        f1_list.append(f1_val)
        ewf1_list.append(ewma_f1)
        
        if compute_formulas:
            formulas = model.to_formula(verbose=False, features_names=col_names)
            formulas_simplified = [non_redundant_clauses(sympy.parse_expr(formula), starting=starting) for formula in formulas]
            clause_activation_info = []
            formulas_active = []
            for f_simp in formulas_simplified:
                info, filtered_expr = clause_activation_status(f_simp, X_train_val, col_names, starting=starting, return_expression=True)
                clause_activation_info.append(info)
                formulas_active.append(non_redundant_clauses(filtered_expr, starting=starting))
            num_clauses_list, avg_clause_length_list = [], []
            for f_active in formulas_active:
                num_clauses, avg_clause_length = clause_stats(f_active, starting=starting)
                num_clauses_list.append(num_clauses)
                avg_clause_length_list.append(avg_clause_length)
            formulas_simplified_str = [str(f_simp) for f_simp in formulas_simplified]
            formulas_active_str = [str(f_active) for f_active in formulas_active]
            active_formulas_list.append(formulas_active_str)
            clauses_lengths_list.append(avg_clause_length_list)
            clauses_numbers_list.append(num_clauses_list)
            if use_wandb:
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
        if use_wandb:
            wandb.summary["finished"] = True
            wandb.summary["time"] = time_elapsed
            run.finish()

    print(f1_list)
    print()
    print(f'Finished trial {trial.number}')
    mean_acc = np.mean(acc_list)
    mean_f1 = np.mean(f1_list)
    mean_ewf1 = np.mean(ewf1_list)
    mean_runtime = np.mean(runtimes)
    std_acc = np.std(acc_list)
    std_f1 = np.std(f1_list)
    std_ewf1 = np.std(ewf1_list)
    std_runtime = np.std(runtimes)
    if compute_formulas:
        mean_num_clauses = np.mean(clauses_numbers_list)
        mean_avg_clause_length = np.mean(clauses_lengths_list)
        std_num_clauses = np.std(clauses_numbers_list)
        std_avg_clause_length = np.std(clauses_lengths_list)

    if use_wandb:
        for run_id in id_list:
            run = wandb.init(id=run_id, project=experiment_name, resume="must")
            wandb.summary["mean_acc"] = mean_acc
            wandb.summary["mean_f1"] = mean_f1
            wandb.summary["mean_ewf1"] = mean_ewf1
            wandb.summary["std_acc"] = std_acc
            wandb.summary["std_f1"] = std_f1
            wandb.summary["std_ewf1"] = std_ewf1
            if compute_formulas:
                wandb.summary["mean_num_clauses"] = mean_num_clauses
                wandb.summary["mean_avg_clause_length"] = mean_avg_clause_length
                wandb.summary["std_num_clauses"] = std_num_clauses
                wandb.summary["std_avg_clause_length"] = std_avg_clause_length
            wandb.summary["pruned"] = pruned
            wandb.summary["repetitions"] = len(acc_list)
            run.finish()
    if verbose:
        print()
        print(f'Mean acc: {mean_acc:.3f} +/- {std_acc:.3f}')
        print(f'Mean f1: {mean_f1:.3f} +/- {std_f1:.3f}')
        print(f'Mean ewf1: {mean_ewf1:.3f} +/- {std_ewf1:.3f}')
        print(f'Mean runtime: {mean_runtime:.3f} +/- {std_runtime:.3f}')
        print("Parameter count: ", model.parameter_count())
        print()
    if best:
        hyper_values = dict(trial.params)
        summary = {'mean_acc': mean_acc,
                    'mean_f1': mean_f1,
                    'mean_ewf1': mean_ewf1,
                    'mean_runtime': mean_runtime,
                    'std_acc': std_acc,
                    'std_f1': std_f1,
                    'std_ewf1': std_ewf1,
                    'std_runtime': std_runtime
                }
        #whole_dict['f1_scores'] = f1_list
        whole_dict['hyperparameters'] = hyper_values
        whole_dict['summary'] =  summary
        whole_dict['parameter_count'] = model.parameter_count()
        if compute_formulas:
            whole_dict["mean_num_clauses"] = mean_num_clauses
            whole_dict["mean_avg_clause_length"] = mean_avg_clause_length
            whole_dict["std_num_clauses"] = std_num_clauses
            whole_dict["std_avg_clause_length"] = std_avg_clause_length
            whole_dict['formulas'] = active_formulas_list
        
        write_results_to_file(whole_dict, RESULTS_FILE)

    return mean_ewf1

if not args.nohypertuning:
    print('Starting hyperparameter tuning.')
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, n_ei_candidates=10, n_startup_trials=n_trials//4, seed=123)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=pruning_warmup, n_startup_trials=pruning_after)
    study = optuna.create_study(direction='maximize', study_name=experiment_name, pruner = pruner, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print()
    print("Best hyperparameters: ", study.best_params)
    print("Best score: ", study.best_value)
    print("--------------------------------------")

    repeats = 10
    seed = set_seed(args.seed)
    train_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size, shuffle=False)
    objective(optuna.trial.FixedTrial(study.best_params), verbose=True, best=True)

else:
    seed = set_seed(args.seed)
    train_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size, shuffle=False)
    repeats = args.repetitions
    hyperparams = get_hyperparameters(RESULTS_FILE, experiment_name, args.group)
    whole_dict['group'] = f'{args.group}_new'
    experiment_name = args.dataset + '_' + args.experimentname + '_new'
    whole_dict['experiment'] = experiment_name
    objective(optuna.trial.FixedTrial(hyperparams), verbose=True, best=True)
