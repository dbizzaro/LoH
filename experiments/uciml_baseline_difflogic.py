import argparse
import numpy as np
import torch
import optuna
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import time
import wandb

import warnings
warnings.filterwarnings("ignore")
import os, sys
sys.path.append('LoH')
DATA_DIR = 'LoH/experiments/data'
RESULTS_FILE = 'LoH/experiments/results/uciml_results_difflogic.json'

from experiments.mllp.mllp_utils import read_csv, DBEncoder
from difflogic import LogicLayer, GroupSum, CompiledLogicNet
from utils import *


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Device: ', device)



def train(model, train_loader, val_loader, device, epochs=50, 
          lr=0.01, epochs_between_eval=10, use_wandb=False):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    n_steps = epochs // epochs_between_eval
    alpha = np.exp(-5. / n_steps)
    ewma_f1 = 0.
    
    for epoch in range(epochs):
        running_loss = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        # Evaluate the model
        if epoch % epochs_between_eval == 0:
            model.eval()
            with torch.no_grad():
                outputs_train, predicted_train, labels_train = predict(model, train_loader, device)
                acc_train, f1_train = eval(predicted_train, labels_train)
                outputs_val, predicted_val, labels_val = predict(model, val_loader, device)
                acc_val, f1_val = eval(predicted_val, labels_val)
                #try:
                #    compiled_model = CompiledLogicNet(model=model, num_bits=64, cpu_compiler='gcc', verbose=False)
                #    compiled_model.compile()
                #    outputs_val_c, predicted_val_c, labels_val_c = predict(compiled_model, val_loader, device, compiled=True)
                #    acc_val_c, f1_val_c = eval(predicted_val_c, labels_val_c)
                #except Exception as e:
                #    print("Compilation failed with error:", e)
                acc_val_c, f1_val_c = np.nan, np.nan

            step = epoch // epochs_between_eval
            ewma_f1 = alpha * ewma_f1 + (1. - alpha) * f1_val
            if use_wandb:
                wandb.log(data={'loss': running_loss,
                            "acc_train": acc_train, "f1_train": f1_train,
                            "acc_val": acc_val, "f1_val": f1_val,
                            "ewma_f1": ewma_f1}, step=step)
            model.train()
                   
    return acc_val, f1_val, acc_train, f1_train, ewma_f1, acc_val_c, f1_val_c
    


def experiment(args, net_structure, grad_factor, best=False):
    logic_layers = []
    logic_layers.append(torch.nn.Flatten())
    for i in range(len(net_structure) - 2):
        logic_layers.append(LogicLayer(in_dim=net_structure[i], out_dim=net_structure[i+1], grad_factor=grad_factor, connections=args.connections, implementation='cuda', device='cuda'))
    model = torch.nn.Sequential(
        *logic_layers,
        GroupSum(net_structure[-1], args.tau, device='cuda')
    )
    total_num_neurons = sum(map(lambda x: x.num_neurons, logic_layers[1:]))
    print(f'total_num_neurons={total_num_neurons}')
    total_num_weights = sum(map(lambda x: x.num_weights, logic_layers[1:])) * 16
    print(f'total_num_weights={total_num_weights}')
    model = model.to(device)
    print(model)

    if best:
        train_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_torch, y_val_torch), batch_size=args.batch_size, shuffle=False)

    acc_val, f1_val, acc_train, f1_train, ewma_f1, acc_symbolic, f1_symbolic = train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.learning_rate,
        epochs_between_eval=args.epochs_between_eval,
        use_wandb=args.wandb
    )

    return acc_val, f1_val, acc_train, f1_train, ewma_f1, total_num_weights, f1_symbolic



def objective(trial, best=False):
    global args

    n_layers = trial.suggest_int('n_layers', 1, 10)
    layers_sizes = []
    previous = args.input_units
    for i in range(n_layers):
        previous = trial.suggest_int(f'n_units_{i}', max(previous // 2 + 1, 16), min(previous*(previous-1)//2, 512))
        layers_sizes.append(previous)
    rounding_to_multiple = lambda n,k: k * ((n + k - 1) // k)
    layers_sizes[-1] = rounding_to_multiple(layers_sizes[-1], args.output_units)
    net_structure = [args.input_units] + layers_sizes + [args.output_units]
    
    grad_factor = trial.suggest_float('grad_factor', 1.0, 2.0)
    args.learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
    args.tau = trial.suggest_float('tau', 1, 100)
    repetitions = args.repetitions

    id_list = []
    acc_list = []
    f1_list = []
    ewf1_list = []
    runtimes = []
    f1_symbolic_list = []
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

        start_time = time.time()
        acc_val, f1_val, acc_train, f1_train, ewma_f1, parameter_count, f1_symbolic  = experiment(args, net_structure, grad_factor, best)
        time_elapsed = time.time() - start_time
        
        acc_list.append(acc_val)
        f1_list.append(f1_val)
        ewf1_list.append(ewma_f1)
        f1_symbolic_list.append(f1_symbolic)
        runtimes.append(time_elapsed)

        trial.report(f1_val, repetition)
        if trial.should_prune():
            pruned = True
            if args.wandb:
                run.finish()
            raise optuna.TrialPruned()

        if args.wandb:
            wandb.summary["finished"] = True
            run.finish()
    
    
    print(f'Finished trial {trial.number}')
    mean_acc = np.mean(acc_list)
    mean_f1 = np.mean(f1_list)
    mean_ewf1 = np.mean(ewf1_list)
    mean_runtime = np.mean(runtimes)
    std_acc = np.std(acc_list)
    std_f1 = np.std(f1_list)
    std_ewf1 = np.std(ewf1_list)
    std_runtime = np.std(runtimes)
    mean_f1_symbolic = np.mean(f1_symbolic_list)
    std_f1_symbolic = np.std(f1_symbolic_list)

    if args.wandb:
        for run_id in id_list:
            run = wandb.init(id=run_id, project=experiment_name, resume="must")
            wandb.summary["mean_acc"] = mean_acc
            wandb.summary["mean_f1"] = mean_f1
            wandb.summary["mean_ewf1"] = mean_ewf1
            wandb.summary["std_acc"] = std_acc
            wandb.summary["std_f1"] = std_f1
            wandb.summary["std_ewf1"] = std_ewf1
            wandb.summary["pruned"] = pruned
            wandb.summary["repetitions"] = len(acc_list)
            wandb.summary["runtime"] = mean_runtime
            wandb.summary["std_runtime"] = std_runtime
            wandb.summary["mean_f1_symbolic"] = mean_f1_symbolic
            wandb.summary["std_f1_symbolic"] = std_f1_symbolic
            run.finish()

    if best:
        summary = {'mean_acc': mean_acc,
                    'mean_f1': mean_f1,
                    'mean_ewf1': mean_ewf1,
                    'mean_runtime': mean_runtime,
                    'std_acc': std_acc,
                    'std_f1': std_f1,
                    'std_ewf1': std_ewf1,
                    'std_runtime': std_runtime,
                    'mean_f1_symbolic': mean_f1_symbolic,
                    'std_f1_symbolic': std_f1_symbolic
                }
        hyper_values = dict(trial.params)
        whole_dict = {'experiment': experiment_name, 
                      'group': args.group, 
                      'data_set': args.data_set,
                      'epochs': args.epochs,
                      'repetitions': repetitions,
                      'seed': args.seed,
                      'batch_size': args.batch_size,
                      'epochs_between_eval': args.epochs_between_eval,
                      'hyperparameters': hyper_values, 
                      'results': summary,
                      'parameter_count': parameter_count
                      }

        write_results_to_file(whole_dict, RESULTS_FILE)

    return mean_ewf1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_set', type=str, default='tic-tac-toe',
                        help='Set the data set for training. All the data sets in the dataset folder are available.')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='Set the total epoch.')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Set the batch size.')
    parser.add_argument('--trials', type=int, default=80, help='Number of trials')
    parser.add_argument('--pruningafter', type=int, default=10, help='Number of trials before pruning')
    #parser.add_argument('--pruningwarmup', type=int, default=10, help='Number of warmup steps before pruning')
    parser.add_argument('--repetitions', type=int, default=1, help='Number of repetitions')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--group', type=str, default='Trials', help='Group name for wandb')
    parser.add_argument('--epochs_between_eval', type=int, default=10, help='Number of epochs between evaluations')
    parser.add_argument('--experimentname', type=str, default='difflogic', help='Name of the experiment')
    parser.add_argument('--wandb', action='store_true', help='Use weights and biases for logging')
    parser.add_argument('--nohypertuning', action='store_true', help='Use fixed hyperparameters instead of tuning')
    #parser.add_argument('--implementation', type=str, default='cuda', choices=['cuda', 'python'], help='`cuda` is the fast implementation and `python` is simpler but much slower.')
    parser.add_argument('--connections', type=str, default='unique', choices=['random', 'unique'])


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
    print('Data discretization and binarization are done.')

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
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=0, n_startup_trials=args.pruningafter)
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
        objective(optuna.trial.FixedTrial(hyperparams), best=True)
