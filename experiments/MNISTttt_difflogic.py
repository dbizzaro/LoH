import argparse
import numpy as np
import torch
import optuna
import json
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import wandb
import warnings
warnings.filterwarnings("ignore")

import os, sys
sys.path.append('LoH')
from experiments.MNISTttt import *
from experiments.mllp.mllp_utils import read_csv, DBEncoder
from difflogic import LogicLayer, GroupSum, CompiledLogicNet

from utils import *


RESULTS_FILE = 'LoH/experiments/results/MNISTttt_results_difflogic.json'
DATA_DIR = 'LoH/experiments/data'


class ConvDLNModel(nn.Module):
    def __init__(self, cnn_model, DLN_model):
        super(ConvDLNModel, self).__init__()
        self.cnn_model = cnn_model
        self.DLN_model = DLN_model
        self.module_list = nn.ModuleList([self.cnn_model, self.DLN_model])
        self.intermediate_units = cnn_model.output_units

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        cnn_output = self.cnn_model(x)
        cnn_output_flat = cnn_output.view(-1, 9*self.intermediate_units)
        andor_output = self.DLN_model(cnn_output_flat)
        return andor_output
    

def train(model, train_loader, val_loader, device, epochs=50, 
          lr_diff=0.01, lr_cnn=0.001, epochs_between_eval=10, use_wandb=False):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
            {'params': model.cnn_model.parameters(), 'lr': lr_cnn},
            {'params': model.DLN_model.parameters(), 'lr': lr_diff},
        ])

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
                
                compiled_model = CompiledLogicNet(model=model.DLN_model, num_bits=64, cpu_compiler='gcc', verbose=False)
                compiled_model.compile()
                outputs_val_c, predicted_val_c, labels_val_c = predict(compiled_model, symbolic_loader, device, compiled=True)
                acc_val_c, f1_val_c = eval(predicted_val_c, labels_val_c)

            step = epoch // epochs_between_eval
            ewma_f1 = alpha * ewma_f1 + (1. - alpha) * f1_val
            if use_wandb:
                wandb.log(data={'loss': running_loss,
                            "acc_train": acc_train, "f1_train": f1_train,
                            "acc_val": acc_val, "f1_val": f1_val,
                            "ewma_f1": ewma_f1, "f1_val_symb": f1_val_c}, step=step)
            model.train()
                   
    return acc_val, f1_val, acc_train, f1_train, ewma_f1, acc_val_c, f1_val_c


def experiment(args, net_structure, grad_factor, best=False):
    cnn = CNN(args.intermediateunits, args.use_softmax)
    logic_layers = []
    logic_layers.append(torch.nn.Flatten())
    for i in range(len(net_structure) - 2):
        logic_layers.append(LogicLayer(in_dim=net_structure[i], out_dim=net_structure[i+1], grad_factor=grad_factor, connections=args.connections, implementation='cuda', device='cuda'))
    DLN_model = torch.nn.Sequential(
        *logic_layers,
        GroupSum(net_structure[-1], args.tau, device='cuda')
    )
    #total_num_neurons = sum(map(lambda x: x.num_neurons, logic_layers[1:]))
    #print(f'total_num_neurons={total_num_neurons}')
    #total_num_weights = sum(map(lambda x: x.num_weights, logic_layers[1:])) * 16
    #print(f'total_num_weights={total_num_weights}')
    #model = model.to(device)
    #print(model)

    net = ConvDLNModel(cnn, DLN_model)
    net.to(device)

    if best:
        train_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_torch, y_val_torch), batch_size=args.batch_size, shuffle=False)

    acc_val, f1_val, acc_train, f1_train, ewma_f1, acc_symbolic, f1_symbolic = train(
        net,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr_cnn=args.lr_cnn,
        lr_diff=args.lr_diff,
        epochs_between_eval=args.epochs_between_eval,
        use_wandb=args.wandb
    )
    return acc_val, f1_val, acc_train, f1_train, ewma_f1, acc_symbolic, f1_symbolic



def objective(trial, best=False):
    global args

    if args.tune_n_bits:
        args.intermediateunits = trial.suggest_int('intermediate_units', 2, 4)

    args.input_units = 9 * args.intermediateunits
    args.output_units = 2
    
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
    args.tau = trial.suggest_float('tau', 1, 100)
    args.lr_diff = trial.suggest_float('lr_diff', 1e-3, 1e-1, log=True)
    args.lr_cnn = trial.suggest_float('lr_cnn', 1e-4, 1e-2, log=True)
    
    repetitions = args.repetitions if not best else 30
    

    id_list = []
    acc_list = []
    f1_list = []
    ewf1_list = []
    f1_sym_list = []
    runtimes = []
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
        acc_val, f1_val, acc_train, f1_train, ewma_f1, acc_symbolic, f1_symbolic  = experiment(args, net_structure, grad_factor, best)
        time_elapsed = time.time() - start_time
        
        acc_list.append(acc_val)
        f1_list.append(f1_val)
        ewf1_list.append(ewma_f1)
        runtimes.append(time_elapsed)
        f1_sym_list.append(f1_symbolic)


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
    mean_f1_sym = np.mean(f1_sym_list)
    std_f1_sym = np.std(f1_sym_list)

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
            wandb.summary["mean_f1_sym"] = mean_f1_sym
            wandb.summary["std_f1_sym"] = std_f1_sym
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
                    'f1_sym_mean': mean_f1_sym, 
                    'f1_sym_std': std_f1_sym
                }
        hyper_values = dict(trial.params)
        whole_dict = {'experiment': experiment_name, 
                      'group': args.group, 
                      'epochs': args.epochs,
                      'repetitions': repetitions,
                      'seed': args.seed,
                      'batch_size': args.batch_size,
                      'epochs_between_eval': args.epochs_between_eval,
                      'hyperparameters': hyper_values, 
                      'results': summary
                      }

        write_results_to_file(whole_dict, RESULTS_FILE)

    return mean_ewf1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST-tictactoe DLN Experiment')
    parser.add_argument('--experimentname', type=str, default='MNIST-TTT', help='Name of the experiment')
    parser.add_argument('--nodrop', type=bool, default=True, help='Do not drop values in one-hot encoding')
    parser.add_argument('--repetitions', type=int, default=2, help='Number of repetitions')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs')
    parser.add_argument('--epochs_between_eval', type=int, default=10, help='Number of epochs between evaluations')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--trials', type=int, default=80, help='Number of trials')
    parser.add_argument('--pruningafter', type=int, default=10, help='Number of trials before pruning')
    #parser.add_argument('--pruningwarmup', type=int, default=20, help='Number of warmup steps before pruning')
    parser.add_argument('--group', type=str, default='DLN_any', help='Group name for wandb')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--intermediateunits', type=int, default=3, help='Number of symbols returned by CNN')
    parser.add_argument('--tune_n_bits', action='store_true', help='Tune the number of outputs of the CNN as hyperparams')
    parser.add_argument('--use_softmax', action='store_true', help='Use softmax instead of sigmoid')
    parser.add_argument('--repetitions_dataset', type=int, default=2, help='Repetitions of tic-tac-toe dataset (with different imgs)')
    parser.add_argument('--nohypertuning', action='store_true', help='Avoid hyperparameter tuning and use sotred values')
    #parser.add_argument('--cnf_or_dnf', type=str, default='any', help='Type of formula to compute (cnf or dnf or any)')
    parser.add_argument('--wandb', action='store_true', help='Use weights and biases for logging')
    parser.add_argument('--connections', type=str, default='unique', choices=['random', 'unique'])
    args = parser.parse_args()

    experiment_name = args.experimentname
    n_epochs = args.epochs
    n_trials = args.trials
    batch_size = args.batch_size
    epochs_between_eval = args.epochs_between_eval
    n_steps = n_epochs // epochs_between_eval
    #args.group = f'{args.group}_{args.cnf_or_dnf}'

    intermediate_units = args.intermediateunits
    use_softmax = args.use_softmax
    seed = set_seed(args.seed)

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    mnist_train = filter_mnist(mnist_train)
    mnist_test = filter_mnist(mnist_test)

    X, Y = get_dataset('tic-tac-toe')
    label_enc = preprocessing.OneHotEncoder(categories='auto', sparse_output=False) if args.nodrop else preprocessing.OrdinalEncoder()
    label_enc.fit(Y.values)
    Y = Y.reset_index(drop=True)
    Y = label_enc.transform(Y.values)

    mapping = {
            'x': 0,  # Example mapping: 'x' -> digit 0
            'o': 1,  # Example mapping: 'o' -> digit 1
            'b': 2   # Example mapping: 'b' -> digit 2
        }
    inverse_mapping = ['X', 'O', 'B']

    X_train_val_symb, X_test_symb, y_train_val_symb, y_test_symb = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train_symb, X_val_symb, y_train_symb, y_val_symb = train_test_split(X_train_val_symb, y_train_val_symb, test_size=0.125, random_state=56)
    input_units = X_train_val_symb.shape[1]
    output_units = y_train_val_symb.shape[1]

    X_train_val, y_train_val = map_tic_tac_toe_to_mnist(X_train_val_symb, y_train_val_symb, mnist_train, mapping, args.repetitions_dataset)
    X_test, y_test = map_tic_tac_toe_to_mnist(X_test_symb, y_test_symb, mnist_test, mapping)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=56)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('Device: ', device)

    X_train_val_torch = torch.tensor(X_train_val, dtype=torch.float)#.to(device)
    y_train_val_torch = torch.tensor(y_train_val, dtype=torch.float)#.to(device)
    X_train_torch = torch.tensor(X_train, dtype=torch.float)#.to(device)
    X_val_torch = torch.tensor(X_val, dtype=torch.float)#.to(device)
    X_test_torch = torch.tensor(X_test, dtype=torch.float)#.to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.float)#.to(device)
    y_test_torch = torch.tensor(y_test, dtype=torch.float)#.to(device)
    y_val_torch = torch.tensor(y_val, dtype=torch.float)#.to(device)


    data_path = os.path.join(DATA_DIR, 'tic-tac-toe.data')
    info_path = os.path.join(DATA_DIR, 'tic-tac-toe.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)
    db_enc = DBEncoder(f_df, discrete=True)
    X_fname = db_enc.X_fname
    y_fname = db_enc.y_fname
    db_enc.fit(X_df, y_df)
    X_symb, y_symb = db_enc.transform(X_df, y_df)

    X_symb_torch = torch.tensor(X_symb, dtype=torch.float)
    y_symb_torch = torch.tensor(y_symb, dtype=torch.float)

    train_val_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_torch, y_val_torch), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size, shuffle=False)
    symbolic_loader = DataLoader(TensorDataset(X_symb_torch, y_symb_torch), batch_size=batch_size, shuffle=False)

    print(f'START {experiment_name}')
    print(args)
    print('')

    if not args.nohypertuning:
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, n_ei_candidates=10, n_startup_trials=n_trials//4, seed=123)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=0, n_startup_trials=args.pruningafter)
        study = optuna.create_study(direction='maximize', study_name=experiment_name, pruner = pruner, sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        print('hyperparameters: ', study.best_params)
        print('best validation f1 value: ', study.best_value)
        print('')


        seed = set_seed(args.seed)
        train_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size, shuffle=False)
        objective(optuna.trial.FixedTrial(study.best_params), best=True)

    else:
        hyperparams = get_hyperparameters(RESULTS_FILE, experiment_name, args.group)
        seed = set_seed(args.seed)
        args.group = f'{args.group}_new'
        train_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size, shuffle=False)
        objective(optuna.trial.FixedTrial(hyperparams), best=True)