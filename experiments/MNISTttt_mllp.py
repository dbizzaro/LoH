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

import sys
sys.path.append('LoH')
from experiments.mllp.mllp_models import MLLP
from experiments.MNISTttt import *
from utils import *

RESULTS_FILE = 'LoH/experiments/results/MNISTttt_results_mllp.json'


class ConvMLLPModel(nn.Module):
    def __init__(self, cnn_model, mllp_model):
        super(ConvMLLPModel, self).__init__()
        self.cnn_model = cnn_model
        self.mllp_model = mllp_model
        self.module_list = nn.ModuleList([self.cnn_model, self.mllp_model])
        self.intermediate_units = cnn_model.output_units

    def forward(self, x, *args, **kwargs):
        x = x.reshape(-1, 1, 28, 28)
        cnn_output = self.cnn_model(x)
        cnn_output_flat = cnn_output.view(-1, 9*self.intermediate_units)
        andor_output = self.mllp_model(cnn_output_flat, *args, **kwargs)
        return andor_output
    
    def binarized_forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        cnn_output = self.cnn_model(x)
        cnn_output_flat = cnn_output.view(-1, 9*self.intermediate_units)
        andor_output = self.mllp_model.binarized_forward(cnn_output_flat)
        return andor_output
    

def get_mllp_formula(combined_model, mnist_dataset, verbose=True):
    cnn_predictions, labels = get_cnn_predictions(combined_model, mnist_dataset)
    neuron_classes = classify_cnn_outputs_sigmoid(cnn_predictions, labels, verbose=verbose)
    prop_names = get_propositions(neuron_classes)
    starting = combined_model.mllp_model.starting
    rules = combined_model.mllp_model.get_rules()
    if verbose:
        print(rules)
        print(prop_names)
    clauses = []
    for value in rules[-1].values():
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
    return formula

def train(model, train_loader, val_loader, device, epochs=50, 
          lr=0.01, lr_decay_epoch=100, lr_decay_rate=0.75, weight_decay=0.0, 
          epochs_between_eval=10, seed=42, criterion=nn.MSELoss(), use_wandb=False):
    
    global mnist_test, mapping, inverse_mapping, X, Y
    
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
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model.forward(x, randomly_binarize=True)
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.mllp_model.clip()
        
        # Change the set of weights to be binarized every epoch.
        model.mllp_model.randomly_binarize_layer_refresh()
        
        # Evaluate the model
        if epoch % epochs_between_eval == 0:
            #model.eval()
            with torch.no_grad():
                outputs_train, predicted_train, labels_train = predict(model, train_loader, device)
                acc_train, f1_train = eval(predicted_train, labels_train)
                outputs_val, predicted_val, labels_val = predict(model, val_loader, device)
                acc_val, f1_val = eval(predicted_val, labels_val)

                outputs_train_b, predicted_train_b, labels_train_b = predict(model, train_loader, device, binarized_forward=True)
                acc_train_b, f1_train_b = eval(predicted_train_b, labels_train_b)
                outputs_val_b, predicted_val_b, labels_val_b = predict(model, val_loader, device, binarized_forward=True)
                acc_val_b, f1_val_b = eval(predicted_val_b, labels_val_b)

                formula = get_mllp_formula(model, mnist_test, verbose=True)
                formula_simplified = non_redundant_clauses(sympy.parse_expr(formula), starting=model.mllp_model.starting)
                acc_symbolic, f1_symbolic = check_formula(formula_simplified, X, Y, mapping, inverse_mapping)
                #except:
                #    formula = ''
                #    formula_simplified = ''
                #    acc_symbolic = np.nan
                #    f1_symbolic = np.nan
                print(epoch, f1_val_b, f1_symbolic)
                print(formula_simplified)
                print('')

            #model.train()

            step = epoch // epochs_between_eval
            ewma_f1_b = alpha * ewma_f1_b + (1. - alpha) * f1_val_b
            if use_wandb:
                wandb.log(data={'loss': running_loss,
                            "acc_train_b": acc_train_b, "f1_train_b": f1_train_b,
                            "acc_val_b": acc_val_b, "f1_val_b": f1_val_b,
                            "acc_train": acc_train, "f1_train": f1_train,
                            "acc_val": acc_val, "f1_val": f1_val,
                            "ewma_f1": ewma_f1_b, 
                            "formula": formula, "formula_simplified": str(formula_simplified), 
                            "acc_symb": acc_symbolic, "f1_symb": f1_symbolic}, step=step)
                   
    return acc_val, acc_val_b, f1_val, f1_val_b, ewma_f1_b, acc_train, acc_train_b, f1_train, f1_train_b, acc_symbolic, f1_symbolic, formula, str(formula_simplified)



def experiment(args, best=False):
    cnn = CNN(args.intermediateunits, args.use_softmax)
    net_structure = [args.input_units] + list(map(int, args.structure.split('_'))) + [args.output_units]
    mllp = MLLP(net_structure,
               device=device,
               random_binarization_rate=args.random_binarization_rate,
               use_not=args.use_not, 
               starting=args.starting)
    
    net = ConvMLLPModel(cnn, mllp)
    net.to(device)

    if best:
        train_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_torch, y_val_torch), batch_size=args.batch_size, shuffle=False)

    acc_val, acc_val_b, f1_val, f1_val_b, ewma_f1_b, acc_train, acc_train_b, f1_train, f1_train_b,  acc_symbolic, f1_symbolic, formula, formula_simplified = train(
        net,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.learning_rate,
        lr_decay_epoch=args.lr_decay_epoch,
        lr_decay_rate=args.lr_decay_rate,
        weight_decay=args.weight_decay,
        epochs_between_eval=args.epochs_between_eval,
        seed=args.seed,
        use_wandb=args.wandb
    )

    #with open(args.crs_file, 'w') as f:
    #    net.concept_rule_set_print(X_train, X_fname, y_fname, f)
    #torch.save(net.state_dict(), args.model)

    return acc_val, acc_val_b, f1_val, f1_val_b, ewma_f1_b, f1_symbolic, formula_simplified


def objective(trial, best=False):
    global args

    if args.tune_n_bits:
        args.intermediateunits = trial.suggest_int('intermediate_units', 2, 4)

    args.input_units = 9 * args.intermediateunits
    args.output_units = 1
    
    
    if args.cnf_or_dnf == 'cnf':
        args.starting = trial.suggest_categorical('starting', ['or'])
        n_layers = trial.suggest_int('n_layers', 1, 1)
    elif args.cnf_or_dnf == 'dnf':
        args.starting = trial.suggest_categorical('starting', ['and'])
        n_layers = trial.suggest_int('n_layers', 1, 1)
    else:
        args.starting = trial.suggest_categorical('starting', ['and', 'or'])
        n_layers = trial.suggest_categorical('n_layers', [1, 3])

    #n_layers = trial.suggest_categorical('n_layers', [1])
    layers_sizes = []
    for i in range(n_layers):
        layers_sizes.append(trial.suggest_int(f'n_units_{i}', 16, 256))
    args.structure = '_'.join(map(str, layers_sizes))
    args.weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
    args.learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    args.random_binarization_rate = trial.suggest_float('p', 0.0, 0.99)
    if args.use_not:
        args.use_not = trial.suggest_categorical('use_not', [True, False])
    
    repetitions = args.repetitions if not best else 30

    id_list = []
    acc_list = []
    f1_list = []
    acc_b_list = []
    f1_b_list = []
    ewf1_b_list = []
    f1_sym_list = []
    formulas_list = []
    pruned = False

    for repetition in range(repetitions):
        
        config = dict(trial.params)
        config["trial.number"] = trial.number
        config["repeat"] = repetition
        group = args.group if args.group != 'Trials' else f'trial_{trial.number}'
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

        acc_val, acc_val_b, f1_val, f1_val_b, ewma_f1_b, f1_symbolic, formula = experiment(args, best)
        
        acc_list.append(acc_val)
        f1_list.append(f1_val)
        acc_b_list.append(acc_val_b)
        f1_b_list.append(f1_val_b)
        ewf1_b_list.append(ewma_f1_b)
        f1_sym_list.append(f1_symbolic)
        formulas_list.append(formula)

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
    mean_ewf1_b = np.mean(ewf1_b_list)
    std_acc = np.std(acc_list)
    std_f1 = np.std(f1_list)
    std_ewf1_b = np.std(ewf1_b_list)
    mean_acc_b = np.mean(acc_b_list)
    mean_f1_b = np.mean(f1_b_list)
    std_acc_b = np.std(acc_b_list)
    std_f1_b = np.std(f1_b_list)
    mean_f1_sym = np.mean(f1_sym_list)
    std_f1_sym = np.std(f1_sym_list)

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
            wandb.summary["mean_f1_sym"] = mean_f1_sym
            wandb.summary["std_f1_sym"] = std_f1_sym
            run.finish()

    if best:
        summary = {'f1_mean': mean_f1, 'f1_std': std_f1,
                  'acc_mean': mean_acc, 'acc_std': std_acc,
                  'ewf1_mean': mean_ewf1_b, 'ewf1_std': std_ewf1_b,
                  'f1_b_mean': mean_f1_b, 'f1_b_std': std_f1_b,
                  'f1_sym_mean': mean_f1_sym, 'f1_sym_std': std_f1_sym
                  }
        hyper_values = dict(trial.params)
        whole_dict = {'experiment': experiment_name, 
                      'group': args.group,  
                      'epochs': args.epochs,
                      'repetitions': len(acc_list),
                      'seed': args.seed,
                      'batch_size': args.batch_size,
                      'lr_decay_rate': args.lr_decay_rate,
                      'lr_decay_epoch': args.lr_decay_epoch,
                      'epochs_between_eval': args.epochs_between_eval,
                      'repetitions_dataset': args.repetitions_dataset,
                      'hyperparameters': hyper_values, 
                      #'results': {'accuracies': acc_list, 'f1_scores': f1_list, 'formulas': list(zip(f1_sym_list, formulas_list))},
                      'summary': summary,
                      }

        write_results_to_file(whole_dict, RESULTS_FILE)

    return mean_ewf1_b



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST-tictactoe MLLP Experiment')
    parser.add_argument('--experimentname', type=str, default='VisualTTT_baseline', help='Name of the experiment')
    parser.add_argument('--nodrop', type=bool, default=True, help='Do not drop values in one-hot encoding')
    parser.add_argument('--repetitions', type=int, default=2, help='Number of repetitions')
    parser.add_argument('--epochs', type=int, default=600, help='Number of epochs')
    parser.add_argument('--epochs_between_eval', type=int, default=10, help='Number of epochs between evaluations')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--trials', type=int, default=80, help='Number of trials')
    parser.add_argument('--pruningafter', type=int, default=10, help='Number of trials before pruning')
    parser.add_argument('--pruningwarmup', type=int, default=50, help='Number of warmup steps before pruning')
    parser.add_argument('--group', type=str, default='mllp', help='Group name for wandb')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--intermediateunits', type=int, default=3, help='Number of symbols returned by CNN')
    parser.add_argument('--tune_n_bits', action='store_true', help='Tune the number of outputs of the CNN as hyperparams')
    parser.add_argument('--use_softmax', action='store_true', help='Use softmax instead of sigmoid')
    parser.add_argument('--repetitions_dataset', type=int, default=2, help='Repetitions of tic-tac-toe dataset (with different imgs)')
    parser.add_argument('--nohypertuning', action='store_true', help='Avoid hyperparameter tuning and use sotred values')
    parser.add_argument('--cnf_or_dnf', type=str, default='any', help='Type of formula to compute (cnf or dnf or any)')
    parser.add_argument('--wandb', action='store_true', help='Use weights and biases for logging')

    parser.add_argument('-lrdr', '--lr_decay_rate', type=float, default=0.75, help='Set the learning rate decay rate.')
    parser.add_argument('-lrde', '--lr_decay_epoch', type=int, default=100, help='Set the learning rate decay epoch.')
    parser.add_argument('--use_not', action="store_true",
                        help='Use the NOT (~) operator in logical rules. '
                             'It will enhance model capability but make the CRS more complex.')

    args = parser.parse_args()

    experiment_name = args.experimentname
    n_epochs = args.epochs
    n_trials = args.trials
    pruning_after = args.pruningafter
    pruning_warmup = args.pruningwarmup
    batch_size = args.batch_size
    epochs_between_eval = args.epochs_between_eval
    n_steps = n_epochs // epochs_between_eval
    nodrop = args.nodrop
    args.group = f'{args.group}_{args.cnf_or_dnf}'

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

    if nodrop:
        mnist_train = filter_mnist(mnist_train)
        mnist_test = filter_mnist(mnist_test)
    else:
        mnist_train = filter_mnist(mnist_train, labels=[0, 1])
        mnist_test = filter_mnist(mnist_test, labels=[0, 1])

    X, Y = get_dataset('tic-tac-toe')
    y_one_hot = len(Y.value_counts()) > 2
    drop = None if args.nodrop else ['b'] * X.shape[1]
    label_enc = preprocessing.OneHotEncoder(categories='auto', sparse_output=False) if y_one_hot else preprocessing.OrdinalEncoder()
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

    if args.wandb:
        api = wandb.Api()

    criterion = torch.nn.BCELoss()
    train_val_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_torch, y_val_torch), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size, shuffle=False)

    print(f'START {experiment_name}')
    print(args)
    print('')

    if not args.nohypertuning:
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, n_ei_candidates=10, n_startup_trials=n_trials//4, seed=123)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=pruning_warmup, n_startup_trials=pruning_after)
        study = optuna.create_study(direction='maximize', study_name=experiment_name, 
                                    pruner = pruner, sampler=sampler) #storage="sqlite:///db.sqlite3", load_if_exists=False)
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
        args.group = f'new_{args.group}'
        train_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size, shuffle=False)
        objective(optuna.trial.FixedTrial(hyperparams), best=True)