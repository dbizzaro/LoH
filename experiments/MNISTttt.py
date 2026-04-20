import argparse
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import optuna
import wandb
import json
import time

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('LoH')

from models import *
from utils import *
from layers import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inverse_mapping = ['X', 'O', 'B']

RESULTS_FILE = 'LoH/experiments/results/MNISTttt_results.json'


def filter_mnist(dataset, labels=[0, 1, 2]):
    # Filter MNIST dataset to get images of 0, 1, and 2
    indices = [i for i, target in enumerate(dataset.targets) if target in labels]
    dataset.data = dataset.data[indices]
    dataset.targets = dataset.targets[indices]
    return dataset


def get_mnist_image(digit, dataset, places):
    """
    Yields the first non-used MNIST image matching the given digit from the dataset.
    """
    try:
        indices = np.where(dataset.targets == digit)[0]
        index = indices[places[digit]]
        places[digit] += 1
        return dataset[index][0][0]
    except:
        print(f'Exceded mnist dataset size, with index {places[digit]}/{len(indices)}')
    

def map_tic_tac_toe_to_mnist(X, y, mnist, mapping, repetitions=1):
    """
    Map the Tic Tac Toe dataset to MNIST images.
    """
    
    mnist_X = []
    mnist_y = []
    places = [0,0,0]
    
    for _ in range(repetitions):
        for index, row in enumerate(X):
            mnist_row = []
            for cell_value in row:
                digit = mapping[cell_value]
                image = get_mnist_image(digit, mnist, places)
                mnist_row.append(image.numpy())
            mnist_X.append(np.stack(mnist_row))
            mnist_y.append(y[index])

    print(f"used {places} out of {np.unique(mnist.targets.numpy(), return_counts=True)[1]} images per category")
    return np.array(mnist_X), np.array(mnist_y)


class CNN(nn.Module):
    def __init__(self, output_units=3, use_softmax=False, embeddings=False, device=None):
        super(CNN, self).__init__()
        self.output_units = output_units
        self.use_softmax = use_softmax
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, output_units)
        self.dropout = nn.Dropout(0.5)
        self.classifier = Classifier(3, device=device, uniform_noise_bounds=[0.75,1.25])
        self.embeddings = embeddings
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        if self.output_units > 1 and self.use_softmax:
            x = torch.sigmoid(x)
            return self.classifier(x, weight_random=0.3)
        elif not self.embeddings:
            return torch.clip(x, 0., 1.)
        else:
            return F.relu(x)
    

class CombinedModel(nn.Module):
    def __init__(self, cnn_model, andor_model):
        super(CombinedModel, self).__init__()
        self.cnn_model = cnn_model
        self.andor_model = andor_model
        self.module_list = nn.ModuleList([self.cnn_model, self.andor_model])
        self.intermediate_units = cnn_model.output_units

    def forward(self, x, *args, **kwargs):
        x = x.reshape(-1, 1, 28, 28)
        cnn_output = self.cnn_model(x)
        cnn_output_flat = cnn_output.view(-1, 9*self.intermediate_units)
        andor_output = self.andor_model(cnn_output_flat, *args, **kwargs)
        return andor_output
    


def get_cnn_predictions(combined_model, mnist_dataset):
    combined_model.cnn_model.eval()
    cnn_predictions = []
    labels = []
    with torch.no_grad():
        for image, label in mnist_dataset:
            image = image.unsqueeze(0).to(device)
            cnn_output = combined_model.cnn_model(image)
            cnn_predictions.append(cnn_output.cpu().numpy()[0])
            labels.append(label)
    return np.array(cnn_predictions), np.array(labels)


def produce_proposition_string(binary_predictions):
    global inverse_mapping
    true_propositions = np.array(inverse_mapping)[binary_predictions]
    false_propositions = np.array(inverse_mapping)[~binary_predictions]
    if len(true_propositions) == 0:
        neuron_string = 'False'
    elif len(true_propositions) == 1:
            neuron_string = true_propositions[0]
    elif len(true_propositions) == 2:
        neuron_string = '~ ' + false_propositions[0]
    else:
        neuron_string = 'True'
    return neuron_string


def classify_cnn_outputs_sigmoid(cnn_predictions, labels, verbose=True):
    mean_cnn_predictions = []
    for label in [0,1,2]:
        mean_cnn_predictions.append(np.mean(cnn_predictions[labels == label], axis=0))
    mean_cnn_predictions = np.stack(mean_cnn_predictions, axis=1)
    if verbose:
        print('Mean CNN predictions for each label:')
        print(mean_cnn_predictions)

    neuron_strings = []
    for neuron, mean_prediction in enumerate(mean_cnn_predictions):
        binary_predictions = mean_prediction > 0.5
        neuron_string = produce_proposition_string(binary_predictions)
        if verbose:
            print(f'Neuron {neuron}: {neuron_string}')
        neuron_strings.append(neuron_string)
    return neuron_strings


def classify_cnn_outputs_softmax(cnn_predictions, labels, verbose=True):
    confusion_list = []
    for label in [0,1,2]:
        predictions = cnn_predictions[labels == label]
        argmaxes = np.argmax(predictions, axis=1)
        counts = np.bincount(argmaxes, minlength=cnn_predictions.shape[1])
        confusion_list.append(counts)
    confusion_matrix = np.stack(confusion_list, axis=1)
    if verbose:
        print('Confusion matrix:')
        print(confusion_matrix)

    max_per_target = np.max(confusion_matrix, axis=0)
    neuron_strings = []
    for neuron, confusion in enumerate(confusion_matrix):
        binary_predictions = max_per_target == confusion
        neuron_string = produce_proposition_string(binary_predictions)
        if verbose:
            print(f'Neuron {neuron}: {neuron_string}')
        neuron_strings.append(neuron_string)
    return neuron_strings
        
def get_propositions(neurons_classes):
    prop_array = []
    for i in range(9):
        sub_list = []
        for val in neurons_classes:
            if val not in {'True', 'False'}:
                sub_list.append(val + f'_{i+1}')
            else:
                sub_list.append(val)        
        prop_array.append(np.array(sub_list))
    prop_array = np.stack(prop_array, axis = 0)
    return prop_array.reshape(-1, 9*len(neurons_classes)).ravel()


def get_formula(combined_model, mnist_dataset, use_softmax, verbose=True):
    cnn_predictions, labels = get_cnn_predictions(combined_model, mnist_dataset)
    if use_softmax:
        neuron_classes = classify_cnn_outputs_softmax(cnn_predictions, labels, verbose=verbose)
    else:
        neuron_classes = classify_cnn_outputs_sigmoid(cnn_predictions, labels, verbose=verbose)
    prop_names = get_propositions(neuron_classes)
    if verbose:
        print(combined_model.andor_model.to_formula(features_names=prop_names)[0])
    return combined_model.andor_model.to_formula(features_names=prop_names)[0]


def produce_substitutions(sample, mapping, inverse_mapping):
    substitutions = {}
    for i in range(len(sample)):
        for key, value in mapping.items():
            substitutions[inverse_mapping[value]+'_'+str(i+1)] = sample[i] == key
    return substitutions

def check_formula(formula, X_test, y_test, mapping, inverse_mapping):
    predictions = []
    for sample in X_test:
        try:
            substitutions = produce_substitutions(sample, mapping, inverse_mapping)
            predictions.append(bool(formula.subs(substitutions))+0)
        except:
            predictions.append(formula+0)
    predictions = np.array(predictions).reshape(-1, 1)
    return eval(predictions, y_test)
    

def objective(trial, verbose=False, best=False):
    global args

    if args.cnf_or_dnf == 'cnf':
        starting = trial.suggest_categorical('starting', ['or'])
        n_layers = trial.suggest_int('n_layers', 2, 2)
    elif args.cnf_or_dnf == 'dnf':
        starting = trial.suggest_categorical('starting', ['and'])
        n_layers = trial.suggest_int('n_layers', 2, 2)
    else:
        starting = trial.suggest_categorical('starting', ['and', 'or'])
        n_layers = trial.suggest_int('n_layers', 2, 3)

    layers_type = trial.suggest_categorical('layer_type', ['without ks', 'with ks'])
    if args.tune_n_bits:
        intermediate_units = trial.suggest_int('intermediate_units', 2, 4)
    else:
        intermediate_units = args.intermediate_units

    if layers_type == 'without ks':
        ks = None
    else:
        ks = []
    n_units = [intermediate_units * 9]
    for i in range(n_layers):
        if i == n_layers - 1:
            n_units.append(output_units)
        else:
            n_units.append(trial.suggest_int(f'n_units_{i}', 16, 256))
        if ks is not None:
            ks.append(trial.suggest_int(f'k_{i}', 4, 10))


    lr_andor = trial.suggest_float('lr', 0.01, 0.2)
    lr_cnn = trial.suggest_float('lr_cnn', 1e-5, 1e-3, log=True)
    noise_w = trial.suggest_float('noise_w', 0.4, 1.2)
    temp = trial.suggest_float('temp', 0.4, 1.2)
    multiplicative_coef_temp = trial.suggest_float('multiplicative_coef_temp', 0.9925, 1.)
    min_temp = trial.suggest_float('min_temp', 0.1, 0.4)

    alpha = np.exp(-6. / n_steps) # decay factor for smoothing of the f1 on the validation set
    pruned = False

    id_list = []
    acc_list = []
    f1_list = []
    ewf1_list = []
    f1_sym_list = []
    formulas_list = []
    runtimes = []

    repetitions = args.repetitions if not best else 30

    for repeat in range(repetitions):
        cnn_model = CNN(intermediate_units, use_softmax, device=device)
        andor_model = AndOrModel(n_units, ks, device, starting)
        
        combined_model = CombinedModel(cnn_model, andor_model).to(device)

        optimizer = torch.optim.Adam([
            {'params': combined_model.cnn_model.parameters(), 'lr': lr_cnn},
            {'params': combined_model.andor_model.parameters(), 'lr': lr_andor},
        ])

        config = dict(trial.params)
        config["trial.number"] = trial.number
        config["repeat"] = repeat
        group = args.group
        job_type = 'Trials' if not best else 'Best'
        name_trial = f'trial_{trial.number}_{repeat}' if not best else f'best_{repeat}'
        
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

        ewma_f1 = 0.0
        temp = trial.params['temp']
        combined_model.train()
        start_time = time.time()

        for epoch in range(n_epochs):
            for i, (features, labels) in enumerate(train_loader):
                features = features.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
            
                outputs = combined_model(features, noise_w, temp)

                loss = criterion(outputs, labels)
            
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            if epoch % epochs_between_eval == 0:
                combined_model.eval()
                with torch.no_grad():
                    outputs_train, predicted_train, labels_train = predict(combined_model, train_loader, device)
                    outputs_val, predicted_val, labels_val = predict(combined_model, val_loader, device)
                    acc_train, f1_train = eval(predicted_train, labels_train)
                    acc_val, f1_val = eval(predicted_val, labels_val)

                    if args.formulas:
                        try:
                            if n_layers == 3:
                                raise Exception('formula for 3 layers not supported')
                            formula = get_formula(combined_model, mnist_test, use_softmax=use_softmax, verbose=verbose)
                            formula_simplified = non_redundant_clauses(sympy.parse_expr(formula), starting=starting)
                            acc_symbolic, f1_symbolic = check_formula(formula_simplified, X, y, mapping, inverse_mapping)
                        except:
                            formula = ''
                            formula_simplified = ''
                            acc_symbolic = np.nan
                            f1_symbolic = np.nan
                    else:
                        formula = ''
                        formula_simplified = ''
                        acc_symbolic = np.nan
                        f1_symbolic = np.nan
            
                combined_model.train()

                step = epoch // epochs_between_eval
                ewma_f1 = alpha * ewma_f1 + (1. - alpha) * f1_val
                if args.wandb:
                    wandb.log(data={"acc_val": acc_val, "f1_val": f1_val, "ewma_f1": ewma_f1, 
                                "loss": loss.item(), "acc_train": acc_train, "f1_train": f1_train,
                                "formula": formula, "formula_simplified": str(formula_simplified), 
                                "acc_symb": acc_symbolic, "f1_symb": f1_symbolic}, step=step)
                if verbose:
                    print(f'Epoch [{epoch}/{n_epochs}], Loss: {loss.item():.5f}')
                    print(f'train acc = {acc_train:.2f} %, val acc = {acc_val:.2f} %')
                    print(f'f1 train = {f1_train:.2f}, f1 val = {f1_val:.2f}')
                    print(f'symbolic acc = {acc_symbolic:.2f} %, symbolic f1 = {f1_symbolic:.3f}')
                    print(formula_simplified)
            
                trial.report(ewma_f1, n_steps * repeat + step)
                if trial.should_prune():
                    pruned = True
                    run.finish()
                    raise optuna.TrialPruned()
        
            if temp > min_temp:
                temp *= multiplicative_coef_temp
        
        time_elapsed = time.time() - start_time
        runtimes.append(time_elapsed)
        if verbose:
            print('Training finished -----------------------------')
        acc_list.append(acc_val)
        f1_list.append(f1_val)
        ewf1_list.append(ewma_f1)
        f1_sym_list.append(f1_symbolic)
        formulas_list.append(str(formula_simplified))
        if args.wandb:
            wandb.summary["time"] = time_elapsed
            wandb.summary["finished"] = True
            run.finish()

    print(f'Finished trial {trial.number}')
    mean_runtime = np.mean(runtimes)
    mean_acc = np.mean(acc_list)
    mean_f1 = np.mean(f1_list)
    mean_ewf1 = np.mean(ewf1_list)
    mean_f1_sym = np.mean(f1_sym_list)
    std_acc = np.std(acc_list)
    std_f1 = np.std(f1_list)
    std_ewf1 = np.std(ewf1_list)
    std_f1_sym = np.std(f1_sym_list)
    std_runtime = np.std(runtimes)
    
    if args.wandb:
        for run_id in id_list:
            run = wandb.init(id=run_id, project=experiment_name, resume="must")
            wandb.summary["mean_acc"] = mean_acc
            wandb.summary["mean_f1"] = mean_f1
            wandb.summary["mean_ewf1"] = mean_ewf1
            wandb.summary["std_acc"] = std_acc
            wandb.summary["std_f1"] = std_f1
            wandb.summary["std_ewf1"] = std_ewf1
            wandb.summary["mean_f1_sym"] = mean_f1_sym
            wandb.summary["std_f1_sym"] = std_f1_sym
            wandb.summary["pruned"] = pruned
            wandb.summary["repetitions"] = len(acc_list)
            wandb.summary["mean_runtime"] = mean_runtime
            wandb.summary["std_runtime"] = std_runtime
            run.finish()


    if best:
        summary = {'f1_mean': mean_f1, 'f1_std': std_f1,
                  'acc_mean': mean_acc, 'acc_std': std_acc,
                  'ewf1_mean': mean_ewf1, 'ewf1_std': std_ewf1,
                  'f1_sym_mean': mean_f1_sym, 'f1_sym_std': std_f1_sym,
                  'runtime_mean': mean_runtime, 'runtime_std': std_runtime,
                  }
        hyper_values = dict(trial.params)
        whole_dict = {'experiment': experiment_name, 
                      'group': args.group,  
                      'epochs': args.epochs,
                      'repetitions': len(acc_list),
                      'seed': args.seed,
                      'batch_size': args.batchsize,
                      'epochs_between_eval': args.epochsbetweeneval,
                      'repetitions_dataset': args.repetitions_dataset,
                      'hyperparameters': hyper_values, 
                      #'results': {'accuracies': acc_list, 'f1_scores': f1_list, 'formulas': list(zip(f1_sym_list, formulas_list))},
                      'summary': summary,
                      'parameter_count': andor_model.parameter_count()
                      }

        write_results_to_file(whole_dict, RESULTS_FILE)

    return mean_ewf1



class CNN_model(torch.nn.Module):
    def __init__(self, intermediate_units, dense_layers, use_softmax=False, embeddings=True):
        super(CNN_model, self).__init__()
        self.CNN = CNN(intermediate_units, use_softmax, embeddings)
        self.dense_layers = torch.nn.Sequential(*dense_layers)
        self.intermediate_units = intermediate_units
    
    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.CNN(x)
        x = x.view(-1, self.intermediate_units*9)
        x = self.dense_layers(x)
        return x
        

def objective_nn(trial, verbose=False, best=False):
    global args
    
    criterion = torch.nn.BCELoss()
    n_images = 9

    intermediate_units = trial.suggest_int('intermediate_units', 4, 32)

    n_layers = trial.suggest_int('n_layers', 1, 3)
    n_units = [intermediate_units*n_images]
    layers = []
    for i in range(n_layers):
        n_units.append(trial.suggest_int(f'n_units_{i}', 4, 128))
        layers.append(torch.nn.Linear(n_units[i], n_units[i+1]))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(n_units[-1], output_units))
    if output_units == 1:
        layers.append(torch.nn.Sigmoid())
    else:
        layers.append(torch.nn.Softmax())
    
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

    alpha = np.exp(-6. / n_steps) # decay factor for smoothing of the f1 on the validation set
    pruned = False

    id_list = []
    acc_list = []
    f1_list = []
    ewf1_list = []

    repetitions = args.repetitions if not best else 30

    for repeat in range(repetitions):

        config = dict(trial.params)
        config["trial.number"] = trial.number
        config["repeat"] = repeat
        group = args.group
        job_type = 'Trials' if not best else 'Best'
        name_trial = f'trial_{trial.number}_{repeat}' if not best else f'best_{repeat}'

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

        model = CNN_model(intermediate_units, layers, use_softmax).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        ewma_f1 = 0.0
    
        model.train()
        for epoch in range(n_epochs):
            for i, (features, labels) in enumerate(train_loader):
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % epochs_between_eval == 0:
                model.eval()
                with torch.no_grad():
                    outputs_train, predicted_train, labels_train = predict(model, train_loader, device)
                    acc_train, f1_train = eval(predicted_train, labels_train)
                    outputs_val, predicted_val, labels_val = predict(model, val_loader, device)
                    acc_val, f1_val = eval(predicted_val, labels_val)
                model.train()

                step = epoch // epochs_between_eval
                ewma_f1 = alpha * ewma_f1 + (1. - alpha) * f1_val

                if args.wandb:
                    wandb.log(data={"acc_val": acc_val, "f1_val": f1_val, "ewma_f1": ewma_f1, 
                                "loss": loss.item(), "acc_train": acc_train, "f1_train": f1_train}, step=step)
                if verbose:
                    print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.5f}')
                    print(f'train acc = {acc_train:.2f} %, val acc = {acc_val:.2f} %')
                    print(f'f1 train = {f1_train:.3f}, f1 val = {f1_val:.3f}')
                
                trial.report(ewma_f1, n_steps * repeat + step)
                if trial.should_prune():
                    pruned = True
                    run.finish()
                    raise optuna.TrialPruned()

                
        acc_list.append(acc_val)
        f1_list.append(f1_val)
        ewf1_list.append(ewma_f1)
        if args.wandb:
            wandb.summary["finished"] = True
            run.finish()
        if verbose:
            print('')

    
    print(f'Finished trial {trial.number}')
    mean_acc = np.mean(acc_list)
    mean_f1 = np.mean(f1_list)
    mean_ewf1 = np.mean(ewf1_list)
    std_acc = np.std(acc_list)
    std_f1 = np.std(f1_list)
    std_ewf1 = np.std(ewf1_list)
    
    if args.wandb:
        for run_id in id_list:
            run = wandb.init(id=run_id, project=experiment_name, resume="must")
            wandb.summary["mean_acc"] = mean_acc
            wandb.summary["mean_f1"] = mean_f1
            wandb.summary["mean_ewf1"] = mean_ewf1
            wandb.summary["std_acc"] = std_acc
            wandb.summary["std_f1"] = std_f1
            wandb.summary["std_ewf1"] = std_ewf1
            run.finish()

    if best:
        summary = {'f1_mean': mean_f1, 'f1_std': std_f1,
                  'acc_mean': mean_acc, 'acc_std': std_acc,
                  'ewf1_mean': mean_ewf1, 'ewf1_std': std_ewf1}
        hyper_values = dict(trial.params)
        whole_dict = {'experiment': experiment_name, 
                      'group': args.group,  
                      'epochs': args.epochs,
                      'repetitions': len(acc_list),
                      'seed': args.seed,
                      'batch_size': args.batchsize,
                      'epochs_between_eval': args.epochsbetweeneval,
                      'hyperparameters': hyper_values, 
                      #'results': {'accuracies': acc_list, 'f1_scores': f1_list},
                      'summary': summary}

        write_results_to_file(whole_dict, RESULTS_FILE)

    return mean_ewf1




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST-tictactoe Experiment')
    parser.add_argument('--experimentname', type=str, default='VisualTTT', help='Name of the experiment')
    parser.add_argument('--nodrop', type=bool, default=True, help='Do not drop values in one-hot encoding')
    parser.add_argument('--repetitions', type=int, default=2, help='Number of repetitions')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs')
    parser.add_argument('--epochsbetweeneval', type=int, default=10, help='Number of epochs between evaluations')
    parser.add_argument('--batchsize', type=int, default=256, help='Batch size')
    parser.add_argument('--trials', type=int, default=80, help='Number of trials')
    parser.add_argument('--pruningafter', type=int, default=20, help='Number of trials before pruning')
    parser.add_argument('--pruningwarmup', type=int, default=20, help='Number of warmup steps before pruning')
    parser.add_argument('--group', type=str, default='Ours', help='Group name for wandb')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--intermediate_units', type=int, default=3, help='Number of symbols returned by CNN')
    parser.add_argument('--tune_n_bits', action='store_true', help='Tune the number of outputs of the CNN as hyperparams')
    parser.add_argument('--softmax', action='store_true', help='Use softmax instead of sigmoid')
    parser.add_argument('--nn_baseline', action='store_true', help='Use standard NN baseline instead of GodelNN')
    parser.add_argument('--repetitions_dataset', type=int, default=2, help='Repetitions of tic-tac-toe dataset (with different imgs)')
    parser.add_argument('--nohypertuning', action='store_true', help='Avoid hyperparameter tuning and use sotred values')
    parser.add_argument('--formulas', action='store_true', help='Compute and store formulas')
    parser.add_argument('--cnf_or_dnf', type=str, default='any', help='Type of formula to compute (cnf or dnf or any)')
    parser.add_argument('--wandb', action='store_true', help='Use weights and biases for logging')

    args = parser.parse_args()

    experiment_name = args.experimentname
    n_epochs = args.epochs
    n_trials = args.trials
    pruning_after = args.pruningafter
    pruning_warmup = args.pruningwarmup
    batch_size = args.batchsize
    epochs_between_eval = args.epochsbetweeneval
    n_steps = n_epochs // epochs_between_eval
    nodrop = args.nodrop
    args.group = f'{args.group}_{args.cnf_or_dnf}'

    intermediate_units = args.intermediate_units
    use_softmax = args.softmax
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

    X, y = get_dataset('tic-tac-toe')
    y_one_hot = len(y.value_counts()) > 2 #or not args.binary_classification
    drop = None if args.nodrop else ['b'] * X.shape[1]
    label_enc = preprocessing.OneHotEncoder(categories='auto', sparse_output=False) if y_one_hot else preprocessing.OrdinalEncoder()
    label_enc.fit(y.values)
    y = y.reset_index(drop=True)
    y = label_enc.transform(y.values)

    mapping = {
            'x': 0,  # Example mapping: 'x' -> digit 0
            'o': 1,  # Example mapping: 'o' -> digit 1
            'b': 2   # Example mapping: 'b' -> digit 2
        }
    inverse_mapping = ['X', 'O', 'B']

    X_train_val_symb, X_test_symb, y_train_val_symb, y_test_symb = train_test_split(X, y, test_size=0.2, random_state=42)
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

    X_train_val_torch = torch.tensor(X_train_val, dtype=torch.float)
    y_train_val_torch = torch.tensor(y_train_val, dtype=torch.float)
    X_train_torch = torch.tensor(X_train, dtype=torch.float)
    X_val_torch = torch.tensor(X_val, dtype=torch.float)
    X_test_torch = torch.tensor(X_test, dtype=torch.float)
    y_train_torch = torch.tensor(y_train, dtype=torch.float)
    y_test_torch = torch.tensor(y_test, dtype=torch.float)
    y_val_torch = torch.tensor(y_val, dtype=torch.float)

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

    if not args.nn_baseline:
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
            args.formulas = True
            train_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size, shuffle=False)
            objective(optuna.trial.FixedTrial(study.best_params), verbose=True, best=True)
        else:
            hyperparams = get_hyperparameters(RESULTS_FILE, experiment_name, args.group)
            args.formulas = True
            seed = set_seed(args.seed)
            args.group = f'new_{args.group}'
            train_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size, shuffle=False)
            objective(optuna.trial.FixedTrial(hyperparams), verbose=True, best=True)
            
    else:
        #nn
        if not args.nohypertuning:
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=pruning_warmup, n_startup_trials=pruning_after)
            study_mlp = optuna.create_study(direction='maximize')
            study_mlp.optimize(objective_nn, n_trials=n_trials, show_progress_bar=False)
            print(study_mlp.best_params)
            print(study_mlp.best_value)
            print('')

            # use the best hyperparameters, retraining on the whole train+validation dataset
            seed = set_seed(args.seed)
            train_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size, shuffle=False)
            objective_nn(optuna.trial.FixedTrial(study_mlp.best_params), verbose=True, best=True)
        else:
            hyperparams =  {'intermediate_units': 22, 'n_layers': 2, 'n_units_0': 41, 'n_units_1': 88, 'lr': 0.0037435600351631323}
            seed = set_seed(args.seed)
            train_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size, shuffle=False)
            objective_nn(optuna.trial.FixedTrial(hyperparams), verbose=True, best=True)


