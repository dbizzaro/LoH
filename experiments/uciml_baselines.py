import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.naive_bayes import BernoulliNB

import warnings
warnings.filterwarnings("ignore")
import sys, os
sys.path.append('LoH')

from utils import *
from experiments.mllp.mllp_utils import read_csv, DBEncoder

DATA_DIR = 'LoH/experiments/data'

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

parser = argparse.ArgumentParser(description='UCIML Experiment')
parser.add_argument('--dataset', type=str, default='tic-tac-toe', help='Name of the dataset')
parser.add_argument('--experimentname', type=str, default='Baselines', help='Name of the experiment')
parser.add_argument('--repetitions', type=int, default=10, help='Number of repetitions')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--epochsbetweeneval', type=int, default=10, help='Number of epochs between evaluations')
parser.add_argument('--batchsize', type=int, default=256, help='Batch size')
parser.add_argument('--trials', type=int, default=80, help='Number of trials')
parser.add_argument('--pruningafter', type=int, default=15, help='Number of trials before pruning')
parser.add_argument('--pruningwarmup', type=int, default=10, help='Number of warmup steps before pruning')
parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

args = parser.parse_args()

dataset_name = args.dataset
experiment_name = dataset_name + '_' + args.experimentname
n_epochs = args.epochs
n_trials = args.trials
pruning_after = args.pruningafter
pruning_warmup = args.pruningwarmup
batch_size = args.batchsize
epochs_between_eval = args.epochsbetweeneval
n_steps = n_epochs // epochs_between_eval
set_seed(args.seed)


print(f'START {dataset_name}')
print(f'Running experiment {experiment_name} on dataset {dataset_name} with {n_trials} trials and {n_epochs} epochs')

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print('Device: ', device)
print('')

data_path = os.path.join(DATA_DIR, dataset_name + '.data')
info_path = os.path.join(DATA_DIR, dataset_name + '.info')
print('Discretizing and binarizing data. Please wait ...')
X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)
db_enc = DBEncoder(f_df, discrete=True)
db_enc.fit(X_df, y_df)
X, y = db_enc.transform(X_df, y_df)
print('Data discretization and binarization are done.')

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=56)
input_units = X_train.shape[1]
output_units = y_train.shape[1]


def objective_dt(trial):
    min_samples_split = trial.suggest_int('min_samples_split', 2, 50)
    max_depth = trial.suggest_int('max_depth', 2, 50)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 50)
    model = DecisionTreeClassifier(min_samples_split=min_samples_split, 
                                   max_depth=max_depth, 
                                   min_samples_leaf=min_samples_leaf, 
                                   random_state=99)
    model.fit(X_train, y_train)
    #score = model.score(X_val, y_val)
    score = f1_score(y_val, model.predict(X_val), average='macro')
    return score

# Optimize DecisionTree
study_dt = optuna.create_study(direction='maximize')
study_dt.optimize(objective_dt, n_trials=n_trials)
print('Best trial for DecisionTree:')
trial_dt = study_dt.best_trial
print(f'  Value: {trial_dt.value}')
print('  Params: ')
for key, value in trial_dt.params.items():
    print(f'    {key}: {value}')
res =[]
set_seed(args.seed)
for i in range(args.repetitions):
    best_dt = DecisionTreeClassifier(min_samples_split=trial_dt.params['min_samples_split'],
                                     max_depth=trial_dt.params['max_depth'],
                                     min_samples_leaf=trial_dt.params['min_samples_leaf'])
    best_dt.fit(X_train_val, y_train_val)
    print(f'Best DecisionTree accuracy = {best_dt.score(X_test, y_test) * 100:.2f} %')
    f1_score_dt = f1_score(y_test, best_dt.predict(X_test), average='macro')
    print(f"Best DecisionTree F1 score = {f1_score_dt:.3f}")
    res.append(f1_score_dt)
print('mean:', np.mean(res))
print('std:', np.std(res))
print('')





def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 50)
    max_depth = trial.suggest_int('max_depth', 2, 50)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 50)
    model = RandomForestClassifier(n_estimators=n_estimators, 
                                    min_samples_split=min_samples_split, 
                                    max_depth=max_depth, 
                                    min_samples_leaf=min_samples_leaf,
                                    n_jobs=-1, random_state=99)
    model.fit(X_train, y_train)
    #score = model.score(X_val, y_val)
    score = f1_score(y_val, model.predict(X_val), average='macro')
    return score

# Optimize RandomForest
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=n_trials)
print('Best trial for RandomForest:')
trial_rf = study_rf.best_trial
print(f'  Value: {trial_rf.value}')
print('  Params: ')
for key, value in trial_rf.params.items():
    print(f'    {key}: {value}')
res =[]
set_seed(args.seed)
for i in range(args.repetitions):
    best_rf = RandomForestClassifier(n_estimators=trial_rf.params['n_estimators'],
                                     min_samples_split=trial_rf.params['min_samples_split'],
                                     max_depth=trial_rf.params['max_depth'],
                                     min_samples_leaf=trial_rf.params['min_samples_leaf'],
                                     n_jobs=-1)
    best_rf.fit(X_train_val, y_train_val)
    print(f'Best RandomForest accuracy = {best_rf.score(X_test, y_test) * 100:.2f} %')
    f1_score_rf = f1_score(y_test, best_rf.predict(X_test), average='macro')
    print(f"Best RandomForest F1 score = {f1_score_rf:.3f}")
    res.append(f1_score_rf)
print('mean:', np.mean(res))
print('std:', np.std(res))
print('')



# XGBoost
def objective_xgb(trial):
    max_depth = trial.suggest_int('max_depth', 5, 20)
    n_estimators = trial.suggest_int('n_estimators', 10, 500)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.2, log=True)
    xgb_clf = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=-1, random_state=99)
    xgb_clf.fit(X_train, y_train)
    #score = xgb_clf.score(X_val, y_val)
    score = f1_score(y_val, xgb_clf.predict(X_val), average='macro')
    return score

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=n_trials)
print('Best trial for XGBoost:')
trial_xgb = study_xgb.best_trial
print(f'  Value: {trial_xgb.value}')
print('  Params: ')
for key, value in trial_xgb.params.items():
    print(f'    {key}: {value}')
res =[]
set_seed(args.seed)
for i in range(args.repetitions):
    xgb_clf = xgb.XGBClassifier(max_depth=trial_xgb.params['max_depth'], n_estimators=trial_xgb.params['n_estimators'], learning_rate=trial_xgb.params['learning_rate'], n_jobs=-1)
    xgb_clf.fit(X_train_val, y_train_val)
    print(f'XGBoost accuracy = {xgb_clf.score(X_test, y_test)*100:.2f} %')
    f1_score_xgb = f1_score(y_test, xgb_clf.predict(X_test), average='macro')
    print(f"XGBoost F1 score = {f1_score_xgb:.3f}")
    res.append(f1_score_xgb)
print('mean:', np.mean(res))
print('std:', np.std(res))
print('')


#Naive Bayes
if output_units > 1:
    # Convert y_train and y_test from one-hot encoding to class labels
    y_train_val_arg = np.argmax(y_train_val, axis=1)
    y_test_arg = np.argmax(y_test, axis=1)
    # Train the model and evaluate
    nb = BernoulliNB()
    clf_nb = nb.fit(X_train_val, y_train_val_arg)
    print("NB Accuracy: ", clf_nb.score(X_test, y_test_arg))
    print("NB F1 Score:", f1_score(y_test_arg, clf_nb.predict(X_test), average='macro'))
else:
    nb = BernoulliNB()
    clf_nb = nb.fit(X_train_val, y_train_val.ravel())
    print("NB Accuracy: ", clf_nb.score(X_test, y_test))
    print("NB F1 Score:", f1_score(y_test, clf_nb.predict(X_test), average='macro'))
print('')



# MLP
X_train_val_torch = torch.tensor(X_train_val, dtype=torch.float)
X_train_torch = torch.tensor(X_train, dtype=torch.float)
X_val_torch = torch.tensor(X_val, dtype=torch.float)
X_test_torch = torch.tensor(X_test, dtype=torch.float)
y_train_val_torch = torch.tensor(y_train_val, dtype=torch.float)
y_train_torch = torch.tensor(y_train, dtype=torch.float)
y_test_torch = torch.tensor(y_test, dtype=torch.float)
y_val_torch = torch.tensor(y_val, dtype=torch.float)

criterion = torch.nn.BCELoss()
train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_torch, y_val_torch), batch_size=batch_size, shuffle=False)

def objective_mlp(trial, verbose=False):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    n_units = [input_units]
    layers = []
    for i in range(n_layers):
        n_units.append(trial.suggest_int(f'n_units_{i}', 4, 128))
        layers.append(torch.nn.Linear(n_units[i], n_units[i+1]))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(n_units[-1], output_units))
    layers.append(torch.nn.Sigmoid())
    model = torch.nn.Sequential(*layers).to(device)
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
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
        if (epoch + 1) % epochs_between_eval == 0:
            model.eval()
            with torch.no_grad():
                outputs_train, predicted_train, labels_train = predict(model, train_loader, device)
                acc_train, f1_train = eval(predicted_train, labels_train)
                outputs_val, predicted_val, labels_val = predict(model, val_loader, device)
                acc_val, f1_val = eval(predicted_val, labels_val)
            model.train()
            trial.report(acc_val, epoch//epochs_between_eval)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if verbose:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.5f}')
                print(f'train acc = {acc_train:.2f} %, val acc = {acc_val:.2f} %')
                print(f'f1 train = {f1_train:.3f}, f1 val = {f1_val:.3f}')   
    return f1_val

pruner = optuna.pruners.MedianPruner(n_warmup_steps=pruning_warmup, n_startup_trials=pruning_after)
study_mlp = optuna.create_study(direction='maximize')
study_mlp.optimize(objective_mlp, n_trials=n_trials, show_progress_bar=False)
print(study_mlp.best_params)
print(study_mlp.best_value)

# use the best hyperparameters, retraining on the whole train+validation dataset
train_loader = DataLoader(TensorDataset(X_train_val_torch, y_train_val_torch), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size, shuffle=False)
set_seed(args.seed)
res = []
for i in range(args.repetitions):
    res.append(objective_mlp(optuna.trial.FixedTrial(study_mlp.best_params), verbose=True))
print('mean:', np.mean(res))
print('std:', np.std(res))
print('')

print('Done!')
print('-------------------------------------------')
print('-------------------------------------------')
print('')
