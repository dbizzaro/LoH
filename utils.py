import torch
import numpy as np
import random
from sklearn.metrics import f1_score   
import sympy
import wandb
import json

from ucimlrepo import fetch_ucirepo


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def non_redundant_clauses(sympy_expr_nf, dnf_or_cnf=None, starting='and'):
    """
    Simplifies a sympy expression by removing redundant clauses.
    Parameters:
        sympy_expr_nf (sympy expression)
        dnf_or_cnf (str): 'dnf' or 'cnf'. If None, inferred from starting
        starting (str): 'and' for DNF, 'or' for CNF.
    Returns:
        sympy expression: the simplified expression
    """
    
    if dnf_or_cnf is None:
        dnf_or_cnf = 'dnf' if starting == 'and' else 'cnf'

    if dnf_or_cnf == 'dnf' and not isinstance(sympy_expr_nf, sympy.logic.boolalg.Or): #single clause
        return sympy_expr_nf 
    if dnf_or_cnf == 'cnf' and not isinstance(sympy_expr_nf, sympy.logic.boolalg.And): #single clause
        return sympy_expr_nf
    
    type_ = type(sympy_expr_nf)
    clauses_list = list(sympy_expr_nf.args)
    clauses = []
    for clause in clauses_list:
        if isinstance(clause, sympy.core.symbol.Symbol) or isinstance(clause, sympy.logic.boolalg.Not):
            clauses.append(clause)
        else:
            clauses.append(clause.args)

    non_redundant_clauses = set(clauses)
    for c1 in clauses:
        for c2 in clauses:
            if c1 != c2:
                try:
                    if set(c2).issubset(set(c1)):  # c2 is a subset of c1
                        non_redundant_clauses.discard(c1)
                except: # either c1 or c2 is a singleton
                    try:
                        if c2 in set(c1):
                            non_redundant_clauses.discard(c1)
                    except:
                        pass
    
    type_clauses = sympy.logic.boolalg.Or if dnf_or_cnf == 'cnf' else sympy.logic.boolalg.And
    new_clauses = []
    for clause in non_redundant_clauses:
        try:
            new_clauses.append(type_clauses(*clause))
        except:
            new_clauses.append(clause)
    return type_(*new_clauses)



def clause_stats(sympy_expr, dnf_or_cnf=None, starting='and'):
    """
    Compute number of clauses and their average length for a CNF or DNF formula.
    Returns:
        num_clauses (int), avg_clause_length (float)
    """
    if dnf_or_cnf is None:
        dnf_or_cnf = 'dnf' if starting == 'and' else 'cnf'
    

    outer_type = sympy.logic.boolalg.And if dnf_or_cnf == 'cnf' else sympy.logic.boolalg.Or
    inner_clause_type = sympy.logic.boolalg.Or if dnf_or_cnf == 'cnf' else sympy.logic.boolalg.And

    if isinstance(sympy_expr, outer_type):
        clauses = list(sympy_expr.args)
    else:
        clauses = [sympy_expr]

    def clause_length(clause):
        if isinstance(clause, inner_clause_type):
            return len(clause.args)
        if isinstance(clause, (sympy.Symbol, sympy.logic.boolalg.Not)):
            return 1
        if hasattr(clause, 'args') and clause.args:
            return len(clause.args)
        return 1

    clause_lengths = [clause_length(clause) for clause in clauses]
    num_clauses = len(clause_lengths)
    avg_clause_length = float(sum(clause_lengths)) / num_clauses if num_clauses else 0.0
    return num_clauses, avg_clause_length


def clause_activation_status(sympy_expr, inputs, feature_names, dnf_or_cnf=None, starting='and', return_expression=False):
    """
    Check whether each clause in a simplified sympy formula is activated by any input example.
    Parameters:
        sympy_expr (sympy expression): simplified formula obtained from non_redundant_clauses
        inputs (np.ndarray or torch.Tensor): binarized inputs used to train/evaluate the model
        feature_names (List[str]): ordering of propositions that matches the columns in inputs
        dnf_or_cnf (str): optional override, either 'dnf' or 'cnf'. If None it is inferred from starting.
        starting (str): 'and' or 'or' describing the first layer of the network (DNF if 'and').
        return_expression (bool): when True, also return the expression composed only of activated clauses.
    Returns:
        List[Dict[str, Any]] or Tuple[List, sympy expression]: activation metadata, and optionally the
        filtered expression with inactive clauses removed.
    """
    if sympy_expr is None:
        return ([], sympy_expr) if return_expression else []
    if inputs is None:
        raise ValueError("inputs cannot be None when checking clause activation.")
    if feature_names is None:
        raise ValueError("feature_names cannot be None when checking clause activation.")
    if dnf_or_cnf is None:
        dnf_or_cnf = 'dnf' if starting == 'and' else 'cnf'
    if isinstance(inputs, torch.Tensor):
        inputs_np = inputs.detach().cpu().numpy()
    else:
        inputs_np = np.asarray(inputs)
    if inputs_np.ndim != 2:
        raise ValueError(f"Expected inputs to be a 2D array, got shape {inputs_np.shape}.")
    if len(feature_names) != inputs_np.shape[1]:
        raise ValueError("Length of feature_names must match number of columns in inputs.")
    inputs_bool = inputs_np.astype(bool)
    outer_type = sympy.logic.boolalg.Or if dnf_or_cnf == 'dnf' else sympy.logic.boolalg.And
    inner_clause_type = sympy.logic.boolalg.And if dnf_or_cnf == 'dnf' else sympy.logic.boolalg.Or
    if isinstance(sympy_expr, outer_type):
        clauses = list(sympy_expr.args)
    else:
        clauses = [sympy_expr]
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    n_samples = inputs_bool.shape[0]

    def literal_values(literal):
        if literal is sympy.true or literal==True:
            return np.ones(n_samples, dtype=bool)
        if literal is sympy.false or literal==False:
            return np.zeros(n_samples, dtype=bool)
        if isinstance(literal, sympy.Symbol):
            key = str(literal)
            if key not in feature_to_idx:
                raise KeyError(f"Literal {key} not found in feature_names.")
            return inputs_bool[:, feature_to_idx[key]]
        if isinstance(literal, sympy.logic.boolalg.Not):
            return np.logical_not(literal_values(literal.args[0]))
        raise TypeError(f"Unsupported literal type: {type(literal)}")

    def clause_values(clause):
        if isinstance(clause, inner_clause_type):
            literals = clause.args
        elif isinstance(clause, (sympy.Symbol, sympy.logic.boolalg.Not, sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)):
            literals = (clause,)
        elif hasattr(clause, 'args') and clause.args:
            literals = clause.args
        else:
            literals = (clause,)
        if dnf_or_cnf == 'dnf':
            values = np.ones(n_samples, dtype=bool)
            for literal in literals:
                values &= literal_values(literal)
        else:
            values = np.zeros(n_samples, dtype=bool)
            for literal in literals:
                values |= literal_values(literal)
        return values

    activation_info = []
    active_clauses = []
    for clause in clauses:
        values = clause_values(clause)
        activated = bool(np.any(values)) if dnf_or_cnf == 'dnf' else (not bool(np.all(values)))
        activation_info.append({
            'clause': str(clause),
            'activated': activated,
            'activation_count': int(np.sum(values))
        })
        if activated:
            active_clauses.append(clause)

    if not return_expression:
        return activation_info

    if not active_clauses:
        filtered_expr = sympy.false if dnf_or_cnf == 'dnf' else sympy.true
    elif len(active_clauses) == 1:
        filtered_expr = active_clauses[0]
    else:
        filtered_expr = outer_type(*active_clauses)
    return activation_info, filtered_expr


def predict(model, loader, device=None, binarized_forward=False, compiled=False):
    """
    Make predictions on a given loader.
    Parameters:
        model (nn.Module): the model to make predictions with
        loader (DataLoader): the data to make predictions on
        device (torch.device): the device to use for predictions
        binarized_forward (bool): use CRS baseline's binarized_forward method.
        compiled (bool): use DiffLogic baseline's discretized model
    Returns:
        all_outputs (np.ndarray): the output of the model for each input in the loader
        all_predicted (np.ndarray): the predicted labels for each input in the loader
        all_ground_truth (np.ndarray): the ground truth labels for each input in the loader
    """
    all_outputs = []
    all_predicted = []
    all_ground_truth = []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        if binarized_forward:
            outputs = model.binarized_forward(features)
        elif compiled:
            features = torch.nn.Flatten()(features).bool().cpu().detach().numpy()
            outputs = model(features)
        else:
            outputs = model(features)
        if type(outputs) == tuple:
            outputs = outputs[0]
        if (labels.ndim == 2 and labels.shape[1] == 1):
            predicted = torch.round(outputs.data)
            ground_truth = labels.reshape(-1, 1)
        elif labels.ndim == 1:
            predicted = torch.round(outputs.data)
            ground_truth = labels
        else:
            predicted = outputs.argmax(dim=1)
            ground_truth = labels.argmax(dim=1)
        all_predicted.append(predicted.cpu().detach().numpy())
        all_ground_truth.append(ground_truth.cpu().detach().numpy())
        all_outputs.append(outputs.cpu().detach().numpy())
    all_predicted = np.concatenate(all_predicted)
    all_ground_truth = np.concatenate(all_ground_truth)
    all_outputs = np.concatenate(all_outputs)
    return all_outputs, all_predicted, all_ground_truth



def eval(predicted, ground_truth):
    '''compute accuracy and f1 score'''
    acc = np.mean(predicted == ground_truth) * 100
    f1 = f1_score(ground_truth, predicted, average='macro')
    return acc, f1



def set_wandb_runs(trial, args, best, repeat, id_list):
    """Set up a wandb run for the given trial.
    Parameters:
        trial (optuna.trial.Trial): the trial to set up the run for
        args (argparse.Namespace): the arguments for the experiment
        best (bool): whether this is the best trial
        repeat (int): the repeat number of the trial
        id_list (list): list to store the run ids
    """
    config = dict(trial.params)
    config["trial number"] = trial.number
    config["repeat"] = repeat
    job_type = 'Trials' if not best else 'Best'
    name_trial = f'trial_{trial.number}_{repeat}' if not best else f'best_{repeat}'
    run = wandb.init(
        project=args.experimentname,
        config=config,
        group=args.group,
        name=name_trial,
        job_type=job_type,
        reinit=True
    )
    id_list.append(run.id)
    return run


def write_results_to_file(results_dict, results_file):
    """Write the results dictionary to a JSON file."""
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):  # Ensure it's a list
                data = [data]
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(results_dict)
    with open(results_file, "w") as f:
        json.dump(data, f, indent=4) # Save back to the file



def get_hyperparameters(results_file, experiment_name, group):
    """Retrieve hyperparameters from a JSON file based on the experiment name and group."""
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):  # Ensure it's a list
                data = [data]
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    for results_dict in data:
        if results_dict['experiment'] == experiment_name and results_dict['group'] == group:
            hyperparams = results_dict['hyperparameters']
            break
    if 'hyperparams' not in locals():
        raise ValueError(f"No hyperparameters found for experiment {experiment_name} and group {group}.")
    return hyperparams



def get_dataset(dataset_name, print_info=False, print_more_info=False):
    """Fetch a dataset from the UCI Machine Learning Repository."""

    datasets = {
    'adult': 2,
    'bank_marketing': 222,
    'banknote': 267,
    'chess': 23, #KRK
    'connect-4': 26,
    'letRecog': 59,
    'magic': 159,
    'tic-tac-toe': 101,
    'wine': 109,
    'mushroom': 73,
    'magic04': 159,
    'nursery': 76
    }

    if print_info:
        print(dataset_name)
    UCI_dataset = fetch_ucirepo(id = datasets[dataset_name])
    X = UCI_dataset.data.features
    y = UCI_dataset.data.targets
    if print_more_info:
        print(UCI_dataset.metadata)
    if print_info:
        print(UCI_dataset.variables)
        print(X.shape, y.shape)
    return X.to_numpy(), y
