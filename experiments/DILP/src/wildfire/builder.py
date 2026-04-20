"""Build dILP wildfire programs with LoH-matched clause spaces via structural restrictions."""

from dataclasses import dataclass
import math

import numpy as np

from src.core import Term, Atom, Clause
from src.ilp import Language_Frame, Program_Template, Rule_Template
from src.wildfire.constants import (
    VISUAL_VARIABLES,
    NON_VISUAL_VARIABLES,
    CANDIDATE_RULE_GROUPS,
    REGIME_FULL_KNOWLEDGE,
    REGIME_SELECT_RELIABLE,
    REGIME_SELECT_ONE_PER_SET,
    REGIME_PARTIAL_FUEL_KNOWN,
    FUEL_KNOWN_INDEX,
)


@dataclass
class WildfireProgramBundle:
    language_frame: Language_Frame
    program_template: Program_Template
    background: list
    positive: list
    negative: list
    constants: list
    train_indices: np.ndarray
    test_indices: np.ndarray
    labels: np.ndarray
    non_visual_features: np.ndarray
    candidate_predicates: list
    candidate_rules: dict
    candidate_groups: dict
    dynamic_predicates: list
    top_predicate: str
    target_predicate: str
    regime: str


def _cand_name(group_name, idx):
    return 'cand_%s_%d' % (group_name, idx)


def _target_atom(name, x0):
    return Atom([x0], name)


def _ground_atom(pred_name, constant):
    return Atom([Term(False, constant)], pred_name)


def _mk_clause(head_atom, body1_atom, body2_atom):
    return Clause(head_atom, [body1_atom, body2_atom])


def _add_and_tree(leaves, prefix, top_atom, target_atom, atom_registry, p_a_list, rules):
    """Add fixed binary AND tree over leaves and return root atom."""
    layer = list(leaves)
    counter = 0
    while len(layer) > 1:
        next_layer = []
        i = 0
        while i < len(layer):
            if i + 1 >= len(layer):
                next_layer.append(layer[i])
                i += 1
                continue
            left = layer[i]
            right = layer[i + 1]
            aux_name = '%s_%d' % (prefix, counter)
            counter += 1
            aux_atom = atom_registry[aux_name]
            p_a_list.append(aux_atom)
            rules[aux_atom] = (
                Rule_Template(0, True, clause_whitelist=[_mk_clause(aux_atom, left, right)]),
                None,
            )
            next_layer.append(aux_atom)
            i += 2
        layer = next_layer

    root = layer[0]
    # Keep two-literal body while fixing target definition.
    rules[target_atom] = (
        Rule_Template(0, True, clause_whitelist=[_mk_clause(target_atom, root, top_atom)]),
        None,
    )
    return root


def build_wildfire_program(table, train_indices, test_indices, regime):
    labels = table['labels']
    non_visual = table['non_visual_features']
    n_examples = len(labels)
    constants = ['e_%04d' % i for i in range(n_examples)]

    x0 = Term(True, 'X_0')

    candidate_rules = {}
    candidate_groups = {}
    candidate_predicates = []
    for group_name in ('fuel', 'dry', 'trigger'):
        preds = []
        for i, rule in enumerate(CANDIDATE_RULE_GROUPS[group_name]):
            pred_name = _cand_name(group_name, i)
            preds.append(pred_name)
            candidate_predicates.append(pred_name)
            candidate_rules[pred_name] = rule
        candidate_groups[group_name] = preds

    top_predicate = 'top'
    target_predicate = 'wfrisk'

    # dILP rules consume only top/candidate predicates. Non-visual and raw visual
    # variables are used upstream to compute candidate predicate valuations.
    extensional_names = [top_predicate] + candidate_predicates

    # Intensional names depend on regime.
    intensional_names = [target_predicate]
    if regime in (REGIME_FULL_KNOWLEDGE, REGIME_SELECT_ONE_PER_SET, REGIME_PARTIAL_FUEL_KNOWN):
        intensional_names.append('fuel_sel')
    if regime in (REGIME_FULL_KNOWLEDGE, REGIME_SELECT_ONE_PER_SET):
        intensional_names.extend(['dry_sel', 'trigger_sel'])

    if regime in (REGIME_SELECT_RELIABLE, REGIME_PARTIAL_FUEL_KNOWN):
        gate_source = []
        if regime == REGIME_SELECT_RELIABLE:
            gate_source.extend(candidate_predicates)
        else:
            gate_source.extend(candidate_groups['dry'] + candidate_groups['trigger'])
        for name in gate_source:
            intensional_names.append('gate_%s' % name)

    # Conservative upper bound for and-tree internal predicates.
    for i in range(0, 64):
        intensional_names.append('and_%d' % i)

    atom_registry = {}
    for name in set(extensional_names + intensional_names):
        atom_registry[name] = _target_atom(name, x0)

    p_e = [atom_registry[name] for name in extensional_names]
    target_atom = atom_registry[target_predicate]

    background = []
    for i, constant in enumerate(constants):
        background.append(_ground_atom(top_predicate, constant))

    positive = []
    negative = []
    train_set = set(train_indices.tolist())
    for i, constant in enumerate(constants):
        if i not in train_set:
            continue
        atom = _ground_atom(target_predicate, constant)
        if labels[i] >= 0.5:
            positive.append(atom)
        else:
            negative.append(atom)

    rules = {}
    p_a = []

    top_atom = atom_registry[top_predicate]

    def add_fixed_selector(sel_name, cand_name):
        sel_atom = atom_registry[sel_name]
        p_a.append(sel_atom)
        rules[sel_atom] = (
            Rule_Template(0, False, clause_whitelist=[_mk_clause(sel_atom, atom_registry[cand_name], top_atom)]),
            None,
        )
        return sel_atom

    def add_choice_selector(sel_name, cand_names):
        sel_atom = atom_registry[sel_name]
        p_a.append(sel_atom)
        choices = [_mk_clause(sel_atom, atom_registry[cand], top_atom) for cand in cand_names]
        rules[sel_atom] = (
            Rule_Template(0, False, clause_whitelist=choices),
            None,
        )
        return sel_atom

    def add_include_gate(gate_name, cand_name):
        gate_atom = atom_registry[gate_name]
        p_a.append(gate_atom)
        rules[gate_atom] = (
            Rule_Template(
                0,
                False,
                clause_whitelist=[
                    _mk_clause(gate_atom, atom_registry[cand_name], top_atom),
                    _mk_clause(gate_atom, top_atom, top_atom),
                ],
            ),
            None,
        )
        return gate_atom

    n_leaves = 0

    if regime == REGIME_FULL_KNOWLEDGE:
        fuel_sel = add_fixed_selector('fuel_sel', candidate_groups['fuel'][0])
        dry_sel = add_fixed_selector('dry_sel', candidate_groups['dry'][0])
        trigger_sel = add_fixed_selector('trigger_sel', candidate_groups['trigger'][0])
        leaves = [fuel_sel, dry_sel, trigger_sel]
        n_leaves = len(leaves)
        _add_and_tree(leaves, 'and', top_atom, target_atom, atom_registry, p_a, rules)

    elif regime == REGIME_SELECT_ONE_PER_SET:
        fuel_sel = add_choice_selector('fuel_sel', candidate_groups['fuel'])
        dry_sel = add_choice_selector('dry_sel', candidate_groups['dry'])
        trigger_sel = add_choice_selector('trigger_sel', candidate_groups['trigger'])
        leaves = [fuel_sel, dry_sel, trigger_sel]
        n_leaves = len(leaves)
        _add_and_tree(leaves, 'and', top_atom, target_atom, atom_registry, p_a, rules)

    elif regime == REGIME_SELECT_RELIABLE:
        leaves = []
        for cand in candidate_predicates:
            gate_name = 'gate_%s' % cand
            leaves.append(add_include_gate(gate_name, cand))
        n_leaves = len(leaves)
        _add_and_tree(leaves, 'and', top_atom, target_atom, atom_registry, p_a, rules)

    elif regime == REGIME_PARTIAL_FUEL_KNOWN:
        fuel_sel = add_fixed_selector('fuel_sel', candidate_groups['fuel'][FUEL_KNOWN_INDEX])
        leaves = [fuel_sel]
        for cand in candidate_groups['dry'] + candidate_groups['trigger']:
            gate_name = 'gate_%s' % cand
            leaves.append(add_include_gate(gate_name, cand))
        n_leaves = len(leaves)
        _add_and_tree(leaves, 'and', top_atom, target_atom, atom_registry, p_a, rules)

    else:
        raise ValueError('Unknown regime: %s' % regime)

    used_aux = []
    for atom in p_a:
        if atom.predicate.startswith('and_') and atom not in rules:
            continue
        used_aux.append(atom)

    language_frame = Language_Frame(target_atom, p_e, constants)
    # Minimal forward depth for acyclic composition:
    # selector/gate layer (1) + binary AND tree depth + target layer (1).
    tree_depth = int(math.ceil(math.log(max(n_leaves, 1), 2))) if n_leaves > 1 else 0
    t_steps = 2 + tree_depth
    program_template = Program_Template(used_aux, rules, T=t_steps)

    return WildfireProgramBundle(
        language_frame=language_frame,
        program_template=program_template,
        background=background,
        positive=positive,
        negative=negative,
        constants=constants,
        train_indices=np.array(train_indices, dtype=np.int32),
        test_indices=np.array(test_indices, dtype=np.int32),
        labels=np.array(labels, dtype=np.float32),
        non_visual_features=np.array(non_visual, dtype=np.float32),
        candidate_predicates=list(candidate_predicates),
        candidate_rules=dict(candidate_rules),
        candidate_groups=dict(candidate_groups),
        dynamic_predicates=list(candidate_predicates),
        top_predicate=top_predicate,
        target_predicate=target_predicate,
        regime=regime,
    )
