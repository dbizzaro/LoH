"""Training/evaluation runner for wildfire dILP experiments."""

import json
import os
import uuid

import numpy as np
import tensorflow as tf

from src.core import Atom, Term
from src.dilp import DILP
from src.wildfire.constants import (
    VISUAL_VARIABLES,
    NON_VISUAL_VARIABLES,
    CANDIDATE_RULE_GROUPS,
    ALL_REGIMES,
)
from src.wildfire.cnn import SimpleCNN
from src.wildfire.data import load_wildfire_table, train_test_split_indices, get_train_val_indices, build_tf_dataset
from src.wildfire.builder import build_wildfire_program
from src.wildfire.fuzzy_compile import FormulaCompiler


def _binary_f1(y_true, y_pred):
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 0.0
    return float((2.0 * tp) / denom)


def _binary_acc(y_true, y_prob):
    y_true = y_true.astype(np.int32)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return float(np.mean(y_true == y_pred))


def _build_atom(pred_name, constant):
    return Atom([Term(False, constant)], pred_name)


def _compute_candidate_values(visual_probs, non_visual_feats, compilers, candidate_predicates):
    env = {
        VISUAL_VARIABLES[0]: visual_probs[:, 0],
        VISUAL_VARIABLES[1]: visual_probs[:, 1],
    }
    for i, name in enumerate(NON_VISUAL_VARIABLES):
        env[name] = non_visual_feats[:, i]

    values = {}
    for pred in candidate_predicates:
        values[pred] = compilers[pred].evaluate(env)
    return values


def _collect_full_dynamic_values(cnn, all_dataset, num_batches, compilers, candidate_predicates, training):
    # all_dataset is created with shuffle=False, so concatenation preserves example order.
    visual_batches = []
    candidate_batches = {pred: [] for pred in candidate_predicates}
    ds_iter = iter(all_dataset)
    for _ in range(num_batches):
        images, non_visual, _, _ = next(ds_iter)
        visual_batch = cnn(images, training=training)
        cand_batch = _compute_candidate_values(visual_batch, non_visual, compilers, candidate_predicates)
        visual_batches.append(visual_batch)
        for pred in candidate_predicates:
            candidate_batches[pred].append(tf.reshape(cand_batch[pred], [-1]))

    if len(visual_batches) == 0:
        visual_full = tf.zeros([0, len(VISUAL_VARIABLES)], dtype=tf.float32)
        candidate_full = {pred: tf.zeros([0], dtype=tf.float32) for pred in candidate_predicates}
    else:
        visual_full = tf.concat(visual_batches, axis=0)
        candidate_full = {pred: tf.concat(candidate_batches[pred], axis=0) for pred in candidate_predicates}

    values_by_pred = {
        VISUAL_VARIABLES[0]: visual_full[:, 0],
        VISUAL_VARIABLES[1]: visual_full[:, 1],
    }
    values_by_pred.update(candidate_full)
    return values_by_pred


def _build_dynamic_index_maps(bundle, valuation_mapping):
    dynamic_indices_by_pred = {}
    for pred in bundle.dynamic_predicates:
        idx = []
        for constant in bundle.constants:
            idx.append(valuation_mapping[_build_atom(pred, constant)])
        dynamic_indices_by_pred[pred] = np.array(idx, dtype=np.int32)

    target_indices = []
    for constant in bundle.constants:
        target_indices.append(valuation_mapping[_build_atom(bundle.target_predicate, constant)])
    target_indices = np.array(target_indices, dtype=np.int32)
    return dynamic_indices_by_pred, target_indices


def _concat_override(dynamic_predicates, dynamic_indices_by_pred, values_by_pred):
    indices = np.concatenate([dynamic_indices_by_pred[pred] for pred in dynamic_predicates], axis=0)
    values = tf.concat([tf.reshape(values_by_pred[pred], [-1]) for pred in dynamic_predicates], axis=0)
    return indices, values


def _evaluate_from_outputs(outputs, target_indices, labels, split_indices, return_details=False):
    split_target_indices = tf.gather(target_indices, split_indices)
    probs = tf.gather(outputs, split_target_indices).numpy()
    preds = (probs >= 0.5).astype(np.int32)
    y_true = labels[split_indices].astype(np.int32)
    acc = float(np.mean(preds == y_true)) * 100.0
    f1 = _binary_f1(y_true, preds)
    if return_details:
        return acc, f1, probs, preds, y_true, split_indices
    return acc, f1


def _extract_selected_rules(dilp):
    selected = {}
    for predicate in dilp.rule_weights:
        shape = dilp.rule_weights[predicate].shape
        rule_weights = tf.reshape(dilp.rule_weights[predicate], [-1])
        weights = tf.reshape(tf.nn.softmax(rule_weights)[:, None], shape)
        pos = np.unravel_index(np.argmax(weights, axis=None), weights.shape)
        clauses = dilp.clause_map[predicate]

        def _clause_to_str(c):
            if c is None:
                return '⊥'
            return str(c)

        selected[str(predicate)] = {
            'tau1': _clause_to_str(clauses[0][pos[0]]),
            'tau2': _clause_to_str(clauses[1][pos[1]]),
            'argmax': [int(pos[0]), int(pos[1])],
        }
    return selected


def run_regime(
    regime,
    dataset_dir=None,
    results_dir='results/wildfire_risk_dilp',
    epochs=30,
    repetitions=5,
    batch_size=256,
    lr_rules=0.08,
    lr_cnn=8e-4,
    dropout=0.15,
    train_ratio=0.75,
    split_seed=42,
    non_visual_seed=42,
    use_persisted_split=True,
    use_minibatch=False,
    print_examples_per_epoch=0,
    print_visual_head_metrics=False,
):
    if regime not in ALL_REGIMES:
        raise ValueError('Unknown regime: %s' % regime)

    os.makedirs(results_dir, exist_ok=True)

    table = load_wildfire_table(dataset_dir=dataset_dir, non_visual_seed=non_visual_seed)
    n_examples = len(table['labels'])
    all_indices = np.arange(n_examples, dtype=np.int32)
    if use_persisted_split:
        train_idx, test_idx = get_train_val_indices(table, train_ratio=train_ratio, seed=split_seed)
    else:
        train_idx, test_idx = train_test_split_indices(n_examples, train_ratio=train_ratio, seed=split_seed)
    train_batch_size = batch_size
    if use_minibatch and train_batch_size > len(train_idx):
        train_batch_size = len(train_idx)
    all_num_batches = int(np.ceil(float(n_examples) / float(batch_size)))
    train_num_batches = int(np.floor(float(len(train_idx)) / float(train_batch_size))) if use_minibatch else 0

    all_dataset = build_tf_dataset(
        table['filepaths'],
        table['non_visual_features'],
        table['labels'],
        all_indices,
        batch_size=batch_size,
        shuffle=False,
    )
    train_dataset = build_tf_dataset(
        table['filepaths'][train_idx],
        table['non_visual_features'][train_idx],
        table['labels'][train_idx],
        train_idx,
        batch_size=train_batch_size,
        shuffle=True,
        drop_remainder=use_minibatch,
    )

    compilers = {}
    for group in ('fuel', 'dry', 'trigger'):
        for i, formula in enumerate(CANDIDATE_RULE_GROUPS[group]):
            pred = 'cand_%s_%d' % (group, i)
            compilers[pred] = FormulaCompiler(formula)

    histories_train = []
    histories_val = []
    selected_rules_runs = []

    for rep in range(repetitions):
        scope_name = 'rule_weights_%s_%d' % (regime, rep)
        bundle = build_wildfire_program(table, train_idx, test_idx, regime)
        dilp = DILP(
            bundle.language_frame,
            bundle.background,
            bundle.positive,
            bundle.negative,
            bundle.program_template,
            scope_name=scope_name,
        )

        dynamic_indices_by_pred, target_indices = _build_dynamic_index_maps(bundle, dilp.valuation_mapping)

        cnn = SimpleCNN(n_output=2, dropout=dropout)
        # build cnn variables
        images, _, _, _ = next(iter(all_dataset))
        _ = cnn(images, training=False)

        minibatch_bundle = None
        minibatch_dilp = None
        minibatch_dynamic_indices_by_pred = None
        minibatch_target_indices = None
        rule_weights_shared = True
        if use_minibatch:
            minibatch_table = {
                'labels': np.zeros(train_batch_size, dtype=np.float32),
                'non_visual_features': np.zeros((train_batch_size, len(NON_VISUAL_VARIABLES)), dtype=np.float32),
            }
            minibatch_train_idx = np.arange(train_batch_size, dtype=np.int32)
            minibatch_bundle = build_wildfire_program(
                minibatch_table,
                minibatch_train_idx,
                np.array([], dtype=np.int32),
                regime,
            )
            minibatch_dilp = DILP(
                minibatch_bundle.language_frame,
                minibatch_bundle.background,
                minibatch_bundle.positive,
                minibatch_bundle.negative,
                minibatch_bundle.program_template,
                scope_name=scope_name,
            )
            minibatch_dynamic_indices_by_pred, minibatch_target_indices = _build_dynamic_index_maps(
                minibatch_bundle,
                minibatch_dilp.valuation_mapping,
            )
            rule_weights_shared = all(
                dilp.rule_weights[predicate] is minibatch_dilp.rule_weights[predicate]
                for predicate in dilp.rule_weights
            )
            if not rule_weights_shared:
                for predicate in dilp.rule_weights:
                    minibatch_dilp.rule_weights[predicate].assign(dilp.rule_weights[predicate])
            rule_vars = list(minibatch_dilp.rule_weights.values())
        else:
            rule_vars = list(dilp.rule_weights.values())
        optimizer_rules = tf.keras.optimizers.Adam(learning_rate=lr_rules)
        optimizer_cnn = tf.keras.optimizers.Adam(learning_rate=lr_cnn)

        train_f1_hist = []
        val_f1_hist = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            if not use_minibatch:
                with tf.GradientTape() as tape:
                    full_values = _collect_full_dynamic_values(
                        cnn,
                        all_dataset,
                        all_num_batches,
                        compilers=compilers,
                        candidate_predicates=bundle.candidate_predicates,
                        training=True,
                    )
                    overrides = _concat_override(bundle.dynamic_predicates, dynamic_indices_by_pred, full_values)
                    outputs = dilp.deduction(valuation_overrides=overrides, verbose=False)
                    train_target_idx = tf.gather(target_indices, bundle.train_indices)
                    train_probs = tf.gather(outputs, train_target_idx)
                    y_train = tf.convert_to_tensor(bundle.labels[bundle.train_indices], dtype=tf.float32)
                    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_train, train_probs))

                variables = rule_vars + cnn.trainable_variables
                grads = tape.gradient(loss, variables)
                n_rule = len(rule_vars)
                rule_pairs = [(g, v) for g, v in zip(grads[:n_rule], rule_vars) if g is not None]
                cnn_pairs = [(g, v) for g, v in zip(grads[n_rule:], cnn.trainable_variables) if g is not None]
                if len(rule_pairs) > 0:
                    optimizer_rules.apply_gradients(rule_pairs)
                if len(cnn_pairs) > 0:
                    optimizer_cnn.apply_gradients(cnn_pairs)
                epoch_loss = float(loss.numpy())

            else:
                batch_losses = []
                train_iter = iter(train_dataset)
                for _ in range(train_num_batches):
                    images, non_visual, y_batch, _ = next(train_iter)
                    with tf.GradientTape() as tape:
                        visual_batch = cnn(images, training=True)
                        cand_batch = _compute_candidate_values(
                            visual_batch,
                            non_visual,
                            compilers,
                            bundle.candidate_predicates,
                        )
                        values_by_pred = {
                            VISUAL_VARIABLES[0]: visual_batch[:, 0],
                            VISUAL_VARIABLES[1]: visual_batch[:, 1],
                        }
                        for pred in bundle.candidate_predicates:
                            values_by_pred[pred] = cand_batch[pred]

                        overrides = _concat_override(
                            minibatch_bundle.dynamic_predicates,
                            minibatch_dynamic_indices_by_pred,
                            values_by_pred,
                        )
                        outputs = minibatch_dilp.deduction(valuation_overrides=overrides, verbose=False)
                        batch_probs = tf.gather(outputs, minibatch_target_indices)
                        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.cast(y_batch, tf.float32), batch_probs))

                    variables = rule_vars + cnn.trainable_variables
                    grads = tape.gradient(loss, variables)
                    n_rule = len(rule_vars)
                    rule_pairs = [(g, v) for g, v in zip(grads[:n_rule], rule_vars) if g is not None]
                    cnn_pairs = [(g, v) for g, v in zip(grads[n_rule:], cnn.trainable_variables) if g is not None]
                    if len(rule_pairs) > 0:
                        optimizer_rules.apply_gradients(rule_pairs)
                    if len(cnn_pairs) > 0:
                        optimizer_cnn.apply_gradients(cnn_pairs)
                    batch_losses.append(float(loss.numpy()))

                if not rule_weights_shared:
                    for predicate in dilp.rule_weights:
                        dilp.rule_weights[predicate].assign(minibatch_dilp.rule_weights[predicate])
                if len(batch_losses) > 0:
                    epoch_loss = float(np.mean(batch_losses))
                else:
                    epoch_loss = float('nan')

            eval_values = _collect_full_dynamic_values(
                cnn,
                all_dataset,
                all_num_batches,
                compilers,
                bundle.candidate_predicates,
                training=False,
            )
            eval_overrides = _concat_override(bundle.dynamic_predicates, dynamic_indices_by_pred, eval_values)
            eval_outputs = dilp.deduction(valuation_overrides=eval_overrides, verbose=False)
            train_acc, train_f1 = _evaluate_from_outputs(
                eval_outputs, target_indices, bundle.labels, bundle.train_indices
            )
            if print_examples_per_epoch > 0:
                val_acc, val_f1, val_probs, val_preds, val_true, val_indices = _evaluate_from_outputs(
                    eval_outputs, target_indices, bundle.labels, bundle.test_indices, return_details=True
                )
            else:
                val_acc, val_f1 = _evaluate_from_outputs(eval_outputs, target_indices, bundle.labels, bundle.test_indices)

            train_f1_hist.append(train_f1)
            val_f1_hist.append(val_f1)
            print('regime=%s rep=%d epoch=%d loss=%.4f train_f1=%.4f val_f1=%.4f' % (
                regime,
                rep + 1,
                epoch + 1,
                epoch_loss,
                train_f1,
                val_f1,
            ))
            if print_visual_head_metrics:
                dense_pred = eval_values[VISUAL_VARIABLES[0]].numpy()
                dry_pred = eval_values[VISUAL_VARIABLES[1]].numpy()
                dense_true = table['visual_features'][:, 0]
                dry_true = table['visual_features'][:, 1]
                dense_acc_train = _binary_acc(dense_true[bundle.train_indices], dense_pred[bundle.train_indices])
                dense_acc_val = _binary_acc(dense_true[bundle.test_indices], dense_pred[bundle.test_indices])
                dry_acc_train = _binary_acc(dry_true[bundle.train_indices], dry_pred[bundle.train_indices])
                dry_acc_val = _binary_acc(dry_true[bundle.test_indices], dry_pred[bundle.test_indices])
                dense_mean_train = float(np.mean(dense_pred[bundle.train_indices]))
                dense_mean_val = float(np.mean(dense_pred[bundle.test_indices]))
                dry_mean_train = float(np.mean(dry_pred[bundle.train_indices]))
                dry_mean_val = float(np.mean(dry_pred[bundle.test_indices]))
                print(
                    '  visual_heads: \n'
                    '  ACCURACY: DENSE FOREST val = %.3f '
                    ' DRY VEGETATION val = %.3f \n'
                    '  MEAN: DENSE FOREST val = %.3f '
                    'DRY VEGETATION val = %.3f' % (
                        dense_acc_val,
                        dry_acc_val,
                        dense_mean_val,
                        dry_mean_val,
                    )
                )
            if print_examples_per_epoch > 0:
                n_preview = min(int(print_examples_per_epoch), len(val_indices))
                preview = []
                for i in range(n_preview):
                    preview.append('idx=%d y=%d p=%.3f pred=%d' % (
                        int(val_indices[i]),
                        int(val_true[i]),
                        float(val_probs[i]),
                        int(val_preds[i]),
                    ))
                print('  val_examples: %s' % ', '.join(preview))

        histories_train.append(train_f1_hist)
        histories_val.append(val_f1_hist)
        selected_rules_runs.append(_extract_selected_rules(dilp))

    histories_train = np.array(histories_train, dtype=np.float32)
    histories_val = np.array(histories_val, dtype=np.float32)

    run_id = '%s_%s' % (regime, uuid.uuid4().hex[:8])
    np.save(os.path.join(results_dir, 'f1_train_hist_%s.npy' % run_id), histories_train)
    np.save(os.path.join(results_dir, 'f1_val_hist_%s.npy' % run_id), histories_val)
    with open(os.path.join(results_dir, 'selected_rules_%s.json' % run_id), 'w') as f:
        json.dump(selected_rules_runs, f, indent=2)

    return {
        'run_id': run_id,
        'regime': regime,
        'f1_train_hist': histories_train,
        'f1_val_hist': histories_val,
        'selected_rules': selected_rules_runs,
        'results_dir': results_dir,
    }
