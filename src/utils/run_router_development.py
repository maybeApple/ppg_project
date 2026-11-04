"""Develop HR routers with nested participant validation on original TRAIN people.

Every outer validation fold has fresh experts, training-only OOF gate examples,
and training-only expert selection. Original source OOF predictions cannot be
reused for gate training here: those experts may have seen outer validation
participants. Frozen encoder embeddings and signal-only features can be reused.

This entry point never opens held-out evaluation references or writes a paper.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits

from src.utils.submission_protocol import (
    FEATURES, assert_fit_scope, fit_expert, grouped_splits, metrics,
    select_experts, sha256, write_json,
)

CLASSICAL = ['peak_based', 'spectral']
PROBES = ['pulseppg_linear', 'pulseppg_ridge', 'pulseppg_random_forest',
          'papagei_linear', 'papagei_ridge', 'papagei_random_forest']
EXPERTS = CLASSICAL + PROBES
METRICS = ['MAE', 'RMSE', 'median_absolute_error', 'p95_absolute_error',
           'catastrophic_error_rate_20bpm']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def log(message):
    print(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {message}', flush=True)


def verify_artifacts(base, artifacts):
    base = Path(base).resolve()
    for relative, expected in artifacts.items():
        path = (base / relative).resolve()
        if not path.is_relative_to(base) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Development artifact changed or escapes run directory: {relative}')


def load_training_cache(cfg):
    """Whitelist signal caches and training targets; no final predictions/labels."""
    source = Path(cfg['source_root'])
    split = read_json(cfg['split_config'])
    train, excluded = split['train_participants'], split['test_participants']
    if not train or len(set(train)) != len(train) or set(train) & set(excluded):
        raise ValueError('Invalid original training-only participant scope')
    preparation = read_json(source / 'manifests/preparation.json')
    source_config_path = source / 'configs/frozen_protocol.json'
    source_split_path = source / 'configs/participant_split.json'
    if (preparation['protocol_sha256'] != sha256(source_config_path)
            or read_json(source_split_path) != split):
        raise ValueError('Frozen source protocol or participant split changed')
    input_hashes = {}
    input_hashes[source_config_path.as_posix()] = sha256(source_config_path)
    input_hashes[source_split_path.as_posix()] = sha256(source_split_path)
    tables = {}
    for filename in ['metadata.csv', 'train_targets.csv', 'classical_predictions.csv', 'router_features.csv']:
        relative = f'features/{filename}'
        path = source / relative
        digest = sha256(path)
        if preparation['artifacts'].get(relative) != digest:
            raise ValueError(f'Unverified frozen signal cache: {filename}')
        input_hashes[path.as_posix()] = digest
        tables[filename] = pd.read_csv(path, dtype={'participant_id': str})
    metadata = tables['metadata.csv']
    if metadata.window_key.isna().any() or metadata.window_key.duplicated().any():
        raise ValueError('Invalid cached window identities')
    mask = metadata.split.eq('train')
    if (set(metadata.loc[mask, 'participant_id']) != set(train)
            or set(metadata.loc[mask, 'participant_id']) & set(excluded)):
        raise ValueError('Original training participant scope changed')
    selected = metadata.loc[mask].copy().reset_index(drop=True)
    target = tables['train_targets.csv']
    if (target.window_key.duplicated().any() or target.participant_id.isna().any()
            or set(target.participant_id) != set(train)
            or set(target.window_key) != set(selected.window_key)):
        raise ValueError('Training targets contain excluded, missing or duplicate rows')
    target_people = target.set_index('window_key').participant_id.reindex(selected.window_key)
    if not np.array_equal(target_people.to_numpy(), selected.participant_id.to_numpy()):
        raise ValueError('Training target participant does not match its exact window identity')
    selected = selected.merge(target[['window_key', 'participant_id', 'y_true']],
                              on=['window_key', 'participant_id'], validate='one_to_one')
    for filename, names in [('classical_predictions.csv', CLASSICAL),
                            ('router_features.csv', FEATURES['motion_quality'])]:
        table = tables[filename]
        if table.window_key.duplicated().any() or set(table.window_key) != set(metadata.window_key):
            raise ValueError('Signal feature keys disagree with frozen metadata')
        selected = selected.merge(table[['window_key', *names]], on='window_key', validate='one_to_one')
    embeddings = {}
    positions = pd.Index(metadata.window_key).get_indexer(selected.window_key)
    for model in ['pulseppg', 'papagei']:
        path = source / f'features/{model}_embeddings.npy'
        receipt_path = source / f'manifests/{model}_embedding.json'
        receipt = read_json(receipt_path)
        digest = sha256(path)
        if (receipt != preparation['checkpoints'][model]
                or preparation['artifacts'].get(f'features/{model}_embeddings.npy') != digest
                or receipt['sha256'] != digest or receipt.get('legacy_cache_used') is not False):
            raise ValueError('Embedding cache lacks current frozen-checkpoint provenance')
        if receipt['inputs']['preprocessing_code_sha256'] != sha256('src/models/common.py'):
            raise ValueError('Frozen feature inference code changed')
        source_inputs = receipt['inputs']
        vendor = 'pulseppg_resnet1d.py' if model == 'pulseppg' else 'papagei_resnet.py'
        if (source_inputs['source_windows_sha256'] != preparation['windows_sha256']
                or source_inputs['n_windows'] != len(metadata)
                or source_inputs['model_code_sha256'] != sha256(Path('src/vendor') / vendor)
                or source_inputs['shared_code_sha256'] != sha256('src/vendor/resnet1d_shared.py')):
            raise ValueError('Embedding signal source or encoder implementation changed')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if array.ndim != 2 or len(array) != len(metadata):
            raise ValueError('Encoder feature row count disagrees with metadata')
        # Frozen encoder values remain unchanged; solve downstream linear
        # systems in float64 to avoid single-precision ill-conditioning.
        embeddings[model] = np.asarray(array[positions], dtype=np.float64)
        if not np.isfinite(embeddings[model]).all():
            raise ValueError('Nonfinite training embeddings')
        input_hashes[path.as_posix()] = digest
        input_hashes[receipt_path.as_posix()] = sha256(receipt_path)
    if not np.isfinite(selected.y_true).all():
        raise ValueError('Nonfinite training target')
    input_hashes[(source / 'manifests/preparation.json').as_posix()] = sha256(source / 'manifests/preparation.json')
    input_hashes[cfg['split_config']] = sha256(cfg['split_config'])
    return selected, embeddings, input_hashes


def fit_probe(name, embeddings, frame, fit_indices, allowed, heldout, cfg):
    rows = frame.iloc[fit_indices]
    groups = rows.participant_id.to_numpy()
    assert_fit_scope(groups, allowed, heldout)
    model, kind = name.split('_', 1)
    x, y = embeddings[model][fit_indices], rows.y_true.to_numpy(float)
    if kind != 'random_forest':
        return fit_expert(x, y, groups, kind, cfg['seed'], allowed, heldout, cfg['inner_folds'])
    estimator = RandomForestRegressor(**cfg['random_forest'], n_jobs=1, random_state=cfg['seed'])
    estimator.fit(x, y)
    record = {'fit_participants': sorted(set(groups)), 'heldout_participants': sorted(set(heldout)),
              'hyperparameters': cfg['random_forest'], 'hyperparameter_selection': 'fixed_protocol_no_search',
              'inner_folds': [], 'preprocessing_fit_scope': 'no_fitted_input_scaler'}
    return estimator, record


def expert_predictions(frame, embeddings, train_indices, predict_indices, cfg):
    allowed = sorted(frame.iloc[train_indices].participant_id.unique())
    heldout = sorted(frame.iloc[predict_indices].participant_id.unique())
    if set(allowed) & set(heldout):
        raise ValueError('Expert fitting crosses participant validation boundary')
    predictions = frame.iloc[predict_indices][['window_key', 'participant_id', 'y_true', *CLASSICAL]].copy()
    estimators, provenance = {}, {}
    for name in PROBES:
        estimator, record = fit_probe(name, embeddings, frame, train_indices, allowed, heldout, cfg)
        estimators[name] = estimator
        provenance[name] = record
        predictions[name] = estimator.predict(embeddings[name.split('_', 1)[0]][predict_indices])
    return predictions, estimators, provenance


def meta_oof(frame, embeddings, indices, cfg):
    """Rebuild expert OOF entirely inside the current outer training people."""
    groups = frame.iloc[indices].participant_id.to_numpy()
    blocks, records = [], []
    for number, (a, b) in enumerate(grouped_splits(groups, cfg['meta_folds']), 1):
        prediction, _, provenance = expert_predictions(frame, embeddings, indices[a], indices[b], cfg)
        prediction['meta_fold'] = number
        blocks.append(prediction)
        records.append({'meta_fold': number,
                        'train_participants': sorted(set(groups[a])),
                        'validation_participants': sorted(set(groups[b])), 'experts': provenance})
    oof = pd.concat(blocks, ignore_index=True)
    if oof.window_key.duplicated().any() or set(oof.window_key) != set(frame.iloc[indices].window_key):
        raise ValueError('Nested gate OOF windows missing or duplicated')
    return oof, records


def fit_routers(oof, frame, cfg):
    from src.routing.adaptive import AdaptiveRouter
    usable = np.isfinite(oof[EXPERTS].to_numpy(float)).all(axis=1)
    oof = oof.loc[usable].reset_index(drop=True)
    allowed = sorted(oof.participant_id.unique())
    selection = select_experts(oof, allowed, candidates=EXPERTS)
    rows = frame.set_index('window_key').loc[oof.window_key].reset_index()
    if (not np.array_equal(rows.participant_id.to_numpy(), oof.participant_id.to_numpy())
            or not np.array_equal(rows.y_true.to_numpy(), oof.y_true.to_numpy())):
        raise ValueError('Gate OOF identities or references disagree with training windows')
    c = oof[selection['best_classical']].to_numpy(float)
    f = oof[selection['best_foundation']].to_numpy(float)
    gates = {}
    diagnostics = {}
    for spec in cfg['candidates']:
        router = AdaptiveRouter(FEATURES[spec['features']], method=spec['method'], seed=cfg['seed'], **cfg['router'])
        router.fit(rows, oof.y_true.to_numpy(float), c, f, rows.participant_id.to_numpy())
        gates[spec['id']] = router
        diagnostics[spec['id']] = router.diagnostics_
    return selection, gates, diagnostics


def evaluate_outer_fold(config_path, number):
    cfg = read_json(config_path)
    root = Path(cfg['output_root'])
    folder = root / f'folds/fold_{number}'
    folder.mkdir(parents=True, exist_ok=True)
    verify_run_inputs(cfg, root)
    receipt = folder / 'complete.json'
    if receipt.exists():
        previous = read_json(receipt)
        if previous['protocol_sha256'] != sha256(root / 'frozen_protocol.json'):
            raise ValueError('Cached development fold used another protocol')
        verify_artifacts(folder, previous['artifacts'])
        return previous
    with threadpool_limits(limits=cfg['threads_per_worker']):
        frame, embeddings, _ = load_training_cache(cfg)
        train, validation = grouped_splits(frame.participant_id.to_numpy(), cfg['outer_folds'])[number-1]
        fit_people = sorted(frame.iloc[train].participant_id.unique())
        validation_people = sorted(frame.iloc[validation].participant_id.unique())
        log(f'fold {number}: fresh nested expert OOF; validation {validation_people}')
        oof, nested_provenance = meta_oof(frame, embeddings, train, cfg)
        selection, gates, diagnostics = fit_routers(oof, frame, cfg)
        log(f'fold {number}: selected {selection["best_classical"]}/{selection["best_foundation"]}; full outer-training expert refit')
        prediction, _, refit = expert_predictions(frame, embeddings, train, validation, cfg)
        inference_features = frame.iloc[validation].reset_index(drop=True)
        prediction = prediction.reset_index(drop=True)
        usable = np.isfinite(prediction[EXPERTS].to_numpy(float)).all(axis=1)
        valid = prediction.loc[usable].reset_index(drop=True)
        x = inference_features.loc[usable].reset_index(drop=True)
        c, f = (valid[selection[k]].to_numpy(float) for k in ['best_classical', 'best_foundation'])
        for alias in ['best_single', 'best_classical', 'best_foundation']:
            valid[alias] = valid[selection[alias]]
        valid['pair_oracle'] = np.where(abs(c-valid.y_true) <= abs(f-valid.y_true), c, f)
        for spec in cfg['candidates']:
            valid[spec['id']] = gates[spec['id']].predict(x, c, f, soft=spec['soft'])
        valid['outer_fold'] = number
        valid.to_csv(folder / 'validation_predictions.csv', index=False)
        oof.to_csv(folder / 'gate_training_oof.csv', index=False)
        write_json(folder / 'selection.json', selection)
        write_json(folder / 'gate_diagnostics.json', diagnostics)
        write_json(folder / 'provenance.json', {'train_participants': fit_people,
            'validation_participants': validation_people, 'meta_oof': nested_provenance,
            'final_expert_refit': refit, 'validation_labels_used_in_fit_or_selection': False,
            'source_training_oof_reused': False})
        verify_run_inputs(cfg, root)
        record = {'outer_fold': number, 'protocol_sha256': sha256(root / 'frozen_protocol.json'),
                  'eligible_windows': len(prediction), 'paired_windows': len(valid),
                  'n_validation_participants': len(validation_people),
                  'artifacts': {p.name: sha256(p) for p in folder.iterdir() if p.is_file() and p != receipt}}
        write_json(receipt, record)
        log(f'fold {number} complete: {len(valid)} common validation windows')
        return record


def summarize_predictions(predictions, cfg):
    if (predictions.window_key.duplicated().any() or predictions.participant_id.isna().any()
            or predictions.groupby('participant_id').outer_fold.nunique().ne(1).any()):
        raise ValueError('Outer folds overlap or participant prediction provenance is invalid')
    models = [*EXPERTS, 'best_single', 'best_classical', 'best_foundation', 'pair_oracle',
              *[s['id'] for s in cfg['candidates']]]
    participant = []
    for name in models:
        if not np.isfinite(predictions[name]).all():
            raise ValueError('Candidate evaluation must use identical finite common windows')
        for person, group in predictions.groupby('participant_id', sort=True):
            participant.append({'model_name': name, 'participant_id': person,
                                **metrics(group.y_true, group[name])})
    participant = pd.DataFrame(participant)
    rows = []
    for name in models:
        per = participant.loc[participant.model_name.eq(name)]
        rows.append({'model_name': name, **metrics(predictions.y_true, predictions[name]),
                     'n_participants': predictions.participant_id.nunique(),
                     **{f'participant_mean_{m}': float(per[m].mean()) for m in METRICS},
                     'participant_SD_MAE': float(per.MAE.std(ddof=1))})
    return pd.DataFrame(rows), participant


def choose_candidate(summary, cfg):
    selection = cfg['selection']
    rows = summary.set_index('model_name')
    reference = rows.loc[selection['guard_reference']]
    choices = [spec['id'] for spec in cfg['candidates']]
    if selection['allow_best_single_fallback']:
        choices.append('best_single')
    eligible, rejected = [], {}
    for name in choices:
        row = rows.loc[name]
        if (row.participant_mean_p95_absolute_error <= reference.participant_mean_p95_absolute_error
                * selection['maximum_relative_participant_p95'] + 1e-12
                and row.participant_mean_catastrophic_error_rate_20bpm <=
                reference.participant_mean_catastrophic_error_rate_20bpm
                + selection['maximum_participant_catastrophic_rate_increase'] + 1e-12):
            eligible.append(name)
        else:
            rejected[name] = 'Fails predeclared participant-P95 or >20 bpm error-rate guard'
    selected = min(eligible, key=lambda name: (rows.loc[name, 'participant_mean_MAE'], name))
    return {'selected_system': selected, 'eligible_candidates': eligible, 'rejected': rejected,
            'source': 'grouped_original_training_participant_development_CV',
            'objective': selection, 'confirmatory_test': False}


def verify_run_inputs(cfg, root):
    frozen = read_json(root / 'frozen_protocol.json')
    if frozen['config'] != cfg:
        raise ValueError('Configuration changed after protocol freeze; use a new run directory')
    for path, expected in {**frozen['input_hashes'], **frozen['implementation_hashes']}.items():
        if sha256(path) != expected:
            raise ValueError(f'Frozen development input or implementation changed: {path}')
    return frozen


def verify_cv(root):
    record = read_json(root / 'cv_complete.json')
    if record['protocol_sha256'] != sha256(root / 'frozen_protocol.json'):
        raise ValueError('Development CV protocol identity changed')
    verify_artifacts(root, record['artifacts'])
    for number, digest in record['fold_receipts'].items():
        path = root / f'folds/fold_{number}/complete.json'
        if sha256(path) != digest:
            raise ValueError('Nested fold receipt changed after CV completion')
        fold = read_json(path)
        if fold['protocol_sha256'] != record['protocol_sha256']:
            raise ValueError('Nested fold uses a different protocol')
        verify_artifacts(path.parent, fold['artifacts'])
    return record


def initialize(config_path):
    cfg = read_json(config_path)
    if (cfg['schema_version'] != 'training_only_router_development_v1'
            or cfg.get('test_evaluation') is not False or cfg.get('paper_updates') is not False
            or cfg.get('experts') != EXPERTS):
        raise ValueError('This entry point is restricted to the declared training-only development protocol')
    root = Path(cfg['output_root'])
    source = Path(cfg['source_root'])
    if (root.resolve().is_relative_to(source.resolve())
            or root.resolve().is_relative_to(Path('paper').resolve())):
        raise ValueError('New development artifacts must not overwrite the frozen source or paper')
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'frozen_protocol.json'
    if path.exists():
        verify_run_inputs(cfg, root)
        return cfg, root
    frame, _, inputs = load_training_cache(cfg)
    implementations = ['src/utils/run_router_development.py', 'src/routing/adaptive.py',
                       'src/utils/submission_protocol.py', 'src/models/common.py']
    frozen = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg,
        'n_training_windows': len(frame), 'training_participants': sorted(frame.participant_id.unique()),
        'excluded_test_participants': read_json(cfg['split_config'])['test_participants'],
        'input_hashes': inputs, 'implementation_hashes': {p: sha256(p) for p in implementations},
        'no_final_test_reference_reads': True, 'no_external_dataset_labels': True,
        'preprocessing': 'verified_frozen_encoder_harmonized_embeddings_no_refit',
        'scope_note': 'outer-CV numbers are model development; selecting among candidates adds optimism'}
    write_json(path, frozen)
    return cfg, root


def run_cv(config_path):
    cfg, root = initialize(config_path)
    receipt = root / 'cv_complete.json'
    if receipt.exists():
        complete = verify_cv(root)
        log('Existing development CV artifacts verified')
        return complete
    completed = []
    with ProcessPoolExecutor(max_workers=cfg['workers']) as executor:
        futures = [executor.submit(evaluate_outer_fold, str(config_path), n)
                   for n in range(1, cfg['outer_folds']+1)]
        for future in as_completed(futures):
            completed.append(future.result())
    verify_run_inputs(cfg, root)
    prediction = pd.concat([pd.read_csv(root / f'folds/fold_{n}/validation_predictions.csv')
                            for n in range(1, cfg['outer_folds']+1)], ignore_index=True)
    summary, participant = summarize_predictions(prediction, cfg)
    choice = choose_candidate(summary, cfg)
    prediction.to_csv(root / 'development_predictions.csv', index=False)
    summary.to_csv(root / 'development_summary.csv', index=False)
    participant.to_csv(root / 'development_participant_metrics.csv', index=False)
    write_json(root / 'development_selection.json', choice)
    paths = ['development_predictions.csv', 'development_summary.csv',
             'development_participant_metrics.csv', 'development_selection.json']
    record = {'status': 'training_only_nested_development_cv_complete',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'fold_receipts': {str(r['outer_fold']): sha256(root / f"folds/fold_{r['outer_fold']}/complete.json") for r in completed},
              'protocol_sha256': sha256(root / 'frozen_protocol.json'),
              'all_outer_validation_people': sorted(prediction.participant_id.unique()),
              'artifacts': {p: sha256(root / p) for p in paths}}
    write_json(receipt, record)
    log(f'Nested CV complete; development-selected system {choice["selected_system"]}')
    return record


def fit_final(config_path):
    cfg, root = initialize(config_path)
    if not (root / 'cv_complete.json').exists():
        raise ValueError('Complete grouped development CV before fitting a candidate for use')
    verify_cv(root)
    folder = root / 'fitted_system'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        previous = read_json(folder / 'complete.json')
        if previous['protocol_sha256'] != sha256(root / 'frozen_protocol.json'):
            raise ValueError('Fitted candidate used a different development protocol')
        verify_artifacts(folder, previous['artifacts'])
        return previous
    with threadpool_limits(limits=cfg['threads_per_worker']):
        frame, embeddings, _ = load_training_cache(cfg)
        indices = np.arange(len(frame))
        log('Final candidate fit: fresh training-only OOF and all-18-participant expert fit')
        oof, provenance = meta_oof(frame, embeddings, indices, cfg)
        selection, gates, diagnostics = fit_routers(oof, frame, cfg)
        estimators, refit = {}, {}
        allowed = sorted(frame.participant_id.unique())
        for name in PROBES:
            estimators[name], refit[name] = fit_probe(name, embeddings, frame, indices, allowed, [], cfg)
        choice = read_json(root / 'development_selection.json')
        bundle = {'schema_version': 'adaptive_hr_bundle_v1', 'experts': estimators, 'gates': gates,
                  'expert_selection': selection, 'system_selection': choice, 'config': cfg,
                  'training_participants': allowed, 'protocol_sha256': sha256(root / 'frozen_protocol.json'),
                  'input_contract': 'same frozen encoder preprocessing, classical estimators and motion/quality features',
                  'status': 'development_candidate_not_final_test_verified'}
        joblib.dump(bundle, folder / 'adaptive_hr_system.joblib')
        write_json(folder / 'expert_selection.json', selection)
        write_json(folder / 'gate_diagnostics.json', diagnostics)
        write_json(folder / 'fit_provenance.json', {'meta_oof': provenance, 'final_experts': refit,
                  'training_participants': allowed, 'final_test_labels_read': False,
                  'source_inference_files_modified': False})
        verify_run_inputs(cfg, root)
        record = {'status': 'development_candidate_fitted_without_final_test',
                  'protocol_sha256': sha256(root / 'frozen_protocol.json'),
                  'artifacts': {p.name: sha256(p) for p in folder.iterdir() if p.is_file() and p.name != 'complete.json'}}
        write_json(folder / 'complete.json', record)
        log('Development candidate serialized; original final test not evaluated')
        return record


def predict_bundle(bundle, frame, embeddings):
    """Run a serialized candidate on aligned signal-only feature rows."""
    if bundle.get('schema_version') != 'adaptive_hr_bundle_v1':
        raise ValueError('Unknown adaptive HR bundle')
    if frame.window_key.isna().any() or frame.window_key.duplicated().any():
        raise ValueError('Unique finite window identities required')
    for model in ['pulseppg', 'papagei']:
        if (np.asarray(embeddings[model]).ndim != 2 or len(embeddings[model]) != len(frame)
                or not np.isfinite(embeddings[model]).all()):
            raise ValueError('Embeddings must align with inference feature rows')
    selection = bundle['expert_selection']
    def expert(name):
        if name in CLASSICAL:
            return frame[name].to_numpy(float)
        return bundle['experts'][name].predict(np.asarray(embeddings[name.split('_', 1)[0]], dtype=np.float64))
    choice = bundle['system_selection']['selected_system']
    if choice == 'best_single':
        return expert(selection['best_single'])
    spec = next(s for s in bundle['config']['candidates'] if s['id'] == choice)
    return bundle['gates'][choice].predict(frame, expert(selection['best_classical']),
                                          expert(selection['best_foundation']), soft=spec['soft'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/router_development.json'))
    parser.add_argument('--stage', choices=['plan', 'cv', 'fit', 'all'], default='all')
    args = parser.parse_args()
    if args.stage == 'plan':
        cfg, root = initialize(args.config)
        print(json.dumps({'status': 'training_only_protocol_frozen', 'output_root': str(root),
                          'outer_folds': cfg['outer_folds'], 'test_evaluation': False}))
        return
    if args.stage in ['cv', 'all']:
        run_cv(args.config)
    if args.stage in ['fit', 'all']:
        fit_final(args.config)


if __name__ == '__main__':
    main()
