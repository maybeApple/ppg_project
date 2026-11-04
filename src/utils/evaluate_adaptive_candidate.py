"""Once-only reassessment of an already frozen development candidate.

All signal-only predictions across requested datasets are serialized and hashed
before reference CSVs are decoded. There is no fitting, candidate reselection or
paper generation. These historically exposed cohorts are reassessments, not new
confirmatory tests; changing a model after reading results needs a new study.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.utils.run_router_development import (CLASSICAL, EXPERTS, PROBES, predict_bundle,
    read_json, verify_artifacts, verify_cv, verify_run_inputs)
from src.utils.predict_adaptive_hr import REFERENCE_FIELD
from src.utils.submission_protocol import FEATURES, metrics, sha256, write_json

REPO = Path(__file__).resolve().parents[2]
MODELS = [*EXPERTS, 'candidate', 'legacy_same_pair_soft', 'preselected_single']


def _hash_array(array):
    array = np.asarray(array)
    digest = hashlib.sha256(str((array.shape, str(array.dtype))).encode())
    for start in range(0, len(array), 8192):
        digest.update(np.ascontiguousarray(array[start:start+8192]).tobytes())
    return digest.hexdigest()


def _job_signature(job):
    signature = {'dataset': job['dataset'], 'reference_hashes': {
        str(Path(p).resolve()): sha256(p) for p in job['reference_paths']}}
    if 'frame' in job:
        signature['frame'] = hashlib.sha256(pd.util.hash_pandas_object(
            job['frame'], index=False).to_numpy().tobytes()).hexdigest()
        signature['embeddings'] = {m: _hash_array(job['embeddings'][m]) for m in ['pulseppg', 'papagei']}
    else:
        paths = [job['metadata_path'], job['features_path'], *job['embedding_paths'].values()]
        if job.get('classical_path'): paths.append(job['classical_path'])
        signature['signal_file_hashes'] = {str(Path(p).resolve()): sha256(p) for p in paths}
        signature['participants'] = job.get('participants')
    return signature


def _validate_signal_frame(frame):
    forbidden = [name for name in frame if REFERENCE_FIELD.search(name)]
    if forbidden:
        raise ValueError(f'Reference/error fields cannot enter signal inference: {forbidden}')
    required = ['window_key', 'participant_id', *CLASSICAL, *FEATURES['motion_quality']]
    if (set(required)-set(frame) or frame.empty or frame.window_key.isna().any()
            or frame.window_key.duplicated().any() or frame.participant_id.isna().any()):
        raise ValueError('Signal feature windows need unique identities and all inference fields')


def _load_job(job):
    if 'frame' in job:
        frame, embeddings = job['frame'].copy(), job['embeddings']
    else:
        metadata = pd.read_csv(job['metadata_path'], dtype={'window_key': str, 'participant_id': str})
        if metadata.window_key.isna().any() or metadata.window_key.duplicated().any():
            raise ValueError('Cached embedding row identities are invalid')
        frame = metadata.copy()
        if job.get('participants') is not None:
            frame = frame.loc[frame.participant_id.isin(job['participants'])].copy()
            if set(frame.participant_id) != set(job['participants']):
                raise ValueError('Requested evaluation participant coverage changed')
        columns = FEATURES['motion_quality'] + ([] if job.get('classical_path') else CLASSICAL)
        tables = [(job['features_path'], columns)]
        if job.get('classical_path'): tables.append((job['classical_path'], CLASSICAL))
        for path, names in tables:
            table = pd.read_csv(path, dtype={'window_key': str})
            if table.window_key.duplicated().any() or set(table.window_key) != set(metadata.window_key):
                raise ValueError('Feature and embedding metadata window keys disagree')
            frame = frame.merge(table[['window_key', *names]], on='window_key', validate='one_to_one')
        positions = pd.Index(metadata.window_key).get_indexer(frame.window_key)
        embeddings = {}
        for model, path in job['embedding_paths'].items():
            matrix = np.load(path, mmap_mode='r', allow_pickle=False)
            if matrix.ndim != 2 or len(matrix) != len(metadata):
                raise ValueError('Embedding dimensions disagree with metadata')
            embeddings[model] = np.asarray(matrix[positions])
    frame = frame.reset_index(drop=True)
    _validate_signal_frame(frame)
    for model in ['pulseppg', 'papagei']:
        matrix = np.asarray(embeddings[model])
        if matrix.ndim != 2 or len(matrix) != len(frame):
            raise ValueError('Embedding rows must align with signal windows')
    return frame, embeddings


def freeze_predictions(dataset_jobs, bundle, output_root, control_name='legacy_mq_soft', *, protocol_context=None):
    """Freeze every dataset before opening any reference values; safe to resume."""
    root = Path(output_root)
    if not dataset_jobs or bundle.get('schema_version') != 'adaptive_hr_bundle_v1':
        raise ValueError('A fitted adaptive bundle and nonempty dataset jobs are required')
    if root.resolve().is_relative_to(REPO/'paper'):
        raise ValueError('Evaluation outputs must not write to paper')
    context = protocol_context or {}
    file_hash = context.get('bundle_file_sha256')
    if file_hash is not None and not re.fullmatch(r'[0-9a-f]{64}', file_hash):
        raise ValueError('Serialized bundle fingerprint must be a SHA-256 digest')
    # Tree node arrays may contain padding whose bytes differ after unpickling.
    # Hashing the unchanged serialized file avoids that irrelevant RAM detail.
    fingerprint = file_hash if file_hash is not None else joblib.hash(bundle)
    protocol = {'schema_version': 'frozen_adaptive_reassessment_v1',
        'bundle_fingerprint': fingerprint,
        'bundle_fingerprint_source': ('serialized_bundle_file_sha256' if file_hash is not None
                                      else 'in_memory_synthetic_helper_object'),
        'bundle_protocol_sha256': bundle.get('protocol_sha256'),
        'selected_system': bundle['system_selection']['selected_system'],
        'control_name': control_name, 'jobs': [_job_signature(job) for job in dataset_jobs],
        'context': context, 'fit_or_selection_performed': False,
        'coverage_rule': 'common_finite_all_eight_new_experts_and_finite_embeddings; no_reference_filtering',
        'historically_exposed_reassessment': True, 'confirmatory': False,
        'wildppg_motion_units': 'conditional_unverified_g_scaling',
        'wildppg_reference': 'automatic_XQRS_with_residual_uncertainty'}
    root.mkdir(parents=True, exist_ok=True)
    path = root/'evaluation_protocol.json'
    if path.exists():
        if read_json(path) != protocol:
            raise ValueError('Frozen evaluation input/cache/model signature changed; choose a new study')
    else:
        write_json(path, protocol)
    receipt_path = root/'inference_freeze.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt['protocol_sha256'] != sha256(path):
            raise ValueError('Frozen inference protocol changed')
        verify_artifacts(root, receipt['artifacts'])
        return receipt
    blocks, coverage = {}, {}
    for job in dataset_jobs:
        dataset = job['dataset']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', dataset):
            raise ValueError('Dataset identity is not a safe directory name')
        frame, embeddings = _load_job(job)
        valid = np.isfinite(frame[CLASSICAL].to_numpy(float)).all(axis=1)
        valid &= ~np.isinf(frame[FEATURES['motion_quality']].to_numpy(float)).any(axis=1)
        for matrix in embeddings.values(): valid &= np.isfinite(matrix).all(axis=1)
        x = frame.loc[valid].reset_index(drop=True)
        arrays = {m: np.asarray(a[valid], dtype=np.float64) for m, a in embeddings.items()}
        prediction = x[['window_key', 'participant_id']].copy()
        for name in CLASSICAL: prediction[name] = x[name].to_numpy(float)
        for name in PROBES:
            prediction[name] = (bundle['experts'][name].predict(arrays[name.split('_',1)[0]])
                                if len(x) else np.array([], dtype=float))
        common = np.isfinite(prediction[EXPERTS].to_numpy(float)).all(axis=1)
        prediction = prediction.loc[common].reset_index(drop=True)
        x = x.loc[common].reset_index(drop=True)
        arrays = {m: a[common] for m, a in arrays.items()}
        if len(x):
            selection = bundle['expert_selection']
            prediction['candidate'] = predict_bundle(bundle, x, arrays)
            prediction['legacy_same_pair_soft'] = bundle['gates'][control_name].predict(x,
                prediction[selection['best_classical']].to_numpy(float),
                prediction[selection['best_foundation']].to_numpy(float), soft=True)
            prediction['preselected_single'] = prediction[selection['best_single']]
            if not np.isfinite(prediction[MODELS].to_numpy(float)).all():
                raise ValueError('Frozen candidate/control produced nonfinite predictions')
        else:
            for alias in ['candidate','legacy_same_pair_soft','preselected_single']: prediction[alias] = []
        blocks.setdefault(dataset, []).append(prediction)
        coverage.setdefault(dataset, []).append({'eligible_windows': len(frame),
            'signal_finite_windows': int(valid.sum()), 'paired_windows': len(prediction),
            'eligible_key_hash': hashlib.sha256('\n'.join(frame.window_key.astype(str)).encode()).hexdigest(),
            'eligible_participants': sorted(frame.participant_id.unique().tolist())})
    artifacts = {}
    for dataset, chunks in blocks.items():
        frame = pd.concat(chunks, ignore_index=True)
        if frame.empty or frame.window_key.duplicated().any():
            raise ValueError('Empty finite coverage or duplicate evaluation window keys across chunks')
        folder = root/dataset
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(folder/'frozen_predictions_without_labels.csv', index=False)
        write_json(folder/'coverage.json', {'chunks': coverage[dataset],
            'eligible_windows': sum(c['eligible_windows'] for c in coverage[dataset]),
            'paired_windows': len(frame), 'paired_participants': int(frame.participant_id.nunique()),
            'mask_uses_reference_values': False})
        for name in ['frozen_predictions_without_labels.csv','coverage.json']:
            artifacts[f'{dataset}/{name}'] = sha256(folder/name)
    if [_job_signature(job) for job in dataset_jobs] != protocol['jobs']:
        raise ValueError('Evaluation cache/reference input changed during inference')
    receipt = {'status': 'all_dataset_predictions_frozen_without_references',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': sha256(path), 'datasets': sorted(blocks), 'artifacts': artifacts,
        'reference_values_decoded': False, 'fit_or_selection_performed': False}
    write_json(receipt_path, receipt)
    return receipt


def evaluate_frozen_predictions(dataset_jobs, output_root):
    root = Path(output_root)
    freeze = read_json(root/'inference_freeze.json')
    protocol = read_json(root/'evaluation_protocol.json')
    if freeze['protocol_sha256'] != sha256(root/'evaluation_protocol.json'):
        raise ValueError('Frozen inference protocol changed')
    verify_artifacts(root, freeze['artifacts'])
    if [_job_signature(job) for job in dataset_jobs] != protocol['jobs']:
        raise ValueError('Frozen evaluation cache/reference signature changed')
    complete_path = root/'evaluation_complete.json'
    if complete_path.exists():
        complete = read_json(complete_path)
        verify_artifacts(root, complete['artifacts'])
        return complete
    artifacts = {}
    for dataset in freeze['datasets']:
        jobs = [job for job in dataset_jobs if job['dataset']==dataset]
        expected = []
        for job in jobs:
            if 'frame' in job: expected.extend(job['frame'].window_key.tolist())
            else:
                metadata = pd.read_csv(job['metadata_path'], dtype={'window_key':str,'participant_id':str})
                if job.get('participants') is not None: metadata=metadata.loc[metadata.participant_id.isin(job['participants'])]
                expected.extend(metadata.window_key.tolist())
        reference_paths = sorted({Path(p) for job in jobs for p in job['reference_paths']})
        reference = pd.concat([pd.read_csv(p, dtype={'window_key':str}) for p in reference_paths], ignore_index=True)
        if (not {'window_key','y_true'} <= set(reference) or reference.window_key.isna().any()
                or reference.window_key.duplicated().any() or set(reference.window_key) != set(expected)
                or len(expected) != len(set(expected)) or not np.isfinite(reference.y_true).all()):
            raise ValueError('Reference window identities/finite targets do not match exact eligible signal windows')
        folder = root/dataset
        prediction = pd.read_csv(folder/'frozen_predictions_without_labels.csv', dtype={'window_key':str,'participant_id':str})
        paired = prediction.merge(reference[['window_key','y_true']], on='window_key', validate='one_to_one')
        if len(paired) != len(prediction): raise ValueError('Paired reference coverage changed')
        interpretation = ('historically exposed reassessment; motion assumes g; automatic ECG reference uncertain'
                          if dataset=='WildPPG-wrist' else 'historically exposed reassessment; not confirmatory')
        participant, summary = [], []
        for name in MODELS:
            for person, group in paired.groupby('participant_id',sort=True):
                participant.append({'model_name':name,'participant_id':person,
                    **metrics(group.y_true,group[name]),'interpretation':interpretation})
            values = [row for row in participant if row['model_name']==name]
            summary.append({'model_name':name,**metrics(paired.y_true,paired[name]),
                'n_participants':int(paired.participant_id.nunique()),
                'participant_mean_MAE':float(np.mean([v['MAE'] for v in values])),
                'interpretation':interpretation})
        paired.to_csv(folder/'paired_predictions.csv',index=False)
        pd.DataFrame(summary).to_csv(folder/'summary.csv',index=False)
        pd.DataFrame(participant).to_csv(folder/'participant_metrics.csv',index=False)
        record={'status':'frozen_candidate_reassessment_complete','dataset':dataset,
            'historically_exposed_reassessment':True,'confirmatory':False,'fit_or_selection_performed':False,
            'inference_freeze_sha256':sha256(root/'inference_freeze.json'),
            'interpretation':interpretation,
            'artifacts':{name:sha256(folder/name) for name in ['paired_predictions.csv','summary.csv','participant_metrics.csv']}}
        write_json(folder/'evaluation_manifest.json',record)
        for name in [*record['artifacts'],'evaluation_manifest.json']: artifacts[f'{dataset}/{name}']=sha256(folder/name)
    complete={'status':'frozen_candidate_reassessment_complete','datasets':freeze['datasets'],
        'inference_freeze_sha256':sha256(root/'inference_freeze.json'),
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),'artifacts':artifacts,
        'fit_or_selection_performed':False,'historically_exposed_reassessment':True,'confirmatory':False}
    write_json(complete_path,complete)
    return complete


def _verified_bundle(development_root):
    root=Path(development_root)
    protocol=read_json(root/'frozen_protocol.json')
    verify_run_inputs(protocol['config'],root)
    verify_cv(root)
    folder=root/'fitted_system'
    complete=read_json(folder/'complete.json')
    if complete.get('status')!='development_candidate_fitted_without_final_test' or complete.get('protocol_sha256')!=sha256(root/'frozen_protocol.json'):
        raise ValueError('Completed frozen candidate fit is required before reassessment')
    verify_artifacts(folder,complete['artifacts'])
    bundle=joblib.load(folder/'adaptive_hr_system.joblib')
    if (bundle['system_selection']!=read_json(root/'development_selection.json')
            or bundle['expert_selection']!=read_json(folder/'expert_selection.json')
            or bundle['protocol_sha256']!=complete['protocol_sha256']):
        raise ValueError('Frozen bundle, model selection and development protocol do not agree')
    files=[root/'frozen_protocol.json',root/'cv_complete.json',root/'development_selection.json',
           folder/'complete.json',folder/'adaptive_hr_system.joblib',folder/'expert_selection.json']
    return bundle,{str(p.resolve()):sha256(p) for p in files}


def _build_jobs(bundle, external_root, galaxy_only):
    source=Path(bundle['config']['source_root'])
    preparation=read_json(source/'manifests/preparation.json')
    split=read_json(source/'configs/participant_split.json')
    inputs={}
    def verified(path,expected=None):
        path=Path(path)
        digest=sha256(path)
        if expected is not None and digest!=expected: raise ValueError(f'Frozen source/cache artifact changed: {path}')
        inputs[str(path.resolve())]=digest
        return path
    source_freeze=verified(source/'manifests/frozen_fit.json')
    verified(source/'manifests/preparation.json')
    verified(source/'configs/participant_split.json')
    for name in ['metadata.csv','router_features.csv','classical_predictions.csv',
                 'pulseppg_embeddings.npy','papagei_embeddings.npy']:
        verified(source/f'features/{name}',preparation['artifacts'][f'features/{name}'])
    reference=verified(source/'evaluation/evaluation_reference.csv',preparation['artifacts']['evaluation/evaluation_reference.csv'])
    jobs=[{'dataset':'GalaxyPPG','metadata_path':source/'features/metadata.csv',
        'features_path':source/'features/router_features.csv','classical_path':source/'features/classical_predictions.csv',
        'embedding_paths':{m:source/f'features/{m}_embeddings.npy' for m in ['pulseppg','papagei']},
        'participants':split['test_participants'],'reference_paths':[reference]}]
    if set(split['test_participants']) & set(bundle['training_participants']): raise ValueError('Candidate training overlaps original test participants')
    if galaxy_only: return jobs,inputs
    for dataset,units_filename,count in [('PPG-DaLiA','ppgdalia_acc_units.json',15),('WildPPG-wrist','wildppg_acc_units.json',16)]:
        root=Path(external_root)/dataset
        config=read_json(verified(root/'frozen_config.json'))
        units=verified(REPO/'configs'/units_filename,config['units_sha256'])
        manifest_path=Path(config['manifest'])
        if not manifest_path.is_absolute(): manifest_path=REPO/manifest_path
        manifest=read_json(verified(manifest_path,config['manifest_sha256']))
        if (config['source_freeze_sha256']!=inputs[str(source_freeze.resolve())]
                or config['preprocessing']!='unchanged_Galaxy_harmonized'
                or manifest['ppg_inverted'] is not False): raise ValueError('External source/preprocessing contract changed')
        receipts=sorted((root/'prepared').glob('*/complete.json'))
        if len(receipts)!=count: raise ValueError('External participant preparation is incomplete')
        for path in receipts:
            receipt=read_json(verified(path))
            expected=receipt['inputs']
            if (expected['source_freeze_sha256']!=config['source_freeze_sha256']
                    or expected['manifest_sha256']!=config['manifest_sha256']
                    or expected['units_sha256']!=config['units_sha256']
                    or expected['implementation_sha256']!=sha256(REPO/'src/utils/run_external_research.py')
                    or receipt['encoder_frozen'] is not True or receipt['no_external_fit'] is not True):
                raise ValueError('External prepared inference provenance changed')
            archive=manifest['participant_archives'][receipt['participant']]
            archive_hash=archive.get('sha256',archive.get('archive_sha256'))
            if not isinstance(archive_hash,str) or expected['archive_sha256']!=archive_hash:
                raise ValueError('External signal archive receipt differs from accepted manifest')
            for name in ['metadata.csv','features.csv','pulseppg.npy','papagei.npy','references.csv']:
                verified(path.parent/name,receipt['artifacts'][name])
            jobs.append({'dataset':dataset,'metadata_path':path.parent/'metadata.csv',
                'features_path':path.parent/'features.csv',
                'embedding_paths':{m:path.parent/f'{m}.npy' for m in ['pulseppg','papagei']},
                'reference_paths':[path.parent/'references.csv']})
    return jobs,inputs


def _reuse_completed_evaluation(root, bundle, context, jobs):
    """Verify completed data using immutable input bytes, without replaying it.

    Historical protocols used an in-memory joblib fingerprint, which is not
    stable for reloaded RF tree padding. Their serialized bundle SHA and every
    other frozen input remain authoritative. Reader-code updates are allowed
    here because no new prediction, reference decode or result write occurs.
    """
    protocol_path = root/'evaluation_protocol.json'
    protocol = read_json(protocol_path)
    stored = protocol.get('context', {})
    bundle_path = str((Path(context['development_root'])/'fitted_system/adaptive_hr_system.joblib').resolve())
    bundle_sha = context['bundle_file_sha256']
    if (context['input_hashes'].get(bundle_path) != bundle_sha
            or stored.get('input_hashes', {}).get(bundle_path) != bundle_sha
            or stored.get('bundle_file_sha256', bundle_sha) != bundle_sha
            or stored.get('input_hashes') != context['input_hashes']
            or stored.get('development_root') != context['development_root']
            or protocol.get('schema_version') != 'frozen_adaptive_reassessment_v1'
            or protocol.get('selected_system') != bundle['system_selection']['selected_system']
            or protocol.get('bundle_protocol_sha256') != bundle.get('protocol_sha256')
            or protocol.get('control_name') != 'legacy_mq_soft'
            or protocol.get('jobs') != [_job_signature(job) for job in jobs]):
        raise ValueError('Completed evaluation frozen input/cache/bundle signature changed')
    for path, digest in context['input_hashes'].items():
        if sha256(path) != digest:
            raise ValueError('Completed evaluation frozen input bytes changed')
    freeze_path = root/'inference_freeze.json'
    freeze = read_json(freeze_path)
    expected_datasets = sorted({job['dataset'] for job in jobs})
    if (freeze.get('protocol_sha256') != sha256(protocol_path)
            or freeze.get('datasets') != expected_datasets
            or freeze.get('status') != 'all_dataset_predictions_frozen_without_references'
            or not freeze.get('artifacts')):
        raise ValueError('Completed evaluation inference freeze changed')
    verify_artifacts(root, freeze['artifacts'])
    complete = read_json(root/'evaluation_complete.json')
    if (complete.get('inference_freeze_sha256') != sha256(freeze_path)
            or complete.get('datasets') != expected_datasets
            or complete.get('status') != 'frozen_candidate_reassessment_complete'
            or not complete.get('artifacts')):
        raise ValueError('Completed evaluation completion/freeze identity changed')
    verify_artifacts(root, complete['artifacts'])
    return complete


def run_evaluation(development_root,external_root,output_root,galaxy_only=False):
    root=Path(output_root).resolve()
    if (root.is_relative_to(Path(development_root).resolve())
            or root.is_relative_to(Path(external_root).resolve())
            or root.is_relative_to(REPO/'experiments/conference_submission_2026_11_01')):
        raise ValueError('Reassessment must use a separate new output directory')
    bundle,inputs=_verified_bundle(development_root)
    jobs,cache_inputs=_build_jobs(bundle,external_root,galaxy_only)
    inputs.update(cache_inputs)
    bundle_path=str((Path(development_root)/'fitted_system/adaptive_hr_system.joblib').resolve())
    if bundle_path not in inputs:
        raise ValueError('Verified serialized bundle SHA is required for production evaluation')
    context={'development_root':str(Path(development_root).resolve()),'input_hashes':inputs,
        'bundle_file_sha256':inputs[bundle_path],
        'implementation_hashes':{str(Path(__file__).resolve()):sha256(__file__)},
        'model_changes_after_reassessment_are_prohibited_in_this_protocol':True}
    if (root/'evaluation_complete.json').exists():
        return _reuse_completed_evaluation(root,bundle,context,jobs)
    with threadpool_limits(limits=2):
        freeze_predictions(jobs,bundle,root,protocol_context=context)
        for path,digest in inputs.items():
            if sha256(path)!=digest: raise ValueError('Input changed after the inference freeze')
        return evaluate_frozen_predictions(jobs,root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-root',type=Path,default=Path('experiments/router_development_2026_11_01_float64'))
    parser.add_argument('--external-root',type=Path,default=Path('experiments/external_validation_2026_11_01'))
    parser.add_argument('--output-root',type=Path,default=Path('experiments/adaptive_candidate_evaluation_2026_11_01'))
    parser.add_argument('--galaxy-only',action='store_true')
    args=parser.parse_args()
    result=run_evaluation(args.development_root,args.external_root,args.output_root,args.galaxy_only)
    print(json.dumps({'status':result['status'],'datasets':result['datasets'],'output_root':str(args.output_root)}))


if __name__=='__main__': main()
