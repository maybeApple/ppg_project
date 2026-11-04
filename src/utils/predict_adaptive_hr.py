"""Predict HR from explicitly keyed signal caches with a fitted local bundle.

This command performs inference only: it never fits, loads reference targets or
computes errors. Embedding metadata must describe the row order of both supplied
NPY files. Feature CSV rows may be a shuffled subset of those window keys.
Use trusted local joblib bundles and the frozen encoder preprocessing/feature
units specified by their input contract.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile

import joblib
import numpy as np
import pandas as pd

from src.utils.run_router_development import CLASSICAL, predict_bundle
from src.utils.submission_protocol import FEATURES, sha256, write_json


REPO = Path(__file__).resolve().parents[2]
REQUIRED_FEATURES = ['window_key', *CLASSICAL, *FEATURES['motion_quality']]
SAFE_METADATA = {'window_key', 'participant_id', 'session_id', 'session_name', 'activity', 'split',
                 'window_index', 'window_start_ms', 'window_end_ms', 'window_length_s', 'stride_s',
                 'ppg_sampling_hz', 'outer_fold'}
# These fields must not accompany inference inputs, even if they would otherwise
# be ignored by a selected gate. Observed PPG peak/spectral disagreement is legal.
REFERENCE_FIELD = re.compile(
    r'(^|_)(y_true|y_pred|label|target|reference|ref|error|err|ecg|ibi|rr|'
    r'beat_interval|valid_beat_count|winner|oracle|native_watch_hr)($|_)', re.IGNORECASE)


def _read_signal_csv(path: Path, *, required: list[str]) -> pd.DataFrame:
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    if (not header or len(set(header)) != len(header)
            or any(not name or name != name.strip() for name in header)):
        raise ValueError(f'CSV headers must be present, unique and unambiguous: {path}')
    forbidden = [name for name in header if REFERENCE_FIELD.search(name)]
    if forbidden:
        raise ValueError(f'Inference CSV must not contain target/reference/error fields: {forbidden}')
    unknown = set(header) - set(REQUIRED_FEATURES) - SAFE_METADATA
    if unknown:
        raise ValueError(f'Inference CSV contains unsupported non-signal columns: {sorted(unknown)}')
    if set(required) - set(header):
        raise ValueError(f'Missing required signal columns: {sorted(set(required)-set(header))}')
    frame = pd.read_csv(path, dtype={'window_key': str})
    if (frame.empty or frame.window_key.isna().any() or frame.window_key.duplicated().any()
            or frame.window_key.str.strip().eq('').any()
            or not frame.window_key.eq(frame.window_key.str.strip()).all()):
        raise ValueError('Window keys must be present, nonempty, unique and unambiguous')
    return frame


def _aligned_embeddings(path: Path, metadata_rows: int, positions: np.ndarray) -> np.ndarray:
    matrix = np.load(path, mmap_mode='r', allow_pickle=False)
    if (matrix.ndim != 2 or matrix.shape[0] != metadata_rows or matrix.shape[1] < 1
            or matrix.dtype.kind not in {'f', 'i', 'u'}):
        raise ValueError('Embedding matrix shape/dtype disagrees with embedding metadata')
    # Validate in bounded chunks instead of allocating a full boolean copy for a
    # long native-data recording. Rejected rows cannot be hidden by subsetting.
    for start in range(0, len(matrix), 8192):
        if not np.isfinite(matrix[start:start+8192]).all():
            raise ValueError('Embedding matrices require finite values on every metadata row')
    return np.asarray(matrix[positions])


def _check_representation_sidecars(paths):
    """Known v3 representations cannot enter this v1-bundle inference CLI.

    Matching dimensions/window keys do not establish preprocessing equivalence.
    A future v3 fitted system needs an explicit v3-aware entrypoint; adding a
    metadata claim to an existing adaptive_hr_bundle_v1 does not upgrade it.
    """
    files = {paths[name].parent / 'protocol.json' for name in
             ['embedding_metadata', 'pulseppg_embeddings', 'papagei_embeddings']}
    hashes = {}
    for path in sorted(files):
        if not path.is_file():
            continue
        hashes[str(path)] = sha256(path)
        try:
            protocol = json.loads(path.read_text(encoding='utf-8-sig'))
        except (ValueError, OSError) as error:
            raise ValueError(f'Cannot validate embedding representation protocol: {path}') from error
        if not isinstance(protocol, dict):
            raise ValueError(f'Embedding representation protocol must be a JSON object: {path}')
        preprocessing = protocol.get('preprocessing', {})
        if (protocol.get('schema_version') == 'anti_alias_preprocessing_v3'
                or isinstance(preprocessing, dict)
                and preprocessing.get('schema_version') == 'anti_alias_preprocessing_v3'):
            raise ValueError('V3 embeddings cannot be used with adaptive_hr_bundle_v1. '
                             'Refit probes/gates and use an explicit v3-aware inference entrypoint; '
                             'matching 512-dimensional arrays or a user-added claim are insufficient.')
    return hashes


def run_inference(bundle_path, features_path, embedding_metadata_path,
                  pulseppg_embeddings_path, papagei_embeddings_path, output_path) -> dict:
    paths = {name: Path(path).resolve() for name, path in {
        'bundle': bundle_path, 'features': features_path, 'embedding_metadata': embedding_metadata_path,
        'pulseppg_embeddings': pulseppg_embeddings_path,
        'papagei_embeddings': papagei_embeddings_path}.items()}
    output = Path(output_path).resolve()
    provenance = output.with_suffix(output.suffix + '.provenance.json')
    if output.suffix.lower() != '.csv':
        raise ValueError('Prediction output must have a .csv extension')
    if output.is_relative_to(REPO / 'paper'):
        raise ValueError('Inference outputs must not update the paper directory')
    if output in paths.values() or provenance in paths.values():
        raise ValueError('Inference output cannot overwrite an input file')
    if output.exists() or provenance.exists():
        raise FileExistsError('Prediction output or provenance already exists; choose a new output path')
    if len(set(paths.values())) != len(paths):
        raise ValueError('Bundle, features, metadata and embedding paths must be distinct')

    input_hashes = {str(path): sha256(path) for path in paths.values()}
    input_hashes.update(_check_representation_sidecars(paths))
    receipt_path = paths['bundle'].parent / 'complete.json'
    receipt = None
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
        if receipt.get('artifacts', {}).get(paths['bundle'].name) != input_hashes[str(paths['bundle'])]:
            raise ValueError('Fitted-system receipt does not verify the supplied bundle hash')
        input_hashes[str(receipt_path)] = sha256(receipt_path)

    frame = _read_signal_csv(paths['features'], required=REQUIRED_FEATURES)
    metadata = _read_signal_csv(paths['embedding_metadata'], required=['window_key'])
    for column in [*CLASSICAL, *FEATURES['motion_quality']]:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    if not np.isfinite(frame[CLASSICAL].to_numpy(float)).all():
        raise ValueError('Classical predictions must be finite; no fallback or window exclusion is performed')
    if np.isinf(frame[FEATURES['motion_quality']].to_numpy(float)).any():
        raise ValueError('Infinite inference features are invalid; NaN may denote an observation to impute')
    positions = pd.Index(metadata.window_key).get_indexer(frame.window_key)
    if np.any(positions < 0):
        raise ValueError('Feature window keys are missing from embedding metadata')
    embeddings = {
        model: _aligned_embeddings(paths[f'{model}_embeddings'], len(metadata), positions)
        for model in ['pulseppg', 'papagei']}
    bundle = joblib.load(paths['bundle'])
    if (receipt is not None and 'protocol_sha256' in receipt
            and bundle.get('protocol_sha256') != receipt['protocol_sha256']):
        raise ValueError('Bundle and fitted-system receipt use different frozen protocols')
    prediction = np.asarray(predict_bundle(bundle, frame, embeddings), dtype=float)
    if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
        raise ValueError('Bundle must produce one finite prediction per requested window')
    for path, expected in input_hashes.items():
        if sha256(path) != expected:
            raise ValueError(f'Inference input changed while predictions were being computed: {path}')

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name+'.', suffix='.tmp',
                                     delete=False) as stream:
        temporary = Path(stream.name)
    try:
        pd.DataFrame({'window_key': frame.window_key,
                      'predicted_hr_bpm': prediction}).to_csv(temporary, index=False)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    record = {'status': 'signal_cache_inference_complete',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'n_predictions': len(frame), 'embedding_metadata_rows': len(metadata),
        'embedding_alignment': 'explicit_unique_window_key_lookup_in_feature_row_order',
        'n_metadata_rows_not_requested': len(metadata)-len(frame),
        'input_hashes': input_hashes, 'output_path': str(output), 'output_sha256': sha256(output),
        'bundle_receipt_verified': receipt is not None,
        'bundle_protocol_sha256': bundle.get('protocol_sha256'),
        'bundle_status': bundle.get('status'),
        'selected_system': bundle['system_selection']['selected_system'],
        'input_contract_declared_by_bundle': bundle.get('input_contract'),
        'input_preprocessing_and_physical_unit_equivalence_verified_by_cli': False,
        'missing_feature_counts': {name: int(frame[name].isna().sum())
                                   for name in FEATURES['motion_quality']},
        'fit_performed': False, 'reference_labels_loaded': False, 'metrics_computed': False,
        'prediction_clipping_or_invalid_expert_fallback': False,
        'implementation_hashes': {str(path): sha256(path) for path in [Path(__file__),
            REPO/'src/utils/run_router_development.py', REPO/'src/routing/adaptive.py']}}
    write_json(provenance, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True, help='Trusted local adaptive_hr_system.joblib')
    parser.add_argument('--features', type=Path, required=True, help='Signal-only feature/classical-prediction CSV')
    parser.add_argument('--embedding-metadata', type=Path, required=True, help='CSV window keys in NPY row order')
    parser.add_argument('--pulseppg-embeddings', type=Path, required=True)
    parser.add_argument('--papagei-embeddings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run_inference(args.bundle, args.features, args.embedding_metadata,
                           args.pulseppg_embeddings, args.papagei_embeddings, args.output)
    print(json.dumps({'status': result['status'], 'n_predictions': result['n_predictions'],
                      'output': result['output_path']}))


if __name__ == '__main__':
    main()
