"""Compare preserved and reassessed systems on exactly the same physical windows.

External frozen-transfer exports namespace participant IDs in the window key;
prepared native caches use local IDs. Only that declared dataset prefix is
removed for comparison. No intersection/subset selection or model fitting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.run_router_development import read_json, verify_artifacts
from src.utils.submission_protocol import metrics, sha256, write_json


def compare(root: Path, source: Path, external: Path):
    complete = read_json(root / 'evaluation_complete.json')
    verify_artifacts(root, complete['artifacts'])
    rows, inputs, mappings = [], {}, {}
    for dataset in ['GalaxyPPG', 'PPG-DaLiA', 'WildPPG-wrist']:
        new_path = root / dataset / 'paired_predictions.csv'
        new = pd.read_csv(new_path)
        if dataset == 'GalaxyPPG':
            old_path = source / 'predictions/final_test_predictions.csv'
            long = pd.read_csv(old_path)
            if long.groupby('window_key').y_true.nunique().ne(1).any():
                raise ValueError('Preserved references disagree across models')
            old = long[['window_key', 'y_true']].drop_duplicates('window_key').merge(
                long.pivot(index='window_key', columns='model_name', values='y_pred'),
                on='window_key', validate='one_to_one')
        else:
            old_path = external / dataset / 'transfer/paired_predictions.csv'
            old = pd.read_csv(old_path)
            prefix = dataset + '::'
            if not old.window_key.str.startswith(prefix).all():
                raise ValueError('Preserved external window-key namespace changed')
            old['window_key'] = old.window_key.str[len(prefix):]
            mappings[dataset] = f'Remove exactly {prefix} from preserved window-key prefix; dataset remains explicit'
        if (new.window_key.duplicated().any() or old.window_key.duplicated().any()
                or set(new.window_key) != set(old.window_key)):
            raise ValueError('Comparison requires the same complete window-key set, never an intersection')
        merged = new[['window_key', 'participant_id', 'y_true', 'candidate', 'legacy_same_pair_soft', 'peak_based']].merge(
            old[['window_key', 'y_true', 'motion_quality_soft', 'best_single', 'peak_based']],
            on='window_key', validate='one_to_one', suffixes=('_new', '_old'))
        if (not np.allclose(merged.y_true_new, merged.y_true_old, rtol=0, atol=1e-8)
                or not np.allclose(merged.peak_based_new, merged.peak_based_old, rtol=0, atol=1e-8)):
            raise ValueError('Physical-window references or preserved classical predictions differ')
        for name, field in [('preserved_primary_soft', 'motion_quality_soft'),
                            ('new_fit_legacy_same_pair_soft', 'legacy_same_pair_soft'),
                            ('frozen_adaptive_candidate', 'candidate'),
                            ('peak_based', 'peak_based_new'),
                            ('preserved_preselected_single', 'best_single')]:
            rows.append({'dataset': dataset, 'model_name': name,
                         **metrics(merged.y_true_new, merged[field]),
                         'n_participants': int(merged.participant_id.nunique())})
        inputs[str(new_path)] = sha256(new_path)
        inputs[str(old_path)] = sha256(old_path)
    frame = pd.DataFrame(rows)
    output = root / 'comparison_with_preserved_runs.csv'
    frame.to_csv(output, index=False)
    write_json(root / 'comparison_with_preserved_runs_manifest.json', {
        'status': 'same_physical_windows_references_and_classical_predictions_verified',
        'inputs': inputs, 'code_sha256': sha256(__file__), 'namespace_mapping': mappings,
        'comparison_uses_complete_equal_key_sets': True, 'model_fit_or_selection': False,
        'paper_updates': False, 'output_sha256': sha256(output),
        'command': 'python -m src.utils.compare_adaptive_candidate'})
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-root', type=Path, default=Path('experiments/adaptive_candidate_evaluation_2026_11_01'))
    parser.add_argument('--source-root', type=Path, default=Path('experiments/conference_submission_2026_11_01'))
    parser.add_argument('--external-root', type=Path, default=Path('experiments/external_validation_2026_11_01'))
    args = parser.parse_args()
    frame = compare(args.evaluation_root, args.source_root, args.external_root)
    print(frame[['dataset', 'model_name', 'MAE', 'p95_absolute_error', 'catastrophic_error_rate_20bpm', 'n_windows']].to_string(index=False))


if __name__ == '__main__':
    main()
