# Week 5-6 External Validation Status

Week 5-6 engineering support is in place, but real external-validation metrics are pending because the repository does not include the PPG-DaLiA or WildPPG wrist raw datasets.

| Dataset | Within-dataset baseline complete? | PulsePPG/PaPaGei complete? | Router transfer complete? | Result table path | Status |
| --- | --- | --- | --- | --- | --- |
| GalaxyPPG | yes | yes | yes | `experiments/week2_galaxyppg_corrected_2026-05-01/`, `experiments/week4_galaxyppg_lightweight_router_2026-05-13/` | completed locally; generated artifacts are gitignored |
| PPG-DaLiA | code ready; smoke tested on synthetic data | pending real data and checkpoint-backed embedding run | pending real data and cross-dataset routing run | `experiments/week5_ppgdalia_external_validation/` | requires real PPG-DaLiA data |
| WildPPG wrist | code ready; smoke tested on synthetic data | pending real data and checkpoint-backed embedding run | pending real data and cross-dataset routing run | `experiments/week6_wildppg_wrist_external_validation/` | requires real WildPPG wrist data |

## Available Code Paths

- PPG-DaLiA loader: `src/data/ppgdalia_loader.py`
- PPG-DaLiA export: `src/data/export_ppgdalia.py`
- WildPPG wrist manifest loader: `src/data/wildppg_loader.py`
- WildPPG wrist export: `src/data/export_wildppg_wrist.py`
- External status/result table builder: `src/utils/build_external_validation_status.py`
- PPG-DaLiA smoke test: `scripts/smoke_ppgdalia_export.py`
- WildPPG wrist smoke test: `scripts/smoke_wildppg_wrist_export.py`

## Result Table Command

After real Week 5/6 runs exist, build the external-validation status/result table with:

```bash
python -m src.utils.build_external_validation_status --experiments-root experiments --output-csv reports/external_validation_result_table.csv --output-md reports/external_validation_result_table.md
```

If run before real external metrics exist, the table will mark PPG-DaLiA and WildPPG wrist rows as `requires_real_data` rather than fabricating values.

