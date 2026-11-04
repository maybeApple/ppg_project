# GalaxyPPG schema notes

These notes are based on the actual files under `data/raw/GalaxyPPG/`.

## Files used for the current pipeline

- `PXX/GalaxyWatch/PPG.csv`
  - Columns: `dataReceived`, `timestamp`, `ppg`, `status`
  - Used fields:
    - `timestamp`: Galaxy Watch timestamp in milliseconds
    - `ppg`: raw watch PPG value
    - `status`: signal status, with `0` and `500` treated as usable by default

- `PXX/PolarH10/HR.csv`
  - Columns: `phoneTimestamp`, `hr`, `hrv`
  - Used fields:
    - `phoneTimestamp`: Polar phone-side timestamp in milliseconds
    - `hr`: chest-belt reference heart rate in bpm

- `PXX/PolarH10/IBI.csv`
  - Columns: `phoneTimestamp`, `duration`
  - Used fields:
    - `phoneTimestamp`: Polar phone-side timestamp in milliseconds
    - `duration`: R-R interval in milliseconds
  - Derived field:
    - `hr_bpm = 60000 / duration`

- `PXX/Event.csv`
  - Columns: `timestamp`, `session`, `status`
  - Used for session boundaries by pairing `ENTER` and `EXIT`

## Time alignment rule

Actual data inspection shows that `PolarH10/HR.csv` and `PolarH10/IBI.csv` are recorded on a clock that is shifted by about `+9` hours relative to Galaxy Watch and Event timestamps.

Current code therefore normalizes Polar timestamps with:

```text
aligned_timestamp_ms = phoneTimestamp - 9 * 60 * 60 * 1000
```

After this correction:

- Polar reference and Galaxy Watch PPG fall on the same Unix-millisecond axis
- session boundaries from `Event.csv` can be applied directly
- overlap is computed with:
  - `start = max(ppg_start, reference_start)`
  - `end = min(ppg_end, reference_end)`

## Session handling

- If `Event.csv` exists, windows are generated strictly inside each session interval.
- If `Event.csv` is missing, the code falls back to one synthetic `full_recording` session on the shared overlap only.

## Known missing data

- `P01`: no Galaxy Watch folder and no `Event.csv`
- `P07`, `P08`: missing `GalaxyWatch/HR.csv`, but `PPG.csv` exists, so the current pipeline still works

## Windowing defaults

- Window length: `10s`
- Default stride: `2s`
- Default label aggregation: `median(reference_hr_in_window)`

The 2-second stride was chosen as a compromise:

- finer temporal resolution than `5s`
- much less redundant than `1s`
- still practical for early experiments with participant-level train/test splits
