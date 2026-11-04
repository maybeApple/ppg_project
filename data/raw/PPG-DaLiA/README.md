# PPG-DaLiA Raw Data Placement

The raw PPG-DaLiA dataset is not bundled in this repository.

Place the participant pickle files in one of these layouts:

```text
data/raw/PPG-DaLiA/
|-- S1.pkl
|-- S2.pkl
`-- ...
```

or:

```text
data/raw/PPG-DaLiA/
|-- S1/
|   `-- S1.pkl
|-- S2/
|   `-- S2.pkl
`-- ...
```

The Week 5 loader expects each pickle to contain the standard PPG-DaLiA keys:

- `signal.wrist.BVP`
- `signal.wrist.ACC`
- `signal.chest.ECG`
- `rpeaks` when available
- `activity` when available

The export script uses wrist BVP as PPG, wrist ACC as motion input, and chest ECG/R-peaks as the reference source.
