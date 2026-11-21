# Morgan fingerprint design for IR spectrum prediction

Code for *Collision-Free Molecular Fingerprint Design for Gas-Phase Infrared Spectrum Prediction*.

A dense network predicts a gas-phase infrared spectrum from the Morgan fingerprint of a molecule. The code builds 48 fingerprints that differ in radius, width, assignment of substructures to columns (hashing, Sort & Slice, unfolded), stored value (binary or counts) and atom description, tunes a network for each, and scores it by five-fold cross-validation under a random and a Butina cluster partition.

## Data

The spectra are not redistributed here. `build_dataset.py` expects a `data/raw/` folder next to the code containing

- `spectra_normalized.xlsx`, one row per compound, with five metadata columns (`Compound`, `NIST ID`, `Source`, `Original Resolution`, `Up or downsampled?`) followed by one column per wavenumber from 580 to 3,846 cm⁻¹ in steps of 2 cm⁻¹
- `SMILES.xlsx`, with `NIST ID` and `SMILES` columns
- `spectra_metadata_all.csv`, one row per archived spectrum, with the columns `nistid`, `source`, `state`, `xunits`, `yunits`, `deltax`, `minx` and `maxx` read from the JCAMP headers

The three property datasets (ESOL, FreeSolv and Lipophilicity from MoleculeNet) are downloaded on first use into `data/external/`. On a machine without internet access, save the DeepChem files `delaney-processed.csv`, `SAMPL.csv` and `Lipophilicity.csv` there as `esol.csv`, `freesolv.csv` and `lipo.csv`.

## Running

```bash
pip install -r requirements.txt
python build_dataset.py
python fingerprint_statistics.py
python compare_fingerprints.py
python imposed_collisions.py --strategies random rare_moved rare_removed
python imposed_collisions.py --base r2_unfolded_bin
python empty_columns.py
python seed_replicates.py
python training_set_size.py
python property_datasets.py
```

`build_dataset.py` writes the 8,085-compound dataset to `data/processed/`. `compare_fingerprints.py` has to run before the four scripts that follow it, which reuse the network settings it selects. Tables are written to `results/`, together with the selected settings (`run_*.json`) and the held-out predictions (`pred_*.npy`) of each run. Finished runs are loaded from there, so an interrupted script continues where it stopped.

Training uses all visible GPUs, with one search trial and one fold per device. `--device cuda:0` or `--device cpu` pins a script to one device, `--scheme random` or `--scheme cluster` restricts it to one partition, and `--only` restricts `compare_fingerprints.py` to the named fingerprints. Install the PyTorch build that matches your CUDA version. The RDKit version affects the fingerprint identifiers and the cluster partition.

## Layout

| File | Contents |
|---|---|
| `paths.py` | locations of the data and results |
| `spectra.py` | reading the source spreadsheets, normalization |
| `preprocess.py` | filters that reduce the source spectra to the dataset |
| `fingerprints.py` | Morgan fingerprints with hashing, Sort & Slice and unfolding, binary or counts, three atom descriptions, MACCS keys, imposed collisions |
| `splits.py` | random and Butina cluster partitions, folds |
| `metrics.py` | SID and SIS |
| `models.py` | dense network and training loop |
| `pipeline.py` | hyperparameter search and cross-validated training of one fingerprint |
| `build_dataset.py` | builds the dataset from the source files |
| `fingerprint_statistics.py` | collisions between substructures and molecules with identical vectors |
| `compare_fingerprints.py` | the 48 fingerprints under both partitions |
| `imposed_collisions.py` | collisions imposed on the unfolded fingerprint |
| `empty_columns.py` | imposed collisions with the emptied columns removed from the input |
| `seed_replicates.py` | repeated training with different random seeds |
| `training_set_size.py` | baseline and unfolded count fingerprint on reduced training sets |
| `property_datasets.py` | eight fingerprints on ESOL, FreeSolv and Lipophilicity |
