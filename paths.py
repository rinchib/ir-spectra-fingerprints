from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
PROCESSED = DATA / "processed"

RESULTS = ROOT / "results"

SPECTRA_XLSX = RAW / "spectra_normalized.xlsx"
SMILES_XLSX = RAW / "SMILES.xlsx"
SPECTRA_METADATA_CSV = RAW / "spectra_metadata_all.csv"

CACHE_Y = CACHE / "Y_raw.npy"
CACHE_META = CACHE / "meta_raw.pkl"
CACHE_WAVENUMBERS = CACHE / "wavenumbers.npy"

CORPUS_SPECTRA = PROCESSED / "corpus_spectra.npy"
CORPUS_TABLE = PROCESSED / "corpus.csv"
CORPUS_WAVENUMBERS = PROCESSED / "wavenumbers.npy"
CORPUS_FUNNEL = PROCESSED / "funnel.json"


def ensure_dirs() -> None:
    for d in (CACHE, PROCESSED, RESULTS):
        d.mkdir(parents=True, exist_ok=True)
