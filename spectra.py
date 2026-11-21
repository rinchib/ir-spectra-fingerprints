from __future__ import annotations

import time

import numpy as np
import pandas as pd
from openpyxl import load_workbook

import paths

N_META_COLUMNS = 5


def load_spectra(refresh: bool = False) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cached = paths.CACHE_Y.exists() and paths.CACHE_META.exists() and paths.CACHE_WAVENUMBERS.exists()
    if cached and not refresh:
        meta = pd.read_pickle(paths.CACHE_META)
        return meta, np.load(paths.CACHE_Y), np.load(paths.CACHE_WAVENUMBERS)

    started = time.time()
    workbook = load_workbook(paths.SPECTRA_XLSX, read_only=True, data_only=True)
    sheet = workbook.active
    rows = sheet.iter_rows(values_only=True)
    header = list(next(rows))

    meta_rows, traces = [], []
    for row in rows:
        meta_rows.append(row[:N_META_COLUMNS])
        traces.append(row[N_META_COLUMNS:])
    workbook.close()

    meta = pd.DataFrame(meta_rows, columns=[str(c) for c in header[:N_META_COLUMNS]])
    meta["NIST ID"] = meta["NIST ID"].astype(str)
    Y = np.asarray(traces, dtype=np.float32)
    wavenumbers = np.asarray([float(c) for c in header[N_META_COLUMNS:]], dtype=np.float64)

    paths.ensure_dirs()
    np.save(paths.CACHE_Y, Y)
    np.save(paths.CACHE_WAVENUMBERS, wavenumbers)
    meta.to_pickle(paths.CACHE_META)
    print(f"  parsed {paths.SPECTRA_XLSX.name} {Y.shape} in {time.time() - started:.0f} s (cached)")

    return meta, Y, wavenumbers


def load_smiles() -> pd.DataFrame:
    df = pd.read_excel(paths.SMILES_XLSX)
    df["NIST ID"] = df["NIST ID"].astype(str)
    return df.set_index("NIST ID")


def load_spectrum_metadata() -> pd.DataFrame:
    df = pd.read_csv(paths.SPECTRA_METADATA_CSV)
    df["nistid"] = df["nistid"].astype(str)
    return df


def sum_normalize(Y: np.ndarray) -> np.ndarray:
    Y = np.clip(np.nan_to_num(Y, nan=0.0), 0.0, None)
    return (Y / np.clip(Y.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)


def max_normalize(Y: np.ndarray) -> np.ndarray:
    Y = np.clip(np.nan_to_num(Y, nan=0.0), 0.0, None)
    return (Y / np.clip(Y.max(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)
