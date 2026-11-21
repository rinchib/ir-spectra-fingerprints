from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ALLOWED_ELEMENTS = frozenset({"C", "H", "O", "N", "Si", "P", "S", "F", "Cl", "Br", "I"})

WAVENUMBER_XUNITS = frozenset({"1/CM", "cm-1"})

ABSORBING_YUNITS = frozenset({"ABSORBANCE", "(micromol/mol)-1m-1 (base 10)"})

SPECTRAL_DUPLICATE_COSINE = 0.999


@dataclass
class FilterResult:
    name: str
    criterion: str
    keep: np.ndarray
    detail: dict = field(default_factory=dict)

    @property
    def n_dropped(self) -> int:
        return int((~self.keep).sum())

    @property
    def n_kept(self) -> int:
        return int(self.keep.sum())


def canonicalize(smiles: Sequence[str]) -> list[str | None]:
    out: list[str | None] = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        out.append(Chem.MolToSmiles(mol) if mol is not None else None)
    return out


def filter_selection_criteria(smiles: Sequence[str]) -> FilterResult:
    keep = np.ones(len(smiles), dtype=bool)
    n_unparseable = n_multifragment = n_element = 0

    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            keep[i] = False
            n_unparseable += 1
        elif "." in s or len(Chem.GetMolFrags(mol)) > 1:
            keep[i] = False
            n_multifragment += 1
        elif {a.GetSymbol() for a in mol.GetAtoms()} - ALLOWED_ELEMENTS:
            keep[i] = False
            n_element += 1

    return FilterResult(
        name="selection criteria",
        criterion="single fragment, elements in C,H,O,N,Si,P,S,F,Cl,Br,I",
        keep=keep,
        detail={
            "unparseable": n_unparseable,
            "mixtures_and_salts": n_multifragment,
            "disallowed_elements": n_element,
        },
    )


def filter_structure_duplicates(canonical: Sequence[str | None]) -> FilterResult:
    keep = np.ones(len(canonical), dtype=bool)
    seen: set[str] = set()
    for i, c in enumerate(canonical):
        if c is None:
            continue
        if c in seen:
            keep[i] = False
        else:
            seen.add(c)

    return FilterResult(
        name="structure duplicates",
        criterion="unique canonical isomeric SMILES",
        keep=keep,
        detail={"collapsed": int((~keep).sum())},
    )


def filter_spectral_duplicates(
    Y: np.ndarray,
    threshold: float = SPECTRAL_DUPLICATE_COSINE,
    max_examples: int = 5,
) -> FilterResult:
    n = Y.shape[0]
    Yc = np.clip(Y, 0.0, None)
    unit = Yc / (np.linalg.norm(Yc, axis=1, keepdims=True) + 1e-12)

    keep = np.ones(n, dtype=bool)
    rows = np.arange(n)
    examples: list[tuple[int, int, float]] = []

    for i in range(n):
        if not keep[i]:
            continue
        sims = unit @ unit[i]
        dup = (rows > i) & keep & (sims >= threshold)
        if dup.any():
            if len(examples) < max_examples:
                j = int(np.flatnonzero(dup)[0])
                examples.append((i, j, float(sims[j])))
            keep[dup] = False

    return FilterResult(
        name="spectral duplicates",
        criterion=f"pairwise trace cosine < {threshold}",
        keep=keep,
        detail={"collapsed": int((~keep).sum()), "examples": examples},
    )


def resolve_ordinate_units(
    nist_ids: Sequence[str],
    sources: Sequence[str],
    metadata: pd.DataFrame,
) -> list[str]:
    qualifying = metadata[
        (metadata["state"] == "gas")
        & (metadata["xunits"].isin(WAVENUMBER_XUNITS))
        & (metadata["deltax"].abs() <= 4.0)
        & (metadata["minx"] <= 580.0)
        & (metadata["maxx"] >= 3846.0)
    ]
    by_key = qualifying.groupby(["nistid", "source"])["yunits"].apply(set)

    units: list[str] = []
    for nist_id, source in zip(nist_ids, sources):
        found = by_key.get((nist_id, source))
        if not found:
            units.append("UNKNOWN")
        elif len(found) == 1:
            units.append(next(iter(found)))
        else:
            units.append("MIXED")
    return units


def filter_ordinate_format(
    Y: np.ndarray,
    ordinate_units: Sequence[str],
    baseline_median: float = 0.5,
) -> FilterResult:
    units = np.asarray(ordinate_units, dtype=object)
    from_metadata = ~np.isin(units, list(ABSORBING_YUNITS))

    trace_median = np.median(np.clip(Y, 0.0, None), axis=1)
    from_shape = trace_median > baseline_median

    if not np.array_equal(from_metadata, from_shape):
        disagree = np.flatnonzero(from_metadata != from_shape)
        raise ValueError(
            "ordinate-format checks disagree on rows "
            f"{disagree.tolist()}: metadata says "
            f"{units[disagree].tolist()}, trace medians are "
            f"{np.round(trace_median[disagree], 3).tolist()}"
        )

    keep = ~from_metadata
    return FilterResult(
        name="ordinate format",
        criterion="absorbance targets only",
        keep=keep,
        detail={
            "dropped_units": sorted(set(units[from_metadata].tolist())),
            "dropped_median_absorbance": np.round(trace_median[from_metadata], 3).tolist(),
        },
    )


def clip_negative_absorbance(Y: np.ndarray) -> tuple[np.ndarray, int]:
    n_negative = int((Y < 0).sum())
    return np.clip(Y, 0.0, None), n_negative


def subset(keep: np.ndarray, *arrays: Iterable) -> tuple:
    idx = np.flatnonzero(keep)
    out = []
    for a in arrays:
        if isinstance(a, np.ndarray):
            out.append(a[idx])
        elif isinstance(a, pd.DataFrame):
            out.append(a.iloc[idx].reset_index(drop=True))
        else:
            out.append([a[i] for i in idx])
    return tuple(out)
