from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.ML.Cluster import Butina

import paths

SCHEMES = ("random", "cluster")

SPLIT_FINGERPRINT_RADIUS = 2
SPLIT_FINGERPRINT_BITS = 2048

CLUSTER_SIMILARITY = 0.65

SEED = 42


def _split_fingerprints(smiles: list[str]) -> list:
    generator = rfg.GetMorganGenerator(
        radius=SPLIT_FINGERPRINT_RADIUS, fpSize=SPLIT_FINGERPRINT_BITS
    )
    fingerprints = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"SMILES does not parse: {s!r}")
        fingerprints.append(generator.GetFingerprint(mol))
    return fingerprints


def butina_groups(smiles: list[str], similarity: float = CLUSTER_SIMILARITY) -> np.ndarray:
    cache = paths.CACHE / f"butina_{len(smiles)}_{str(similarity).replace('.', 'p')}.npy"
    if cache.exists():
        return np.load(cache)

    fingerprints = _split_fingerprints(smiles)
    n = len(fingerprints)

    distances: list[float] = []
    for i in range(1, n):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        distances.extend(1.0 - s for s in similarities)

    clusters = Butina.ClusterData(distances, n, 1.0 - similarity, isDistData=True)

    labels = np.empty(n, dtype=int)
    for label, members in enumerate(clusters):
        for index in members:
            labels[index] = label

    paths.ensure_dirs()
    np.save(cache, labels)
    print(f"  Butina: {n:,} molecules -> {len(clusters):,} clusters at Tanimoto >= {similarity}")
    return labels


def make_groups(smiles: list[str], scheme: str) -> np.ndarray:
    if scheme == "random":
        return np.arange(len(smiles))
    if scheme == "cluster":
        return butina_groups(smiles)
    raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")


def grouped_kfold(groups: np.ndarray, k: int = 5, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    order = order[np.argsort(-sizes[order], kind="stable")]

    load = np.zeros(k)
    fold_of_group = np.empty(len(unique), dtype=int)
    for group_index in order:
        fold = int(np.argmin(load))
        fold_of_group[group_index] = fold
        load[fold] += sizes[group_index]

    fold_of_row = fold_of_group[inverse]
    all_rows = np.arange(len(groups))
    return [
        (np.sort(all_rows[fold_of_row != fold]), np.sort(all_rows[fold_of_row == fold]))
        for fold in range(k)
    ]


def single_split(
    groups: np.ndarray,
    seed: int = SEED,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))

    n_test_target = test_frac * len(groups)
    n_val_target = val_frac * len(groups)

    part = np.zeros(len(unique), dtype=int)
    n_test = n_val = 0
    for group_index in order:
        if n_test < n_test_target:
            part[group_index] = 2
            n_test += sizes[group_index]
        elif n_val < n_val_target:
            part[group_index] = 1
            n_val += sizes[group_index]

    part_of_row = part[inverse]
    all_rows = np.arange(len(groups))
    return (
        np.sort(all_rows[part_of_row == 0]),
        np.sort(all_rows[part_of_row == 1]),
        np.sort(all_rows[part_of_row == 2]),
    )


def inner_split(
    train_rows: np.ndarray,
    groups: np.ndarray,
    fold: int,
    val_frac: float = 0.18,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    fold_groups = groups[train_rows]
    unique = np.unique(fold_groups)

    rng = np.random.default_rng(seed + 1 + fold)
    rng.shuffle(unique)

    target = val_frac * len(train_rows)
    chosen: set[int] = set()
    count = 0
    for group in unique:
        if count >= target:
            break
        chosen.add(int(group))
        count += int((fold_groups == group).sum())

    is_val = np.isin(fold_groups, list(chosen))
    return np.sort(train_rows[~is_val]), np.sort(train_rows[is_val])


def describe(groups: np.ndarray, scheme: str) -> str:
    unique, counts = np.unique(groups, return_counts=True)
    singletons = int((counts == 1).sum())
    return (
        f"{scheme}: {len(groups):,} compounds in {len(unique):,} groups, "
        f"{singletons:,} of size 1, largest {counts.max():,}"
    )
