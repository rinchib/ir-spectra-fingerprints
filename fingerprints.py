from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as rfg

RDLogger.DisableLog("rdApp.*")

POOLINGS = ("hash", "sort_slice", "unfolded")
INVARIANTS = ("default", "feature", "vibrational")
COUNT_SCALINGS = ("raw", "log1p", "sqrt")


@dataclass(frozen=True)
class MorganConfig:
    radius: int = 2
    n_bits: int | None = 2048
    counts: bool = False
    invariants: str = "default"
    chirality: bool = False
    pooling: str = "hash"
    count_scaling: str = "raw"

    def __post_init__(self) -> None:
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {self.pooling!r}")
        if self.invariants not in INVARIANTS:
            raise ValueError(f"invariants must be one of {INVARIANTS}, got {self.invariants!r}")
        if self.count_scaling not in COUNT_SCALINGS:
            raise ValueError(f"count_scaling must be one of {COUNT_SCALINGS}")
        if self.pooling == "hash" and not self.n_bits:
            raise ValueError("hash pooling needs an explicit n_bits")
        if self.pooling == "unfolded" and self.n_bits is not None:
            raise ValueError("unfolded sets its own width; leave n_bits as None")
        if self.radius < 0:
            raise ValueError("radius must be non-negative")

    @property
    def name(self) -> str:
        parts = [f"r{self.radius}"]
        parts.append("unfolded" if self.n_bits is None else f"b{self.n_bits}")
        parts.append("count" if self.counts else "bin")
        if self.counts and self.count_scaling != "raw":
            parts.append(self.count_scaling)
        if self.invariants != "default":
            parts.append(self.invariants)
        if self.chirality:
            parts.append("chiral")
        if self.pooling == "sort_slice":
            parts.append(self.pooling)
        return "_".join(parts)

    def replace(self, **changes) -> "MorganConfig":
        return replace(self, **changes)


BASELINE = MorganConfig(radius=2, n_bits=2048, counts=False, pooling="hash")


_HYBRIDIZATION = {
    Chem.HybridizationType.S: 0,
    Chem.HybridizationType.SP: 1,
    Chem.HybridizationType.SP2: 2,
    Chem.HybridizationType.SP3: 3,
    Chem.HybridizationType.SP3D: 4,
    Chem.HybridizationType.SP3D2: 5,
}


def pack_vibrational(atom: Chem.Atom) -> int:
    mass = int(round(atom.GetMass()))
    degree = min(atom.GetDegree(), 7)
    n_hydrogen = min(atom.GetTotalNumHs(), 7)
    hybridization = _HYBRIDIZATION.get(atom.GetHybridization(), 7)
    aromatic = int(atom.GetIsAromatic())
    in_ring = int(atom.IsInRing())
    charge = min(max(atom.GetFormalCharge(), -4), 3) + 4

    return (
        (mass & 0x1FF)
        | (degree << 9)
        | (n_hydrogen << 12)
        | (hybridization << 15)
        | (aromatic << 18)
        | (in_ring << 19)
        | (charge << 20)
    )


def mix32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value


def vibrational_invariants(mol: Chem.Mol) -> list[int]:
    return [mix32(pack_vibrational(atom)) for atom in mol.GetAtoms()]


def _make_generator(config: MorganConfig):
    kwargs = {
        "radius": config.radius,
        "fpSize": config.n_bits or 2048,
        "includeChirality": config.chirality,
    }
    if config.invariants == "feature":
        kwargs["atomInvariantsGenerator"] = rfg.GetMorganFeatureAtomInvGen()
    return rfg.GetMorganGenerator(**kwargs)


def parse_molecules(smiles: Sequence[str]) -> list[Chem.Mol]:
    molecules = []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"SMILES at index {i} does not parse: {s!r}")
        molecules.append(mol)
    return molecules


class MorganFeaturizer:
    def __init__(self, config: MorganConfig):
        self.config = config
        self._generator = _make_generator(config)
        self.vocabulary_: dict[int, int] | None = None
        self.n_features_: int | None = None
        self.n_substructures_seen_: int | None = None
        self.substructure_document_frequency_: Counter | None = None


    def sparse_counts(self, molecules: Iterable[Chem.Mol]) -> list[dict[int, int]]:
        out = []
        for mol in molecules:
            if self.config.invariants == "vibrational":
                fingerprint = self._generator.GetSparseCountFingerprint(
                    mol, customAtomInvariants=vibrational_invariants(mol)
                )
            else:
                fingerprint = self._generator.GetSparseCountFingerprint(mol)
            out.append(dict(fingerprint.GetNonzeroElements()))
        return out


    def fit(
        self, molecules: Sequence[Chem.Mol], collect_diagnostics: bool = True
    ) -> "MorganFeaturizer":
        config = self.config

        if config.pooling == "hash":
            self.n_features_ = config.n_bits
            if collect_diagnostics:
                sparse = self.sparse_counts(molecules)
                self.substructure_document_frequency_ = Counter(
                    identifier for molecule in sparse for identifier in molecule
                )
                self.n_substructures_seen_ = len(self.substructure_document_frequency_)
            return self

        sparse = self.sparse_counts(molecules)
        document_frequency = Counter(
            identifier for molecule in sparse for identifier in molecule
        )
        self.substructure_document_frequency_ = document_frequency
        self.n_substructures_seen_ = len(document_frequency)

        if config.pooling == "sort_slice":
            ranked = sorted(document_frequency.items(), key=lambda kv: (-kv[1], kv[0]))
            kept = [identifier for identifier, _ in ranked[: config.n_bits]]
        else:
            kept = sorted(document_frequency)

        self.vocabulary_ = {identifier: column for column, identifier in enumerate(kept)}
        self.n_features_ = len(kept)
        return self


    def transform(self, molecules: Sequence[Chem.Mol]) -> np.ndarray:
        if self.n_features_ is None:
            raise RuntimeError("call fit() before transform()")

        config = self.config
        X = np.zeros((len(molecules), self.n_features_), dtype=np.float32)

        if config.pooling == "hash":
            for row, mol in enumerate(molecules):
                if config.invariants == "vibrational":
                    invariants = vibrational_invariants(mol)
                    fingerprint = (
                        self._generator.GetCountFingerprintAsNumPy(mol, customAtomInvariants=invariants)
                        if config.counts
                        else self._generator.GetFingerprintAsNumPy(mol, customAtomInvariants=invariants)
                    )
                else:
                    fingerprint = (
                        self._generator.GetCountFingerprintAsNumPy(mol)
                        if config.counts
                        else self._generator.GetFingerprintAsNumPy(mol)
                    )
                X[row] = fingerprint
        else:
            vocabulary = self.vocabulary_
            assert vocabulary is not None
            for row, counts in enumerate(self.sparse_counts(molecules)):
                for identifier, count in counts.items():
                    column = vocabulary.get(identifier)
                    if column is not None:
                        X[row, column] = count if config.counts else 1.0

        if config.counts:
            if config.count_scaling == "log1p":
                X = np.log1p(X, dtype=np.float32)
            elif config.count_scaling == "sqrt":
                X = np.sqrt(X, dtype=np.float32)

        return X

    def fit_transform(self, molecules: Sequence[Chem.Mol]) -> np.ndarray:
        return self.fit(molecules).transform(molecules)


    def coverage(self, molecules: Sequence[Chem.Mol]) -> float:
        if self.config.pooling == "hash":
            return 1.0
        vocabulary = self.vocabulary_ or {}
        seen = kept = 0
        for counts in self.sparse_counts(molecules):
            for identifier in counts:
                seen += 1
                kept += identifier in vocabulary
        return kept / seen if seen else 1.0


@dataclass(frozen=True)
class MACCSConfig:
    counts: bool = False

    radius: int | None = None
    n_bits: int = 166
    invariants: str = "maccs"
    chirality: bool = False
    pooling: str = "curated"
    count_scaling: str = "raw"

    @property
    def name(self) -> str:
        return "maccs_count" if self.counts else "maccs_bin"

    def replace(self, **changes) -> "MACCSConfig":
        return replace(self, **changes)


class MACCSFeaturizer:
    def __init__(self, config: MACCSConfig | None = None):
        from rdkit.Chem import MACCSkeys

        self.config = config or MACCSConfig()
        self._maccs = MACCSkeys
        self._patterns = {
            key: Chem.MolFromSmarts(smarts)
            for key, (smarts, _) in MACCSkeys.smartsPatts.items()
            if smarts != "?"
        }
        self.n_features_ = 166
        self.vocabulary_ = {key: key - 1 for key in range(1, 167)}
        self.n_substructures_seen_ = 166
        self.substructure_document_frequency_ = None

    def fit(self, molecules: Sequence[Chem.Mol], collect_diagnostics: bool = True) -> "MACCSFeaturizer":
        if collect_diagnostics:
            frequency: Counter = Counter()
            for mol in molecules:
                for bit in self._maccs.GenMACCSKeys(mol).GetOnBits():
                    frequency[bit] += 1
            self.substructure_document_frequency_ = frequency
        return self

    def transform(self, molecules: Sequence[Chem.Mol]) -> np.ndarray:
        X = np.zeros((len(molecules), 166), dtype=np.float32)
        for row, mol in enumerate(molecules):
            bits = self._maccs.GenMACCSKeys(mol)
            for bit in bits.GetOnBits():
                if 1 <= bit <= 166:
                    X[row, bit - 1] = 1.0
            if not self.config.counts:
                continue
            for key, pattern in self._patterns.items():
                if pattern is None or not (1 <= key <= 166) or X[row, key - 1] == 0.0:
                    continue
                X[row, key - 1] = float(len(mol.GetSubstructMatches(pattern, uniquify=True)))
        return X

    def fit_transform(self, molecules: Sequence[Chem.Mol]) -> np.ndarray:
        return self.fit(molecules, collect_diagnostics=False).transform(molecules)

    def coverage(self, molecules: Sequence[Chem.Mol]) -> float:
        morgan = MorganFeaturizer(BASELINE).fit(molecules)
        seen = morgan.n_substructures_seen_ or 0
        return 166 / seen if seen else 1.0


COLLISION_STRATEGIES = ("random", "rare_moved", "rare_removed")


def _default_collision_base() -> MorganConfig:
    return MorganConfig(radius=2, n_bits=None, counts=True, pooling="unfolded")


@dataclass(frozen=True)
class ImposedCollisionConfig:
    base: MorganConfig = field(default_factory=_default_collision_base)
    fraction: float = 0.0
    strategy: str = "random"
    merge_seed: int = 0

    def __post_init__(self) -> None:
        if self.base.pooling != "unfolded":
            raise ValueError("ImposedCollisionConfig needs a unfolded base")
        if not 0.0 <= self.fraction < 1.0:
            raise ValueError("fraction must lie in [0, 1)")
        if self.strategy not in COLLISION_STRATEGIES:
            raise ValueError(f"strategy must be one of {COLLISION_STRATEGIES}")

    @property
    def radius(self) -> int:
        return self.base.radius

    @property
    def n_bits(self) -> int | None:
        return self.base.n_bits

    @property
    def counts(self) -> bool:
        return self.base.counts

    @property
    def invariants(self) -> str:
        return self.base.invariants

    @property
    def chirality(self) -> bool:
        return self.base.chirality

    @property
    def pooling(self) -> str:
        return "unfolded"

    @property
    def count_scaling(self) -> str:
        return self.base.count_scaling

    @property
    def name(self) -> str:
        moved = int(round(100 * self.fraction))
        suffix = f"_m{self.merge_seed}" if self.merge_seed else ""
        return f"{self.base.name}_moved{moved:03d}_{self.strategy}{suffix}"

    def replace(self, **changes) -> "ImposedCollisionConfig":
        return replace(self, **changes)


class ImposedCollisionFeaturizer:
    def __init__(self, config: ImposedCollisionConfig):
        self.config = config
        self._inner = MorganFeaturizer(config.base)
        self.vocabulary_: dict[int, int] | None = None
        self.n_features_: int | None = None
        self.n_substructures_seen_: int | None = None
        self.substructure_document_frequency_: Counter | None = None
        self.n_moved_: int | None = None

    def fit(
        self, molecules: Sequence[Chem.Mol], collect_diagnostics: bool = True
    ) -> "ImposedCollisionFeaturizer":
        self._inner.fit(molecules)
        original = self._inner.vocabulary_
        frequency = self._inner.substructure_document_frequency_
        assert original is not None and frequency is not None

        identifiers = sorted(original)
        width = len(identifiers)
        n_move = min(int(round(self.config.fraction * width)), max(width - 1, 0))
        rng = np.random.default_rng(self.config.merge_seed)
        column_of = {identifier: original[identifier] for identifier in identifiers}

        if n_move > 0:
            if self.config.strategy == "random":
                moved = rng.choice(width, size=n_move, replace=False)
            else:
                by_prevalence = sorted(
                    range(width), key=lambda i: (frequency[identifiers[i]], identifiers[i])
                )
                moved = np.array(by_prevalence[:n_move])
            if self.config.strategy == "rare_removed":
                for source in moved:
                    del column_of[identifiers[source]]
            else:
                staying = np.setdiff1d(np.arange(width), moved)
                destinations = rng.choice(staying, size=n_move, replace=True)
                for source, destination in zip(moved, destinations):
                    column_of[identifiers[source]] = original[identifiers[destination]]

        self.vocabulary_ = column_of
        self.n_features_ = width
        self.n_substructures_seen_ = width
        self.substructure_document_frequency_ = frequency
        self.n_moved_ = int(n_move)
        return self

    def transform(self, molecules: Sequence[Chem.Mol]) -> np.ndarray:
        if self.vocabulary_ is None or self.n_features_ is None:
            raise RuntimeError("call fit() before transform()")
        base = self.config.base
        X = np.zeros((len(molecules), self.n_features_), dtype=np.float32)
        for row, counts in enumerate(self._inner.sparse_counts(molecules)):
            for identifier, count in counts.items():
                column = self.vocabulary_.get(identifier)
                if column is None:
                    continue
                if base.counts:
                    X[row, column] += count
                else:
                    X[row, column] = 1.0
        if base.counts:
            if base.count_scaling == "log1p":
                X = np.log1p(X, dtype=np.float32)
            elif base.count_scaling == "sqrt":
                X = np.sqrt(X, dtype=np.float32)
        return X

    def fit_transform(self, molecules: Sequence[Chem.Mol]) -> np.ndarray:
        return self.fit(molecules).transform(molecules)

    def coverage(self, molecules: Sequence[Chem.Mol]) -> float:
        return self._inner.coverage(molecules)

    def collision_stats(self) -> dict:
        if self.vocabulary_ is None:
            raise RuntimeError("call fit() before requesting collision statistics")
        occupied = Counter(self.vocabulary_.values())
        colliding = sum(count for count in occupied.values() if count > 1)
        width = self.n_features_ or 0
        return {
            "n_substructures": width,
            "n_features": width,
            "n_occupied_columns": len(occupied),
            "n_colliding_substructures": colliding,
            "substructure_collision_rate": colliding / width if width else 0.0,
            "n_moved": self.n_moved_,
        }


def build_featurizer(config):
    if isinstance(config, MACCSConfig):
        return MACCSFeaturizer(config)
    if isinstance(config, ImposedCollisionConfig):
        return ImposedCollisionFeaturizer(config)
    return MorganFeaturizer(config)


def folding_collision_stats(featurizer: MorganFeaturizer) -> dict:
    if isinstance(featurizer, ImposedCollisionFeaturizer):
        return featurizer.collision_stats()

    frequency = featurizer.substructure_document_frequency_
    if frequency is None:
        raise RuntimeError("call fit() before requesting collision statistics")

    n_substructures = len(frequency)
    config = featurizer.config

    if config.pooling == "hash":
        width = config.n_bits or 0
        occupied = Counter(identifier % width for identifier in frequency)
        colliding = sum(count for count in occupied.values() if count > 1)
        return {
            "n_substructures": n_substructures,
            "n_features": width,
            "n_occupied_columns": len(occupied),
            "n_colliding_substructures": colliding,
            "substructure_collision_rate": colliding / n_substructures if n_substructures else 0.0,
        }

    return {
        "n_substructures": n_substructures,
        "n_features": featurizer.n_features_,
        "n_occupied_columns": featurizer.n_features_,
        "n_colliding_substructures": 0,
        "substructure_collision_rate": 0.0,
    }


def identical_row_groups(X: np.ndarray) -> list[np.ndarray]:
    contiguous = np.ascontiguousarray(X)
    buckets: dict[bytes, list[int]] = {}
    for index, row in enumerate(contiguous):
        buckets.setdefault(row.tobytes(), []).append(index)
    return [np.array(rows) for rows in buckets.values() if len(rows) > 1]
