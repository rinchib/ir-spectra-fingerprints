from __future__ import annotations

import numpy as np
import pandas as pd

import metrics
import paths
from fingerprints import (
    BASELINE,
    MorganConfig,
    MorganFeaturizer,
    folding_collision_stats,
    identical_row_groups,
    parse_molecules,
)


def design_grid() -> list[tuple[str, MorganConfig]]:
    grid: list[tuple[str, MorganConfig]] = [("baseline", BASELINE)]

    for radius in (0, 1, 3, 4, 6):
        grid.append(("radius", MorganConfig(radius=radius, n_bits=2048)))
    for radius in (0, 1, 2, 3, 4, 6):
        grid.append(("counts", MorganConfig(radius=radius, n_bits=2048, counts=True)))

    for n_bits in (512, 1024, 4096, 8192, 16384):
        grid.append(("width", MorganConfig(radius=2, n_bits=n_bits)))
    for n_bits in (512, 1024, 4096, 8192, 16384):
        grid.append(("width", MorganConfig(radius=2, n_bits=n_bits, counts=True)))

    for counts in (False, True):
        grid.append(("pooling", MorganConfig(radius=2, n_bits=2048, counts=counts, pooling="sort_slice")))
        grid.append(("pooling", MorganConfig(radius=2, n_bits=None, counts=counts, pooling="unfolded")))

    for invariants in ("feature", "vibrational"):
        for counts in (False, True):
            grid.append(
                ("invariants", MorganConfig(radius=2, n_bits=2048, counts=counts, invariants=invariants))
            )

    grid.append(("chirality", MorganConfig(radius=2, n_bits=2048, chirality=True)))
    grid.append(("chirality", MorganConfig(radius=2, n_bits=2048, counts=True, chirality=True)))

    return grid


def main() -> int:
    paths.ensure_dirs()

    corpus = pd.read_csv(paths.CORPUS_TABLE)
    Y = np.load(paths.CORPUS_SPECTRA)
    molecules = parse_molecules(corpus["canonical_smiles"].tolist())
    print(f"Corpus: {len(molecules):,} compounds, spectra {Y.shape}\n")

    header = (
        f"{'axis':<11s} {'configuration':<34s} {'dim':>6s} {'subs':>7s} "
        f"{'sub coll':>9s} {'cover':>7s} {'groups':>7s} {'ambig':>7s} {'ceiling':>8s}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for axis, config in design_grid():
        featurizer = MorganFeaturizer(config)
        X = featurizer.fit_transform(molecules)

        collisions = folding_collision_stats(featurizer)
        coverage = featurizer.coverage(molecules)
        groups = identical_row_groups(X)
        ceiling = metrics.ambiguity_ceiling(Y, groups)

        row = {
            "axis": axis,
            "config": config.name,
            "radius": config.radius,
            "n_bits": config.n_bits,
            "counts": config.counts,
            "invariants": config.invariants,
            "chirality": config.chirality,
            "pooling": config.pooling,
            "n_features": X.shape[1],
            "density": float((X > 0).mean()),
            "substructure_coverage": coverage,
            **collisions,
            **ceiling,
        }
        rows.append(row)

        print(
            f"{axis:<11s} {config.name:<34s} {X.shape[1]:6d} "
            f"{collisions['n_substructures']:7d} "
            f"{100 * collisions['substructure_collision_rate']:8.1f}% "
            f"{100 * coverage:6.1f}% {len(groups):7d} "
            f"{100 * ceiling['fraction_ambiguous']:6.2f}% "
            f"{ceiling['ceiling_on_ambiguous']:8.4f}"
        )

    table = pd.DataFrame(rows)
    out = paths.RESULTS / "fingerprint_statistics.csv"
    table.to_csv(out, index=False)
    print(f"\nWrote {out}")

    def lookup(name: str) -> pd.Series:
        return table.loc[table["config"] == name].iloc[0]

    baseline = lookup(BASELINE.name)
    print("\nWhat moves molecule-level ambiguity, against the ECFP4 baseline")
    print(
        f"  {'baseline ' + BASELINE.name:<34s} "
        f"{100 * baseline['fraction_ambiguous']:5.2f}% ambiguous  "
        f"ceiling {baseline['ceiling_overall']:.4f}"
    )
    for label, name in [
        ("counts instead of presence", "r2_b2048_count"),
        ("unfolded, still binary", "r2_unfolded_bin"),
        ("unfolded with counts", "r2_unfolded_count"),
        ("eight times the width", "r2_b8192_bin"),
        ("radius 6", "r6_b2048_bin"),
        ("pharmacophore invariants", "r2_b2048_bin_feature"),
        ("vibrational invariants", "r2_b2048_bin_vibrational"),
        ("stereochemistry included", "r2_b2048_bin_chiral"),
        ("stereochemistry with counts", "r2_b2048_count_chiral"),
    ]:
        if name not in set(table["config"]):
            continue
        row = lookup(name)
        delta = 100 * (row["fraction_ambiguous"] - baseline["fraction_ambiguous"])
        print(
            f"  {label:<34s} {100 * row['fraction_ambiguous']:5.2f}% ambiguous  "
            f"ceiling {row['ceiling_overall']:.4f}  ({delta:+.2f} points)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
