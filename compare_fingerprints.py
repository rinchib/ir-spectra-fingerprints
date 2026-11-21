from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import metrics
import paths
import pipeline
import splits
from fingerprints import (
    BASELINE,
    MACCSConfig,
    MorganConfig,
    parse_molecules,
)


def fingerprint_grid() -> list[MorganConfig]:
    grid = [BASELINE]

    grid += [
        MorganConfig(radius=2, n_bits=2048, counts=True),
        MorganConfig(radius=2, n_bits=2048, counts=True, count_scaling="log1p"),
        MorganConfig(radius=2, n_bits=2048, counts=True, count_scaling="sqrt"),
    ]

    for radius in (1, 3, 4, 6):
        grid.append(MorganConfig(radius=radius, n_bits=2048))
        grid.append(MorganConfig(radius=radius, n_bits=2048, counts=True))

    for n_bits in (1024, 4096, 8192, 16384):
        grid.append(MorganConfig(radius=2, n_bits=n_bits))
        grid.append(MorganConfig(radius=2, n_bits=n_bits, counts=True))
    for n_bits in (4096, 8192):
        grid.append(MorganConfig(radius=2, n_bits=n_bits, pooling="sort_slice"))
        grid.append(MorganConfig(radius=2, n_bits=n_bits, counts=True, pooling="sort_slice"))

    for counts in (False, True):
        grid.append(MorganConfig(radius=2, n_bits=2048, counts=counts, pooling="sort_slice"))
        grid.append(MorganConfig(radius=2, n_bits=None, counts=counts, pooling="unfolded"))

    for radius in (1, 3, 4, 6):
        for counts in (False, True):
            grid.append(MorganConfig(radius=radius, n_bits=None, counts=counts,
                                     pooling="unfolded"))

    grid.append(MACCSConfig(counts=False))
    grid.append(MACCSConfig(counts=True))

    for invariants in ("feature", "vibrational"):
        for counts in (False, True):
            grid.append(
                MorganConfig(radius=2, n_bits=2048, counts=counts, invariants=invariants)
            )

    grid.append(MorganConfig(radius=2, n_bits=16384, counts=True,
                             invariants="vibrational"))
    for radius in (1, 2, 3):
        grid.append(MorganConfig(radius=radius, n_bits=None, counts=True,
                                 invariants="vibrational", pooling="unfolded"))

    grid.append(MorganConfig(radius=2, n_bits=2048, chirality=True))
    grid.append(MorganConfig(radius=2, n_bits=2048, counts=True, chirality=True))

    seen, unique = set(), []
    for config in grid:
        if config.name not in seen:
            seen.add(config.name)
            unique.append(config)
    return unique


def summarize(records: list[dict], scheme: str) -> pd.DataFrame:
    rows = []
    for record in records:
        config = record["config"]
        rows.append(
            {
                "config": record["config_name"],
                "radius": config.get("radius"),
                "n_bits": config.get("n_bits"),
                "counts": config.get("counts"),
                "count_scaling": config.get("count_scaling"),
                "invariants": config.get("invariants"),
                "chirality": config.get("chirality"),
                "pooling": config.get("pooling"),
                "n_features": record["fold_scores"][0]["n_features"],
                "SIS_b10": record["fold_mean"],
                "SIS_b10_sd": record["fold_std"],
                "SIS_b15": record["pooled"]["SIS_b15"],
                "cosine": record["pooled"]["cosine"],
                "MAE": record["pooled"]["MAE"],
                "tuned": record["tuned"],
            }
        )
    table = pd.DataFrame(rows).sort_values("SIS_b10", ascending=False).reset_index(drop=True)
    out = paths.RESULTS / f"comparison_{scheme}.csv"
    table.to_csv(out, index=False)
    print(f"\nWrote {out}")
    return table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="both", choices=splits.SCHEMES + ("both",))
    parser.add_argument("--trials", type=int, default=pipeline.DEFAULT_TRIALS)
    parser.add_argument("--folds", type=int, default=pipeline.DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=splits.SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--only", nargs="*", help="restrict to these fingerprint names")
    args = parser.parse_args()

    corpus = pd.read_csv(paths.CORPUS_TABLE)
    Y = metrics.sum_normalize(np.load(paths.CORPUS_SPECTRA)).astype(np.float32)
    smiles = corpus["canonical_smiles"].tolist()
    molecules = parse_molecules(smiles)

    devices = pipeline.available_devices(args.device)
    print(f"Corpus: {len(molecules):,} compounds | devices: {len(devices)} x {', '.join(devices)}")

    schemes = list(splits.SCHEMES) if args.scheme == "both" else [args.scheme]

    for scheme in schemes:
        configs = fingerprint_grid()
        if args.only:
            configs = [c for c in configs if c.name in set(args.only)]
        print(f"\nComparing {len(configs)} fingerprints [{scheme}] at {args.trials} trials each\n")

        started = time.time()
        records = []
        for index, config in enumerate(configs, start=1):
            print(f"[{index}/{len(configs)}] ", end="")
            records.append(
                pipeline.run(
                    config,
                    molecules,
                    Y,
                    smiles,
                    scheme=scheme,
                    n_trials=args.trials,
                    n_folds=args.folds,
                    seed=args.seed,
                    device=args.device,
                    use_cache=not args.no_cache,
                )
            )

        table = summarize(records, scheme)
        print(f"\n{'config':<38s} {'dim':>7s} {'SIS_b10':>9s} {'sd':>7s} {'vs base':>8s}")
        print("-" * 74)
        base = table.loc[table["config"] == BASELINE.name, "SIS_b10"]
        base_score = float(base.iloc[0]) if len(base) else float("nan")
        for _, row in table.iterrows():
            delta = row["SIS_b10"] - base_score
            print(
                f"{row['config']:<38s} {row['n_features']:>7d} {row['SIS_b10']:>9.4f} "
                f"{row['SIS_b10_sd']:>7.4f} {delta:>+8.4f}"
            )
        print(f"\n{scheme}: {len(configs)} fingerprints in {(time.time() - started) / 60:.0f} min")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
