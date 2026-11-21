from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import metrics
import paths
import pipeline
import splits
from compare_fingerprints import fingerprint_grid
from fingerprints import BASELINE, parse_molecules

DEFAULT_CONFIGS = (BASELINE.name, "r2_unfolded_count")
DEFAULT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def tuned_params(name: str, scheme: str) -> dict:
    source = paths.RESULTS / f"run_{name}__{scheme}.json"
    if not source.exists():
        raise SystemExit(f"missing {source.name}; run compare_fingerprints.py for {name} first")
    return json.loads(source.read_text(encoding="utf-8"))["params"]


def fraction_tag(fraction: float) -> str:
    return "" if fraction >= 1.0 else f"frac{int(round(100 * fraction)):03d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="both", choices=splits.SCHEMES + ("both",))
    parser.add_argument("--configs", nargs="*", default=list(DEFAULT_CONFIGS),
                        help="first one is treated as the baseline the others are compared with")
    parser.add_argument("--fractions", nargs="*", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--folds", type=int, default=pipeline.DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=splits.SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    fractions = sorted({float(f) for f in args.fractions})
    for f in fractions:
        if not 0.0 < f <= 1.0:
            raise SystemExit(f"fraction {f} is outside (0, 1]")

    grid = {c.name: c for c in fingerprint_grid()}
    for name in args.configs:
        if name not in grid:
            raise SystemExit(f"{name!r} is not one of the compared fingerprints")

    corpus = pd.read_csv(paths.CORPUS_TABLE)
    Y = metrics.sum_normalize(np.load(paths.CORPUS_SPECTRA)).astype(np.float32)
    smiles = corpus["canonical_smiles"].tolist()
    molecules = parse_molecules(smiles)
    n_total = len(molecules)

    devices = pipeline.available_devices(args.device)
    print(f"Corpus: {n_total:,} compounds | devices: {len(devices)} x {', '.join(devices)}")

    schemes = list(splits.SCHEMES) if args.scheme == "both" else [args.scheme]

    for scheme in schemes:
        print(f"\n{'=' * 70}\n{scheme}: {len(args.configs)} configurations x {len(fractions)} "
              f"training fractions, architecture fixed per configuration\n{'=' * 70}")
        started = time.time()
        rows = []
        for name in args.configs:
            config = grid[name]
            params = tuned_params(name, scheme)
            for fraction in fractions:
                record = pipeline.run(
                    config, molecules, Y, smiles, scheme=scheme, n_folds=args.folds,
                    seed=args.seed, device=args.device, use_cache=not args.no_cache,
                    params=params, tag=fraction_tag(fraction), train_fraction=fraction,
                )
                n_train = [s.get("n_train", n_total - s["n_test"]) for s in record["fold_scores"]]
                rows.append({
                    "scheme": scheme, "config": name, "fraction": fraction,
                    "n_train_mean": float(np.mean(n_train)),
                    "n_features": record["fold_scores"][0]["n_features"],
                    "SIS_b10": record["fold_mean"], "SIS_b10_sd": record["fold_std"],
                    "SIS_b15": record["pooled"]["SIS_b15"],
                    "cosine": record["pooled"]["cosine"], "MAE": record["pooled"]["MAE"],
                })

        table = pd.DataFrame(rows)
        baseline = args.configs[0]
        base_by_fraction = table[table["config"] == baseline].set_index("fraction")["SIS_b10"]
        table["gain_vs_baseline"] = [
            row["SIS_b10"] - base_by_fraction.get(row["fraction"], np.nan) for _, row in table.iterrows()
        ]
        out = paths.RESULTS / f"training_set_size_{scheme}.csv"
        table.to_csv(out, index=False)

        print(f"\n  {'config':<42s}{'fraction':>9s}{'n_train':>9s}{'SIS_b10':>9s}{'sd':>8s}{'gain':>9s}")
        print("  " + "-" * 86)
        for _, row in table.iterrows():
            gain = "" if row["config"] == baseline else f"{row['gain_vs_baseline']:>+9.4f}"
            print(f"  {row['config']:<42s}{row['fraction']:>9.2f}{row['n_train_mean']:>9,.0f}"
                  f"{row['SIS_b10']:>9.4f}{row['SIS_b10_sd']:>8.4f}{gain:>9s}")

        for name in args.configs[1:]:
            gains = table[table["config"] == name].sort_values("fraction")
            if len(gains) >= 2:
                first, last = gains.iloc[0], gains.iloc[-1]
                print(f"\n  {name}: gain over {baseline} moves from {first['gain_vs_baseline']:+.4f} "
                      f"at {first['fraction']:.0%} of the data to {last['gain_vs_baseline']:+.4f} at "
                      f"{last['fraction']:.0%}")

        print(f"\n  wrote {out.name}   ({(time.time() - started) / 60:.0f} min)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
