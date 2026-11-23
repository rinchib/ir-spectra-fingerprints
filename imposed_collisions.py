from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import metrics
import paths
import pipeline
import splits
from fingerprints import (
    COLLISION_STRATEGIES,
    ImposedCollisionConfig,
    MorganConfig,
    build_featurizer,
    folding_collision_stats,
    identical_row_groups,
    parse_molecules,
)

BASES = {
    "r2_unfolded_count": MorganConfig(
        radius=2, n_bits=None, counts=True, pooling="unfolded"
    ),
    "r2_unfolded_bin": MorganConfig(
        radius=2, n_bits=None, counts=False, pooling="unfolded"
    ),
}

DEFAULT_FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def unfolded_params(base_name: str, scheme: str) -> dict:
    source = paths.RESULTS / f"run_{base_name}__{scheme}.json"
    if not source.exists():
        raise SystemExit(
            f"missing {source.name}; the imposed collisions reuse the settings selected for "
            f"the unfolded fingerprint, so run compare_fingerprints.py for {base_name} first"
        )
    return json.loads(source.read_text(encoding="utf-8"))["params"]


def corpus_diagnostics(config, molecules) -> dict:
    featurizer = build_featurizer(config)
    X = featurizer.fit_transform(molecules)
    stats = folding_collision_stats(featurizer)
    groups = identical_row_groups(X)
    n_ambiguous = int(sum(len(group) for group in groups))
    return {
        "n_features": int(X.shape[1]),
        "n_substructures": stats["n_substructures"],
        "n_occupied_columns": stats["n_occupied_columns"],
        "n_colliding_substructures": stats["n_colliding_substructures"],
        "substructure_collision_rate": stats["substructure_collision_rate"],
        "n_moved": stats.get("n_moved", 0),
        "n_ambiguous_groups": len(groups),
        "n_ambiguous": n_ambiguous,
        "fraction_ambiguous": n_ambiguous / len(molecules),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="both", choices=splits.SCHEMES + ("both",))
    parser.add_argument("--base", default="r2_unfolded_count", choices=sorted(BASES))
    parser.add_argument("--fractions", nargs="*", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--strategies", nargs="*", default=["random"], choices=COLLISION_STRATEGIES)
    parser.add_argument("--merge-seed", type=int, default=0)
    parser.add_argument("--tune", action="store_true",
                        help="search again at every share moved instead of keeping "
                             "the settings of the unfolded fingerprint")
    parser.add_argument("--trials", type=int, default=pipeline.DEFAULT_TRIALS)
    parser.add_argument("--folds", type=int, default=pipeline.DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=splits.SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    base = BASES[args.base]
    fractions = sorted({float(f) for f in args.fractions})
    for f in fractions:
        if not 0.0 <= f < 1.0:
            raise SystemExit(f"fraction {f} is outside [0, 1)")

    corpus = pd.read_csv(paths.CORPUS_TABLE)
    Y = metrics.sum_normalize(np.load(paths.CORPUS_SPECTRA)).astype(np.float32)
    smiles = corpus["canonical_smiles"].tolist()
    molecules = parse_molecules(smiles)

    devices = pipeline.available_devices(args.device)
    print(f"Corpus: {len(molecules):,} compounds | devices: {len(devices)} x {', '.join(devices)}")

    def config_for(strategy: str, fraction: float):
        if fraction == 0.0:
            return base
        return ImposedCollisionConfig(base=base, fraction=fraction, strategy=strategy,
                                   merge_seed=args.merge_seed)

    print("\nWhat each share moved does to the fingerprint (fitted on the whole dataset)")
    header = (f"  {'strategy':<17s}{'f':>5s}{'moved':>7s}{'occupied':>9s}"
              f"{'sub coll':>10s}{'ambiguous':>11s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    diagnostics: dict[tuple[str, float], dict] = {}
    for strategy in args.strategies:
        for fraction in fractions:
            if fraction == 0.0 and (strategy, 0.0) in diagnostics:
                continue
            key = (strategy, fraction)
            diagnostics[key] = corpus_diagnostics(config_for(*key), molecules)
            d = diagnostics[key]
            print(f"  {strategy:<17s}{fraction:>5.2f}{d['n_moved']:>7,d}{d['n_occupied_columns']:>9,d}"
                  f"{100 * d['substructure_collision_rate']:>9.1f}%"
                  f"{d['n_ambiguous']:>7,d} ({100 * d['fraction_ambiguous']:4.2f}%)")

    schemes = list(splits.SCHEMES) if args.scheme == "both" else [args.scheme]

    for scheme in schemes:
        shared = None if args.tune else unfolded_params(args.base, scheme)
        mode = f"tuned per share at {args.trials} trials" if args.tune else "fixed architecture"
        print(f"\n{'=' * 70}\n{scheme}: {len(fractions)} shares x {len(args.strategies)} "
              f"strateg{'ies' if len(args.strategies) > 1 else 'y'}, {mode}\n{'=' * 70}")
        if shared is not None:
            print(f"  settings of the unfolded fingerprint: {shared}\n")

        started = time.time()
        rows = []
        for strategy in args.strategies:
            for fraction in fractions:
                config = config_for(strategy, fraction)
                record = pipeline.run(
                    config, molecules, Y, smiles, scheme=scheme,
                    n_trials=args.trials, n_folds=args.folds, seed=args.seed,
                    device=args.device, use_cache=not args.no_cache, params=shared,
                )
                d = diagnostics[(strategy, fraction)] if fraction > 0 else diagnostics[(args.strategies[0], 0.0)]
                rows.append({
                    "scheme": scheme,
                    "base": args.base,
                    "strategy": strategy if fraction > 0 else "none",
                    "fraction": fraction,
                    "config": record["config_name"],
                    **d,
                    "SIS_b10": record["fold_mean"],
                    "SIS_b10_sd": record["fold_std"],
                    "SIS_b15": record["pooled"]["SIS_b15"],
                    "cosine": record["pooled"]["cosine"],
                    "MAE": record["pooled"]["MAE"],
                    "tuned": record["tuned"],
                })

        table = pd.DataFrame(rows)
        suffix = f"_m{args.merge_seed}" if args.merge_seed else ""
        if args.base != "r2_unfolded_count":
            suffix = "_bin" + suffix
        out = paths.RESULTS / f"imposed_collisions_{scheme}{suffix}.csv"
        table.to_csv(out, index=False)

        print(f"\n  {'strategy':<17s}{'f':>5s}{'sub coll':>10s}{'ambiguous':>10s}"
              f"{'SIS_b10':>9s}{'sd':>8s}{'vs f=0':>9s}")
        print("  " + "-" * 66)
        endpoint = float(table.loc[table["fraction"] == 0.0, "SIS_b10"].iloc[0])
        for _, row in table.iterrows():
            print(f"  {row['strategy']:<17s}{row['fraction']:>5.2f}"
                  f"{100 * row['substructure_collision_rate']:>9.1f}%"
                  f"{row['n_ambiguous']:>10,d}{row['SIS_b10']:>9.4f}{row['SIS_b10_sd']:>8.4f}"
                  f"{row['SIS_b10'] - endpoint:>+9.4f}")

        for strategy in args.strategies:
            subset = table[(table["strategy"] == strategy) | (table["fraction"] == 0.0)]
            if len(subset) >= 3:
                rho = spearmanr(subset["substructure_collision_rate"], subset["SIS_b10"]).statistic
                print(f"\n  {strategy}: Spearman(collision rate, SIS_b10) over "
                      f"{len(subset)} shares = {rho:+.3f}")

        print(f"\n  wrote {out.name}   ({(time.time() - started) / 60:.0f} min)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
