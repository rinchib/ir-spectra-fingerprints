from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import metrics
import paths
import pipeline
import splits
from fingerprints import MorganConfig, parse_molecules

REPLICATED = {
    "random": MorganConfig(radius=1, n_bits=None, counts=True, pooling="unfolded"),
    "cluster": MorganConfig(radius=2, n_bits=16384, counts=True,
                            invariants="vibrational", pooling="sort_slice"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="both", choices=splits.SCHEMES + ("both",))
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4],
                        help="training seeds for the new replicates; the original run "
                             "(seed 42) is replicate zero")
    parser.add_argument("--trials", type=int, default=pipeline.DEFAULT_TRIALS)
    parser.add_argument("--folds", type=int, default=pipeline.DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=splits.SEED,
                        help="fold seed, which every replicate shares")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if args.seed in args.seeds:
        raise SystemExit(f"--seeds must not include the fold seed {args.seed}, which is replicate zero")

    corpus = pd.read_csv(paths.CORPUS_TABLE)
    Y = metrics.sum_normalize(np.load(paths.CORPUS_SPECTRA)).astype(np.float32)
    Y64 = Y.astype(np.float64)
    smiles = corpus["canonical_smiles"].tolist()
    molecules = parse_molecules(smiles)

    devices = pipeline.available_devices(args.device)
    print(f"Corpus: {len(molecules):,} compounds | devices: {len(devices)} x {', '.join(devices)}")

    schemes = list(splits.SCHEMES) if args.scheme == "both" else [args.scheme]

    for scheme in schemes:
        config = REPLICATED[scheme]
        name = config.name

        print(f"\n{'=' * 70}\n{scheme}: {name}, {1 + len(args.seeds)} replicates on identical folds"
              f"\n{'=' * 70}")
        started = time.time()

        members = []
        original = pipeline.run(config, molecules, Y, smiles, scheme=scheme, n_trials=args.trials,
                                n_folds=args.folds, seed=args.seed, device=args.device,
                                use_cache=not args.no_cache)
        params = original["params"]
        print(f"  architecture: {params}\n")
        members.append(("original", args.seed, original))
        for s in args.seeds:
            record = pipeline.run(config, molecules, Y, smiles, scheme=scheme, n_folds=args.folds,
                                  seed=args.seed, device=args.device,
                                  use_cache=not args.no_cache, params=params,
                                  tag=f"seed{s}", training_seed=s)
            members.append((f"seed{s}", s, record))

        normalized = [metrics.sum_normalize(r["predictions"].astype(np.float64)) for _, _, r in members]
        per_compound = np.column_stack([metrics.per_spectrum_sis(Y64, p) for p in normalized])
        rhos = [
            float(spearmanr(per_compound[:, i], per_compound[:, j]).statistic)
            for i in range(len(members)) for j in range(i + 1, len(members))
        ]

        member_rows = []
        for (label, s, record), column in zip(members, per_compound.T):
            member_rows.append({
                "scheme": scheme, "config": name, "replicate": label, "training_seed": s,
                "SIS_b10": record["fold_mean"], "SIS_b10_sd": record["fold_std"],
                "per_compound_mean": float(column.mean()),
            })
        member_table = pd.DataFrame(member_rows)
        singles = member_table["SIS_b10"].to_numpy()
        print(f"  single-model SIS_b10 across replicates: mean {singles.mean():.4f}, "
              f"sd {singles.std(ddof=0):.4f}, range {singles.min():.4f} to {singles.max():.4f}")
        print(f"  per-compound correlation between replicates: median {np.median(rhos):.3f}, "
              f"range {min(rhos):.3f} to {max(rhos):.3f}")

        out = paths.RESULTS / f"seed_replicates_{scheme}.csv"
        member_table.to_csv(out, index=False, float_format="%.6f")
        print(f"  wrote {out.name}")
        print(f"  {scheme} done in {(time.time() - started) / 60:.0f} min")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
