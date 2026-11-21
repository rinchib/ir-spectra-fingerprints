from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import fingerprints
import metrics
import paths
import pipeline
import splits
from fingerprints import (
    ImposedCollisionConfig,
    ImposedCollisionFeaturizer,
    MorganConfig,
    parse_molecules,
)

BASES = {
    "r2_unfolded_count": MorganConfig(radius=2, n_bits=None, counts=True, pooling="unfolded"),
    "r2_unfolded_bin": MorganConfig(radius=2, n_bits=None, counts=False, pooling="unfolded"),
}
DEFAULT_FRACTIONS = (0.5, 0.9, 0.95)


@dataclass(frozen=True)
class CompactCollisionConfig(ImposedCollisionConfig):
    @property
    def name(self) -> str:
        return super().name + "_compact"


class CompactCollisionFeaturizer(ImposedCollisionFeaturizer):
    def fit(self, molecules, collect_diagnostics: bool = True) -> "CompactCollisionFeaturizer":
        super().fit(molecules, collect_diagnostics=collect_diagnostics)
        used = sorted(set(self.vocabulary_.values()))
        renumber = {column: i for i, column in enumerate(used)}
        self.vocabulary_ = {identifier: renumber[column] for identifier, column in self.vocabulary_.items()}
        self.n_features_ = len(used)
        return self


_original_build = fingerprints.build_featurizer


def build_featurizer(config):
    if isinstance(config, CompactCollisionConfig):
        return CompactCollisionFeaturizer(config)
    return _original_build(config)


fingerprints.build_featurizer = build_featurizer
pipeline.build_featurizer = build_featurizer


def unfolded_params(base_name: str, scheme: str) -> dict:
    source = paths.RESULTS / f"run_{base_name}__{scheme}.json"
    if not source.exists():
        raise SystemExit(f"missing {source.name}; run compare_fingerprints.py for {base_name} first")
    return json.loads(source.read_text(encoding="utf-8"))["params"]


def check_compaction(base: MorganConfig, fraction: float, molecules) -> dict:
    kept = ImposedCollisionFeaturizer(ImposedCollisionConfig(base=base, fraction=fraction, strategy="random"))
    compact = CompactCollisionFeaturizer(CompactCollisionConfig(base=base, fraction=fraction, strategy="random"))
    X_kept = kept.fit_transform(molecules)
    X_compact = compact.fit_transform(molecules)
    in_use = sorted(set(kept.vocabulary_.values()))
    if not np.array_equal(X_kept[:, in_use], X_compact) or X_kept[:, np.setdiff1d(np.arange(X_kept.shape[1]), in_use)].any():
        raise SystemExit(f"compaction changed the fingerprint values at fraction {fraction}")
    return {"columns kept": int(X_kept.shape[1]), "columns compact": int(X_compact.shape[1])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="both", choices=splits.SCHEMES + ("both",))
    parser.add_argument("--base", default="r2_unfolded_count", choices=sorted(BASES))
    parser.add_argument("--fractions", nargs="*", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--folds", type=int, default=pipeline.DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=splits.SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rerun-kept", action="store_true",
                        help="train the runs with the empty columns kept again, under the tag 'rerun'")
    args = parser.parse_args()

    base = BASES[args.base]
    corpus = pd.read_csv(paths.CORPUS_TABLE)
    Y = metrics.sum_normalize(np.load(paths.CORPUS_SPECTRA)).astype(np.float32)
    smiles = corpus["canonical_smiles"].tolist()
    molecules = parse_molecules(smiles)
    print(f"Corpus: {len(molecules):,} compounds | devices: {', '.join(pipeline.available_devices(args.device))}")

    print("\nColumns on the whole corpus, kept and removed")
    widths = {}
    for fraction in args.fractions:
        widths[fraction] = check_compaction(base, fraction, molecules)
        print(f"  {fraction:.2f}: {widths[fraction]['columns kept']:,} -> {widths[fraction]['columns compact']:,}")

    schemes = list(splits.SCHEMES) if args.scheme == "both" else [args.scheme]
    rows = []
    for scheme in schemes:
        params = unfolded_params(args.base, scheme)
        print(f"\n{'=' * 70}\n{scheme}: architecture of the unfolded fingerprint {params}\n{'=' * 70}")
        started = time.time()
        for fraction in args.fractions:
            kept_config = ImposedCollisionConfig(base=base, fraction=fraction, strategy="random")
            compact_config = CompactCollisionConfig(base=base, fraction=fraction, strategy="random")
            common = dict(scheme=scheme, n_folds=args.folds, seed=args.seed, device=args.device, params=params)
            if args.rerun_kept:
                kept = pipeline.run(kept_config, molecules, Y, smiles, use_cache=True, tag="rerun", **common)
            else:
                kept = pipeline.run(kept_config, molecules, Y, smiles, use_cache=True, **common)
            compact = pipeline.run(compact_config, molecules, Y, smiles, use_cache=True, **common)
            rows.append({
                "scheme": scheme,
                "base": args.base,
                "share moved": fraction,
                "columns in input, kept": kept["fold_scores"][0]["n_features"],
                "columns in input, removed": compact["fold_scores"][0]["n_features"],
                "SIS kept": kept["fold_mean"],
                "SIS kept sd": kept["fold_std"],
                "SIS removed": compact["fold_mean"],
                "SIS removed sd": compact["fold_std"],
                "removed minus kept": compact["fold_mean"] - kept["fold_mean"],
                "largest fold difference": float(np.max(np.abs(
                    np.array([s["SIS_b10"] for s in compact["fold_scores"]])
                    - np.array([s["SIS_b10"] for s in kept["fold_scores"]])))),
                "kept from": "rerun" if args.rerun_kept else "cache",
            })
            r = rows[-1]
            print(f"  {fraction:.2f}: kept {r['SIS kept']:.4f}  removed {r['SIS removed']:.4f}  "
                  f"difference {r['removed minus kept']:+.4f}")
        print(f"  ({(time.time() - started) / 60:.0f} min)")

    table = pd.DataFrame(rows)
    suffix = "" if args.base == "r2_unfolded_count" else "_bin"
    out = paths.RESULTS / f"empty_columns{suffix}.csv"
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(table.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
