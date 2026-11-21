from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import paths
import preprocess
import spectra

EXPECTED_FINAL_COUNT = 8085


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-parse the .xlsx sources")
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="warn instead of failing if the corpus is not 8,085 compounds",
    )
    args = parser.parse_args()

    paths.ensure_dirs()
    funnel: list[dict] = []

    def record(stage: str, criterion: str, remaining: int, detail: dict | None = None) -> None:
        funnel.append(
            {"stage": stage, "criterion": criterion, "remaining": remaining, "detail": detail or {}}
        )
        print(f"  {stage:22s} {criterion:52s} {remaining:>6,d}")

    print("Loading sources")
    meta, Y, wavenumbers = spectra.load_spectra(refresh=args.refresh)
    smiles_table = spectra.load_smiles()
    spectrum_metadata = spectra.load_spectrum_metadata()

    nist_ids = meta["NIST ID"].tolist()
    missing = set(nist_ids) - set(smiles_table.index)
    if missing:
        raise ValueError(f"{len(missing)} compounds have no SMILES, e.g. {sorted(missing)[:5]}")
    smiles = smiles_table.loc[nist_ids, "SMILES"].tolist()

    if len(smiles) != Y.shape[0]:
        raise ValueError(f"SMILES ({len(smiles)}) and spectra ({Y.shape[0]}) disagree in length")
    print(
        f"  {Y.shape[0]:,} compounds x {Y.shape[1]:,} points, "
        f"{wavenumbers[0]:.0f}-{wavenumbers[-1]:.0f} cm-1 "
        f"at {wavenumbers[1] - wavenumbers[0]:.0f} cm-1"
    )

    print("\nFunnel")
    record("loaded", "one resampled gas-phase spectrum per compound", Y.shape[0])

    Y, n_negative = preprocess.clip_negative_absorbance(Y)
    print(
        f"  clipped {n_negative:,} negative absorbances to zero "
        f"({100 * n_negative / Y.size:.4f}% of points)"
    )

    ordinate_units = preprocess.resolve_ordinate_units(nist_ids, meta["Source"].tolist(), spectrum_metadata)
    unresolved = [u for u in ordinate_units if u in ("UNKNOWN", "MIXED")]
    if unresolved:
        raise ValueError(f"could not resolve ordinate units for {len(unresolved)} compounds")

    result = preprocess.filter_selection_criteria(smiles)
    Y, meta, smiles, ordinate_units = preprocess.subset(result.keep, Y, meta, smiles, ordinate_units)
    record(result.name, result.criterion, result.n_kept, result.detail)

    canonical = preprocess.canonicalize(smiles)
    result = preprocess.filter_structure_duplicates(canonical)
    Y, meta, smiles, canonical, ordinate_units = preprocess.subset(
        result.keep, Y, meta, smiles, canonical, ordinate_units
    )
    record(result.name, result.criterion, result.n_kept, result.detail)

    result = preprocess.filter_spectral_duplicates(Y)
    for i, j, cosine in result.detail.get("examples", []):
        print(f"      e.g. kept {meta['NIST ID'][i]} / dropped {meta['NIST ID'][j]} (cosine {cosine:.5f})")
    Y, meta, smiles, canonical, ordinate_units = preprocess.subset(
        result.keep, Y, meta, smiles, canonical, ordinate_units
    )
    record(result.name, result.criterion, result.n_kept, result.detail)

    result = preprocess.filter_ordinate_format(Y, ordinate_units)
    dropped = meta.loc[~result.keep, ["Compound", "NIST ID", "Source"]]
    for _, row in dropped.iterrows():
        print(f"      dropped {row['NIST ID']} ({row['Source']}) {row['Compound']} - transmittance")
    Y, meta, smiles, canonical, ordinate_units = preprocess.subset(
        result.keep, Y, meta, smiles, canonical, ordinate_units
    )
    record(result.name, result.criterion, result.n_kept, result.detail)

    n_final = Y.shape[0]
    if n_final != EXPECTED_FINAL_COUNT:
        message = f"corpus is {n_final:,} compounds, expected {EXPECTED_FINAL_COUNT:,}"
        if not args.allow_count_mismatch:
            raise SystemExit(f"\nFAILED: {message}")
        print(f"\nWARNING: {message}")

    if len(set(canonical)) != n_final:
        raise SystemExit("FAILED: canonical SMILES are not unique after cleaning")
    if not np.isfinite(Y).all() or (Y < 0).any():
        raise SystemExit("FAILED: spectra contain negative or non-finite values")

    corpus = pd.DataFrame(
        {
            "nist_id": meta["NIST ID"].values,
            "name": meta["Compound"].values,
            "smiles": smiles,
            "canonical_smiles": canonical,
            "source": meta["Source"].values,
            "ordinate_units": ordinate_units,
            "original_resolution_cm": meta["Original Resolution"].values,
            "resampling": meta["Up or downsampled?"].values,
        }
    )

    np.save(paths.CORPUS_SPECTRA, Y.astype(np.float32))
    np.save(paths.CORPUS_WAVENUMBERS, wavenumbers)
    corpus.to_csv(paths.CORPUS_TABLE, index=False, encoding="utf-8")
    paths.CORPUS_FUNNEL.write_text(json.dumps(funnel, indent=2), encoding="utf-8")

    print(f"\nWrote {n_final:,} compounds to {paths.PROCESSED}")
    print(f"  {paths.CORPUS_SPECTRA.name}      {Y.shape} float32, max-normalized")
    print(f"  {paths.CORPUS_TABLE.name}             {corpus.shape[1]} columns")
    print(f"  {paths.CORPUS_WAVENUMBERS.name}        {wavenumbers.shape[0]} points")
    print(f"  {paths.CORPUS_FUNNEL.name}             {len(funnel)} stages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
