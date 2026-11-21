from __future__ import annotations

import argparse
import json
import time
import urllib.request

import numpy as np
import optuna
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.ML.Cluster import Butina
from torch import nn

import paths
import pipeline
import splits
from fingerprint_statistics import design_grid
from fingerprints import (
    BASELINE,
    MorganConfig,
    build_featurizer,
    folding_collision_stats,
    identical_row_groups,
)

RDLogger.DisableLog("rdApp.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)

EXTERNAL = paths.DATA / "external"
BUCKET = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"

DATASETS = {
    "esol": ("delaney-processed.csv", "smiles", "measured log solubility in mols per litre", "log mol/L"),
    "freesolv": ("SAMPL.csv", "smiles", "expt", "kcal/mol"),
    "lipo": ("Lipophilicity.csv", "smiles", "exp", "logD"),
}

TRAIN_CONFIGS = [
    BASELINE,
    MorganConfig(radius=2, n_bits=2048, counts=True),
    MorganConfig(radius=2, n_bits=None, counts=False, pooling="unfolded"),
    MorganConfig(radius=2, n_bits=None, counts=True, pooling="unfolded"),
    MorganConfig(radius=1, n_bits=None, counts=True, pooling="unfolded"),
    MorganConfig(radius=2, n_bits=16384, counts=True),
    MorganConfig(radius=2, n_bits=2048, counts=False, pooling="sort_slice"),
    MorganConfig(radius=2, n_bits=2048, counts=True, pooling="sort_slice"),
]

DEFAULT_TRIALS = 30
MAX_EPOCHS = 200
PATIENCE = 20


def load_dataset(name: str) -> tuple[list[str], np.ndarray, list[Chem.Mol]]:
    filename, smiles_column, target_column, _ = DATASETS[name]
    path = EXTERNAL / f"{name}.csv"
    if not path.exists():
        EXTERNAL.mkdir(parents=True, exist_ok=True)
        url = BUCKET + filename
        try:
            print(f"  downloading {url}")
            urllib.request.urlretrieve(url, path)
        except Exception as error:
            raise SystemExit(
                f"could not download {url} ({error}); fetch it by hand and save it as {path}"
            ) from error

    frame = pd.read_csv(path)
    canonical, targets = [], []
    for s, y in zip(frame[smiles_column], frame[target_column]):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None or mol.GetNumAtoms() == 0 or not np.isfinite(y):
            continue
        canonical.append(Chem.MolToSmiles(mol))
        targets.append(float(y))
    table = pd.DataFrame({"smiles": canonical, "y": targets}).groupby("smiles", as_index=False)["y"].mean()
    smiles = table["smiles"].tolist()
    molecules = [Chem.MolFromSmiles(s) for s in smiles]
    return smiles, table["y"].to_numpy(dtype=np.float64), molecules


def cluster_groups(smiles: list[str], name: str) -> np.ndarray:
    cache = paths.CACHE / f"butina_{name}_{len(smiles)}_0p65.npy"
    if cache.exists():
        return np.load(cache)
    fingerprints = splits._split_fingerprints(smiles)
    n = len(fingerprints)
    distances: list[float] = []
    for i in range(1, n):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        distances.extend(1.0 - s for s in similarities)
    clusters = Butina.ClusterData(distances, n, 1.0 - splits.CLUSTER_SIMILARITY, isDistData=True)
    labels = np.empty(n, dtype=int)
    for label, members in enumerate(clusters):
        for index in members:
            labels[index] = label
    paths.ensure_dirs()
    np.save(cache, labels)
    return labels


def make_groups(smiles: list[str], scheme: str, name: str) -> np.ndarray:
    if scheme == "random":
        return np.arange(len(smiles))
    return cluster_groups(smiles, name)


def scalar_ceiling(y: np.ndarray, groups: list[np.ndarray]) -> dict:
    if not groups:
        return {"n_ambiguous_groups": 0, "n_ambiguous": 0, "fraction_ambiguous": 0.0,
                "within_group_rmse": float("nan"), "irreducible_rmse": 0.0}
    squared = np.concatenate([(y[g] - y[g].mean()) ** 2 for g in groups])
    return {
        "n_ambiguous_groups": len(groups),
        "n_ambiguous": int(len(squared)),
        "fraction_ambiguous": float(len(squared) / len(y)),
        "within_group_rmse": float(np.sqrt(squared.mean())),
        "irreducible_rmse": float(np.sqrt(squared.sum() / len(y))),
    }


def run_statistics(datasets: list[str]) -> pd.DataFrame:
    rows = []
    for name in datasets:
        smiles, y, molecules = load_dataset(name)
        print(f"\n{name}: {len(molecules):,} molecules, target sd {y.std():.3f} {DATASETS[name][3]}")
        header = (f"  {'axis':<11s}{'configuration':<34s}{'dim':>7s}{'sub coll':>10s}"
                  f"{'ambig':>8s}{'floor':>8s}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for axis, config in design_grid():
            featurizer = build_featurizer(config)
            X = featurizer.fit_transform(molecules)
            collisions = folding_collision_stats(featurizer)
            ceiling = scalar_ceiling(y, identical_row_groups(X))
            rows.append({
                "dataset": name, "n_molecules": len(molecules), "target_sd": float(y.std()),
                "axis": axis, "config": config.name, "radius": config.radius,
                "n_bits": config.n_bits, "counts": config.counts, "invariants": config.invariants,
                "chirality": config.chirality, "pooling": config.pooling,
                "n_features": int(X.shape[1]), **collisions, **ceiling,
            })
            print(f"  {axis:<11s}{config.name:<34s}{X.shape[1]:>7,d}"
                  f"{100 * collisions['substructure_collision_rate']:>9.1f}%"
                  f"{100 * ceiling['fraction_ambiguous']:>7.2f}%"
                  f"{ceiling['within_group_rmse']:>8.3f}")
    table = pd.DataFrame(rows)
    out = paths.RESULTS / "properties_statistics.csv"
    table.to_csv(out, index=False)
    print(f"\nwrote {out.name}")

    print("\nPer dataset (baseline -> unfolded binary -> counts at 2,048)")
    for name in datasets:
        sub = table[table["dataset"] == name].set_index("config")
        def amb(cfg: str) -> str:
            return f"{100 * sub.loc[cfg, 'fraction_ambiguous']:5.2f}%"
        print(f"  {name:<9s} ambiguous: baseline {amb('r2_b2048_bin')}  unfolded {amb('r2_unfolded_bin')}"
              f"  counts {amb('r2_b2048_count')}  |  substructures sharing a bit at 2,048: "
              f"{100 * sub.loc['r2_b2048_bin', 'substructure_collision_rate']:.1f}%")
    return table


class ScalarMLP(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.BatchNorm1d(size), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _tensor(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32, device=device)


@torch.no_grad()
def predict_scalar(model: nn.Module, X: np.ndarray, device: str, mean: float, sd: float) -> np.ndarray:
    model.eval()
    out = model(_tensor(X, device)).float().cpu().numpy()
    return out * sd + mean


def fit_scalar(
    X_train: np.ndarray, y_train: np.ndarray, params: dict, seed: int, device: str,
    X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
    n_epochs: int | None = None, on_epoch=None,
) -> tuple[nn.Module, int, float, float, float]:
    mean, sd = float(y_train.mean()), float(y_train.std() or 1.0)
    torch.manual_seed(seed)
    model = ScalarMLP(X_train.shape[1], tuple([params["width"]] * params["n_layers"]),
                      params["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"],
                                 weight_decay=params["weight_decay"])
    X_t, y_t = _tensor(X_train, device), _tensor((y_train - mean) / sd, device)
    n_rows = len(X_t)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    validating = X_val is not None and y_val is not None
    total = n_epochs if n_epochs is not None else MAX_EPOCHS
    best_rmse, best_state, best_epoch, n_worse = np.inf, None, 0, 0

    for epoch in range(total):
        model.train()
        order = torch.randperm(n_rows, generator=generator).to(device)
        batch_size = params["batch_size"]
        for start in range(0, max(1, n_rows // batch_size) * batch_size, batch_size):
            rows = order[start:start + batch_size]
            if len(rows) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(X_t[rows]), y_t[rows])
            loss.backward()
            optimizer.step()
        if not validating:
            continue
        rmse = float(np.sqrt(np.mean((predict_scalar(model, X_val, device, mean, sd) - y_val) ** 2)))
        if on_epoch is not None:
            on_epoch(epoch, rmse)
        if rmse < best_rmse - 1e-6:
            best_rmse, best_epoch, n_worse = rmse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            n_worse += 1
            if n_worse >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, (best_epoch if validating else total), (best_rmse if validating else np.nan), mean, sd


def featurize(config, molecules, fit_rows: np.ndarray) -> np.ndarray:
    featurizer = build_featurizer(config)
    featurizer.fit([molecules[i] for i in fit_rows], collect_diagnostics=False)
    return featurizer.transform(molecules)


def tune_scalar(config, molecules, y, groups, n_trials: int, seed: int, device: str) -> dict:
    train_rows, val_rows, _ = splits.single_split(groups, seed=seed)
    X = featurize(config, molecules, train_rows)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "width": trial.suggest_int("width", 128, 2048, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.6),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        }

        def report(epoch: int, rmse: float) -> None:
            trial.report(rmse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        _, best_epoch, rmse, _, _ = fit_scalar(X[train_rows], y[train_rows], params, seed, device,
                                               X[val_rows], y[val_rows], on_epoch=report)
        trial.set_user_attr("best_epoch", int(best_epoch))
        return float(rmse)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best["best_epoch"] = int(study.best_trial.user_attrs.get("best_epoch", 0))
    best["val_rmse"] = float(study.best_value)
    return best


def evaluate_scalar(config, molecules, y, groups, params: dict, seed: int, device: str,
                    n_folds: int = pipeline.DEFAULT_FOLDS) -> dict:
    folds = splits.grouped_kfold(groups, k=n_folds, seed=seed)
    predictions = np.full(len(y), np.nan)
    fold_rmse, n_features = [], None
    for fold, (train_rows, test_rows) in enumerate(folds):
        fold_seed = seed + 1000 * fold
        inner_train, inner_val = splits.inner_split(train_rows, groups, fold, seed=seed)
        X_inner = featurize(config, molecules, inner_train)
        _, best_epoch, _, _, _ = fit_scalar(X_inner[inner_train], y[inner_train], params, fold_seed,
                                            device, X_inner[inner_val], y[inner_val])
        X = featurize(config, molecules, train_rows)
        model, _, _, mean, sd = fit_scalar(X[train_rows], y[train_rows], params, fold_seed, device,
                                           n_epochs=max(1, best_epoch + 1))
        predictions[test_rows] = predict_scalar(model, X[test_rows], device, mean, sd)
        fold_rmse.append(float(np.sqrt(np.mean((predictions[test_rows] - y[test_rows]) ** 2))))
        n_features = int(X.shape[1])
    residual = predictions - y
    return {
        "fold_rmse": fold_rmse,
        "rmse_mean": float(np.mean(fold_rmse)),
        "rmse_sd": float(np.std(fold_rmse, ddof=0)),
        "pooled_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "pooled_mae": float(np.abs(residual).mean()),
        "r2": float(1.0 - (residual ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "n_features": n_features,
        "predictions": predictions,
    }


def run_training(datasets: list[str], schemes: list[str], n_trials: int, seed: int,
                 device: str, use_cache: bool) -> pd.DataFrame:
    cache_dir = paths.RESULTS / "properties_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in datasets:
        smiles, y, molecules = load_dataset(name)
        unit = DATASETS[name][3]
        for scheme in schemes:
            groups = make_groups(smiles, scheme, name)
            print(f"\n{'=' * 70}\n{name} [{scheme}]: {len(y):,} molecules in {len(np.unique(groups)):,} "
                  f"groups, {len(TRAIN_CONFIGS)} configurations at {n_trials} trials each\n{'=' * 70}")
            rows, per_compound = [], {"smiles": smiles, "y": y}
            for config in TRAIN_CONFIGS:
                cache = cache_dir / f"{name}__{config.name}__{scheme}.json"
                if use_cache and cache.exists():
                    record = json.loads(cache.read_text(encoding="utf-8"))
                    print(f"  {config.name:<40s} loaded from cache  RMSE {record['rmse_mean']:.3f}")
                else:
                    started = time.time()
                    params = tune_scalar(config, molecules, y, groups, n_trials, seed, device)
                    outcome = evaluate_scalar(config, molecules, y, groups, params, seed, device)
                    predictions = outcome.pop("predictions")
                    record = {"dataset": name, "scheme": scheme, "config": config.name,
                              "params": params, **outcome, "predictions": predictions.tolist()}
                    cache.write_text(json.dumps(record), encoding="utf-8")
                    print(f"  {config.name:<40s} RMSE {record['rmse_mean']:.3f} +/- {record['rmse_sd']:.3f}"
                          f"  ({(time.time() - started) / 60:.1f} min)")
                per_compound[config.name] = np.asarray(record["predictions"])
                rows.append({"dataset": name, "scheme": scheme, "config": config.name,
                             "n_features": record["n_features"], "rmse_mean": record["rmse_mean"],
                             "rmse_sd": record["rmse_sd"], "pooled_rmse": record["pooled_rmse"],
                             "pooled_mae": record["pooled_mae"], "r2": record["r2"],
                             "fold_rmse": json.dumps(record["fold_rmse"]), "unit": unit})

            table = pd.DataFrame(rows)
            base = table.loc[table["config"] == BASELINE.name].iloc[0]
            base_folds = np.array(json.loads(base["fold_rmse"]))
            table["rmse_vs_baseline"] = table["rmse_mean"] - base["rmse_mean"]
            table["folds_better_than_baseline"] = [
                int((np.array(json.loads(f)) < base_folds).sum()) for f in table["fold_rmse"]
            ]
            out = paths.RESULTS / f"properties_{name}_{scheme}.csv"
            table.to_csv(out, index=False)
            pd.DataFrame(per_compound).to_csv(
                paths.RESULTS / f"properties_predictions_{name}_{scheme}.csv", index=False, float_format="%.5f"
            )
            print(f"\n  {'config':<40s}{'dim':>7s}{'RMSE':>8s}{'sd':>7s}{'vs base':>9s}{'folds<base':>11s}")
            print("  " + "-" * 82)
            for _, row in table.sort_values("rmse_mean").iterrows():
                print(f"  {row['config']:<40s}{row['n_features']:>7,d}{row['rmse_mean']:>8.3f}"
                      f"{row['rmse_sd']:>7.3f}{row['rmse_vs_baseline']:>+9.3f}"
                      f"{row['folds_better_than_baseline']:>8d}/5")
            print(f"  wrote {out.name}")
            summary.extend(table.drop(columns=["fold_rmse"]).to_dict("records"))

    summary_table = pd.DataFrame(summary)
    out = paths.RESULTS / "properties_summary.csv"
    summary_table.to_csv(out, index=False)
    print(f"\nwrote {out.name}")
    return summary_table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="both", choices=("statistics", "train", "both"))
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--scheme", default="both", choices=splits.SCHEMES + ("both",))
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=splits.SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    device = pipeline.available_devices(args.device)[0]
    schemes = list(splits.SCHEMES) if args.scheme == "both" else [args.scheme]
    print(f"device: {device}")

    if args.part in ("statistics", "both"):
        run_statistics(args.datasets)
    if args.part in ("train", "both"):
        run_training(args.datasets, schemes, args.trials, args.seed, device, not args.no_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
