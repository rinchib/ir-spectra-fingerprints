from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Sequence

import numpy as np
import optuna
import torch
from rdkit import Chem

import metrics
import models
import paths
import splits
from fingerprints import MorganConfig, build_featurizer

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_TRIALS = 50
DEFAULT_FOLDS = 5


def available_devices(device: str | None = None) -> list[str]:
    if device is not None:
        return [device]
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def _featurize(
    config: MorganConfig,
    molecules: Sequence[Chem.Mol],
    fit_rows: np.ndarray,
) -> np.ndarray:
    featurizer = build_featurizer(config)
    featurizer.fit([molecules[i] for i in fit_rows], collect_diagnostics=False)
    return featurizer.transform(molecules)


def _training_config(params: dict, seed: int, device: str | None = None) -> models.TrainingConfig:
    kwargs = dict(
        hidden=tuple([params["width"]] * params["n_layers"]),
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        batch_size=params["batch_size"],
        seed=seed,
    )
    if device is not None:
        kwargs["device"] = device
    return models.TrainingConfig(**kwargs)


def tune(
    config: MorganConfig,
    molecules: Sequence[Chem.Mol],
    Y: np.ndarray,
    groups: np.ndarray,
    n_trials: int = DEFAULT_TRIALS,
    seed: int = splits.SEED,
    device: str | None = None,
    verbose: bool = True,
) -> dict:
    devices = available_devices(device)
    train_rows, val_rows, _ = splits.single_split(groups, seed=seed)
    X = _featurize(config, molecules, train_rows)
    X_train, Y_train = X[train_rows], Y[train_rows]
    X_val, Y_val = X[val_rows], Y[val_rows]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "width": trial.suggest_int("width", 128, 2048, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.6),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        }
        trial_device = devices[trial.number % len(devices)]
        result = models.train(
            X_train,
            Y_train,
            _training_config(params, seed, trial_device),
            X_val=X_val,
            Y_val=Y_val,
            on_epoch=lambda epoch, score: _report(trial, epoch, score),
        )
        trial.set_user_attr("best_epoch", int(result.best_epoch))
        trial.set_user_attr("device", trial_device)
        return float(result.best_score)

    def _report(trial: optuna.Trial, epoch: int, score: float) -> None:
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    started = time.time()

    def progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        done = len([t for t in study.trials if t.state.is_finished()])
        state = trial.state.name.lower()
        detail = f"SID {-trial.value:.4f}" if trial.value is not None else state
        print(
            f"    trial {done}/{n_trials}  {detail}"
            f"  (best {-study.best_value:.4f})"
            f"  {time.time() - started:.0f}s",
            flush=True,
        )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=len(devices),
        show_progress_bar=False,
        callbacks=[progress] if verbose else None,
    )

    best = dict(study.best_params)
    best["best_epoch"] = int(study.best_trial.user_attrs.get("best_epoch", 0))
    if verbose:
        print(
            f"  tuned in {time.time() - started:.0f}s over {n_trials} trials "
            f"(best validation SID {-study.best_value:.4f}, epoch {best['best_epoch']})"
        )
    return best


def _subsample_groups(
    train_rows: np.ndarray, groups: np.ndarray, fraction: float, seed: int
) -> np.ndarray:
    fold_groups = groups[train_rows]
    unique = np.unique(fold_groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    target = fraction * len(train_rows)
    chosen: set[int] = set()
    count = 0
    for group in unique:
        if count >= target:
            break
        chosen.add(int(group))
        count += int((fold_groups == group).sum())

    keep = np.isin(fold_groups, list(chosen))
    return np.sort(train_rows[keep])


def evaluate(
    config: MorganConfig,
    molecules: Sequence[Chem.Mol],
    Y: np.ndarray,
    groups: np.ndarray,
    params: dict,
    n_folds: int = DEFAULT_FOLDS,
    seed: int = splits.SEED,
    device: str | None = None,
    verbose: bool = True,
    training_seed: int | None = None,
    train_fraction: float = 1.0,
) -> dict:
    devices = available_devices(device)
    folds = splits.grouped_kfold(groups, k=n_folds, seed=seed)
    predictions = np.full_like(Y, np.nan, dtype=np.float32)
    init_seed = seed if training_seed is None else training_seed

    def run_fold(item: tuple[int, tuple[np.ndarray, np.ndarray]]) -> dict:
        fold, (train_rows, test_rows) = item
        fold_device = devices[fold % len(devices)]
        fold_seed = init_seed + 1000 * fold
        if train_fraction < 1.0:
            train_rows = _subsample_groups(train_rows, groups, train_fraction, seed + 7919 + fold)
        inner_train, inner_val = splits.inner_split(train_rows, groups, fold, seed=seed)

        X_inner = _featurize(config, molecules, inner_train)
        early = models.train(
            X_inner[inner_train],
            Y[inner_train],
            _training_config(params, fold_seed, fold_device),
            X_val=X_inner[inner_val],
            Y_val=Y[inner_val],
        )

        X = _featurize(config, molecules, train_rows)
        training_config = _training_config(params, fold_seed, fold_device)
        final = models.train(
            X[train_rows],
            Y[train_rows],
            training_config,
            n_epochs=max(1, early.best_epoch + 1),
        )
        predictions[test_rows] = models.predict(final.model, X[test_rows], training_config.device)

        score = metrics.all_metrics(Y[test_rows], predictions[test_rows])
        score["fold"] = fold
        score["n_test"] = len(test_rows)
        score["n_train"] = len(train_rows)
        score["n_features"] = int(X.shape[1])
        score["epochs"] = int(early.best_epoch + 1)
        score["device"] = fold_device
        if verbose:
            print(
                f"  fold {fold}: SIS_b10 {score['SIS_b10']:.4f}  "
                f"({len(test_rows):,} held out, {score['epochs']} epochs, {fold_device})"
            )
        return score

    if len(devices) > 1:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            fold_scores = list(pool.map(run_fold, enumerate(folds)))
    else:
        fold_scores = [run_fold(item) for item in enumerate(folds)]

    fold_scores.sort(key=lambda s: s["fold"])

    if np.isnan(predictions).any():
        raise RuntimeError("some compounds were never held out; folds do not cover the corpus")

    headline = np.array([s["SIS_b10"] for s in fold_scores])
    pooled = metrics.all_metrics(Y, predictions)

    return {
        "fold_scores": fold_scores,
        "fold_mean": float(headline.mean()),
        "fold_std": float(headline.std(ddof=0)),
        "pooled": pooled,
        "predictions": predictions,
    }


def run(
    config: MorganConfig,
    molecules: Sequence[Chem.Mol],
    Y: np.ndarray,
    smiles: list[str],
    scheme: str = "cluster",
    n_trials: int = DEFAULT_TRIALS,
    n_folds: int = DEFAULT_FOLDS,
    seed: int = splits.SEED,
    device: str | None = None,
    use_cache: bool = True,
    verbose: bool = True,
    params: dict | None = None,
    tag: str = "",
    training_seed: int | None = None,
    train_fraction: float = 1.0,
) -> dict:
    paths.ensure_dirs()
    stem = f"{config.name}__{scheme}" + (f"__{tag}" if tag else "")
    result_path = paths.RESULTS / f"run_{stem}.json"
    prediction_path = paths.RESULTS / f"pred_{stem}.npy"

    if use_cache and result_path.exists() and prediction_path.exists():
        if verbose:
            print(f"{config.name} [{scheme}]: loaded from cache")
        record = json.loads(result_path.read_text(encoding="utf-8"))
        record["predictions"] = np.load(prediction_path)
        return record

    if verbose:
        print(f"{config.name} [{scheme}]" + (f" ({tag})" if tag else ""))
    groups = splits.make_groups(smiles, scheme)

    tuned_here = params is None
    if tuned_here:
        params = tune(config, molecules, Y, groups, n_trials=n_trials, seed=seed,
                      device=device, verbose=verbose)
    outcome = evaluate(config, molecules, Y, groups, params, n_folds=n_folds, seed=seed,
                       device=device, verbose=verbose, training_seed=training_seed,
                       train_fraction=train_fraction)

    predictions = outcome.pop("predictions")
    record = {
        "config": asdict(config),
        "config_name": config.name,
        "scheme": scheme,
        "params": params,
        "tuned": tuned_here,
        "tag": tag,
        "n_trials": n_trials if tuned_here else 0,
        "n_folds": n_folds,
        "seed": seed,
        "training_seed": training_seed,
        "train_fraction": train_fraction,
        **outcome,
    }
    result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    np.save(prediction_path, predictions)
    record["predictions"] = predictions

    if verbose:
        print(f"  SIS_b10 {record['fold_mean']:.4f} +/- {record['fold_std']:.4f}\n")
    return record
