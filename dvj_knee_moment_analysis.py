#!/usr/bin/env python3
"""Minimal, manuscript-aligned analysis framework for DVJ knee moments.

This file contains no participant data, trained weights, figures, or numerical
results. It expects a prepared point-level feature table rather than raw EMG,
kinematic, or kinetic files. Before creating that table, the analyst must:

* synchronize EMG, kinematics, and knee moments;
* map left and right limbs to one anatomical sign convention;
* apply the prespecified EMG and knee-moment normalization; and
* time-normalize every limb trial to the configured number of points.

The implementation covers:
  Input 1: six-muscle sEMG RMS
  Input 2a: Input 1 + sagittal knee angle and angular velocity
  Input 2b: Input 1 + sagittal hip/knee/ankle angles and velocities
  Input 3: five time-domain features per muscle, PCA, then SVR
  Input 4: supervised 1D-FCN features, then SVR

All learned transformations are fitted without access to the outer test set.
PCA and standardization are refitted inside every inner grouped CV split. For
Input 4, the EMG scaler and supervised FCN are also refitted in every inner
split before selecting the downstream SVR hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from itertools import product
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


MUSCLES = ("rf", "bf", "ta", "gas", "sol", "gmax")
TD_FEATURES = ("rms", "mav", "wl", "zc", "ssc")
META_COLUMNS = ("subject_id", "trial_id", "condition", "side", "time_index")
TARGET_COLUMN = "knee_moment_x"

RMS_COLUMNS = tuple(f"rms_{muscle}" for muscle in MUSCLES)
KNEE_COLUMNS = ("knee_angle_x", "knee_velocity_x")
FULL_KINEMATIC_COLUMNS = (
    "hip_angle_x",
    "knee_angle_x",
    "ankle_angle_x",
    "hip_velocity_x",
    "knee_velocity_x",
    "ankle_velocity_x",
)
TD_COLUMNS = tuple(
    f"{feature}_{muscle}" for muscle in MUSCLES for feature in TD_FEATURES
)


@dataclass(frozen=True)
class AnalysisConfig:
    seed: int = 2026
    n_time_points: int = 101
    pca_components: int = 7
    svr_epsilon: float = 0.01
    svr_c: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    svr_gamma: tuple[object, ...] = ("scale", 0.01, 0.1, 1.0)
    inner_folds: int = 3
    within_folds: int = 5
    within_group_level: str = "subject"
    inner_group_level: str = "subject"
    cnn_epochs: int = 50
    cnn_batch_size: int = 16
    cnn_learning_rate: float = 1e-3


@dataclass(frozen=True)
class DataSplit:
    scenario: str
    held_out: str
    train_index: np.ndarray
    test_index: np.ndarray
    outer_group_level: str


def set_seed(seed: int) -> None:
    """Set deterministic seeds for the packages used by the framework."""

    random.seed(seed)
    np.random.seed(seed)


def validate_config(config: AnalysisConfig) -> None:
    """Fail early for invalid settings rather than during model fitting."""

    if config.n_time_points < 2:
        raise ValueError("n_time_points must be at least 2")
    if config.pca_components < 1:
        raise ValueError("pca_components must be at least 1")
    if config.inner_folds < 2 or config.within_folds < 2:
        raise ValueError("inner_folds and within_folds must both be at least 2")
    if config.within_group_level not in {"subject", "trial"}:
        raise ValueError("within_group_level must be 'subject' or 'trial'")
    if config.inner_group_level not in {"subject", "trial"}:
        raise ValueError("inner_group_level must be 'subject' or 'trial'")
    if config.cnn_epochs < 1 or config.cnn_batch_size < 1:
        raise ValueError("cnn_epochs and cnn_batch_size must both be positive")
    if config.cnn_learning_rate <= 0.0:
        raise ValueError("cnn_learning_rate must be positive")
    if config.svr_epsilon < 0.0:
        raise ValueError("svr_epsilon must be non-negative")
    if not config.svr_c or any(value <= 0.0 for value in config.svr_c):
        raise ValueError("all SVR C values must be positive")


def preflight_dependencies(path: Path, selected_inputs: Sequence[str]) -> None:
    """Check optional dependencies before any long-running analysis starts."""

    if not path.is_file():
        raise FileNotFoundError(f"feature table not found: {path}")
    if "input4" in selected_inputs and find_spec("torch") is None:
        raise RuntimeError(
            "Input 4 was selected, but PyTorch is not installed. Install "
            "PyTorch or omit input4 with --inputs."
        )
    if path.suffix.lower() in {".parquet", ".pq"}:
        if find_spec("pyarrow") is None and find_spec("fastparquet") is None:
            raise RuntimeError(
                "Parquet input requires pyarrow or fastparquet; alternatively "
                "supply the feature table as CSV."
            )


def linear_envelope(
    signal: np.ndarray,
    fs_hz: float = 1000.0,
    envelope_hz: float = 6.0,
) -> np.ndarray:
    """Mean-centre, rectify, and zero-phase low-pass filter one EMG channel.

    Acquisition band-pass filtering (20--450 Hz in the study) is assumed to
    have already been applied by the acquisition system. Apply an additional
    band-pass before this function only when working from unfiltered raw data.
    """

    if fs_hz <= 0.0:
        raise ValueError("fs_hz must be positive")
    if not 0.0 < envelope_hz < fs_hz / 2.0:
        raise ValueError("envelope_hz must lie between 0 and the Nyquist frequency")

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 16 or not np.isfinite(values).all():
        raise ValueError("signal must be a finite one-dimensional array")
    centred = values - values.mean()
    rectified = np.abs(centred)
    sos = butter(4, envelope_hz, btype="lowpass", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, rectified)


def _window_bounds(n_samples: int, window: int, step: int) -> Iterator[slice]:
    if window <= 1 or step <= 0 or n_samples < window:
        raise ValueError("invalid window, step, or signal length")
    for start in range(0, n_samples - window + 1, step):
        yield slice(start, start + window)


def extract_window_features(
    filtered_emg: np.ndarray,
    fs_hz: float = 1000.0,
    window_ms: int = 100,
    overlap: float = 0.5,
    zc_threshold: float = 0.0,
    ssc_threshold: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract conventional five-feature EMG windows from one channel.

    `filtered_emg` should be mean-centred, band-pass-filtered EMG rather than a
    non-negative envelope, because RMS, zero crossings, and slope sign changes
    are conventionally calculated from the bipolar signal. Do not use a linear
    envelope for ZC or SSC.
    """

    if fs_hz <= 0.0:
        raise ValueError("fs_hz must be positive")
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    if zc_threshold < 0.0 or ssc_threshold < 0.0:
        raise ValueError("ZC and SSC thresholds must be non-negative")

    x = np.asarray(filtered_emg, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("filtered_emg must be a finite one-dimensional array")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    window = int(round(window_ms * fs_hz / 1000.0))
    step = max(1, int(round(window * (1.0 - overlap))))
    rows: list[dict[str, float]] = []
    centres: list[float] = []

    for bounds in _window_bounds(x.size, window, step):
        segment = x[bounds]
        difference = np.diff(segment)
        sign_change = segment[:-1] * segment[1:] < 0
        zc = np.sum(sign_change & (np.abs(difference) >= zc_threshold))

        if difference.size >= 2:
            slope_change = difference[:-1] * difference[1:] < 0
            slope_size = np.maximum(
                np.abs(difference[:-1]), np.abs(difference[1:])
            )
            ssc = np.sum(slope_change & (slope_size >= ssc_threshold))
        else:
            ssc = 0

        rows.append(
            {
                "rms": float(np.sqrt(np.mean(segment**2))),
                "mav": float(np.mean(np.abs(segment))),
                "wl": float(np.sum(np.abs(difference))),
                "zc": float(zc),
                "ssc": float(ssc),
            }
        )
        centres.append((bounds.start + bounds.stop - 1) / (2.0 * fs_hz))

    return pd.DataFrame(rows, columns=TD_FEATURES), np.asarray(centres)


def time_normalise(values: np.ndarray, n_points: int = 101) -> np.ndarray:
    """Linearly resample a trajectory or feature matrix to `n_points`."""

    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
        squeeze = True
    elif array.ndim == 2:
        squeeze = False
    else:
        raise ValueError("values must be one- or two-dimensional")
    if array.shape[0] < 2 or not np.isfinite(array).all():
        raise ValueError("values must contain at least two finite time points")

    old_time = np.linspace(0.0, 1.0, array.shape[0])
    new_time = np.linspace(0.0, 1.0, n_points)
    resampled = np.column_stack(
        [np.interp(new_time, old_time, array[:, column]) for column in range(array.shape[1])]
    )
    return resampled[:, 0] if squeeze else resampled


def feature_columns(input_name: str) -> tuple[str, ...]:
    mapping = {
        "input1": RMS_COLUMNS,
        "input2a": RMS_COLUMNS + KNEE_COLUMNS,
        "input2b": RMS_COLUMNS + FULL_KINEMATIC_COLUMNS,
        "input3": TD_COLUMNS,
        "input4": RMS_COLUMNS,
    }
    try:
        return mapping[input_name]
    except KeyError as exc:
        raise ValueError(f"unknown input configuration: {input_name}") from exc


def load_feature_table(
    path: Path,
    selected_inputs: Sequence[str],
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Load and strictly validate the standardized point-level analysis table."""

    if not path.is_file():
        raise FileNotFoundError(f"feature table not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise ValueError("--data must be a CSV or Parquet file")

    if frame.empty:
        raise ValueError("feature table is empty")

    required = set(META_COLUMNS) | {TARGET_COLUMN}
    for name in selected_inputs:
        required.update(feature_columns(name))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    frame = frame.loc[
        :,
        list(
            dict.fromkeys(
                list(META_COLUMNS) + sorted(required - set(META_COLUMNS))
            )
        ),
    ].copy()

    for column in ("subject_id", "trial_id", "condition", "side"):
        if frame[column].isna().any():
            raise ValueError(f"{column} contains missing values")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"{column} contains empty identifiers")

    frame["time_index"] = pd.to_numeric(frame["time_index"], errors="raise")
    if not np.isfinite(frame["time_index"].to_numpy(dtype=float)).all():
        raise ValueError("time_index contains NaN or infinite values")

    numeric_columns = sorted(required - set(META_COLUMNS))
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("feature table contains NaN or infinite values")

    sequence_columns = ["subject_id", "condition", "trial_id", "side"]
    duplicate_columns = sequence_columns + ["time_index"]
    duplicate_mask = frame.duplicated(duplicate_columns, keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, duplicate_columns].head(5)
        raise ValueError(
            "duplicate sequence/time rows detected; examples: "
            f"{examples.to_dict(orient='records')}"
        )

    frame = frame.sort_values(duplicate_columns, kind="stable").reset_index(drop=True)
    frame["_sequence"] = list(
        frame.loc[:, sequence_columns].itertuples(index=False, name=None)
    )
    frame["_trial_group"] = list(
        frame.loc[:, ["subject_id", "condition", "trial_id"]].itertuples(
            index=False, name=None
        )
    )
    frame["_subject_group"] = frame["subject_id"]

    sequence_lengths = frame.groupby("_sequence", sort=False).size()
    invalid_lengths = sequence_lengths[sequence_lengths.ne(config.n_time_points)]
    if not invalid_lengths.empty:
        examples = {str(key): int(value) for key, value in invalid_lengths.head(5).items()}
        raise ValueError(
            f"every limb trial must contain exactly {config.n_time_points} time "
            f"points; invalid examples: {examples}"
        )

    time_grids = [
        group["time_index"].to_numpy(dtype=float)
        for _, group in frame.groupby("_sequence", sort=False)
    ]
    reference_grid = time_grids[0]
    if any(
        not np.allclose(grid, reference_grid, rtol=1e-7, atol=1e-10)
        for grid in time_grids[1:]
    ):
        raise ValueError("all limb trials must use the same ordered time_index grid")

    return frame


def grouping_column(level: str) -> str:
    """Return the internal column used for subject- or trial-level grouping."""

    mapping = {"subject": "_subject_group", "trial": "_trial_group"}
    try:
        return mapping[level]
    except KeyError as exc:
        raise ValueError(f"unknown grouping level: {level}") from exc


def grouping_values(frame: pd.DataFrame, level: str) -> np.ndarray:
    """Return stable integer codes suitable for grouped cross-validation."""

    codes, _ = pd.factorize(frame[grouping_column(level)], sort=True)
    return codes


def make_splits(
    frame: pd.DataFrame,
    scenarios: Iterable[str],
    config: AnalysisConfig,
) -> Iterator[DataSplit]:
    """Yield within-condition, LOSO, and LOCO outer validation splits."""

    scenarios = tuple(scenarios)
    if "within" in scenarios:
        for condition, subset in frame.groupby("condition", sort=True):
            groups = grouping_values(subset, config.within_group_level)
            n_groups = np.unique(groups).size
            if n_groups < 2:
                raise ValueError(
                    f"condition {condition!r} has fewer than two independent "
                    f"{config.within_group_level} groups"
                )
            splitter = GroupKFold(n_splits=min(config.within_folds, n_groups))
            for fold, (train_local, test_local) in enumerate(
                splitter.split(subset, groups=groups), start=1
            ):
                yield DataSplit(
                    "within",
                    f"condition={condition};fold={fold}",
                    subset.index.to_numpy()[train_local],
                    subset.index.to_numpy()[test_local],
                    config.within_group_level,
                )

    if "loso" in scenarios:
        for condition, subset in frame.groupby("condition", sort=True):
            subjects = sorted(subset["subject_id"].unique())
            if len(subjects) < 2:
                raise ValueError(
                    f"condition {condition!r} requires at least two subjects for LOSO"
                )
            for subject in subjects:
                test_mask = subset["subject_id"].eq(subject)
                yield DataSplit(
                    "loso",
                    f"condition={condition};subject={subject}",
                    subset.index[~test_mask].to_numpy(),
                    subset.index[test_mask].to_numpy(),
                    "subject",
                )

    if "loco" in scenarios:
        conditions = sorted(frame["condition"].unique())
        if len(conditions) < 2:
            raise ValueError("LOCO requires at least two conditions")
        for condition in conditions:
            test_mask = frame["condition"].eq(condition)
            yield DataSplit(
                "loco",
                f"condition={condition}",
                frame.index[~test_mask].to_numpy(),
                frame.index[test_mask].to_numpy(),
                "condition",
            )


def _svr_pipeline(input_name: str, config: AnalysisConfig) -> Pipeline:
    steps: list[tuple[str, object]] = [("scale", StandardScaler())]
    if input_name == "input3":
        steps.append(
            (
                "pca",
                PCA(n_components=config.pca_components, random_state=config.seed),
            )
        )
    steps.append(
        (
            "svr",
            SVR(kernel="rbf", epsilon=config.svr_epsilon),
        )
    )
    return Pipeline(steps)


def fit_svr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    input_name: str,
    config: AnalysisConfig,
    n_jobs: int,
) -> tuple[Pipeline, dict[str, object]]:
    """Tune C and gamma with training-only grouped cross-validation."""

    estimator = _svr_pipeline(input_name, config)
    n_groups = np.unique(groups).size
    if n_groups < 2:
        raise ValueError(
            "inner grouped cross-validation requires at least two independent groups"
        )

    if input_name == "input3" and config.pca_components > x_train.shape[1]:
        raise ValueError(
            "pca_components cannot exceed the number of Input 3 features"
        )

    inner_cv = GroupKFold(n_splits=min(config.inner_folds, n_groups))
    search = GridSearchCV(
        estimator,
        param_grid={"svr__C": config.svr_c, "svr__gamma": config.svr_gamma},
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(x_train, y_train, groups=groups)
    best = {
        "C": float(search.best_params_["svr__C"]),
        "gamma": search.best_params_["svr__gamma"],
    }
    return search.best_estimator_, best


def _stack_sequences(
    frame: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[np.ndarray]]:
    positions: list[np.ndarray] = []
    lengths: set[int] = set()
    for _, group in frame.groupby("_sequence", sort=False):
        index = group.index.to_numpy()
        positions.append(index)
        lengths.add(index.size)
    if len(lengths) != 1:
        raise ValueError("Input 4 requires an equal number of time points per limb trial")

    x_sequences = np.stack([x[index].T for index in positions])
    y_sequences = None if y is None else np.stack([y[index] for index in positions])
    return x_sequences, y_sequences, positions


def learn_fcn_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: AnalysisConfig,
    device_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the supervised FCN on outer-training trials and return 32-D features."""

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("Input 4 requires PyTorch; install it before use") from exc

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested, but CUDA is unavailable")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    class FCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(len(RMS_COLUMNS), 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Conv1d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.30),
            )
            self.head = nn.Conv1d(32, 1, kernel_size=1)

        def forward(self, values: object) -> tuple[object, object]:
            features = self.encoder(values)
            prediction = self.head(features).squeeze(1)
            return prediction, features

    rms_columns = list(RMS_COLUMNS)
    scaler = StandardScaler().fit(train.loc[:, rms_columns])
    train_x = scaler.transform(train.loc[:, rms_columns])
    test_x = scaler.transform(test.loc[:, rms_columns])
    train_y = train[TARGET_COLUMN].to_numpy(dtype=np.float32)

    # Reindex locally so sequence positions address the corresponding arrays.
    train_local = train.reset_index(drop=True)
    test_local = test.reset_index(drop=True)
    train_sequences, target_sequences, _ = _stack_sequences(
        train_local, train_x, train_y
    )
    assert target_sequences is not None

    dataset = TensorDataset(
        torch.as_tensor(train_sequences, dtype=torch.float32),
        torch.as_tensor(target_sequences, dtype=torch.float32),
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=min(config.cnn_batch_size, len(dataset)),
        shuffle=True,
        generator=loader_generator,
    )
    model = FCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.cnn_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_function = nn.MSELoss()

    model.train()
    for _ in range(config.cnn_epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(batch_x)
            loss = loss_function(prediction, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    def encode(local_frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
        encoded = np.empty((len(local_frame), 32), dtype=np.float32)
        _, _, sequence_positions = _stack_sequences(local_frame, values)
        model.eval()
        with torch.no_grad():
            for position in sequence_positions:
                sequence = torch.as_tensor(
                    values[position].T[None, ...], dtype=torch.float32, device=device
                )
                _, features = model(sequence)
                encoded[position] = features.squeeze(0).T.cpu().numpy()
        return encoded

    return encode(train_local, train_x), encode(test_local, test_x)


def fit_predict_fcn_svr(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: AnalysisConfig,
    device: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Tune and fit the FCN-SVR stack without inner-fold label leakage.

    A fresh EMG scaler and supervised FCN are fitted on each inner-training
    partition. The corresponding inner-validation partition is only encoded
    and scored. After selecting C and gamma, a final scaler/FCN/SVR stack is
    fitted on all outer-training data and applied to the untouched outer test.
    """

    groups = grouping_values(train, config.inner_group_level)
    n_groups = np.unique(groups).size
    if n_groups < 2:
        raise ValueError(
            "Input 4 inner validation requires at least two independent groups"
        )

    inner_cv = GroupKFold(n_splits=min(config.inner_folds, n_groups))
    encoded_folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for inner_train_index, inner_valid_index in inner_cv.split(train, groups=groups):
        inner_train = train.iloc[inner_train_index].copy()
        inner_valid = train.iloc[inner_valid_index].copy()
        encoded_train, encoded_valid = learn_fcn_features(
            inner_train,
            inner_valid,
            config,
            device,
        )
        encoded_folds.append(
            (
                encoded_train,
                inner_train[TARGET_COLUMN].to_numpy(dtype=float),
                encoded_valid,
                inner_valid[TARGET_COLUMN].to_numpy(dtype=float),
            )
        )

    candidate_scores: list[tuple[float, float, object]] = []
    for c_value, gamma_value in product(config.svr_c, config.svr_gamma):
        squared_error_sum = 0.0
        observation_count = 0
        for encoded_train, y_inner_train, encoded_valid, y_inner_valid in encoded_folds:
            estimator = _svr_pipeline("input4", config)
            estimator.set_params(svr__C=c_value, svr__gamma=gamma_value)
            estimator.fit(encoded_train, y_inner_train)
            inner_prediction = estimator.predict(encoded_valid)
            squared_error_sum += float(
                np.sum(np.square(y_inner_valid - inner_prediction))
            )
            observation_count += int(y_inner_valid.size)
        pooled_rmse = float(np.sqrt(squared_error_sum / observation_count))
        candidate_scores.append((pooled_rmse, float(c_value), gamma_value))

    _, best_c, best_gamma = min(candidate_scores, key=lambda item: item[0])
    encoded_train, encoded_test = learn_fcn_features(train, test, config, device)
    final_estimator = _svr_pipeline("input4", config)
    final_estimator.set_params(svr__C=best_c, svr__gamma=best_gamma)
    final_estimator.fit(
        encoded_train,
        train[TARGET_COLUMN].to_numpy(dtype=float),
    )
    prediction = final_estimator.predict(encoded_test)
    return prediction, {"C": best_c, "gamma": best_gamma}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return prediction metrics, with NRMSE explicitly based on the test range."""

    if y_true.size < 2:
        raise ValueError("regression metrics require at least two observations")

    dynamic_range = float(np.max(y_true) - np.min(y_true))
    if dynamic_range <= 0.0:
        raise ValueError("NRMSE is undefined because the test target range is zero")
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "target_range": dynamic_range,
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "nrmse_test_range": rmse / dynamic_range,
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_split(
    frame: pd.DataFrame,
    split: DataSplit,
    input_name: str,
    config: AnalysisConfig,
    n_jobs: int,
    device: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    train = frame.loc[split.train_index].copy()
    test = frame.loc[split.test_index].copy()
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=float)

    if input_name == "input4":
        prediction, best = fit_predict_fcn_svr(train, test, config, device)
    else:
        columns = feature_columns(input_name)
        x_train = train.loc[:, list(columns)].to_numpy(dtype=float)
        x_test = test.loc[:, list(columns)].to_numpy(dtype=float)
        estimator, best = fit_svr(
            x_train,
            y_train,
            grouping_values(train, config.inner_group_level),
            input_name,
            config,
            n_jobs,
        )
        prediction = estimator.predict(x_test)

    metrics = regression_metrics(y_test, prediction)
    summary: dict[str, object] = {
        "scenario": split.scenario,
        "held_out": split.held_out,
        "input": input_name,
        "outer_group_level": split.outer_group_level,
        "inner_group_level": config.inner_group_level,
        "n_train_subjects": int(train["subject_id"].nunique()),
        "n_test_subjects": int(test["subject_id"].nunique()),
        "n_train_physical_trials": int(train["_trial_group"].nunique()),
        "n_test_physical_trials": int(test["_trial_group"].nunique()),
        "n_train_limb_trials": int(train["_sequence"].nunique()),
        "n_test_limb_trials": int(test["_sequence"].nunique()),
        "n_test_points": int(len(test)),
        "target_range_test": metrics["target_range"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "nrmse_test_range": metrics["nrmse_test_range"],
        "r2": metrics["r2"],
        "best_c": best["C"],
        "best_gamma": best["gamma"],
    }
    predictions = test.loc[:, list(META_COLUMNS)].copy()
    predictions.insert(len(predictions.columns), "scenario", split.scenario)
    predictions.insert(len(predictions.columns), "held_out", split.held_out)
    predictions.insert(len(predictions.columns), "input", input_name)
    predictions.insert(len(predictions.columns), "observed", y_test)
    predictions.insert(len(predictions.columns), "predicted", prediction)
    predictions.insert(len(predictions.columns), "residual", y_test - prediction)
    return summary, predictions


def aggregate_prediction_metrics(
    predictions: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Calculate pooled out-of-fold metrics without averaging fold-level R-squared."""

    rows: list[dict[str, object]] = []
    grouper: str | list[str]
    grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for keys, subset in predictions.groupby(grouper, sort=True, dropna=False):
        key_values = (keys,) if len(group_columns) == 1 else tuple(keys)
        metrics = regression_metrics(
            subset["observed"].to_numpy(dtype=float),
            subset["predicted"].to_numpy(dtype=float),
        )
        row: dict[str, object] = dict(zip(group_columns, key_values))
        row.update(
            {
                "n_subjects": int(subset["subject_id"].nunique()),
                "n_physical_trials": int(
                    subset.loc[:, ["subject_id", "condition", "trial_id"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "n_limb_trials": int(
                    subset.loc[
                        :, ["subject_id", "condition", "trial_id", "side"]
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
                "n_points": int(len(subset)),
                "target_range": metrics["target_range"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "nrmse_pooled_test_range": metrics["nrmse_test_range"],
                "r2_pooled": metrics["r2"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="CSV/Parquet feature table")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs"))
    parser.add_argument(
        "--inputs",
        nargs="+",
        choices=("input1", "input2a", "input2b", "input3", "input4"),
        default=("input1", "input2a", "input2b", "input3", "input4"),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("within", "loso", "loco"),
        default=("within", "loso", "loco"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="PyTorch device for Input 4")
    parser.add_argument("--n-time-points", type=int, default=101)
    parser.add_argument("--pca-components", type=int, default=7)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--within-folds", type=int, default=5)
    parser.add_argument(
        "--within-group-level",
        choices=("subject", "trial"),
        default="subject",
        help=(
            "independent unit for within-condition CV; subject is the "
            "conservative default"
        ),
    )
    parser.add_argument(
        "--inner-group-level",
        choices=("subject", "trial"),
        default="subject",
        help="independent unit used for inner SVR hyperparameter tuning",
    )
    parser.add_argument("--cnn-epochs", type=int, default=50)
    parser.add_argument("--cnn-batch-size", type=int, default=16)
    parser.add_argument("--cnn-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="also save point-level observed and predicted values",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the input schema without fitting models",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow this run to replace analysis files already in output-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_inputs = tuple(dict.fromkeys(args.inputs))
    selected_scenarios = tuple(dict.fromkeys(args.scenarios))
    config = AnalysisConfig(
        seed=args.seed,
        n_time_points=args.n_time_points,
        pca_components=args.pca_components,
        inner_folds=args.inner_folds,
        within_folds=args.within_folds,
        within_group_level=args.within_group_level,
        inner_group_level=args.inner_group_level,
        cnn_epochs=args.cnn_epochs,
        cnn_batch_size=args.cnn_batch_size,
        cnn_learning_rate=args.cnn_learning_rate,
    )
    validate_config(config)
    set_seed(config.seed)
    preflight_inputs: Sequence[str] = () if args.validate_only else selected_inputs
    preflight_dependencies(args.data, preflight_inputs)
    frame = load_feature_table(args.data, selected_inputs, config)
    if args.validate_only:
        print(
            "Validation passed: "
            f"{len(frame)} rows, {frame['_sequence'].nunique()} limb trials, "
            f"{frame['subject_id'].nunique()} subjects."
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    managed_names = (
        "metrics.csv",
        "metrics.partial.csv",
        "pooled_metrics.csv",
        "pooled_metrics_by_condition.csv",
        "predictions.csv",
        "run_config.json",
        "run_status.json",
    )
    managed_paths = [args.output_dir / name for name in managed_names]
    existing_paths = [path for path in managed_paths if path.exists()]
    if existing_paths and not args.overwrite:
        existing_names = ", ".join(path.name for path in existing_paths)
        raise FileExistsError(
            f"output-dir already contains analysis files ({existing_names}); "
            "choose another directory or pass --overwrite"
        )
    if args.overwrite:
        for path in existing_paths:
            path.unlink()

    run_config: dict[str, object] = {
        "data": str(args.data.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "inputs": list(selected_inputs),
        "scenarios": list(selected_scenarios),
        "n_jobs": args.n_jobs,
        "device": args.device,
        "save_predictions": bool(args.save_predictions),
        "analysis_config": asdict(config),
    }
    write_json(args.output_dir / "run_config.json", run_config)
    write_json(
        args.output_dir / "run_status.json",
        {"status": "running", "completed_evaluations": 0},
    )

    summaries: list[dict[str, object]] = []
    prediction_tables: list[pd.DataFrame] = []
    try:
        for split in make_splits(frame, selected_scenarios, config):
            for input_name in selected_inputs:
                summary, predictions = evaluate_split(
                    frame, split, input_name, config, args.n_jobs, args.device
                )
                summaries.append(summary)
                prediction_tables.append(predictions)
                pd.DataFrame(summaries).to_csv(
                    args.output_dir / "metrics.partial.csv", index=False
                )
                write_json(
                    args.output_dir / "run_status.json",
                    {
                        "status": "running",
                        "completed_evaluations": len(summaries),
                        "last_scenario": split.scenario,
                        "last_held_out": split.held_out,
                        "last_input": input_name,
                    },
                )
    except Exception as exc:
        write_json(
            args.output_dir / "run_status.json",
            {
                "status": "failed",
                "completed_evaluations": len(summaries),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    try:
        all_predictions = pd.concat(prediction_tables, ignore_index=True)
        duplicate_oof_columns = ["scenario", "input", *META_COLUMNS]
        if all_predictions.duplicated(duplicate_oof_columns).any():
            raise RuntimeError(
                "duplicate out-of-fold predictions detected for the same "
                "scenario/input/row"
            )

        pd.DataFrame(summaries).to_csv(args.output_dir / "metrics.csv", index=False)
        aggregate_prediction_metrics(
            all_predictions,
            ("scenario", "input"),
        ).to_csv(args.output_dir / "pooled_metrics.csv", index=False)
        aggregate_prediction_metrics(
            all_predictions,
            ("scenario", "input", "condition"),
        ).to_csv(
            args.output_dir / "pooled_metrics_by_condition.csv", index=False
        )
        if args.save_predictions:
            all_predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    except Exception as exc:
        write_json(
            args.output_dir / "run_status.json",
            {
                "status": "failed",
                "completed_evaluations": len(summaries),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    (args.output_dir / "metrics.partial.csv").unlink(missing_ok=True)
    write_json(
        args.output_dir / "run_status.json",
        {
            "status": "completed",
            "completed_evaluations": len(summaries),
            "saved_predictions": bool(args.save_predictions),
        },
    )
    print(
        f"Analysis completed: {len(summaries)} evaluations written to "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
