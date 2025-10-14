# python
import numpy as np
import pandas as pd
import warnings
from typing import List, Tuple, Dict, Any

import ruptures as rpt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)
from sklearn.neighbors import LocalOutlierFactor
import plotly.graph_objs as go

warnings.filterwarnings("ignore", category=UserWarning)


# ======================================================
# Funzioni generiche (INVARIATE)
# ======================================================

def _to_float_array(a) -> np.ndarray:
    if a is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(a)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    if arr.dtype == object:
        try:
            return arr.astype(float)
        except Exception:
            return np.asarray([float(x) for x in arr], dtype=float)
    return arr.astype(float, copy=False)


def _extract_pc_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.upper().startswith("PC")]


def _ensure_time_column(df: pd.DataFrame) -> pd.Series:
    if 'Time' in df.columns:
        return pd.Series(df['Time'].values, name='Time')
    return pd.Series(range(len(df)), name='Time')


def _build_window_matrix(series: np.ndarray, w: int, end: int = None) -> Tuple[np.ndarray, np.ndarray]:
    series = _to_float_array(series)
    if end is None:
        end = len(series)
    X, y = [], []
    for i in range(w, end):
        X.append(series[i - w:i])
        y.append(series[i])
    if not X:
        return np.asarray([], dtype=float).reshape(0, w), np.asarray([], dtype=float)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _compute_fixed_threshold(errors_train: np.ndarray) -> float:
    errors_train = _to_float_array(errors_train)
    if errors_train.size == 0:
        return 0.0
    return float(np.mean(errors_train) + 3.0 * np.std(errors_train))


def calculate_variable_thresholds(errors: np.ndarray, pen: int = 1) -> np.ndarray:
    errors = _to_float_array(errors)
    n = errors.size
    if n == 0:
        return np.asarray([], dtype=float)
    if np.allclose(errors, errors[0]):
        thr = float(errors[0] + 3.0 * 0.0)
        return np.full(n, thr, dtype=float)

    try:
        algo = rpt.Pelt(model='l2').fit(errors)
        bkps = algo.predict(pen=pen)
    except Exception:
        return np.full(n, _compute_fixed_threshold(errors), dtype=float)

    var_thr = np.zeros(n, dtype=float)
    start = 0
    for b in bkps:
        seg = errors[start:b]
        if seg.size == 0:
            thr = 0.0
        else:
            thr = float(np.mean(seg) + 3.0 * np.std(seg))
        var_thr[start:b] = thr
        start = b
    return var_thr


def _generate_figures_per_pc(time_idx: np.ndarray,
                             pc_series: np.ndarray,
                             errors: np.ndarray,
                             fixed_thr: float,
                             var_thr: np.ndarray,
                             pc_name: str) -> List[go.Figure]:
    errors = _to_float_array(errors)
    var_thr = _to_float_array(var_thr)
    if var_thr.size != errors.size:
        var_thr = np.full_like(errors, fill_value=_compute_fixed_threshold(errors))
    squared_errors = errors ** 2
    anomalies = errors > var_thr

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=time_idx, y=errors, mode='lines', name='Errore'))
    fig1.add_trace(go.Scatter(x=time_idx, y=[fixed_thr] * len(errors),
                              mode='lines', name='Soglia fissa', line=dict(dash='dash')))
    fig1.add_trace(go.Scatter(x=time_idx, y=var_thr,
                              mode='lines', name='Soglia variabile', line=dict(dash='dot')))
    fig1.update_layout(title=f"{pc_name} - Errori")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time_idx, y=squared_errors, mode='lines', name='Errore^2'))
    fig2.add_trace(go.Scatter(x=time_idx, y=[fixed_thr ** 2] * len(errors),
                              mode='lines', name='Soglia fissa^2', line=dict(dash='dash')))
    fig2.add_trace(go.Scatter(x=time_idx, y=var_thr ** 2,
                              mode='lines', name='Soglia variabile^2', line=dict(dash='dot')))
    fig2.update_layout(title=f"{pc_name} - Squared Errors")

    fig3 = go.Figure()
    full_time = np.arange(len(pc_series))
    fig3.add_trace(go.Scatter(x=full_time, y=pc_series, mode='lines', name=pc_name))
    if errors.size > 0:
        e = errors.copy()
        s = pc_series
        if np.std(e) > 1e-12 and np.std(s) > 1e-12:
            e = (e - np.mean(e)) / (np.std(e) + 1e-12)
            e = e * (np.std(s) * 0.5) + np.mean(s)
        else:
            e = e * 0.0 + np.mean(s)
        fig3.add_trace(go.Scatter(x=time_idx, y=e,
                                  mode='lines', name='Errore (scalato)', line=dict(color='orange')))
        anomaly_time = time_idx[anomalies]
        anomaly_vals = np.take(pc_series, anomaly_time, mode='clip')
        fig3.add_trace(go.Scatter(x=anomaly_time, y=anomaly_vals,
                                  mode='markers', name='Anomalie', marker=dict(color='red', size=6)))
    fig3.update_layout(title=f"{pc_name} - Serie & Anomalie")

    return [fig1, fig2, fig3]


# ======================================================
# 🆕 Helper per Estrarre Anomaly Data
# ======================================================

def _extract_anomaly_data(errors_all: Dict[str, np.ndarray],
                          thresholds_all: Dict[str, Dict],
                          time_all: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """
    Estrae dati strutturati delle anomalie da errors/thresholds.

    Returns:
        Dict {PC: {times: list, errors: list, threshold: float}}
    """
    anomaly_data = {}

    for pc, errors in errors_all.items():
        var_thr = thresholds_all[pc].get('variable', np.array([]))
        if var_thr.size != errors.size:
            var_thr = np.full_like(errors, thresholds_all[pc]['fixed'])

        anomalies_mask = errors > var_thr
        time_idx = time_all.get(pc, np.arange(len(errors)))

        anomaly_data[pc] = {
            'times': time_idx[anomalies_mask].tolist(),
            'errors': errors[anomalies_mask].tolist(),
            'threshold': float(thresholds_all[pc]['fixed'])
        }

    return anomaly_data


# ======================================================
# Linear Regression
# ======================================================

def train_linear_regression(df: pd.DataFrame, n: int, w: int):
    pc_cols = _extract_pc_columns(df)
    models: Dict[str, Any] = {}
    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        end_train = min(n, len(series))
        X, y = _build_window_matrix(series, w, end=end_train)
        if X.shape[0] > 0:
            model = LinearRegression()
            model.fit(X, y)
            models[pc] = model
        else:
            models[pc] = None
    return models, {"type": "linreg", "n": n, "w": w}


def detect_anomalies_linear(df: pd.DataFrame, models: Dict[str, Any], n: int, w: int):
    pc_cols = _extract_pc_columns(df)
    time_full = _ensure_time_column(df)
    figs_all, errors_all, thresholds_all, time_all = [], {}, {}, {}

    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        model = models.get(pc)
        preds, idxs = [], []
        if model is not None:
            for i in range(w, len(series)):
                x = series[i - w:i].reshape(1, -1)
                preds.append(float(model.predict(x)))
                idxs.append(i)
        preds = np.asarray(preds, dtype=float)
        real = series[w:]
        if preds.size != real.size:
            errors = np.asarray([], dtype=float)
            idxs = []
        else:
            errors = np.abs(real - preds)
        train_cut = max(0, min(errors.size, n - w))
        train_errors = errors[:train_cut]
        fixed_thr = _compute_fixed_threshold(train_errors)
        var_thr = calculate_variable_thresholds(errors, pen=1)
        errors_all[pc] = errors
        thresholds_all[pc] = {"fixed": fixed_thr, "variable": var_thr}
        time_idx = time_full.iloc[idxs].values if len(time_full) == len(series) else np.arange(w, len(series))
        time_all[pc] = time_idx
        figs = _generate_figures_per_pc(time_idx, series, errors, fixed_thr, var_thr, pc)
        figs_all.extend(figs)

    # 🆕 Estrai anomaly data
    anomaly_data = _extract_anomaly_data(errors_all, thresholds_all, time_all)

    return figs_all, anomaly_data


# ======================================================
# Linear Regression + Bagging
# ======================================================

def train_linear_regression_bagging(df: pd.DataFrame, n: int, w: int,
                                    num_models: int = 10, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    pc_cols = _extract_pc_columns(df)
    bag_models: Dict[str, List[Any]] = {}

    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        end_train = min(n, len(series))
        X, y = _build_window_matrix(series, w, end=end_train)
        models_pc: List[Any] = []
        if X.shape[0] > 0:
            n_samples = X.shape[0]
            for _ in range(num_models):
                if n_samples > 1:
                    idx = rng.integers(0, n_samples, size=n_samples)
                    Xb, yb = X[idx], y[idx]
                else:
                    Xb, yb = X, y
                m = LinearRegression()
                m.fit(Xb, yb)
                models_pc.append(m)
        bag_models[pc] = models_pc

    return bag_models, {"type": "linreg_bagging", "n": n, "w": w, "num_models": num_models}


def detect_anomalies_linear_bagging(df: pd.DataFrame, bag_models: Dict[str, List[Any]], n: int, w: int):
    pc_cols = _extract_pc_columns(df)
    time_full = _ensure_time_column(df)
    figs_all, errors_all, thresholds_all, time_all = [], {}, {}, {}

    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        models_pc = bag_models.get(pc, [])
        preds, idxs = [], []
        if models_pc:
            for i in range(w, len(series)):
                x = series[i - w:i].reshape(1, -1)
                pred = np.mean([float(m.predict(x)) for m in models_pc])
                preds.append(pred)
                idxs.append(i)
        preds = np.asarray(preds, dtype=float)
        real = series[w:]
        if preds.size != real.size:
            errors = np.asarray([], dtype=float)
            idxs = []
        else:
            errors = np.abs(real - preds)
        train_cut = max(0, min(errors.size, n - w))
        train_errors = errors[:train_cut]
        fixed_thr = _compute_fixed_threshold(train_errors)
        var_thr = calculate_variable_thresholds(errors, pen=1)
        errors_all[pc] = errors
        thresholds_all[pc] = {"fixed": fixed_thr, "variable": var_thr}
        time_idx = time_full.iloc[idxs].values if len(time_full) == len(series) else np.arange(w, len(series))
        time_all[pc] = time_idx
        figs = _generate_figures_per_pc(time_idx, series, errors, fixed_thr, var_thr, pc)
        figs_all.extend(figs)

    # 🆕 Estrai anomaly data
    anomaly_data = _extract_anomaly_data(errors_all, thresholds_all, time_all)

    return figs_all, anomaly_data


# ======================================================
# Alberi + Bagging (RF / GB / ET)
# ======================================================

def _train_tree_bagging(df: pd.DataFrame, n: int, w: int, num_models: int,
                        model_cls, model_kwargs: Dict[str, Any], seed: int = 123):
    rng = np.random.default_rng(seed)
    pc_cols = _extract_pc_columns(df)
    bag_models: Dict[str, List[Any]] = {}

    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        end_train = min(n, len(series))
        X, y = _build_window_matrix(series, w, end=end_train)
        models_pc: List[Any] = []
        if X.shape[0] > 0:
            n_samples = X.shape[0]
            for _ in range(num_models):
                if n_samples > 1:
                    idx = rng.integers(0, n_samples, size=n_samples)
                    Xb, yb = X[idx], y[idx]
                else:
                    Xb, yb = X, y
                m = model_cls(**model_kwargs)
                m.fit(Xb, yb)
                models_pc.append(m)
        bag_models[pc] = models_pc

    return bag_models


def _detect_tree_bagging(df: pd.DataFrame, bag_models: Dict[str, List[Any]], n: int, w: int):
    pc_cols = _extract_pc_columns(df)
    time_full = _ensure_time_column(df)
    figs_all, errors_all, thresholds_all, time_all = [], {}, {}, {}

    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        models_pc = bag_models.get(pc, [])
        preds, idxs = [], []
        if models_pc:
            for i in range(w, len(series)):
                x = series[i - w:i].reshape(1, -1)
                pred = np.mean([float(m.predict(x)) for m in models_pc])
                preds.append(pred)
                idxs.append(i)
        preds = np.asarray(preds, dtype=float)
        real = series[w:]
        if preds.size != real.size:
            errors = np.asarray([], dtype=float)
            idxs = []
        else:
            errors = np.abs(real - preds)
        train_cut = max(0, min(errors.size, n - w))
        train_errors = errors[:train_cut]
        fixed_thr = _compute_fixed_threshold(train_errors)
        var_thr = calculate_variable_thresholds(errors, pen=1)
        errors_all[pc] = errors
        thresholds_all[pc] = {"fixed": fixed_thr, "variable": var_thr}
        time_idx = time_full.iloc[idxs].values if len(time_full) == len(series) else np.arange(w, len(series))
        time_all[pc] = time_idx
        figs = _generate_figures_per_pc(time_idx, series, errors, fixed_thr, var_thr, pc)
        figs_all.extend(figs)

    # 🆕 Estrai anomaly data
    anomaly_data = _extract_anomaly_data(errors_all, thresholds_all, time_all)

    return figs_all, anomaly_data


def train_random_forest_bagging(df: pd.DataFrame, n: int, w: int, num_models: int = 10,
                                n_estimators: int = 100, max_depth: int = 10, min_samples_split: int = 2):
    models = _train_tree_bagging(
        df, n, w, num_models,
        RandomForestRegressor,
        dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=None)
    )
    return models, {"type": "rf_bagging", "n": n, "w": w}


def detect_anomalies_random_forest_bagging(df: pd.DataFrame, models: Dict[str, List[Any]], n: int, w: int):
    return _detect_tree_bagging(df, models, n, w)


def train_gradient_boosting_bagging(df: pd.DataFrame, n: int, w: int, num_models: int = 10,
                                    n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
    models = _train_tree_bagging(
        df, n, w, num_models,
        GradientBoostingRegressor,
        dict(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=None)
    )
    return models, {"type": "gb_bagging", "n": n, "w": w}


def detect_anomalies_gradient_boosting_bagging(df: pd.DataFrame, models: Dict[str, List[Any]], n: int, w: int):
    return _detect_tree_bagging(df, models, n, w)


def train_extra_trees_bagging(df: pd.DataFrame, n: int, w: int, num_models: int = 10,
                              n_estimators: int = 100, max_depth: int = 10, min_samples_split: int = 2):
    models = _train_tree_bagging(
        df, n, w, num_models,
        ExtraTreesRegressor,
        dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=None)
    )
    return models, {"type": "et_bagging", "n": n, "w": w}


def detect_anomalies_extra_trees_bagging(df: pd.DataFrame, models: Dict[str, List[Any]], n: int, w: int):
    return _detect_tree_bagging(df, models, n, w)


# ======================================================
# Local Outlier Factor
# ======================================================

def train_lof(df: pd.DataFrame, n: int, w: int,
              n_neighbors: int = 20, contamination: float = 0.05):
    pc_cols = _extract_pc_columns(df)
    models: Dict[str, Any] = {}
    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        end_train = min(n, len(series))
        X_train, _ = _build_window_matrix(series, w, end=end_train)
        if X_train.shape[0] > 0:
            model = LocalOutlierFactor(
                n_neighbors=max(1, int(n_neighbors)),
                contamination=float(contamination),
                novelty=True
            )
            model.fit(X_train)
            models[pc] = model
        else:
            models[pc] = None
    return models, {"type": "lof", "n": n, "w": w,
                    "n_neighbors": n_neighbors, "contamination": contamination}


def detect_anomalies_lof(df: pd.DataFrame, models: Dict[str, Any], n: int, w: int):
    pc_cols = _extract_pc_columns(df)
    time_full = _ensure_time_column(df)
    figs_all, errors_all, thresholds_all, time_all = [], {}, {}, {}

    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        model: LocalOutlierFactor = models.get(pc)
        windows, idxs = [], []
        for i in range(w, len(series)):
            windows.append(series[i - w:i])
            idxs.append(i)
        if not windows or model is None:
            errors = np.asarray([], dtype=float)
            time_idx = np.asarray([], dtype=float)
            fixed_thr = 0.0
            var_thr = np.asarray([], dtype=float)
        else:
            X = np.asarray(windows, dtype=float)
            errors = -model.score_samples(X)
            train_cut = max(0, min(errors.size, n - w))
            train_errors = errors[:train_cut]
            fixed_thr = _compute_fixed_threshold(train_errors)
            var_thr = calculate_variable_thresholds(errors, pen=1)
            time_idx = time_full.iloc[idxs].values if len(time_full) == len(series) else np.asarray(idxs)

        errors_all[pc] = errors
        thresholds_all[pc] = {"fixed": fixed_thr, "variable": var_thr}
        time_all[pc] = time_idx
        figs = _generate_figures_per_pc(time_idx, series, errors, fixed_thr, var_thr, pc)
        figs_all.extend(figs)

    # 🆕 Estrai anomaly data
    anomaly_data = _extract_anomaly_data(errors_all, thresholds_all, time_all)

    return figs_all, anomaly_data


# ======================================================
# Matrix Profile (STUMP)
# ======================================================

def detect_matrix_profile(df: pd.DataFrame, n: int, w: int):
    try:
        import stumpy as stp
    except Exception as e:
        raise RuntimeError("Il pacchetto 'stumpy' è richiesto per Matrix Profile.") from e

    pc_cols = _extract_pc_columns(df)
    figs_all, errors_all, thresholds_all, time_all = [], {}, {}, {}

    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        if len(series) < w + 2:
            errors = np.asarray([], dtype=float)
            fixed_thr = 0.0
            var_thr = np.asarray([], dtype=float)
            time_idx = np.asarray([], dtype=float)
        else:
            mp = stp.stump(series, m=w)
            errors_full = _to_float_array(mp[:, 0])
            errors = errors_full[:-1] if errors_full.size > 0 else errors_full
            train_cut = max(0, min(errors.size, n - w))
            train_errors = errors[:train_cut]
            fixed_thr = _compute_fixed_threshold(train_errors)
            var_thr = calculate_variable_thresholds(errors, pen=1)
            time_idx = np.arange(w, len(series))

        errors_all[pc] = errors
        thresholds_all[pc] = {"fixed": fixed_thr, "variable": var_thr}
        time_all[pc] = time_idx
        figs = _generate_figures_per_pc(time_idx, series, errors, fixed_thr, var_thr, pc)
        figs_all.extend(figs)

    # 🆕 Estrai anomaly data
    anomaly_data = _extract_anomaly_data(errors_all, thresholds_all, time_all)

    return figs_all, anomaly_data


# ======================================================
# Clustering: DBSCAN
# ======================================================

def detect_anomalies_dbscan(df: pd.DataFrame, n: int, w: int,
                            eps: float = 0.25, min_samples: int = 15, knn_k: int | None = None):
    from sklearn.neighbors import NearestNeighbors

    pc_cols = _extract_pc_columns(df)
    time_full = _ensure_time_column(df)
    figs_all, errors_all, thresholds_all, time_all = [], {}, {}, {}

    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        windows, idxs = [], []
        for i in range(w, len(series)):
            windows.append(series[i - w:i])
            idxs.append(i)

        if not windows:
            errors = np.asarray([], dtype=float)
            fixed_thr = 0.0
            var_thr = np.asarray([], dtype=float)
            time_idx = np.asarray([], dtype=float)
        else:
            X_all = np.asarray(windows, dtype=float)
            n_train_windows = max(0, min(X_all.shape[0], n - w))
            X_train = X_all[:n_train_windows] if n_train_windows > 0 else None

            k = int(knn_k if knn_k is not None else max(1, min_samples))
            if X_train is None or X_train.shape[0] < k:
                nbrs = NearestNeighbors(n_neighbors=min(k, max(1, X_all.shape[0]))).fit(X_all)
                dists, _ = nbrs.kneighbors(X_all)
                kth = dists[:, -1] if dists.ndim == 2 and dists.shape[1] > 0 else np.zeros(X_all.shape[0])
                errors = _to_float_array(kth)
                train_cut = max(0, min(errors.size, n - w))
                train_errors = errors[:train_cut]
            else:
                nbrs = NearestNeighbors(n_neighbors=min(k, X_train.shape[0])).fit(X_train)
                dists, _ = nbrs.kneighbors(X_all)
                kth = dists[:, -1] if dists.ndim == 2 and dists.shape[1] > 0 else np.zeros(X_all.shape[0])
                errors = _to_float_array(kth)
                train_errors = errors[:n_train_windows]

            fixed_thr = _compute_fixed_threshold(train_errors)
            var_thr = calculate_variable_thresholds(errors, pen=1)
            time_idx = time_full.iloc[idxs].values if len(time_full) == len(series) else np.asarray(idxs)

        errors_all[pc] = errors
        thresholds_all[pc] = {"fixed": fixed_thr, "variable": var_thr, "eps_ref": float(eps)}
        time_all[pc] = time_idx
        figs = _generate_figures_per_pc(time_idx, series, errors, fixed_thr, var_thr, pc)
        figs_all.extend(figs)

    # 🆕 Estrai anomaly data
    anomaly_data = _extract_anomaly_data(errors_all, thresholds_all, time_all)

    return figs_all, anomaly_data


# ======================================================
# Clustering: OPTICS
# ======================================================

def detect_anomalies_optics(df: pd.DataFrame, n: int, w: int,
                            min_samples: int = 5, xi: float = 0.01, min_cluster_size: float | int = 0.1):
    from sklearn.cluster import OPTICS

    pc_cols = _extract_pc_columns(df)
    time_full = _ensure_time_column(df)
    figs_all, errors_all, thresholds_all, time_all = [], {}, {}, {}

    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        windows, idxs = [], []
        for i in range(w, len(series)):
            windows.append(series[i - w:i])
            idxs.append(i)

        if not windows:
            errors = np.asarray([], dtype=float)
            fixed_thr = 0.0
            var_thr = np.asarray([], dtype=float)
            time_idx = np.asarray([], dtype=float)
        else:
            X_all = np.asarray(windows, dtype=float)
            n_train_windows = max(0, min(X_all.shape[0], n - w))

            optics = OPTICS(min_samples=max(2, int(min_samples)),
                            xi=float(xi),
                            min_cluster_size=min_cluster_size)
            optics.fit(X_all)
            reach = np.full(X_all.shape[0], np.inf, dtype=float)
            reach[optics.ordering_] = optics.reachability_
            finite = reach[np.isfinite(reach)]
            if finite.size == 0:
                reach[:] = 0.0
            else:
                reach[~np.isfinite(reach)] = np.max(finite)
            errors = _to_float_array(reach)

            train_errors = errors[:n_train_windows]
            fixed_thr = _compute_fixed_threshold(train_errors)
            var_thr = calculate_variable_thresholds(errors, pen=1)
            time_idx = time_full.iloc[idxs].values if len(time_full) == len(series) else np.asarray(idxs)

        errors_all[pc] = errors
        thresholds_all[pc] = {"fixed": fixed_thr, "variable": var_thr}
        time_all[pc] = time_idx
        figs = _generate_figures_per_pc(time_idx, series, errors, fixed_thr, var_thr, pc)
        figs_all.extend(figs)

    # 🆕 Estrai anomaly data
    anomaly_data = _extract_anomaly_data(errors_all, thresholds_all, time_all)

    return figs_all, anomaly_data


# ======================================================
# Clustering: K-Means
# ======================================================

def train_kmeans(df: pd.DataFrame, n: int, w: int, n_clusters: int = 3, random_state: int = 42):
    from sklearn.cluster import KMeans
    pc_cols = _extract_pc_columns(df)
    models: Dict[str, Any] = {}

    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        windows = []
        for i in range(w, min(n, len(series))):
            windows.append(series[i - w:i])
        X_train = np.asarray(windows, dtype=float)
        if X_train.shape[0] == 0:
            models[pc] = None
            continue
        k = int(max(1, n_clusters))
        model = KMeans(n_clusters=k, n_init="auto", random_state=int(random_state))
        model.fit(X_train)
        models[pc] = model

    return models, {"type": "kmeans", "n": n, "w": w, "n_clusters": n_clusters, "random_state": random_state}


def detect_anomalies_kmeans(df: pd.DataFrame, models: Dict[str, Any], n: int, w: int):
    pc_cols = _extract_pc_columns(df)
    time_full = _ensure_time_column(df)
    figs_all, errors_all, thresholds_all, time_all = [], {}, {}, {}

    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        model = models.get(pc)
        windows, idxs = [], []
        for i in range(w, len(series)):
            windows.append(series[i - w:i])
            idxs.append(i)

        if not windows or model is None:
            errors = np.asarray([], dtype=float)
            fixed_thr = 0.0
            var_thr = np.asarray([], dtype=float)
            time_idx = np.asarray([], dtype=float)
        else:
            X_all = np.asarray(windows, dtype=float)
            dists = model.transform(X_all)
            errors = np.min(dists, axis=1)
            train_cut = max(0, min(errors.size, n - w))
            train_errors = errors[:train_cut]
            fixed_thr = _compute_fixed_threshold(train_errors)
            var_thr = calculate_variable_thresholds(errors, pen=1)
            time_idx = time_full.iloc[idxs].values if len(time_full) == len(series) else np.asarray(idxs)

        errors_all[pc] = errors
        thresholds_all[pc] = {"fixed": fixed_thr, "variable": var_thr}
        time_all[pc] = time_idx
        figs = _generate_figures_per_pc(time_idx, series, errors, fixed_thr, var_thr, pc)
        figs_all.extend(figs)

    # 🆕 Estrai anomaly data
    anomaly_data = _extract_anomaly_data(errors_all, thresholds_all, time_all)

    return figs_all, anomaly_data
