import numpy as np
import pandas as pd
import joblib
import ruptures as rpt
import warnings
from typing import List, Tuple, Dict, Any
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
# Utilità interne
# ======================================================

def _to_float_array(a) -> np.ndarray:
    if a is None:
        return np.array([], dtype=float)
    arr = np.asarray(a)
    if arr.dtype == object:
        # forza conversione, ignora elementi non convertibili
        arr = arr.astype(float, copy=False)
    else:
        arr = arr.astype(float, copy=False)
    return arr


def _extract_pc_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.upper().startswith("PC")]


def _ensure_time_column(df: pd.DataFrame) -> pd.Series:
    if 'Time' in df.columns:
        return df['Time']
    return pd.Series(range(len(df)), name='Time')


def _build_window_matrix(series: np.ndarray, w: int, end: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: finestre (i-w:i) per i in [w, end)
    y: valore series[i]
    """
    if end is None:
        end = len(series)
    X, y = [], []
    for i in range(w, end):
        X.append(series[i - w:i])
        y.append(series[i])
    if not X:
        return np.empty((0, w), dtype=float), np.empty((0,), dtype=float)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _compute_fixed_threshold(errors_train: np.ndarray) -> float:
    errors_train = _to_float_array(errors_train)
    if errors_train.size == 0:
        return 0.0
    return float(np.mean(errors_train) + 3.0 * np.std(errors_train))


def calculate_variable_thresholds(errors: np.ndarray, pen: int = 1) -> np.ndarray:
    errors = _to_float_array(errors)
    if errors.size == 0:
        return np.array([], dtype=float)
    if np.allclose(errors, errors[0]):
        return np.full_like(errors, errors[0] + 1e-6)

    try:
        algo = rpt.Pelt(model="rbf").fit(errors)
        bkps = algo.predict(pen=pen)
    except Exception:
        bkps = [len(errors)]

    var_thr = np.zeros_like(errors)
    start = 0
    for b in bkps:
        seg = errors[start:b]
        if seg.size == 0:
            thr_val = 0.0
        else:
            mean_seg = float(np.mean(seg))
            std_seg = float(np.std(seg))
            thr_val = mean_seg + 3.0 * std_seg
        var_thr[start:b] = thr_val
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
        var_thr = np.full_like(errors, fixed_thr)
    squared_errors = errors ** 2
    anomalies = errors > var_thr

    # Figura 1: errori + soglie
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=time_idx, y=errors, mode='lines', name='Errore'))
    fig1.add_trace(go.Scatter(x=time_idx, y=[fixed_thr] * len(errors),
                              mode='lines', name='Soglia Fissa', line=dict(dash='dash')))
    fig1.add_trace(go.Scatter(x=time_idx, y=var_thr,
                              mode='lines', name='Soglia Variabile', line=dict(dash='dot')))
    fig1.update_layout(title=f"{pc_name} - Errori")

    # Figura 2: squared errors
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time_idx, y=squared_errors, mode='lines', name='Errore^2'))
    fig2.add_trace(go.Scatter(x=time_idx, y=[fixed_thr ** 2] * len(errors),
                              mode='lines', name='(Soglia Fissa)^2', line=dict(dash='dash')))
    fig2.add_trace(go.Scatter(x=time_idx, y=var_thr ** 2,
                              mode='lines', name='(Soglia Var)^2', line=dict(dash='dot')))
    fig2.update_layout(title=f"{pc_name} - Squared Errors")

    # Figura 3: serie + anomalie
    fig3 = go.Figure()
    full_time = np.arange(len(pc_series))
    fig3.add_trace(go.Scatter(x=full_time, y=pc_series, mode='lines', name=pc_name))
    if errors.size > 0:
        rng = np.ptp(errors)
        if rng < 1e-12:
            rng = 1.0
        norm_err = (errors - np.min(errors)) / (rng + 1e-9)
        scaled_err = norm_err * (np.ptp(pc_series) * 0.25 if np.ptp(pc_series) > 0 else 1.0) + np.min(pc_series)
        # allineamento: assumiamo time_idx riferito a posizioni di predizione
        fig3.add_trace(go.Scatter(x=time_idx, y=scaled_err,
                                  mode='lines', name='Errore (scalato)', line=dict(color='orange')))
        anomaly_time = time_idx[anomalies]
        anomaly_vals = pc_series[anomaly_time]
        fig3.add_trace(go.Scatter(x=anomaly_time, y=anomaly_vals,
                                  mode='markers', marker=dict(color='red', size=6),
                                  name='Anomalie'))
    fig3.update_layout(title=f"{pc_name} - Serie & Anomalie")

    return [fig1, fig2, fig3]

# ======================================================
# Modelli di Regressione
# ======================================================

def train_linear_regression(df: pd.DataFrame, n: int, w: int):
    pc_cols = _extract_pc_columns(df)
    models = {}
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
    figs_all, errors_all, thresholds_all = [], {}, {}
    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        model = models.get(pc)
        preds, idxs = [], []
        if model is not None:
            for i in range(w, len(series)):
                x = series[i - w:i].reshape(1, -1)
                preds.append(model.predict(x)[0])
                idxs.append(i)
        preds = np.asarray(preds, dtype=float)
        real = series[w:]
        if preds.size != real.size:
            errors = np.array([], dtype=float)
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
        figs = _generate_figures_per_pc(
            time_idx=time_idx,
            pc_series=series,
            errors=errors,
            fixed_thr=fixed_thr,
            var_thr=var_thr,
            pc_name=pc
        )
        figs_all.extend(figs)
    return figs_all, errors_all, thresholds_all

# ======================================================
# Linear Regression + Bagging
# ======================================================

def train_linear_regression_bagging(df: pd.DataFrame, n: int, w: int,
                                    num_models: int = 10, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    pc_cols = _extract_pc_columns(df)
    bag_models = {}
    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        end_train = min(n, len(series))
        X, y = _build_window_matrix(series, w, end=end_train)
        models_pc = []
        if X.shape[0] > 0:
            for _ in range(num_models):
                # bootstrap
                idx = rng.integers(0, X.shape[0], X.shape[0])
                m = LinearRegression()
                m.fit(X[idx], y[idx])
                models_pc.append(m)
        bag_models[pc] = models_pc
    return bag_models, {"type": "linreg_bagging", "n": n, "w": w, "num_models": num_models}


def detect_anomalies_linear_bagging(df: pd.DataFrame, bag_models: Dict[str, List[Any]], n: int, w: int):
    pc_cols = _extract_pc_columns(df)
    time_full = _ensure_time_column(df)
    figs_all, errors_all, thresholds_all = [], {}, {}
    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        models_pc = bag_models.get(pc, [])
        preds, idxs = [], []
        if models_pc:
            for i in range(w, len(series)):
                x = series[i - w:i].reshape(1, -1)
                p = np.mean([m.predict(x)[0] for m in models_pc])
                preds.append(p)
                idxs.append(i)
        preds = np.asarray(preds, dtype=float)
        real = series[w:]
        if preds.size != real.size:
            errors = np.array([], dtype=float)
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
        figs = _generate_figures_per_pc(time_idx, series, errors, fixed_thr, var_thr, pc)
        figs_all.extend(figs)
    return figs_all, errors_all, thresholds_all

# ======================================================
# Alberi + Bagging (RF / GB / ET)
# ======================================================

def _train_tree_bagging(df: pd.DataFrame, n: int, w: int, num_models: int,
                        model_cls, model_kwargs: Dict[str, Any], seed: int = 123):
    rng = np.random.default_rng(seed)
    pc_cols = _extract_pc_columns(df)
    bag_models = {}
    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        end_train = min(n, len(series))
        X, y = _build_window_matrix(series, w, end=end_train)
        models_pc = []
        if X.shape[0] > 0:
            for _ in range(num_models):
                idx = rng.integers(0, X.shape[0], X.shape[0])
                m = model_cls(**model_kwargs)
                m.fit(X[idx], y[idx])
                models_pc.append(m)
        bag_models[pc] = models_pc
    return bag_models


def _detect_tree_bagging(df: pd.DataFrame, bag_models: Dict[str, List[Any]], n: int, w: int):
    pc_cols = _extract_pc_columns(df)
    time_full = _ensure_time_column(df)
    figs_all, errors_all, thresholds_all = [], {}, {}
    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        models_pc = bag_models.get(pc, [])
        preds, idxs = [], []
        if models_pc:
            for i in range(w, len(series)):
                x = series[i - w:i].reshape(1, -1)
                p = np.mean([m.predict(x)[0] for m in models_pc])
                preds.append(p)
                idxs.append(i)
        preds = np.asarray(preds, dtype=float)
        real = series[w:]
        if preds.size != real.size:
            errors = np.array([], dtype=float)
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
        figs = _generate_figures_per_pc(time_idx, series, errors, fixed_thr, var_thr, pc)
        figs_all.extend(figs)
    return figs_all, errors_all, thresholds_all


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
    models = {}
    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        end_train = min(n, len(series))
        windows = []
        for i in range(w, end_train):
            windows.append(series[i - w:i])
        X = np.asarray(windows, dtype=float)
        if X.shape[0] == 0:
            models[pc] = None
            continue
        nn = min(n_neighbors, max(2, X.shape[0] - 1))
        if nn < 2:
            models[pc] = None
            continue
        lof = LocalOutlierFactor(n_neighbors=nn,
                                 contamination=contamination,
                                 novelty=True)
        lof.fit(X)
        models[pc] = lof
    return models, {"type": "lof", "n": n, "w": w,
                    "n_neighbors": n_neighbors, "contamination": contamination}


def detect_anomalies_lof(df: pd.DataFrame, models: Dict[str, Any], n: int, w: int):
    pc_cols = _extract_pc_columns(df)
    time_full = _ensure_time_column(df)
    figs_all, errors_all, thresholds_all = [], {}, {}
    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        model = models.get(pc)
        windows, idxs = [], []
        for i in range(w, len(series)):
            windows.append(series[i - w:i])
            idxs.append(i)
        if not windows or model is None:
            errors = np.array([], dtype=float)
            fixed_thr = 0.0
            var_thr = np.array([], dtype=float)
            time_idx = np.array([], dtype=int)
        else:
            Xw = np.asarray(windows, dtype=float)
            # decision_function: valori >0 normali; usiamo errore = -score
            scores = model.decision_function(Xw)
            errors = -_to_float_array(scores)
            train_cut = max(0, min(errors.size, n - w))
            train_errors = errors[:train_cut]
            fixed_thr = _compute_fixed_threshold(train_errors)
            var_thr = calculate_variable_thresholds(errors, pen=1)
            time_idx = time_full.iloc[idxs].values if len(time_full) == len(series) else np.array(idxs)
        errors_all[pc] = errors
        thresholds_all[pc] = {"fixed": fixed_thr, "variable": var_thr}
        figs = _generate_figures_per_pc(time_idx, series, errors, fixed_thr, var_thr, pc)
        figs_all.extend(figs)
    return figs_all, errors_all, thresholds_all

# ======================================================
# Matrix Profile (STUMP)
# ======================================================

def detect_matrix_profile(df: pd.DataFrame, n: int, w: int):
    try:
        import stumpy
    except ImportError as e:
        raise ImportError("Per usare il Matrix Profile installare 'stumpy' (pip install stumpy).") from e

    pc_cols = _extract_pc_columns(df)
    time_full = _ensure_time_column(df)
    figs_all, errors_all, thresholds_all = [], {}, {}

    for pc in pc_cols:
        series = _to_float_array(df[pc].values)
        if len(series) < w + 2:
            mp_vals = np.array([], dtype=float)
            fixed_thr = 0.0
            var_thr = np.array([], dtype=float)
            time_idx = np.array([], dtype=int)
        else:
            # stumpy.stump restituisce (len(series)-w+1, 4+)
            mp = stumpy.stump(series, m=w)
            mp_vals = _to_float_array(mp[:, 0])
            # sostituiamo NaN (serie costante) con 0
            if np.isnan(mp_vals).any():
                mp_vals = np.nan_to_num(mp_vals, nan=0.0, posinf=np.nanmax(mp_vals[np.isfinite(mp_vals)] + 1.0 if np.isfinite(mp_vals).any() else 1.0), neginf=0.0)
            # Le distanze più alte = più anomale -> usiamo direttamente mp_vals come errors
            L = mp_vals.size
            # allineamento: primo profilo corrisponde alla finestra che termina a indice w-1
            # usiamo l'indice della fine della finestra (w-1 ... w-1+L-1)
            end_indices = np.arange(w - 1, w - 1 + L)
            time_idx = time_full.iloc[end_indices].values if len(time_full) == len(series) else end_indices
            # parte di training: finestre che terminano prima di n
            train_mask = end_indices < n
            train_errors = mp_vals[train_mask]
            fixed_thr = _compute_fixed_threshold(train_errors)
            var_thr = calculate_variable_thresholds(mp_vals, pen=1)
        errors_all[pc] = mp_vals
        thresholds_all[pc] = {"fixed": fixed_thr, "variable": var_thr}
        figs = _generate_figures_per_pc(
            time_idx=time_idx,
            pc_series=series,
            errors=mp_vals,
            fixed_thr=fixed_thr,
            var_thr=var_thr if var_thr.size == mp_vals.size else np.full_like(mp_vals, fixed_thr),
            pc_name=pc
        )
        figs_all.extend(figs)

    return figs_all, errors_all, thresholds_all

# ======================================================
# Salvataggio / Caricamento
# ======================================================

def save_models(models: Dict[str, Any], path: str):
    joblib.dump(models, path)
def load_models(path: str) -> Dict[str, Any]:
    return joblib.load(path)
