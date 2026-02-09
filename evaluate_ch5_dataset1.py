#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# === Import dal tuo progetto ===
from logic.pca_utils import convert_to_wide_format, add_sin_cos_columns

from logic.anomaly_detection import (
    train_linear_regression, detect_anomalies_linear,
    train_random_forest_bagging, detect_anomalies_random_forest_bagging,
    train_extra_trees_bagging, detect_anomalies_extra_trees_bagging,
    train_gradient_boosting_bagging, detect_anomalies_gradient_boosting_bagging,
    train_lof, detect_anomalies_lof
)

# -----------------------------
# Utils
# -----------------------------

def read_angoli_file(path: str) -> pd.DataFrame:
    """
    Legge un file angoli_*.txt. In molti tuoi file c'è una riga di header.
    Prova prima con skiprows=1; se fallisce, riprova senza.
    Output atteso: colonne ["Residuo", "Phi", "Psi"].
    """
    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            skiprows=1,
            header=None,
            names=["Residuo", "Phi", "Psi"],
            engine="python"
        )
        # Se legge male (0 righe), riprova
        if df.empty:
            raise ValueError("Empty after skiprows=1")
        return df
    except Exception:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            skiprows=0,
            header=None,
            names=["Residuo", "Phi", "Psi"],
            engine="python"
        )
        return df


def load_raw_folder(folder_path: str) -> pd.DataFrame:
    """
    Carica tutti i file angoli_*.txt della cartella e restituisce df_wide
    usando la tua convert_to_wide_format (come nel tool).
    """
    files = sorted(glob.glob(os.path.join(folder_path, "angoli_*.txt")))
    if not files:
        raise RuntimeError(f"Nessun file angoli_*.txt trovato in: {folder_path}")

    all_dfs = [read_angoli_file(fp) for fp in files]
    df_wide = convert_to_wide_format(all_dfs)
    return df_wide


def make_pca_df(df_wide: pd.DataFrame, use_sincos: bool, n_components: int = 3) -> pd.DataFrame:
    """
    Converte df_wide in matrice (time x features), opzionalmente applica sin/cos,
    poi PCA e restituisce dataframe con PC1..PCk + Time.
    """
    # df_wide ha righe come (residuo, angolo) e colonne time_*
    df_data = df_wide.drop(["residuo", "angolo"], axis=1)  # (features x time)
    X = df_data.T  # (time x features)

    if use_sincos:
        X = add_sin_cos_columns(X)

    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X)

    pca_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(n_components)])
    # Asse temporale: usiamo indice 0..T-1 (coerente col change-point)
    pca_df["Time"] = np.arange(len(pca_df), dtype=int)
    return pca_df


def build_step_labels(T: int, change_point: int) -> np.ndarray:
    """
    Label binaria: 0 fino a change_point-1, 1 da change_point in poi.
    """
    y = np.zeros(T, dtype=int)
    if change_point < 0:
        change_point = 0
    if change_point > T:
        change_point = T
    y[change_point:] = 1
    return y


def anomaly_data_to_pred(T: int, anomaly_data: dict) -> np.ndarray:
    """
    anomaly_data: dict per PC -> contiene 'times' (lista time index).
    Aggregazione OR: se una PC segna anomalia a t => y_pred[t]=1.
    """
    y_pred = np.zeros(T, dtype=int)

    for pc, info in anomaly_data.items():
        for t in info.get("times", []):
            try:
                t_int = int(round(float(t)))
            except Exception:
                continue
            if 0 <= t_int < T:
                y_pred[t_int] = 1

    return y_pred


def eval_one_model(model_name: str, pca_df: pd.DataFrame, change_point: int,
                  train_fn, detect_fn, train_n: int, w: int) -> dict:
    """
    Esegue train+detect usando le funzioni del progetto e calcola metriche.
    """
    model, _ = train_fn(pca_df, n=train_n, w=w)
    _, anomaly_data = detect_fn(pca_df, model, n=train_n, w=w)

    T = len(pca_df)
    y_true = build_step_labels(T, change_point=change_point)
    y_pred = anomaly_data_to_pred(T, anomaly_data)

    P = precision_score(y_true, y_pred, zero_division=0)
    R = recall_score(y_true, y_pred, zero_division=0)
    F1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # [[TN, FP],[FN, TP]]

    return {
        "model": model_name,
        "precision": float(P),
        "recall": float(R),
        "f1": float(F1),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "T": int(T),
        "change_point": int(change_point),
        "train_n": int(train_n),
        "w": int(w),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cap 5: valutazione F1 su Dataset 1 (raw) con e senza sin/cos"
    )
    parser.add_argument("--dataset_folder", type=str, default="data/12-06",
                        help="Cartella del dataset raw (contiene angoli_*.txt). Esempio: data/12-06")
    parser.add_argument("--change_point", type=int, default=198,
                        help="Indice temporale da cui iniziano le anomalie (default 198).")
    parser.add_argument("--train_n", type=int, default=150,
                        help="Numero di time step usati per train (baseline: 150).")
    parser.add_argument("--w", type=int, default=10,
                        help="Window size per i modelli che la usano (default 10).")
    parser.add_argument("--out_csv", type=str, default="results_ch5_dataset1.csv",
                        help="Path del CSV di output.")
    args = parser.parse_args()

    # 1) Carica raw e crea df_wide
    df_wide = load_raw_folder(args.dataset_folder)

    # quick check
    n_times = df_wide.shape[1] - 2  # meno residuo/angolo
    print(f"✅ Dataset: {args.dataset_folder}")
    print(f"✅ Time steps stimati (colonne time_*): {n_times}")

    # 2) Lista modelli (pochi ma utili)
    model_specs = [
        ("rf_bagging", train_random_forest_bagging, detect_anomalies_random_forest_bagging),
        ("extra_trees_bagging", train_extra_trees_bagging, detect_anomalies_extra_trees_bagging),
        ("gb_bagging", train_gradient_boosting_bagging, detect_anomalies_gradient_boosting_bagging),
        ("lof", train_lof, detect_anomalies_lof),
    ]

    rows = []

    # 3) Esegui due condizioni: senza e con sin/cos
    for use_sincos in [False, True]:
        pca_df = make_pca_df(df_wide, use_sincos=use_sincos, n_components=3)

        for (name, train_fn, detect_fn) in model_specs:
            metrics = eval_one_model(
                model_name=name,
                pca_df=pca_df,
                change_point=args.change_point,
                train_fn=train_fn,
                detect_fn=detect_fn,
                train_n=args.train_n,
                w=args.w
            )
            metrics["sincos"] = bool(use_sincos)
            rows.append(metrics)

    out = pd.DataFrame(rows).sort_values(by=["model", "sincos"]).reset_index(drop=True)
    print("\n=== RISULTATI ===")
    print(out[["model", "sincos", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]])

    out.to_csv(args.out_csv, index=False)
    print(f"\n✅ Salvato CSV: {args.out_csv}")


if __name__ == "__main__":
    main()