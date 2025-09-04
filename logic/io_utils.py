import os
import json
import joblib
import pandas as pd

# --- Gestione cartelle ---
def ensure_dir(path: str):
    """Crea la directory se non esiste."""
    os.makedirs(path, exist_ok=True)


# --- DataFrame ---
def save_dataframe(df: pd.DataFrame, path: str, name: str):
    """Salva un DataFrame come CSV."""
    ensure_dir(path)
    file_path = os.path.join(path, f"{name}.csv")
    df.to_csv(file_path, index=False)
    return file_path

def load_dataframe(path: str, name: str):
    """Carica un DataFrame CSV se esiste."""
    file_path = os.path.join(path, f"{name}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None


# --- Modelli ML ---
def save_model(model, path: str, name: str):
    """Salva un modello sklearn (o compatibile) come .pkl."""
    ensure_dir(path)
    file_path = os.path.join(path, f"{name}.pkl")
    joblib.dump(model, file_path)
    return file_path

def load_model(path: str, name: str):
    """Carica un modello salvato se esiste."""
    file_path = os.path.join(path, f"{name}.pkl")
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None


# --- JSON (per configurazioni, metriche, cluster labels ecc.) ---
def save_json(obj, path: str, name: str):
    """Salva un oggetto Python come JSON."""
    ensure_dir(path)
    file_path = os.path.join(path, f"{name}.json")
    with open(file_path, "w") as f:
        json.dump(obj, f, indent=4)
    return file_path

def load_json(path: str, name: str):
    """Carica un JSON se esiste."""
    file_path = os.path.join(path, f"{name}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return None


# --- Grafici Plotly ---
def save_figure(fig, path: str, name: str, fmt: str = "png"):
    """Salva una figura Plotly in formato immagine o HTML."""
    ensure_dir(path)
    file_path = os.path.join(path, f"{name}.{fmt}")
    if fmt == "html":
        fig.write_html(file_path)
    else:
        fig.write_image(file_path)
    return file_path