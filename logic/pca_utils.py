import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def add_sin_cos_columns(df):
    """
    Converte le colonne angolari di un DataFrame in coordinate cartesiane sin/cos.
    df: righe = campioni (tempo), colonne = angoli (es. 'Phi','Psi',...)
    Restituisce DataFrame con colonne [<col>_cos, <col>_sin, ...]
    """
    new_columns = {}
    for col in df.columns:
        vals = df[col].astype(float).values
        new_columns[f'{col}_cos'] = np.cos(np.deg2rad(vals))
        new_columns[f'{col}_sin'] = np.sin(np.deg2rad(vals))
    return pd.DataFrame(new_columns, index=df.index)


def compute_windowed_pca(df_angles, window_size, n_components=3, use_sin_cos=True, overlap=False):
    """
    Applica PCA su finestre temporali (righe) dei dati angolari.
    df_angles: DataFrame con colonne angolari (es. ['Phi','Psi']), righe = istanti/frame
    window_size: lunghezza della finestra in number of rows (frame)
    n_components: numero desiderato di componenti principali
    use_sin_cos: se True, converte ogni colonna angolare in cos/sin prima di PCA
    overlap: se True le finestre sono scorrevoli (stride=1), altrimenti non sovrapposte (stride=window_size)
    Ritorna DataFrame con colonne PC1..PCk e colonna 'Time'
    """
    N = df_angles.shape[0]
    if N == 0 or window_size <= 0 or window_size > N:
        return pd.DataFrame()

    step = 1 if overlap else window_size
    pca_results = []

    for start in range(0, N - window_size + 1, step):
        window = df_angles.iloc[start:start + window_size].copy()

        if use_sin_cos:
            window_trans = add_sin_cos_columns(window)
        else:
            window_trans = window.copy()

        n_samples, n_features = window_trans.shape
        k_max = min(n_samples, n_features)
        k = min(n_components, k_max)

        if k < 1:
            continue

        pca = PCA(n_components=k)
        transformed = pca.fit_transform(window_trans)
        pca_results.extend(transformed)

    if len(pca_results) == 0:
        return pd.DataFrame()

    k_final = len(pca_results[0])
    cols = [f'PC{i+1}' for i in range(k_final)]
    pca_df = pd.DataFrame(pca_results, columns=cols)
    pca_df['Time'] = np.arange(len(pca_df))
    return pca_df