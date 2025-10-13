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

def normalize_angles(df):
    """
    Normalizza gli angoli in [-180, 180] gradi.
    df: DataFrame con colonne angolari (es. ['Phi', 'Psi'])
    """
    df_norm = df.copy()
    for col in df_norm.columns:
        # Porta l'angolo in [-180, 180]
        df_norm[col] = ((df_norm[col] + 180) % 360) - 180
    return df_norm


def convert_to_wide_format(all_dfs):
    """
    Converte da long format (lista di DataFrame singoli)
    a wide format (residui × istanti temporali).

    Input:
        all_dfs: lista di DataFrame [df_time0, df_time1, ...]
                 Ogni df ha colonne ['Residuo', 'Phi', 'Psi']

    Output:
        DataFrame wide con colonne ['residuo', 'angolo', 'time_0', 'time_1', ...]
    """
    data_list = []

    for file_idx, df in enumerate(all_dfs):
        time_label = f'time_{file_idx}'

        # Aggiungi Phi
        for _, row in df.iterrows():
            data_list.append([row['Residuo'], 'Phi', time_label, row['Phi']])

        # Aggiungi Psi
        for _, row in df.iterrows():
            data_list.append([row['Residuo'], 'Psi', time_label, row['Psi']])

    # Crea DataFrame combinato
    combined_df = pd.DataFrame(data_list, columns=['residuo', 'angolo', 'tempo', 'valore'])

    # Pivot
    pivot_df = combined_df.pivot_table(
        index=['residuo', 'angolo'],
        columns='tempo',
        values='valore'
    ).reset_index()

    # Ordina colonne temporali
    time_columns = sorted(
        [c for c in pivot_df.columns if c.startswith('time_')],
        key=lambda x: int(x.split('_')[1])
    )

    return pivot_df[['residuo', 'angolo'] + time_columns]


def compute_windowed_pca(df_angles, window_size, n_components=3, use_sin_cos=True, overlap=False):
    """
    Applica PCA su finestre temporali dei dati angolari.
    Ritorna DataFrame con 1 riga per finestra (media delle PC).

    Args:
        df_angles: DataFrame con colonne ['Phi', 'Psi'] e opzionalmente 'Time'
        window_size: Dimensione della finestra mobile
        n_components: Numero di componenti principali
        use_sin_cos: Se True, converte angoli in sin/cos
        overlap: Se True, finestre sovrapposte (step=1), altrimenti step=window_size

    Returns:
        DataFrame con colonne ['PC1', 'PC2', ..., 'Time']
    """
    # 🔧 Estrai Time se presente, altrimenti usa indici sequenziali
    has_time = 'Time' in df_angles.columns
    if has_time:
        time_values = df_angles['Time'].values  # ✅ Usa Time originale (0-240)
        df_data = df_angles[['Phi', 'Psi']].copy()  # Estrai solo angoli
    else:
        time_values = np.arange(len(df_angles))  # Fallback: 0, 1, 2, ...
        df_data = df_angles.copy()

    N = df_data.shape[0]
    if N == 0 or window_size <= 0 or window_size > N:
        return pd.DataFrame()

    step = 1 if overlap else window_size
    pca_results = []
    window_times = []  # 🔧 Tempo rappresentativo di ogni finestra

    for start in range(0, N - window_size + 1, step):
        window = df_data.iloc[start:start + window_size].copy()

        # 🔧 Normalizza angoli PRIMA di Sin/Cos
        window_norm = normalize_angles(window)

        if use_sin_cos:
            window_trans = add_sin_cos_columns(window_norm)
        else:
            window_trans = window_norm.copy()

        n_samples, n_features = window_trans.shape
        k_max = min(n_samples, n_features)
        k = min(n_components, k_max)

        if k < 1:
            continue

        pca = PCA(n_components=k)
        transformed = pca.fit_transform(window_trans)  # Shape: (window_size, k)

        # Vettore PC per finestra (media delle righe)
        pc_vector = transformed.mean(axis=0)  # Shape: (k,)
        pca_results.append(pc_vector)

        # 🔧 Time rappresentativo della finestra
        if has_time:
            # Usa il valore mediano di Time nella finestra
            window_time_vals = time_values[start:start + window_size]
            window_times.append(np.median(window_time_vals))
        else:
            # Usa l'indice centrale della finestra
            window_times.append(start + window_size // 2)

    if len(pca_results) == 0:
        return pd.DataFrame()

    k_final = len(pca_results[0])
    cols = [f'PC{i + 1}' for i in range(k_final)]
    pca_df = pd.DataFrame(pca_results, columns=cols)
    pca_df['Time'] = window_times  # ✅ Usa tempo reale delle finestre
    return pca_df
