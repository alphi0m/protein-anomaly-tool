import pandas as pd
from sklearn.decomposition import PCA
from .sin_cos import add_sin_cos_columns

def compute_windowed_pca(df, window_size, n_components=3, use_sin_cos=True):
    """
    Applica la PCA a finestre scorrevoli dei dati angolari.
    """
    pca = PCA(n_components=n_components)
    pca_results = []

    for start in range(0, len(df.columns), window_size):
        end = start + window_size
        if end <= len(df.columns):
            window_data = df.iloc[:, start:end].T  # Trasposta per avere time sulle righe

            if use_sin_cos:
                window_data = add_sin_cos_columns(window_data)

            pca_result = pca.fit_transform(window_data)
            pca_results.extend(pca_result)

    pca_df = pd.DataFrame(pca_results, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['Time'] = range(len(pca_df))
    return pca_df