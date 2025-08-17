import numpy as np
import pandas as pd

def add_sin_cos_columns(df):
    """
    Converte le colonne angolari di un DataFrame in coordinate cartesiane sin/cos.
    """
    new_columns = {}
    for col in df.columns:
        new_columns[f'{col}_cos'] = np.cos(np.deg2rad(df[col]))
        new_columns[f'{col}_sin'] = np.sin(np.deg2rad(df[col]))

    return pd.DataFrame(new_columns, index=df.index)