
import numpy as np

def create_features_and_target(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def create_features(data, window_size):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
    return np.array(X)

def compute_prediction_errors(models, pca_df, n, w):
    errors_abs, errors_sq = {}, {}

    for pc in ['PC1', 'PC2', 'PC3']:
        data = pca_df[pc].values
        model = models[pc]
        X = create_features(data[:n], w)
        y_true = data[w:n]
        y_pred = model.predict(X)

        error_abs = np.abs(y_true - y_pred)
        error_sq = (y_true - y_pred) ** 2

        errors_abs[pc] = error_abs
        errors_sq[pc] = error_sq

    return errors_abs, errors_sq
