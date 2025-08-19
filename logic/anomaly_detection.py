import numpy as np
import joblib
import ruptures as rpt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import plotly.graph_objs as go

# ===============================
# Funzioni generali
# ===============================

def create_features_and_target(data, window_size):
    """Crea feature X e target y da una serie temporale usando finestre scorrevoli."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def create_features(data, window_size):
    """Crea solo feature X (senza target) da una serie temporale."""
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
    return np.array(X)


def calculate_variable_thresholds(errors, pen=1):
    """Calcola soglie variabili con segmentazione tramite PELT (ruptures)."""
    errors = np.array(errors).reshape(-1, 1)
    algo = rpt.Pelt(model="l2", min_size=10)
    algo.fit(errors)
    result = algo.predict(pen=pen)

    variable_thresholds = np.zeros_like(errors)
    start = 0
    for cp in result:
        segment = errors[start:cp]
        threshold = np.mean(segment) + 3 * np.std(segment)
        variable_thresholds[start:cp] = threshold
        start = cp
    return variable_thresholds.flatten(), result


# ===============================
# Linear Regression
# ===============================

def train_linear_regression(pca_df, n=180, w=20):

    models = {}
    train_thresholds = {}
    train_squared_thresholds = {}

    for pc in ['PC1', 'PC2', 'PC3']:
        # Creazione feature e target
        X, y = create_features_and_target(pca_df[pc][:n].values, w)

        # Addestramento modello
        model = LinearRegression()
        model.fit(X, y)

        # Salvataggio
        joblib.dump(model, f'model_lr_{pc}.joblib')
        models[pc] = model

        # Calcolo errori di training
        errors, sq_errors = [], []
        for X_window, real_value in zip(X, y):
            y_pred = model.predict([X_window])[0]
            errors.append(abs(real_value - y_pred))
            sq_errors.append((real_value - y_pred) ** 2)

        # Soglie fisse (media + 3*std)
        train_thresholds[pc] = np.mean(errors) + 3 * np.std(errors)
        train_squared_thresholds[pc] = np.mean(sq_errors) + 3 * np.std(sq_errors)

    return models, train_thresholds, train_squared_thresholds


def detect_anomalies_linear(pca_df, models, n=180, w=20):

    test_time = np.arange(n, len(pca_df))

    prediction_errors = {pc: [] for pc in ['PC1', 'PC2', 'PC3']}
    squared_errors = {pc: [] for pc in ['PC1', 'PC2', 'PC3']}
    variable_thresholds = {}
    variable_squared_thresholds = {}

    for pc in ['PC1', 'PC2', 'PC3']:
        # Carico modello
        model = models.get(pc, joblib.load(f'model_lr_{pc}.joblib'))

        # Costruisco feature test
        test_features = create_features(pca_df[pc][n - w:].values, w)

        # Predizioni ed errori
        for i in range(len(test_features)):
            X_window = test_features[i]
            y_true = pca_df[pc][n + i]
            y_pred = model.predict([X_window])[0]

            error = abs(y_true - y_pred)
            sq_error = (y_true - y_pred) ** 2

            prediction_errors[pc].append(error)
            squared_errors[pc].append(sq_error)

        # Soglie variabili con ruptures
        variable_thresholds[pc], _ = calculate_variable_thresholds(prediction_errors[pc])
        variable_squared_thresholds[pc], _ = calculate_variable_thresholds(squared_errors[pc])

    # === Plotly grafici ===
    figs = []
    for i, pc in enumerate(['PC1', 'PC2', 'PC3']):
        # Prediction Errors
        fig_error = go.Figure()
        fig_error.add_trace(go.Scatter(x=test_time, y=prediction_errors[pc], mode='lines', name=f'Error {pc}'))
        fig_error.add_trace(go.Scatter(x=test_time, y=variable_thresholds[pc], mode='lines',
                                       name='Variable Threshold', line=dict(dash='dot')))
        fig_error.update_layout(title=f"Prediction Error {pc}", xaxis_title="Time", yaxis_title="Error",
                                plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")
        figs.append(fig_error)

        # Squared Errors
        fig_sq_error = go.Figure()
        fig_sq_error.add_trace(go.Scatter(x=test_time, y=squared_errors[pc], mode='lines', name=f'Squared Error {pc}'))
        fig_sq_error.add_trace(go.Scatter(x=test_time, y=variable_squared_thresholds[pc], mode='lines',
                                          name='Variable Threshold', line=dict(dash='dot')))
        fig_sq_error.update_layout(title=f"Squared Error {pc}", xaxis_title="Time", yaxis_title="Squared Error",
                                   plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")
        figs.append(fig_sq_error)

    return figs, prediction_errors, squared_errors

# -------------------------
# Linear Regression + Bagging
# -------------------------
def train_linear_regression_bagging(pca_df, n=180, w=20, num_models=10):

    all_models = {pc: [] for pc in ['PC1','PC2','PC3']}
    train_thresholds = {}
    train_squared_thresholds = {}

    for pc in ['PC1','PC2','PC3']:
        X, y = create_features_and_target(pca_df[pc][:n].values, w)
        for _ in range(num_models):
            X_resampled, y_resampled = resample(X, y)
            model = LinearRegression()
            model.fit(X_resampled, y_resampled)
            all_models[pc].append(model)

        # Errori di training
        errors, sq_errors = [], []
        for X_window, real_value in zip(X, y):
            preds = [m.predict([X_window])[0] for m in all_models[pc]]
            mean_pred = np.mean(preds)
            errors.append(abs(mean_pred - real_value))
            sq_errors.append((mean_pred - real_value)**2)

        train_thresholds[pc] = np.mean(errors) + 3 * np.std(errors)
        train_squared_thresholds[pc] = np.mean(sq_errors) + 3 * np.std(sq_errors)

    return all_models, train_thresholds, train_squared_thresholds

def detect_anomalies_linear_bagging(pca_df, all_models, n=180, w=20):

    import numpy as np
    import plotly.graph_objs as go

    prediction_errors = {pc: [] for pc in ['PC1','PC2','PC3']}
    squared_errors = {pc: [] for pc in ['PC1','PC2','PC3']}
    thresholds = {}
    squared_thresholds = {}
    variable_thresholds = {}
    variable_squared_thresholds = {}
    change_points = {}
    predictions = {pc: [] for pc in ['PC1','PC2','PC3']}
    prediction_intervals = {pc: [] for pc in ['PC1','PC2','PC3']}

    test_time = np.arange(n, len(pca_df))

    # Calcolo predizioni, errori e intervalli
    for pc in ['PC1','PC2','PC3']:
        test_features = create_features(pca_df[pc][n - w:].values, w)

        for i, X_window in enumerate(test_features):
            preds = [m.predict([X_window])[0] for m in all_models[pc]]
            mean_pred = np.mean(preds)
            lower_bound = np.percentile(preds, 2.5)
            upper_bound = np.percentile(preds, 97.5)

            real_value = pca_df[pc][n + i]

            predictions[pc].append(mean_pred)
            prediction_intervals[pc].append((lower_bound, upper_bound))

            prediction_errors[pc].append(abs(mean_pred - real_value))
            squared_errors[pc].append((mean_pred - real_value)**2)

        # Soglie fisse
        thresholds[pc] = np.mean(prediction_errors[pc]) + 3 * np.std(prediction_errors[pc])
        squared_thresholds[pc] = np.mean(squared_errors[pc]) + 3 * np.std(squared_errors[pc])

        # Soglie variabili
        variable_thresholds[pc], change_points[pc] = calculate_variable_thresholds(prediction_errors[pc])
        variable_squared_thresholds[pc], _ = calculate_variable_thresholds(squared_errors[pc])

    # =========================
    # Creazione grafici Plotly
    # =========================
    figs = []

    for pc in ['PC1','PC2','PC3']:
        # Prediction Error
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(x=test_time, y=prediction_errors[pc], mode='lines', name='Prediction Error'))
        fig_err.add_trace(go.Scatter(x=test_time, y=variable_thresholds[pc], mode='lines', name='Variable Threshold', line=dict(dash='dash')))
        fig_err.add_hline(y=thresholds[pc], line=dict(color='red', dash='dot'), annotation_text='Fixed Threshold')
        fig_err.update_layout(title=f'Prediction Error - {pc}', xaxis_title='Time', yaxis_title='Error',
                              plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")
        figs.append(fig_err)

        # Squared Error
        fig_sq = go.Figure()
        fig_sq.add_trace(go.Scatter(x=test_time, y=squared_errors[pc], mode='lines', name='Squared Error'))
        fig_sq.add_trace(go.Scatter(x=test_time, y=variable_squared_thresholds[pc], mode='lines', name='Variable Threshold', line=dict(dash='dash')))
        fig_sq.add_hline(y=squared_thresholds[pc], line=dict(color='red', dash='dot'), annotation_text='Fixed Threshold')
        fig_sq.update_layout(title=f'Squared Error - {pc}', xaxis_title='Time', yaxis_title='Squared Error',
                             plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")
        figs.append(fig_sq)

        # Prediction con intervallo
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=test_time, y=predictions[pc], mode='lines', name='Mean Prediction'))
        fig_pred.add_trace(go.Scatter(x=test_time, y=pca_df[pc][n:].values, mode='lines', name='Real', line=dict(color='red')))
        lower_bounds = [interval[0] for interval in prediction_intervals[pc]]
        upper_bounds = [interval[1] for interval in prediction_intervals[pc]]
        fig_pred.add_trace(go.Scatter(x=np.concatenate([test_time, test_time[::-1]]),
                                      y=np.concatenate([lower_bounds, upper_bounds[::-1]]),
                                      fill='toself', fillcolor='rgba(128,128,128,0.3)',
                                      line=dict(color='rgba(255,255,255,0)'),
                                      name='Prediction Interval'))
        fig_pred.update_layout(title=f'Prediction with Interval - {pc}', xaxis_title='Time', yaxis_title='Value',
                               plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")
        figs.append(fig_pred)

    return figs, prediction_errors, squared_errors, thresholds, squared_thresholds, variable_thresholds, variable_squared_thresholds

# -------------------------
# Randrom Forest Regression + Bagging
# -------------------------

def train_random_forest_bagging(pca_df, n=180, w=20, num_models=10):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.utils import resample
    import joblib

    all_models = {'PC1': [], 'PC2': [], 'PC3': []}
    train_prediction_errors = {'PC1': [], 'PC2': [], 'PC3': []}
    train_squared_errors = {'PC1': [], 'PC2': [], 'PC3': []}
    train_thresholds = {}
    train_squared_thresholds = {}

    for pc in ['PC1', 'PC2', 'PC3']:
        X, y = create_features_and_target(pca_df[pc][:n].values, w)
        for _ in range(num_models):
            X_res, y_res = resample(X, y)
            model = RandomForestRegressor()
            model.fit(X_res, y_res)
            all_models[pc].append(model)

        # Calcolo errori training
        for X_window, real_value in zip(X, y):
            preds = [model.predict([X_window])[0] for model in all_models[pc]]
            mean_pred = np.mean(preds)
            train_prediction_errors[pc].append(abs(mean_pred - real_value))
            train_squared_errors[pc].append((real_value - mean_pred) ** 2)

        # Soglie fisse
        train_thresholds[pc] = np.mean(train_prediction_errors[pc]) + 3 * np.std(train_prediction_errors[pc])
        train_squared_thresholds[pc] = np.mean(train_squared_errors[pc]) + 3 * np.std(train_squared_errors[pc])

    return all_models, train_thresholds, train_squared_thresholds

def detect_anomalies_random_forest_bagging(pca_df, all_models, n=180, w=20):
    import plotly.graph_objs as go
    import numpy as np

    prediction_errors = {'PC1': [], 'PC2': [], 'PC3': []}
    squared_errors = {'PC1': [], 'PC2': [], 'PC3': []}
    predictions = {'PC1': [], 'PC2': [], 'PC3': []}
    prediction_intervals = {'PC1': [], 'PC2': [], 'PC3': []}
    thresholds, squared_thresholds, variable_thresholds, variable_squared_thresholds = {}, {}, {}, {}
    test_time = np.arange(n, len(pca_df))

    for pc in ['PC1','PC2','PC3']:
        test_features_pc = create_features(pca_df[pc][n-w:].values, w)
        for i, X_window in enumerate(test_features_pc):
            preds = [model.predict([X_window])[0] for model in all_models[pc]]
            mean_pred = np.mean(preds)
            lower = np.percentile(preds, 2.5)
            upper = np.percentile(preds, 97.5)
            predictions[pc].append(mean_pred)
            prediction_intervals[pc].append((lower, upper))

            real_value = pca_df[pc][n+i]
            prediction_errors[pc].append(abs(mean_pred - real_value))
            squared_errors[pc].append((real_value - mean_pred)**2)

        # Soglie fisse e variabili
        thresholds[pc] = np.mean(prediction_errors[pc]) + 3*np.std(prediction_errors[pc])
        squared_thresholds[pc] = np.mean(squared_errors[pc]) + 3*np.std(squared_errors[pc])
        variable_thresholds[pc], _ = calculate_variable_thresholds(prediction_errors[pc])
        variable_squared_thresholds[pc], _ = calculate_variable_thresholds(squared_errors[pc])

    # === Creazione grafici Plotly ===
    figs = []
    for pc in ['PC1','PC2','PC3']:
        # Prediction Error
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(x=test_time, y=prediction_errors[pc], mode='lines', name='Prediction Error'))
        fig_err.add_trace(go.Scatter(x=test_time, y=variable_thresholds[pc], mode='lines', name='Variable Threshold', line=dict(dash='dash')))
        fig_err.add_hline(y=thresholds[pc], line=dict(color='red', dash='dot'), annotation_text='Fixed Threshold')
        fig_err.update_layout(title=f'Prediction Error - {pc}', plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")
        figs.append(fig_err)

        # Squared Error
        fig_sq = go.Figure()
        fig_sq.add_trace(go.Scatter(x=test_time, y=squared_errors[pc], mode='lines', name='Squared Error'))
        fig_sq.add_trace(go.Scatter(x=test_time, y=variable_squared_thresholds[pc], mode='lines', name='Variable Threshold', line=dict(dash='dash')))
        fig_sq.add_hline(y=squared_thresholds[pc], line=dict(color='red', dash='dot'), annotation_text='Fixed Threshold')
        fig_sq.update_layout(title=f'Squared Error - {pc}', plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")
        figs.append(fig_sq)

        # Predizione con intervalli
        lower_bounds = [interval[0] for interval in prediction_intervals[pc]]
        upper_bounds = [interval[1] for interval in prediction_intervals[pc]]
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=test_time, y=predictions[pc], mode='lines', name='Mean Prediction'))
        fig_pred.add_trace(go.Scatter(x=test_time, y=pca_df[pc][n:], mode='lines', name='Real', line=dict(color='red')))
        fig_pred.add_trace(go.Scatter(x=test_time, y=lower_bounds, mode='lines', name='Lower Bound', line=dict(dash='dot')))
        fig_pred.add_trace(go.Scatter(x=test_time, y=upper_bounds, mode='lines', name='Upper Bound', line=dict(dash='dot')))
        fig_pred.update_layout(title=f'Prediction with Intervals - {pc}', plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")
        figs.append(fig_pred)

    return figs, prediction_errors, squared_errors, thresholds, squared_thresholds, variable_thresholds, variable_squared_thresholds, predictions, prediction_intervals


# -------------------------
# Gradient Boosting + Bagging
# -------------------------

def train_gradient_boosting_bagging(pca_df, n=180, w=20, num_models=10):

    def create_features_and_target(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    all_models = {'PC1': [], 'PC2': [], 'PC3': []}
    train_thresholds = {}
    train_squared_thresholds = {}

    prediction_errors_train = {'PC1': [], 'PC2': [], 'PC3': []}
    squared_errors_train = {'PC1': [], 'PC2': [], 'PC3': []}

    for pc in ['PC1','PC2','PC3']:
        X, y = create_features_and_target(pca_df[pc][:n].values, w)
        for _ in range(num_models):
            X_res, y_res = resample(X, y)
            model = GradientBoostingRegressor()
            model.fit(X_res, y_res)
            all_models[pc].append(model)

        # Errori sul training set
        for X_window, real_value in zip(X, y):
            preds = [model.predict([X_window])[0] for model in all_models[pc]]
            mean_pred = np.mean(preds)
            prediction_errors_train[pc].append(abs(mean_pred - real_value))
            squared_errors_train[pc].append((real_value - mean_pred)**2)

        # Soglie fisse
        train_thresholds[pc] = np.mean(prediction_errors_train[pc]) + 3*np.std(prediction_errors_train[pc])
        train_squared_thresholds[pc] = np.mean(squared_errors_train[pc]) + 3*np.std(squared_errors_train[pc])

    return all_models, train_thresholds, train_squared_thresholds

def detect_anomalies_gradient_boosting_bagging(pca_df, all_models, n=180, w=20):
    prediction_errors = {'PC1': [], 'PC2': [], 'PC3': []}
    squared_errors = {'PC1': [], 'PC2': [], 'PC3': []}
    predictions = {'PC1': [], 'PC2': [], 'PC3': []}
    prediction_intervals = {'PC1': [], 'PC2': [], 'PC3': []}
    thresholds = {}
    squared_thresholds = {}
    variable_thresholds = {}
    variable_squared_thresholds = {}
    change_points = {}

    test_time = np.arange(n, len(pca_df))

    for pc in ['PC1','PC2','PC3']:
        test_features_pc = create_features(pca_df[pc][n-w:].values, w)
        for i, X_window in enumerate(test_features_pc):
            preds = [model.predict([X_window])[0] for model in all_models[pc]]
            mean_pred = np.mean(preds)
            lower = np.percentile(preds, 2.5)
            upper = np.percentile(preds, 97.5)
            predictions[pc].append(mean_pred)
            prediction_intervals[pc].append((lower, upper))

            real_value = pca_df[pc][n+i]
            prediction_errors[pc].append(abs(mean_pred - real_value))
            squared_errors[pc].append((real_value - mean_pred)**2)

        # Soglie fisse
        thresholds[pc] = np.mean(prediction_errors[pc]) + 3*np.std(prediction_errors[pc])
        squared_thresholds[pc] = np.mean(squared_errors[pc]) + 3*np.std(squared_errors[pc])
        # Soglie variabili
        variable_thresholds[pc], _ = calculate_variable_thresholds(prediction_errors[pc])
        variable_squared_thresholds[pc], _ = calculate_variable_thresholds(squared_errors[pc])

    # === Creazione grafici Plotly ===
    figs = []
    for pc in ['PC1','PC2','PC3']:
        # Prediction Error
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(x=test_time, y=prediction_errors[pc], mode='lines', name='Prediction Error'))
        fig_err.add_trace(go.Scatter(x=test_time, y=variable_thresholds[pc], mode='lines', name='Variable Threshold', line=dict(dash='dash')))
        fig_err.add_hline(y=thresholds[pc], line=dict(color='red', dash='dot'), annotation_text='Fixed Threshold')
        fig_err.update_layout(title=f'Prediction Error - {pc}', plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")
        figs.append(fig_err)

        # Squared Error
        fig_sq = go.Figure()
        fig_sq.add_trace(go.Scatter(x=test_time, y=squared_errors[pc], mode='lines', name='Squared Error'))
        fig_sq.add_trace(go.Scatter(x=test_time, y=variable_squared_thresholds[pc], mode='lines', name='Variable Threshold', line=dict(dash='dash')))
        fig_sq.add_hline(y=squared_thresholds[pc], line=dict(color='red', dash='dot'), annotation_text='Fixed Threshold')
        fig_sq.update_layout(title=f'Squared Error - {pc}', plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")
        figs.append(fig_sq)

        # Prediction con intervalli
        lower_bounds = [interval[0] for interval in prediction_intervals[pc]]
        upper_bounds = [interval[1] for interval in prediction_intervals[pc]]
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=test_time, y=predictions[pc], mode='lines', name='Mean Prediction'))
        fig_pred.add_trace(go.Scatter(x=test_time, y=pca_df[pc][n:], mode='lines', name='Real', line=dict(color='red')))
        fig_pred.add_trace(go.Scatter(x=test_time, y=lower_bounds, mode='lines', name='Lower Bound', line=dict(dash='dot')))
        fig_pred.add_trace(go.Scatter(x=test_time, y=upper_bounds, mode='lines', name='Upper Bound', line=dict(dash='dot')))
        fig_pred.update_layout(title=f'Prediction with Intervals - {pc}', plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")
        figs.append(fig_pred)

    return figs, prediction_errors, squared_errors, thresholds, squared_thresholds, variable_thresholds, variable_squared_thresholds, predictions, prediction_intervals

# -------------------------
# Extra Trees Regressor + Bagging
# -------------------------

def train_extra_trees_bagging(pca_df, n=180, w=20, num_models=10):

    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.utils import resample
    import numpy as np

    def create_features_and_target(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    models = {'PC1': [], 'PC2': [], 'PC3': []}
    train_thresholds = {}
    train_squared_thresholds = {}

    for pc in ['PC1', 'PC2', 'PC3']:
        X, y = create_features_and_target(pca_df[pc][:n].values, w)
        for _ in range(num_models):
            X_res, y_res = resample(X, y)
            model = ExtraTreesRegressor()
            model.fit(X_res, y_res)
            models[pc].append(model)

        # Errori sul training
        pred_errors, sq_errors = [], []
        for X_window, real_val in zip(X, y):
            preds = [m.predict([X_window])[0] for m in models[pc]]
            mean_pred = np.mean(preds)
            pred_errors.append(abs(mean_pred - real_val))
            sq_errors.append((real_val - mean_pred)**2)

        train_thresholds[pc] = np.mean(pred_errors) + 3*np.std(pred_errors)
        train_squared_thresholds[pc] = np.mean(sq_errors) + 3*np.std(sq_errors)

    return models, train_thresholds, train_squared_thresholds


def detect_anomalies_extra_trees_bagging(pca_df, all_models, n=180, w=20):


    prediction_errors = {pc: [] for pc in ['PC1','PC2','PC3']}
    squared_errors = {pc: [] for pc in ['PC1','PC2','PC3']}
    predictions = {pc: [] for pc in ['PC1','PC2','PC3']}
    prediction_intervals = {pc: [] for pc in ['PC1','PC2','PC3']}
    thresholds, squared_thresholds, variable_thresholds, variable_squared_thresholds = {}, {}, {}, {}
    test_time = np.arange(n, len(pca_df))

    for pc in ['PC1','PC2','PC3']:
        test_features_pc = create_features(pca_df[pc][n-w:].values, w)
        for i, X_window in enumerate(test_features_pc):
            preds = [m.predict([X_window])[0] for m in all_models[pc]]
            mean_pred = np.mean(preds)
            lower = np.percentile(preds, 2.5)
            upper = np.percentile(preds, 97.5)
            predictions[pc].append(mean_pred)
            prediction_intervals[pc].append((lower, upper))

            real_value = pca_df[pc][n+i]
            prediction_errors[pc].append(abs(mean_pred - real_value))
            squared_errors[pc].append((real_value - mean_pred)**2)

        # Soglie fisse e variabili
        thresholds[pc] = np.mean(prediction_errors[pc]) + 3*np.std(prediction_errors[pc])
        squared_thresholds[pc] = np.mean(squared_errors[pc]) + 3*np.std(squared_errors[pc])
        variable_thresholds[pc], _ = calculate_variable_thresholds(prediction_errors[pc])
        variable_squared_thresholds[pc], _ = calculate_variable_thresholds(squared_errors[pc])

    # Creazione grafici Plotly
    figs = []
    for pc in ['PC1','PC2','PC3']:
        # Prediction Error
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(x=test_time, y=prediction_errors[pc], mode='lines', name='Prediction Error'))
        fig_err.add_trace(go.Scatter(x=test_time, y=variable_thresholds[pc], mode='lines', name='Variable Threshold', line=dict(dash='dash')))
        fig_err.add_hline(y=thresholds[pc], line=dict(color='red', dash='dot'), annotation_text='Fixed Threshold')
        fig_err.update_layout(title=f'Prediction Error - {pc}', plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")
        figs.append(fig_err)

        # Squared Error
        fig_sq = go.Figure()
        fig_sq.add_trace(go.Scatter(x=test_time, y=squared_errors[pc], mode='lines', name='Squared Error'))
        fig_sq.add_trace(go.Scatter(x=test_time, y=variable_squared_thresholds[pc], mode='lines', name='Variable Threshold', line=dict(dash='dash')))
        fig_sq.add_hline(y=squared_thresholds[pc], line=dict(color='red', dash='dot'), annotation_text='Fixed Threshold')
        fig_sq.update_layout(title=f'Squared Error - {pc}', plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")
        figs.append(fig_sq)

        # Predizione con intervalli
        lower_bounds = [interval[0] for interval in prediction_intervals[pc]]
        upper_bounds = [interval[1] for interval in prediction_intervals[pc]]
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=test_time, y=predictions[pc], mode='lines', name='Mean Prediction'))
        fig_pred.add_trace(go.Scatter(x=test_time, y=pca_df[pc][n:], mode='lines', name='Real', line=dict(color='red')))
        fig_pred.add_trace(go.Scatter(x=test_time, y=lower_bounds, mode='lines', name='Lower Bound', line=dict(dash='dot')))
        fig_pred.add_trace(go.Scatter(x=test_time, y=upper_bounds, mode='lines', name='Upper Bound', line=dict(dash='dot')))
        fig_pred.update_layout(title=f'Prediction with Intervals - {pc}', plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")
        figs.append(fig_pred)

    return figs, prediction_errors, squared_errors, thresholds, squared_thresholds, variable_thresholds, variable_squared_thresholds, predictions, prediction_intervals