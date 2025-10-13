import base64
import io
import traceback
import numpy as np
import pandas as pd
from dash import Input, Output, State, html, dash_table, dcc, no_update
import plotly.express as px
import plotly.graph_objs as go

from logic.pca_utils import compute_windowed_pca, convert_to_wide_format
from logic.clustering_utils import dbscan_clustering, optics_clustering, spectral_clustering
from logic.anomaly_detection import (
    train_linear_regression, detect_anomalies_linear,
    train_linear_regression_bagging, detect_anomalies_linear_bagging,
    train_random_forest_bagging, detect_anomalies_random_forest_bagging,
    train_gradient_boosting_bagging, detect_anomalies_gradient_boosting_bagging,
    train_extra_trees_bagging, detect_anomalies_extra_trees_bagging,
    train_lof, detect_anomalies_lof,
    detect_matrix_profile,
    detect_anomalies_dbscan, detect_anomalies_optics,
    train_kmeans, detect_anomalies_kmeans
)


def apply_light_theme(fig, title=None, reverse_y=False):
    """
    Applica tema chiaro al grafico.

    Args:
        fig: Figura Plotly
        title: Titolo del grafico
        reverse_y: Se True, inverte l'asse Y
    """
    layout_updates = {
        'title': title,
        'plot_bgcolor': "#ffffff",
        'paper_bgcolor': "#ffffff",
        'font': dict(color="#222"),
        'xaxis': dict(color="#222", gridcolor="#ddd"),
        'yaxis': dict(color="#222", gridcolor="#ddd"),
        'legend': dict(font=dict(color="#222"))
    }

    # Aggiungi inversione asse Y se richiesto
    if reverse_y:
        layout_updates['yaxis']['autorange'] = 'reversed'

    fig.update_layout(**layout_updates)
    return fig


def register_callbacks(app):

    # === CALLBACK 1: Analizza Dati Raw ===
    @app.callback(
        Output('raw-output', 'children'),
        Output('stored-raw-data', 'data'),
        Output('raw-section', 'style'),
        Output('preprocessing-section', 'style'),
        Input('analyze-raw-button', 'n_clicks'),
        State('upload-files', 'contents'),
        State('upload-files', 'filename')
    )
    def analyze_raw_data(n_clicks, contents_list, filenames):
        if not n_clicks or n_clicks == 0:
            return "", None, {'display': 'none'}, {'display': 'none'}

        if not contents_list:
            return html.Div("Nessun file caricato.", style={'color': 'red'}), None, \
                {'display': 'none'}, {'display': 'none'}

        try:
            # 1️⃣ LOGICA DEL PROF: costruisci long format diretto
            data_list = []

            for i, (contents, filename) in enumerate(sorted(zip(contents_list, filenames), key=lambda x: x[1])):
                try:
                    header, content_string = contents.split(',')
                    decoded = base64.b64decode(content_string)
                    data = pd.read_csv(
                        io.StringIO(decoded.decode('utf-8', errors="ignore")),
                        sep=r"\s+",
                        skiprows=1,
                        header=None,
                        names=["residuo", "Phi", "Psi"]
                    )

                    time_label = f'time_{i}'

                    # Aggiungi Phi
                    for _, row in data.iterrows():
                        data_list.append([row['residuo'], 'Phi', time_label, row['Phi']])

                    # Aggiungi Psi
                    for _, row in data.iterrows():
                        data_list.append([row['residuo'], 'Psi', time_label, row['Psi']])

                except Exception as e:
                    return html.Div(f"Errore nel file {filename}: {str(e)}", style={'color': 'red'}), None, \
                        {'display': 'none'}, {'display': 'none'}

            # 2️⃣ Creazione DataFrame long
            combined_df = pd.DataFrame(data_list, columns=['residuo', 'angolo', 'tempo', 'valore'])

            # 3️⃣ Pivot → wide format
            pivot_df = combined_df.pivot_table(
                index=['residuo', 'angolo'],
                columns='tempo',
                values='valore'
            ).reset_index()

            # 4️⃣ Ordina colonne temporali
            time_columns = sorted(
                [c for c in pivot_df.columns if c.startswith('time_')],
                key=lambda x: int(x.split('_')[1])
            )
            reordered_df = pivot_df[['residuo', 'angolo'] + time_columns]

            # 5️⃣ Salva CSV (formato WIDE come il prof)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = f'combined_series_sorted_{timestamp}.csv'
            reordered_df.to_csv(output_csv, index=False)
            print(f"✅ CSV salvato: {output_csv}")

            # 6️⃣ Statistiche e plot (usa combined_df long format)
            # 🔧 CALCOLA STATISTICHE PER TEMPO (non per angolo!)
            time_columns = sorted(
                [c for c in combined_df['tempo'].unique()],
                key=lambda x: int(x.split('_')[1])
            )

            # Separa Phi e Psi
            phi_data = combined_df[combined_df['angolo'] == 'Phi']['valore'].values
            psi_data = combined_df[combined_df['angolo'] == 'Psi']['valore'].values

            # Crea DataFrame per statistiche
            df_stats = pd.DataFrame({
                'Componente': ['Phi', 'Psi'],
                'mean': [phi_data.mean(), psi_data.mean()],
                'max': [phi_data.max(), psi_data.max()],
                'min': [phi_data.min(), psi_data.min()],
                'var': [phi_data.var(), psi_data.var()],
                'std': [phi_data.std(), psi_data.std()]
            })

            table = dash_table.DataTable(
                data=df_stats.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df_stats.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center', 'padding': '8px'},  # ✅ Centra i dati
                style_header={'fontWeight': 'bold', 'backgroundColor': '#7a42ff', 'color': 'white'},
                style_data={'backgroundColor': '#ffffff', 'color': '#222'},
            )

            # 7️⃣ Plot temporale (usa tempo numerico estratto da time_label)
            combined_df['Time'] = combined_df['tempo'].str.split('_').str[1].astype(int)

            fig_temporal = go.Figure()
            for angolo in ['Phi', 'Psi']:
                df_angolo = combined_df[combined_df['angolo'] == angolo]
                fig_temporal.add_trace(go.Scatter(
                    x=df_angolo['Time'],
                    y=df_angolo['valore'],
                    mode='lines',
                    name=angolo
                ))
            fig_temporal = apply_light_theme(fig_temporal, "Andamento temporale angoli")

            # 8️⃣ Scatter 2D (Phi vs Psi per ogni tempo)
            phi_df = combined_df[combined_df['angolo'] == 'Phi'][['residuo', 'tempo', 'valore', 'Time']].rename(
                columns={'valore': 'Phi'})
            psi_df = combined_df[combined_df['angolo'] == 'Psi'][['residuo', 'tempo', 'valore']].rename(
                columns={'valore': 'Psi'})
            scatter_df = phi_df.merge(psi_df, on=['residuo', 'tempo'])

            fig_scatter = px.scatter(
                scatter_df,
                x="Phi",
                y="Psi",
                color="Time",
                labels={"Phi": "Phi (°)", "Psi": "Psi (°)"},
                title="Scatter 2D: Phi vs Psi"
            )
            fig_scatter = apply_light_theme(fig_scatter)

            # 9️⃣ Salva wide format nello store (per PCA)
            stored_data = reordered_df.to_json(date_format='iso', orient='split')

            return html.Div([
                html.H3("📊 Statistiche Dati Raw"),
                table,
                html.Hr(),
                dcc.Graph(figure=fig_temporal),
                html.Hr(),
                dcc.Graph(figure=fig_scatter)
            ]), stored_data, {'display': 'block'}, {'display': 'block'}

        except Exception as e:
            tb = traceback.format_exc()
            return html.Div([
                html.B("Errore nell'analisi raw:"),
                html.Pre(str(e)),
                html.Pre(tb)
            ], style={'color': 'red'}), None, {'display': 'none'}, {'display': 'none'}

    # === CALLBACK 2: Toggle campo num_components ===
    @app.callback(
        Output('num-components-container', 'style'),
        Input('pca-toggle', 'value')
    )
    def toggle_num_components(pca_toggle):
        if pca_toggle and 'pca' in pca_toggle:
            return {'display': 'block'}
        return {'display': 'none'}


    # === CALLBACK 3: Applica Preprocessing (PCA) ===
    @app.callback(
        Output('pca-output', 'children'),
        Output('stored-pca-data', 'data'),
        Output('pca-section', 'style'),
        Output('clustering-section', 'style'),
        Output('anomaly-section', 'style'),
        Input('apply-preprocessing-button', 'n_clicks'),
        State('stored-raw-data', 'data'),
        State('pca-toggle', 'value'),
        State('num-components', 'value'),
        State('preprocessing-options', 'value')
    )
    def apply_preprocessing(n_clicks, stored_raw_json, pca_toggle, num_components, preprocessing_options):
        if not n_clicks or n_clicks == 0:
            return "", None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

        if not stored_raw_json:
            return html.Div("Nessun dato raw disponibile. Esegui prima 'Analizza Dati Raw'.", style={'color': 'red'}), \
                None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

        pca_enabled = bool(pca_toggle and 'pca' in pca_toggle)

        if not pca_enabled:
            # Se PCA non è abilitata, usa i dati raw come fallback
            df_raw = pd.read_json(io.StringIO(stored_raw_json), orient='split')
            return html.Div("PCA non abilitata. Utilizza i dati raw per clustering/anomaly.",
                            style={'color': 'orange'}), \
                df_raw.to_json(date_format='iso', orient='split'), \
                {'display': 'none'}, {'display': 'block'}, {'display': 'block'}

        try:
            num_components = int(num_components) if num_components else 3
            if num_components < 1:
                num_components = 3
        except Exception:
            num_components = 3

        use_sin_cos = bool(preprocessing_options and 'sincos' in preprocessing_options)

        try:
            # Carica wide format dal store
            df_wide = pd.read_json(io.StringIO(stored_raw_json), orient='split')

            # Rimuovi metadata
            df_pca = df_wide.drop(['residuo', 'angolo'], axis=1)

            # Trasponi: (residui, tempi) → (tempi, residui)
            window_data = df_pca.T  # Ora le righe sono i tempi, le colonne sono i residui

            # Sin/Cos (opzionale)
            if use_sin_cos:
                from logic.pca_utils import add_sin_cos_columns
                window_trans = add_sin_cos_columns(window_data)
            else:
                window_trans = window_data

            # PCA globale (tutti i tempi insieme)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=num_components)
            pca_result = pca.fit_transform(window_trans)  # Shape: (n_tempi, n_components)

            # Crea DataFrame risultati
            pc_cols = [f'PC{i + 1}' for i in range(num_components)]
            pca_df = pd.DataFrame(pca_result, columns=pc_cols)

            # Estrai Time dai nomi colonne (time_0 → 0, time_1 → 1, ...)
            time_values = [int(col.split('_')[1]) for col in df_pca.columns]
            pca_df['Time'] = time_values  # ✅ Usa tempo reale (0-240)

            if pca_df.empty:
                return html.Div("PCA non ha prodotto risultati.", style={'color': 'red'}), None, \
                    {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

            # Statistiche PC
            stats = pd.DataFrame({
                "Media": pca_df[pc_cols].mean(),
                "Massimo": pca_df[pc_cols].max().round(4),
                "Minimo": pca_df[pc_cols].min().round(4),
                "Varianza": pca_df[pc_cols].var().round(4),
                "Deviazione Std": pca_df[pc_cols].std().round(4),
            })
            stats.insert(0, "Componente", stats.index)

            table = dash_table.DataTable(
                data=stats.to_dict('records'),
                columns=[{"name": i, "id": i} for i in stats.columns],
                style_table={'overflowX': 'hidden'},
                style_cell={'textAlign': 'center', 'padding': '8px'},
                style_header={'backgroundColor': '#7a42ff', 'color': 'white'},
                style_data={'color': '#222', 'backgroundColor': '#ffffff'},
                page_action='none'
            )

            # Grafico temporale
            fig_temporal = go.Figure()
            for pc in pc_cols:
                fig_temporal.add_trace(go.Scatter(x=pca_df['Time'], y=pca_df[pc], mode='lines', name=pc))

            # 🔧 Applica tema con asse Y invertito
            fig_temporal = apply_light_theme(fig_temporal, "Andamento temporale PC", reverse_y=True)

            # Scatter 3D (PC1, PC2, PC3)
            fig_scatter_3d = go.Figure()
            if len(pc_cols) >= 3:
                fig_scatter_3d.add_trace(go.Scatter3d(
                    x=pca_df[pc_cols[0]],
                    y=pca_df[pc_cols[1]],
                    z=pca_df[pc_cols[2]],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=pca_df['Time'],
                        colorscale='Viridis',
                        colorbar=dict(title='Time'),
                        showscale=True
                    ),
                    text=pca_df['Time'],
                    hovertemplate=f'{pc_cols[0]}: %{{x:.2f}}<br>{pc_cols[1]}: %{{y:.2f}}<br>{pc_cols[2]}: %{{z:.2f}}<br>Time: %{{text}}<extra></extra>'
                ))
                fig_scatter_3d.update_layout(
                    title="Scatter Plot 3D (PC1, PC2, PC3)",
                    scene=dict(
                        xaxis_title=pc_cols[0],
                        yaxis_title=pc_cols[1],
                        zaxis_title=pc_cols[2],
                        xaxis=dict(gridcolor="#ddd", backgroundcolor="#fff"),
                        yaxis=dict(gridcolor="#ddd", backgroundcolor="#fff"),
                        zaxis=dict(gridcolor="#ddd", backgroundcolor="#fff")
                    ),
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    font=dict(color="#222")
                )

            # Parallel coordinates plot
            dimensions = []
            for col in pc_cols:
                dimensions.append(dict(
                    label=col,
                    values=pca_df[col]
                ))
            dimensions.append(dict(
                label='Time',
                values=pca_df['Time']
            ))

            fig_parallel = go.Figure(data=go.Parcoords(
                line=dict(
                    color=pca_df['Time'],
                    colorscale='Viridis',
                    showscale=True,
                    cmin=pca_df['Time'].min(),
                    cmax=pca_df['Time'].max()
                ),
                dimensions=dimensions
            ))
            fig_parallel = apply_light_theme(fig_parallel, "Parallel Coordinates Plot")

            return html.Div([
                table,
                dcc.Graph(figure=fig_temporal),
                html.Hr(style={'margin': '30px 0'}),
                dcc.Graph(figure=fig_scatter_3d) if len(pc_cols) >= 3 else html.Div(),
                dcc.Graph(figure=fig_parallel),
            ]), pca_df.to_json(date_format='iso', orient='split'), \
                {'display': 'block'}, {'display': 'block'}, {'display': 'block'}

        except Exception as e:
            tb = traceback.format_exc()
            return html.Div([
                html.B("Errore nel preprocessing PCA:"),
                html.Pre(str(e)),
                html.Pre(tb)
            ], style={'color': 'red'}), None, \
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

    # === CALLBACK 4: Lista file caricati ===
    @app.callback(
        Output('file-list', 'children'),
        Input('upload-files', 'filename')
    )
    def update_file_list(filenames):
        if filenames:
            return [
                html.Li(f"📄 {name}", style={
                    'padding': '4px 10px',
                    'backgroundColor': '#efe6ff',
                    'borderRadius': '15px',
                    'whiteSpace': 'nowrap',
                    'boxShadow': '0 0 5px rgba(122, 66, 255, 0.35)',
                    'marginRight': '8px',
                    'color': '#333',
                    'fontWeight': '600',
                }) for name in filenames
            ]
        return html.Li("Nessun file caricato", style={'color': '#888', 'fontStyle': 'italic'})

    # === CALLBACK 5: Dropdown dinamico modelli anomaly ===
    @app.callback(
        Output('anomaly-algorithm', 'options'),
        Output('anomaly-algorithm', 'value'),
        Input('anomaly-category', 'value'),
        State('anomaly-algorithm', 'value')
    )
    def update_anomaly_model_options(category, current_value):
        if category == 'regression':
            opts = [
                {'label': 'Linear Regression', 'value': 'linreg'},
                {'label': 'Linear Regression + Bagging (Sin/Cos)', 'value': 'linreg_bagging'},
                {'label': 'Random Forest + Bagging', 'value': 'rf_bagging'},
                {'label': 'Gradient Boosting + Bagging', 'value': 'gb_bagging'},
                {'label': 'Extra Trees + Bagging', 'value': 'et_bagging'},
            ]
        elif category == 'distance':
            opts = [
                {'label': 'Local Outlier Factor (LOF)', 'value': 'lof'},
                {'label': 'Matrix Profile (STUMP)', 'value': 'matrix_profile'},
            ]
        elif category == 'clustering':
            opts = [
                {'label': 'DBSCAN (vector score)', 'value': 'dbscan_vector'},
                {'label': 'OPTICS (reachability score)', 'value': 'optics_vector'},
                {'label': 'K-Means (distance score)', 'value': 'kmeans_vector'},
            ]
        else:
            opts = [{'label': 'Linear Regression', 'value': 'linreg'}]

        values = {o['value'] for o in opts}
        new_value = current_value if current_value in values else opts[0]['value']
        return opts, new_value

    # === CALLBACK 6: Toggle gruppi parametri anomaly ===
    @app.callback(
        Output('params-bagging', 'style'),
        Output('params-rf', 'style'),
        Output('params-gb', 'style'),
        Output('params-et', 'style'),
        Output('params-lof', 'style'),
        Output('params-mp', 'style'),
        Output('params-dbscan-clustering', 'style'),
        Output('params-optics-clustering', 'style'),
        Output('params-kmeans-clustering', 'style'),
        Input('anomaly-algorithm', 'value')
    )
    def toggle_anomaly_param_groups(algorithm):
        bagging_models = {'linreg_bagging', 'rf_bagging', 'gb_bagging', 'et_bagging'}

        def style(show):
            return {'marginBottom': '15px'} if show else {'display': 'none', 'marginBottom': '15px'}

        return (
            style(algorithm in bagging_models),
            style(algorithm == 'rf_bagging'),
            style(algorithm == 'gb_bagging'),
            style(algorithm == 'et_bagging'),
            style(algorithm == 'lof'),
            style(algorithm == 'matrix_profile'),
            style(algorithm == 'dbscan_vector'),
            style(algorithm == 'optics_vector'),
            style(algorithm == 'kmeans_vector'),
        )


    # --- Clustering ---
    @app.callback(
        Output('clustering-output', 'children'),
        Input('proceed-button', 'n_clicks'),
        State('stored-pca-data', 'data'),
        State('clustering-algorithm', 'value')
    )
    def perform_clustering(n_clicks, stored_pca_json, algorithm):
        if not n_clicks or not stored_pca_json:
            return no_update
        try:
            pca_df = pd.read_json(io.StringIO(stored_pca_json), orient='split')
            pc_cols = [c for c in pca_df.columns if c.startswith("PC")]

            if algorithm == 'dbscan':
                pca_df = dbscan_clustering(pca_df[pc_cols])
            elif algorithm == 'optics':
                pca_df = optics_clustering(pca_df[pc_cols])
            elif algorithm == 'spectral':
                pca_df = spectral_clustering(pca_df[pc_cols])

            fig_cluster = go.Figure()
            for cluster in pca_df['Cluster'].unique():
                cluster_points = pca_df[pca_df['Cluster'] == cluster]
                fig_cluster.add_trace(go.Scatter3d(
                    x=cluster_points[pc_cols[0]],
                    y=cluster_points[pc_cols[1]] if len(pc_cols) > 1 else [0]*len(cluster_points),
                    z=cluster_points[pc_cols[2]] if len(pc_cols) > 2 else [0]*len(cluster_points),
                    mode='markers',
                    marker=dict(size=4),
                    name=f'Cluster {cluster}'
                ))
            fig_cluster = apply_light_theme(fig_cluster, f"{algorithm.upper()} Clustering")
            return html.Div([dcc.Graph(figure=fig_cluster)])
        except Exception as e:
            tb = traceback.format_exc()
            return html.Div([html.B("Errore clustering:"), html.Pre(str(e)), html.Pre(tb)])

    # --- Anomaly Detection ---
    @app.callback(
        Output('anomaly-output', 'children'),
        Input('anomaly-button', 'n_clicks'),
        State('stored-pca-data', 'data'),
        State('anomaly-algorithm', 'value'),
        # Parametri comuni
        State('param-train-n', 'value'),
        State('param-window-size', 'value'),
        State('param-pen', 'value'),
        # Bagging
        State('param-num-models', 'value'),
        # RF
        State('rf-n-estimators', 'value'),
        State('rf-max-depth', 'value'),
        State('rf-min-split', 'value'),
        # GB
        State('gb-n-estimators', 'value'),
        State('gb-learning-rate', 'value'),
        State('gb-max-depth', 'value'),
        # ET
        State('et-n-estimators', 'value'),
        State('et-max-depth', 'value'),
        State('et-min-split', 'value'),
        # LOF
        State('lof-n-neighbors', 'value'),
        State('lof-contamination', 'value'),
        # --- aggiunte: stati parametri clustering ---
        State('dbscan-eps', 'value'),
        State('dbscan-min-samples', 'value'),
        State('dbscan-knn-k', 'value'),
        State('optics-min-samples', 'value'),
        State('optics-xi', 'value'),
        State('optics-min-cluster-size', 'value'),
        State('kmeans-n-clusters', 'value'),
        State('kmeans-random-state', 'value'),
    )
    def perform_anomaly_detection(n_clicks, stored_pca_json, model_name,
                                  n_train, w_size, pen_value,
                                  num_models,
                                  rf_n_estimators, rf_max_depth, rf_min_split,
                                  gb_n_estimators, gb_learning_rate, gb_max_depth,
                                  et_n_estimators, et_max_depth, et_min_split,
                                  lof_n_neighbors, lof_contamination,
                                  db_eps, db_min_samples, db_knn_k,
                                  op_min_samples, op_xi, op_min_cluster_size,
                                  km_n_clusters, km_random_state):
        if not n_clicks or not stored_pca_json:
            return no_update

        def int_or(v, d):
            try:
                return d if v in (None, '') else int(v)
            except Exception:
                return d

        def float_or(v, d):
            try:
                return d if v in (None, '') else float(v)
            except Exception:
                return d

        n_train = int_or(n_train, 180)
        w_size = int_or(w_size, 20)
        pen_value = int_or(pen_value, 1)
        num_models = int_or(num_models, 10)

        if n_train < 30:
            return html.Div("n troppo basso (>=30).", style={'color': 'red'})
        if w_size < 3:
            return html.Div("w troppo basso (>=3).", style={'color': 'red'})

        try:
            import logic.anomaly_detection as ad_mod
            original_calc = ad_mod.calculate_variable_thresholds

            def _calc_with_custom_pen(errors, pen=pen_value):
                return original_calc(errors, pen=pen_value)

            ad_mod.calculate_variable_thresholds = _calc_with_custom_pen

            pca_df = pd.read_json(io.StringIO(stored_pca_json), orient='split')
            if n_train >= len(pca_df) - w_size - 5:
                return html.Div("n troppo grande rispetto ai dati.", style={'color': 'red'})

            # --- Regressione e Bagging (esistente) ---
            if model_name == "linreg":
                models, *_ = train_linear_regression(pca_df, n=n_train, w=w_size)
                figs, *_ = detect_anomalies_linear(pca_df, models, n=n_train, w=w_size)

            elif model_name == "linreg_bagging":
                models, *_ = train_linear_regression_bagging(pca_df, n=n_train, w=w_size, num_models=num_models)
                figs, *_ = detect_anomalies_linear_bagging(pca_df, models, n=n_train, w=w_size)

            elif model_name == "rf_bagging":
                models, *_ = train_random_forest_bagging(pca_df, n=n_train, w=w_size, num_models=num_models)
                figs, *_ = detect_anomalies_random_forest_bagging(pca_df, models, n=n_train, w=w_size)

            elif model_name == "gb_bagging":
                models, *_ = train_gradient_boosting_bagging(pca_df, n=n_train, w=w_size, num_models=num_models)
                figs, *_ = detect_anomalies_gradient_boosting_bagging(pca_df, models, n=n_train, w=w_size)

            elif model_name == "et_bagging":
                models, *_ = train_extra_trees_bagging(pca_df, n=n_train, w=w_size, num_models=num_models)
                figs, *_ = detect_anomalies_extra_trees_bagging(pca_df, models, n=n_train, w=w_size)

            # --- Distanza (esistente) ---
            elif model_name == "lof":
                n_neighbors = int_or(lof_n_neighbors, 20)
                contamination = float_or(lof_contamination, 0.05)
                models, *_ = train_lof(pca_df, n=n_train, w=w_size,
                                       n_neighbors=n_neighbors,
                                       contamination=contamination)
                figs, *_ = detect_anomalies_lof(pca_df, models, n=n_train, w=w_size)

            elif model_name == "matrix_profile":
                figs, *_ = detect_matrix_profile(pca_df, n=n_train, w=w_size)

            # --- Clustering-based (nuovi) ---
            elif model_name == "dbscan_vector":
                eps = float_or(db_eps, 0.25)
                min_samples = int_or(db_min_samples, 15)
                knn_k = None if db_knn_k in (None, '') else int_or(db_knn_k, None)
                figs, *_ = detect_anomalies_dbscan(
                    pca_df, n=n_train, w=w_size,
                    eps=eps, min_samples=min_samples, knn_k=knn_k
                )

            elif model_name == "optics_vector":
                min_s = int_or(op_min_samples, 5)
                xi = float_or(op_xi, 0.01)
                min_clu = float_or(op_min_cluster_size, 0.1)
                figs, *_ = detect_anomalies_optics(
                    pca_df, n=n_train, w=w_size,
                    min_samples=min_s, xi=xi, min_cluster_size=min_clu
                )

            elif model_name == "kmeans_vector":
                k = int_or(km_n_clusters, 3)
                rs = int_or(km_random_state, 42)
                models, *_ = train_kmeans(pca_df, n=n_train, w=w_size, n_clusters=k, random_state=rs)
                figs, *_ = detect_anomalies_kmeans(pca_df, models, n=n_train, w=w_size)

            else:
                return html.Div("Modello non supportato.", style={'color': 'red'})

            figs = [apply_light_theme(f) for f in figs]
            return html.Div([dcc.Graph(figure=f) for f in figs])

        except Exception as e:
            tb = traceback.format_exc()
            return html.Div([html.B("Errore anomaly detection:"), html.Pre(str(e)), html.Pre(tb)])
        finally:
            try:
                import logic.anomaly_detection as ad_mod_final
                if 'original_calc' in locals():
                    ad_mod_final.calculate_variable_thresholds = original_calc
            except Exception:
                pass
