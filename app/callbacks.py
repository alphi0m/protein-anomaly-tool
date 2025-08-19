import base64
import io
import traceback
import numpy as np
import pandas as pd
from dash import Input, Output, State, html, dash_table, dcc, no_update
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler

from logic.pca_utils import compute_windowed_pca
from logic.clustering_utils import dbscan_clustering, optics_clustering, spectral_clustering

from logic.anomaly_detection import (
    train_linear_regression, detect_anomalies_linear, train_linear_regression_bagging, detect_anomalies_linear_bagging,
    train_random_forest_bagging, detect_anomalies_random_forest_bagging, train_gradient_boosting_bagging,
    detect_anomalies_gradient_boosting_bagging, train_extra_trees_bagging, detect_anomalies_extra_trees_bagging
)


def register_callbacks(app):

    # Mostra/nasconde input numero componenti
    @app.callback(
        Output('num-components-container', 'style'),
        Input('pca-toggle', 'value')
    )
    def toggle_num_components(pca_value):
        if pca_value and 'pca' in pca_value:
            return {'display': 'block', 'marginTop': '10px'}
        return {'display': 'none'}

    # Lista dei file caricati
    @app.callback(
        Output('file-list', 'children'),
        Input('upload-files', 'filename')
    )
    def update_file_list(filenames):
        if filenames:
            return [
                html.Li(f"📄 {name}", style={
                    'padding': '4px 10px',
                    'backgroundColor': '#3a2a8f',
                    'borderRadius': '15px',
                    'whiteSpace': 'nowrap',
                    'boxShadow': '0 0 5px rgba(187, 134, 252, 0.5)',
                    'marginRight': '8px',
                    'color': '#e0dfff',
                    'fontWeight': '600',
                }) for name in filenames
            ]
        return html.Li("Nessun file caricato", style={'color': '#888', 'fontStyle': 'italic'})

    # ----------------- CALLBACK ANALISI (PCA / RAW) -----------------
    @app.callback(
        Output('analysis-output', 'children'),
        Output('stored-pca-data', 'data'),
        Input('analyze-button', 'n_clicks'),
        State('upload-files', 'contents'),
        State('upload-files', 'filename'),
        State('pca-toggle', 'value'),
        State('num-components', 'value'),
        State('preprocessing-options', 'value')
    )
    def analyze(n_clicks, contents_list, filenames, pca_toggle, num_components, preprocessing_options):
        if not n_clicks or n_clicks == 0:
            return "", None

        if not contents_list:
            return html.Div("Nessun file caricato.", style={'color': '#ffcccc'}), None

        try:
            num_components = int(num_components) if num_components else 3
            if num_components < 1:
                num_components = 3
        except Exception:
            num_components = 3

        pca_enabled = bool(pca_toggle and 'pca' in pca_toggle)
        use_sin_cos = bool(preprocessing_options and 'sincos' in preprocessing_options)

        # Caricamento file
        all_dfs = []
        for contents, filename in zip(contents_list, filenames):
            try:
                header, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8', errors="ignore")),
                    sep=r"\s+",
                    skiprows=1,
                    header=None,
                    names=["Residuo", "Phi", "Psi"]
                )
                all_dfs.append(df)
            except Exception as e:
                return html.Div(f"Errore nel file {filename}: {str(e)}", style={'color': '#ffaaaa'}), None

        df_all = pd.concat(all_dfs, ignore_index=True)

        # ---------- PCA ----------
        if pca_enabled:
            try:
                n_frames = df_all.shape[0]
                if n_frames < 2:
                    return html.Div("Dataset troppo piccolo per PCA.", style={'color': '#ffaaaa'}), None

                window_size = min(30, n_frames)
                overlap = False

                pca_df = compute_windowed_pca(
                    df_all[["Phi", "Psi"]],
                    window_size=window_size,
                    n_components=num_components,
                    use_sin_cos=use_sin_cos,
                    overlap=overlap
                )

                if pca_df.empty:
                    return html.Div("PCA non ha prodotto risultati.", style={'color': '#ffaaaa'}), None

                pc_cols = [c for c in pca_df.columns if c.startswith("PC")]

                # Stats
                stats = pd.DataFrame({
                    "Media": pca_df[pc_cols].mean(),
                    "Massimo": pca_df[pc_cols].max().round(4),
                    "Minimo": pca_df[pc_cols].min().round(4),
                    "Varianza": pca_df[pc_cols].var().round(4),
                    "Deviazione Std": pca_df[pc_cols].std().round(4),
                })
                stats.insert(0, "Componente", stats.index)

                # Grafico temporale
                fig_temporal = go.Figure()
                for pc in pc_cols:
                    fig_temporal.add_trace(go.Scatter(x=pca_df['Time'], y=pca_df[pc], mode='lines', name=pc))
                fig_temporal.update_layout(
                    title='Andamento temporale PC',
                    xaxis_title='Time',
                    yaxis_title='Valore',
                    plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white"
                )

                # 3D PCA
                x, y, z = pca_df[pc_cols[0]], \
                          pca_df[pc_cols[1]] if len(pc_cols) > 1 else np.zeros(len(pca_df)), \
                          pca_df[pc_cols[2]] if len(pc_cols) > 2 else np.zeros(len(pca_df))
                fig3d = go.Figure(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers+lines',
                    marker=dict(size=4, color=pca_df["Time"], colorscale='Viridis')
                ))
                fig3d.update_layout(scene=dict(
                    xaxis_title=pc_cols[0],
                    yaxis_title=pc_cols[1] if len(pc_cols) > 1 else '',
                    zaxis_title=pc_cols[2] if len(pc_cols) > 2 else ''
                ), plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")

                # Parallel coords
                scaler = StandardScaler()
                pca_norm = pd.DataFrame(scaler.fit_transform(pca_df[pc_cols]), columns=pc_cols)
                pca_norm["Time"] = pca_df["Time"]
                fig_parallel = px.parallel_coordinates(
                    pca_norm, dimensions=pc_cols, color="Time",
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title="Parallel Coordinates"
                )
                fig_parallel.update_layout(plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")

                # Radar chart
                fig_radar = go.Figure()
                for i in range(len(pca_norm)):
                    fig_radar.add_trace(go.Scatterpolar(
                        r=pca_norm.iloc[i][pc_cols].values.tolist(),
                        theta=pc_cols, fill='toself', opacity=0.15
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
                    showlegend=False, title="Radar Chart",
                    plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white"
                )

                # Tabella
                table = dash_table.DataTable(
                    data=stats.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in stats.columns],
                    style_table={'overflowX': 'hidden'},
                    style_cell={'textAlign': 'center', 'padding': '8px'},
                    style_header={'backgroundColor': '#7e57c2', 'color': 'white'},
                    style_data={'color': 'white', 'backgroundColor': '#333'},
                    page_action='none'
                )

                return html.Div([
                    html.H3("Statistiche PCA"),
                    table,
                    dcc.Graph(figure=fig_temporal),
                    dcc.Graph(figure=fig3d),
                    dcc.Graph(figure=fig_parallel),
                    dcc.Graph(figure=fig_radar)
                ]), pca_df.to_json(date_format='iso', orient='split')

            except Exception as e:
                tb = traceback.format_exc()
                return html.Div([
                    html.B("Errore PCA:"), html.Pre(str(e)), html.Pre(tb)
                ], style={'color': '#ffaaaa'}), None

        # ---------- RAW (Phi/Psi) ----------
        else:
            df_num = df_all[["Phi", "Psi"]]
            stats = pd.DataFrame({
                "Media": df_num.mean(),
                "Massimo": df_num.max(),
                "Minimo": df_num.min(),
                "Varianza": df_num.var(),
                "Deviazione Std": df_num.std(),
            })
            stats.insert(0, "Componente", stats.index)

            fig_line = px.line(df_all.iloc[::10, :], x=df_all.index[::10], y=["Phi", "Psi"],
                               title="Andamento Phi e Psi (campionato)")
            fig_line.update_layout(plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")

            fig_hist = px.histogram(df_num.melt(var_name="Componente", value_name="Valore"),
                                    x="Valore", color="Componente", nbins=40, barmode="overlay",
                                    title="Distribuzione Phi/Psi")
            fig_hist.update_layout(plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")

            fig_temporal = go.Figure()
            for col in ["Phi", "Psi"]:
                fig_temporal.add_trace(go.Scatter(x=df_num.index, y=df_num[col], mode='lines', name=col))
            fig_temporal.update_layout(title='Andamento temporale Phi/Psi',
                                       xaxis_title='Frame', yaxis_title='Valore',
                                       plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")

            table = dash_table.DataTable(
                data=stats.to_dict('records'),
                columns=[{"name": i, "id": i} for i in stats.columns],
                style_cell={'textAlign': 'center', 'padding': '8px'},
                style_header={'backgroundColor': '#7e57c2', 'color': 'white'},
                style_data={'color': 'white', 'backgroundColor': '#333'}
            )

            return html.Div([
                html.H3("Statistiche dataset"),
                table,
                dcc.Graph(figure=fig_temporal),
                dcc.Graph(figure=fig_line),
                dcc.Graph(figure=fig_hist)
            ]), None

    # ----------------- CALLBACK CLUSTERING -----------------
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

            # Clustering
            if algorithm == 'dbscan':
                pca_df = dbscan_clustering(pca_df[pc_cols])
            elif algorithm == 'optics':
                pca_df = optics_clustering(pca_df[pc_cols])
            elif algorithm == 'spectral':
                pca_df = spectral_clustering(pca_df[pc_cols])

            # Grafico cluster
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
            fig_cluster.update_layout(
                title=f"{algorithm.upper()} Clustering",
                plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white"
            )

            return html.Div([dcc.Graph(figure=fig_cluster)])

        except Exception as e:
            tb = traceback.format_exc()
            return html.Div([
                html.B("Errore clustering:"), html.Pre(str(e)), html.Pre(tb)
            ], style={'color': '#ffaaaa'})

    # ----------------- CALLBACK ANOMALY DETECTION -----------------
    @app.callback(
        Output('anomaly-output', 'children'),
        Input('anomaly-button', 'n_clicks'),
        State('stored-pca-data', 'data'),
        State('anomaly-algorithm', 'value')
    )
    def perform_anomaly_detection(n_clicks, stored_pca_json, model_name):
        if not n_clicks or not stored_pca_json:
            return no_update

        try:
            # Recupera dati PCA
            pca_df = pd.read_json(io.StringIO(stored_pca_json), orient='split')
            pc_cols = [c for c in pca_df.columns if c.startswith("PC")]

            # ------------------- Linear Regression semplice -------------------
            if model_name == "linreg":
                models, train_thresholds, train_squared_thresholds = train_linear_regression(pca_df)
                figs, errors, sq_errors = detect_anomalies_linear(pca_df, models)
                return html.Div([dcc.Graph(figure=fig) for fig in figs])

            # ------------- Linear Regression + Bagging (Sin/Cos) -------------
            elif model_name == "linreg_bagging":
                models, train_thresholds, train_squared_thresholds = train_linear_regression_bagging(pca_df)
                figs, errors, sq_errors, thresholds, sq_thresholds, variable_thresholds, variable_sq_thresholds = detect_anomalies_linear_bagging(
                    pca_df, models)
                return html.Div([dcc.Graph(figure=fig) for fig in figs])

            # ---------------- Random Forest Regression + Bagging ----------------
            elif model_name == "rf_bagging":
                models, train_thresholds, train_squared_thresholds = train_random_forest_bagging(pca_df)
                figs, errors, sq_errors, thresholds, sq_thresholds, variable_thresholds, variable_sq_thresholds = detect_anomalies_random_forest_bagging(
                    pca_df, models)
                return html.Div([dcc.Graph(figure=fig) for fig in figs])

            # ------------- Gradient Boosting + Bagging ----------------
            elif model_name == "gb_bagging":
                models, train_thresholds, train_squared_thresholds = train_gradient_boosting_bagging(pca_df)
                figs, errors, sq_errors, thresholds, sq_thresholds, variable_thresholds, variable_sq_thresholds, predictions, prediction_intervals = detect_anomalies_gradient_boosting_bagging(
                    pca_df, models)
                return html.Div([dcc.Graph(figure=fig) for fig in figs])

            # ------------- Extra Tree Regressor + Bagging ----------------
            elif model_name == "et_bagging":
                models, train_thresholds, train_squared_thresholds = train_extra_trees_bagging(pca_df)
                figs, errors, sq_errors, thresholds, sq_thresholds, variable_thresholds, variable_sq_thresholds, predictions, prediction_intervals = detect_anomalies_extra_trees_bagging(
                    pca_df, models)
                return html.Div([dcc.Graph(figure=fig) for fig in figs])

            # Altri modelli ancora non implementati
            else:
                return html.Div("Modello non ancora implementato.")

        except Exception as e:
            tb = traceback.format_exc()
            return html.Div([
                html.B("Errore anomaly detection:"),
                html.Pre(str(e)),
                html.Pre(tb, style={'whiteSpace': 'pre-wrap', 'maxHeight': '300px', 'overflowY': 'auto'})
            ], style={'color': '#ffaaaa'})