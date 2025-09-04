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
    train_linear_regression, detect_anomalies_linear,
    train_linear_regression_bagging, detect_anomalies_linear_bagging,
    train_random_forest_bagging, detect_anomalies_random_forest_bagging,
    train_gradient_boosting_bagging, detect_anomalies_gradient_boosting_bagging,
    train_extra_trees_bagging, detect_anomalies_extra_trees_bagging
)


# ----------- Funzione helper per tema chiaro dei grafici -----------
def apply_light_theme(fig, title=None):
    fig.update_layout(
        title=title,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#222"),
        xaxis=dict(color="#222", gridcolor="#ddd"),
        yaxis=dict(color="#222", gridcolor="#ddd"),
        legend=dict(font=dict(color="#222"))
    )
    return fig


def register_callbacks(app):

    # ----------------- MOSTRA INPUT NUM COMPONENTI -----------------
    @app.callback(
        Output('num-components-container', 'style'),
        Input('pca-toggle', 'value')
    )
    def toggle_num_components(pca_value):
        if pca_value and 'pca' in pca_value:
            return {'display': 'block', 'marginTop': '10px'}
        return {'display': 'none'}

    # ----------------- LISTA FILE CARICATI -----------------
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

    # ----------------- ANALISI (PCA o RAW) -----------------
    @app.callback(
        Output('analysis-output', 'children'),
        Output('stored-pca-data', 'data'),
        Output('pca-section', 'style'),
        Output('clustering-section', 'style'),
        Output('anomaly-section', 'style'),
        Input('analyze-button', 'n_clicks'),
        State('upload-files', 'contents'),
        State('upload-files', 'filename'),
        State('pca-toggle', 'value'),
        State('num-components', 'value'),
        State('preprocessing-options', 'value')
    )
    def analyze(n_clicks, contents_list, filenames, pca_toggle, num_components, preprocessing_options):
        if not n_clicks or n_clicks == 0:
            return "", None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

        if not contents_list:
            return html.Div("Nessun file caricato.", style={'color': 'red'}), None, \
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

        try:
            num_components = int(num_components) if num_components else 3
            if num_components < 1:
                num_components = 3
        except Exception:
            num_components = 3

        pca_enabled = bool(pca_toggle and 'pca' in pca_toggle)
        use_sin_cos = bool(preprocessing_options and 'sincos' in preprocessing_options)

        # ----------------- CARICAMENTO FILE -----------------
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
                return html.Div(f"Errore nel file {filename}: {str(e)}", style={'color': 'red'}), None, \
                    {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

        df_all = pd.concat(all_dfs, ignore_index=True)

        # ----------------- PCA -----------------
        if pca_enabled:
            try:
                n_frames = df_all.shape[0]
                if n_frames < 2:
                    return html.Div("Dataset troppo piccolo per PCA.", style={'color': 'red'}), None, \
                        {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

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
                    return html.Div("PCA non ha prodotto risultati.", style={'color': 'red'}), None, \
                        {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

                pc_cols = [c for c in pca_df.columns if c.startswith("PC")]

                # STATISTICHE
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

                # GRAFICO TEMPORALE
                fig_temporal = go.Figure()
                for pc in pc_cols:
                    fig_temporal.add_trace(go.Scatter(x=pca_df['Time'], y=pca_df[pc], mode='lines', name=pc))
                fig_temporal = apply_light_theme(fig_temporal, "Andamento temporale PC")

                return html.Div([
                    table,
                    dcc.Graph(figure=fig_temporal),
                ]), pca_df.to_json(date_format='iso', orient='split'), \
                    {'display': 'block'}, {'display': 'block'}, {'display': 'block'}

            except Exception as e:
                tb = traceback.format_exc()
                return html.Div([html.B("Errore PCA:"), html.Pre(str(e)), html.Pre(tb)]), None, \
                    {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

        # ----------------- RAW -----------------
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

            table = dash_table.DataTable(
                data=stats.to_dict('records'),
                columns=[{"name": i, "id": i} for i in stats.columns],
                style_cell={'textAlign': 'center', 'padding': '8px'},
                style_header={'backgroundColor': '#7a42ff', 'color': 'white'},
                style_data={'color': '#222', 'backgroundColor': '#ffffff'}
            )

            fig_line = px.line(df_all.iloc[::10, :], x=df_all.index[::10], y=["Phi", "Psi"],
                               title="Andamento Phi e Psi (campionato)")
            fig_line = apply_light_theme(fig_line)

            return html.Div([table, dcc.Graph(figure=fig_line)]), None, \
                {'display': 'block'}, {'display': 'block'}, {'display': 'block'}

    # ----------------- CLUSTERING -----------------
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

            # clustering
            if algorithm == 'dbscan':
                pca_df = dbscan_clustering(pca_df[pc_cols])
            elif algorithm == 'optics':
                pca_df = optics_clustering(pca_df[pc_cols])
            elif algorithm == 'spectral':
                pca_df = spectral_clustering(pca_df[pc_cols])

            # grafico cluster
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

    # ----------------- ANOMALY DETECTION -----------------
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
            pca_df = pd.read_json(io.StringIO(stored_pca_json), orient='split')

            if model_name == "linreg":
                models, *_ = train_linear_regression(pca_df)
                figs, _, _ = detect_anomalies_linear(pca_df, models)

            elif model_name == "linreg_bagging":
                models, *_ = train_linear_regression_bagging(pca_df)
                figs, *_ = detect_anomalies_linear_bagging(pca_df, models)

            elif model_name == "rf_bagging":
                models, *_ = train_random_forest_bagging(pca_df)
                figs, *_ = detect_anomalies_random_forest_bagging(pca_df, models)

            elif model_name == "gb_bagging":
                models, *_ = train_gradient_boosting_bagging(pca_df)
                figs, *_ = detect_anomalies_gradient_boosting_bagging(pca_df, models)

            elif model_name == "et_bagging":
                models, *_ = train_extra_trees_bagging(pca_df)
                figs, *_ = detect_anomalies_extra_trees_bagging(pca_df, models)

            else:
                return html.Div("Modello non ancora implementato.")

            figs = [apply_light_theme(fig) for fig in figs]
            return html.Div([dcc.Graph(figure=fig) for fig in figs])

        except Exception as e:
            tb = traceback.format_exc()
            return html.Div([html.B("Errore anomaly detection:"), html.Pre(str(e)), html.Pre(tb)])