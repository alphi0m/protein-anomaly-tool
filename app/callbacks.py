import base64
import io
import pandas as pd
from dash import Input, Output, State, html, dash_table, dcc
import plotly.express as px

def register_callbacks(app):
    @app.callback(
        Output('num-components-container', 'style'),
        Input('pca-toggle', 'value')
    )
    def toggle_num_components(pca_value):
        if 'pca' in pca_value:
            return {'display': 'block', 'marginTop': '10px'}
        return {'display': 'none'}

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

    @app.callback(
        Output('analysis-output', 'children'),
        Input('analyze-button', 'n_clicks'),
        State('upload-files', 'contents'),
        State('upload-files', 'filename')
    )
    def analyze(n_clicks, contents_list, filenames):
        if n_clicks > 0:
            if not contents_list:
                return "Nessun file caricato."

            all_dfs = []
            for contents, filename in zip(contents_list, filenames):
                try:
                    content_type, content_string = contents.split(',')
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
                    return f"Errore nel file {filename}: {e}"

            df_all = pd.concat(all_dfs, ignore_index=True)
            df_num = df_all[["Phi", "Psi"]]

            stats = pd.DataFrame({
                "Media": df_num.mean(),
                "Massimo": df_num.max(),
                "Minimo": df_num.min(),
                "Varianza": df_num.var(),
                "Deviazione Std": df_num.std()
            })
            stats.insert(0, "Componente", stats.index)
            stats.reset_index(drop=True, inplace=True)

            lunghezza = len(df_num)

            # Downsampling per grafico lineare
            df_sampled = df_all.iloc[::10, :]

            fig_line = px.line(
                df_sampled,
                x=df_sampled.index,
                y=["Phi", "Psi"],
                labels={"x": "Frame", "value": "Angolo (°)", "variable": "Componente"},
                title="Andamento Phi e Psi nel tempo (campionato)"
            )
            fig_line.update_layout(plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")

            fig_hist = px.histogram(
                df_num.melt(var_name="Componente", value_name="Valore"),
                x="Valore",
                color="Componente",
                barmode="overlay",
                nbins=40,
                title="Distribuzione degli angoli"
            )
            fig_hist.update_layout(plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e", font_color="white")

            return html.Div([
                html.H3(f"Statistiche del dataset ({len(filenames)} file)"),
                html.P(f"Lunghezza dinamica totale: {lunghezza}"),

                html.Div(
                    dash_table.DataTable(
                        data=stats.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in stats.columns],
                        style_table={'overflowX': 'hidden'},
                        style_cell={
                            'textAlign': 'center',
                            'padding': '8px',
                            'minWidth': '100px',
                            'maxWidth': '150px',
                            'whiteSpace': 'normal',
                            'height': 'auto',
                        },
                        style_header={
                            'backgroundColor': '#7e57c2',
                            'color': 'white',
                            'fontWeight': 'bold',
                            'whiteSpace': 'normal',
                            'textAlign': 'center',
                            'border': 'none',
                        },
                        style_data={
                            'color': 'white',
                            'backgroundColor': '#333',
                        },
                        style_data_conditional=[
                            {'if': {'row_index': 'odd'}, 'backgroundColor': '#2a2a2a'},
                            {'if': {'row_index': 'even'}, 'backgroundColor': '#1e1e1e'},
                        ],
                        page_action='none',
                    ),
                    style={
                        'minWidth': '600px',
                        'maxWidth': '100%',
                        'overflowX': 'auto',
                        'borderRadius': '10px',
                        'boxShadow': '0 4px 10px rgba(187, 134, 252, 0.4)',
                        'overflow': 'hidden',
                        'marginTop': '20px',
                    }
                ),

                html.Hr(),
                dcc.Graph(figure=fig_line),
                html.Hr(),
                dcc.Graph(figure=fig_hist)
            ])
        return ""