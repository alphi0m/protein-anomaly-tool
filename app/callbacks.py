from dash import Input, Output, State, ctx

def register_callbacks(app):
    @app.callback(
        Output('num-components-container', 'style'),
        Input('analysis-options', 'value')
    )
    def toggle_num_components(options):
        if 'pca' in options:
            return {'display': 'block', 'marginTop': '10px'}
        return {'display': 'none'}

    @app.callback(
        Output('analysis-output', 'children'),
        Input('analyze-button', 'n_clicks'),
        State('upload-file', 'filename')
    )
    def analyze(n_clicks, filename):
        if n_clicks > 0:
            if filename:
                return f"Analisi in corso sul file: {filename}"
            else:
                return "Nessun file caricato."
        return ""