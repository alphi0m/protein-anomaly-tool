from dash import Input, Output, State, ctx

def register_callbacks(app):
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