from dash import html, dcc

layout = html.Div([

    html.Div([
        html.H1("Protein Anomaly Detector", className="app-title"),
        html.P("Carica una simulazione molecolare per analizzare le dinamiche della proteina.",
               className="description"),

        dcc.Upload(
            id='upload-files',
            multiple=True,
            children=html.Div([
                html.Div(id='upload-text', children=[
                    'Trascina qui i file o ',
                    html.A('selezionali dal tuo computer')
                ], style={'fontWeight': 'bold', 'color': '#bb86fc'}),

                html.Ul(id='file-list', style={
                    'listStyleType': 'none',
                    'paddingLeft': '0',
                    'marginTop': '10px',
                    'display': 'flex',
                    'gap': '10px',
                    'overflowX': 'auto',
                    'maxWidth': '100%',
                    'color': '#d1b3ff',
                    'fontSize': '14px',
                    'textAlign': 'left',
                }),
            ]),
            style={
                'width': '100%',
                'height': '130px',
                'lineHeight': '20px',
                'borderWidth': '2px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'cursor': 'pointer',
                'backgroundColor': '#2a2a2a',
                'color': '#bb86fc',
                'userSelect': 'none',
            }
        ),

        html.Button("Analizza", id='analyze-button', n_clicks=0, className="analyze-button"),

    ], className='top-container'),


    html.Div([

        # Sinistra: output analisi (statistiche + grafici)
        html.Div(id='analysis-output', className='left-panel'),

        # Destra: pannello opzioni + bottone procedi
        html.Div([
            html.H4("Opzioni di analisi", className="panel-title"),

            dcc.Checklist(
                id='preprocessing-options',
                options=[{'label': 'Applica Sin/Cos', 'value': 'sincos'}],
                value=[],
                labelStyle={'display': 'block'},
                inputStyle={"margin-right": "10px"}
            ),

            dcc.Checklist(
                id='pca-toggle',
                options=[{'label': 'Abilita PCA', 'value': 'pca'}],
                value=[],
                labelStyle={'display': 'block'},
                inputStyle={"margin-right": "10px"}
            ),

            html.Div([
                html.Label("Numero di componenti principali:"),
                dcc.Input(
                    id='num-components',
                    type='number',
                    min=1,
                    max=20,
                    step=1,
                    value=3
                )
            ], id='num-components-container', style={'display': 'none'}),

            html.Button("Procedi a Clustering", id='proceed-button', n_clicks=0, className="proceed-button")

        ], className='right-panel'),

    ], className='bottom-container'),

])