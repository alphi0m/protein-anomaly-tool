from dash import html, dcc

layout = html.Div([

    # Sezione superiore: caricamento file + opzioni PCA/SinCos + Analizza
    html.Div([
        html.H1("Protein Anomaly Detector", className="app-title"),
        html.P(
            "Carica una simulazione molecolare per analizzare le dinamiche della proteina.",
            className="description"
        ),

        dcc.Upload(
            id='upload-files',
            multiple=True,
            children=html.Div([
                html.Div(id='upload-text', children=[
                    'Trascina qui i file o ',
                    html.A('selezionali dal tuo computer')
                ], style={'fontWeight': 'bold', 'color': '#bb86fc'}),

                html.Ul(
                    id='file-list',
                    style={
                        'listStyleType': 'none',
                        'paddingLeft': '0',
                        'marginTop': '10px',
                        'display': 'flex',
                        'gap': '10px',
                        'overflowX': 'auto',
                        'overflowY': 'hidden',
                        'flexWrap': 'nowrap',
                        'whiteSpace': 'nowrap',
                        'maxWidth': '100%',
                        'maxHeight': '50px',
                        'color': '#d1b3ff',
                        'fontSize': '14px',
                        'textAlign': 'left',
                    }
                )
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

        # Opzioni sin/cos e PCA
        html.Div([
            dcc.Checklist(
                id='preprocessing-options',
                options=[{'label': 'Applica Sin/Cos', 'value': 'sincos'}],
                value=[],
                labelStyle={'display': 'inline-block', 'marginRight': '20px'}
            ),

            dcc.Checklist(
                id='pca-toggle',
                options=[{'label': 'Abilita PCA', 'value': 'pca'}],
                value=[],
                labelStyle={'display': 'inline-block', 'marginRight': '20px'}
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
            ], id='num-components-container', style={'display': 'none', 'marginTop': '5px'}),

            html.Button("Analizza", id='analyze-button', n_clicks=0,
                        className="analyze-button", style={'marginTop': '10px'})
        ], style={'marginTop': '10px'}),

        # Store per salvare PCA dati per clustering/anomaly detection
        dcc.Store(id='stored-pca-data')
    ], className='top-container'),


    # Sezione inferiore
    html.Div([

        # Colonna sinistra con tre aree distinte
        html.Div([
            html.H3("Risultati PCA"),
            html.Div(id='analysis-output', className='analysis-panel'),

            html.Hr(),

            html.H3("Risultati Clustering"),
            html.Div(id='clustering-output', className='clustering-panel'),

            html.Hr(),

            html.H3("Anomaly Detection"),
            html.Div(id='anomaly-output', className='anomaly-panel'),
        ], className='left-panel'),

        # Colonna destra: opzioni clustering + anomaly detection
        html.Div([

            # Clustering
            html.H4("Opzioni Clustering", className="panel-title"),

            html.Label("Seleziona algoritmo di clustering:"),
            dcc.Dropdown(
                id='clustering-algorithm',
                options=[
                    {'label': 'DBSCAN', 'value': 'dbscan'},
                    {'label': 'OPTICS', 'value': 'optics'},
                    {'label': 'Spectral Clustering', 'value': 'spectral'}
                ],
                value='dbscan',
                clearable=False
            ),

            html.Div(id='clustering-parameters', style={'marginTop': '10px'}),

            html.Button("Procedi a Clustering", id='proceed-button', n_clicks=0,
                        className="proceed-button", style={'marginTop': '10px'}),

            html.Hr(),

            # Anomaly detection
            html.H4("Opzioni Anomaly Detection", className="panel-title"),

            html.Label("Seleziona algoritmo di anomaly detection:"),
            dcc.Dropdown(
                id='anomaly-algorithm',
                options=[
                    {'label': 'Linear Regression', 'value': 'linreg'},
                    {'label': 'Linear Regression + Bagging (Sin/Cos)', 'value': 'linreg_bagging'},
                    {'label': 'LSTM + Bagging (Sin/Cos)', 'value': 'lstm_bagging'},
                    {'label': 'Random Forest Regression + Bagging', 'value': 'rf_bagging'},
                    {'label': 'Gradient Boosting Regressor + Bagging', 'value': 'gb_bagging'},
                    {'label': 'Extra Trees Regressor + Bagging', 'value': 'et_bagging'}
                ],
                value='linreg',
                clearable=False
            ),

            html.Div(id='anomaly-parameters', style={'marginTop': '10px'}),

            html.Button("Applica Anomaly Detection", id='anomaly-button', n_clicks=0,
                        className="anomaly-button", style={'marginTop': '10px'})
        ], className='right-panel'),

    ], className='bottom-container'),

])