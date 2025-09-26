from dash import html, dcc

layout = html.Div([

    # === BOX 1: Caricamento file ===
    html.Div([
        html.H1("Protein Anomaly Detector", className="app-title"),
        html.P(
            "Carica una simulazione molecolare per analizzare le dinamiche della proteina.",
            className="description"
        ),
        dcc.Upload(
            id='upload-files',
            multiple=True,
            className='dash-uploader',
            children=html.Div([
                html.Div(
                    id='upload-text',
                    children=[
                        "Trascina qui i file o ",
                        html.A("selezionali dal tuo computer")
                    ],
                    style={'fontWeight': 'bold', 'color': '#7a42ff'}
                ),
                html.Ul(id='file-list')
            ]),
            style={
                'width': '100%',
                'height': '250px',
                'lineHeight': '20px',
                'borderWidth': '2px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'cursor': 'pointer',
                'backgroundColor': '#f8f8f8',
                'color': '#7a42ff',
                'userSelect': 'none',
            }
        ),
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
                    value=3,
                    placeholder='es. 3',
                    style={'width': '120px'}
                )
            ], id='num-components-container', style={'display': 'none', 'marginTop': '5px'}),
        ], style={'marginTop': '10px'}),
        html.Button(
            "Analizza",
            id='analyze-button',
            n_clicks=0,
            className="analyze-button",
            style={'marginTop': '10px'}
        ),
        dcc.Store(id='stored-pca-data')
    ], className='box'),

    # === BOX 2: Risultati (PCA / Statistiche) ===
    html.Div([
        html.H3("Risultati"),
        dcc.Loading(
            id="loading-pca",
            type="circle",
            children=html.Div(id='analysis-output', className='analysis-panel')
        )
    ], id="pca-section", className="box", style={'display': 'none'}),

    # === BOX 3: Clustering ===
    html.Div([
        html.H4("Clustering", className="panel-title"),
        html.Label("Seleziona algoritmo di clustering:"),
        dcc.Dropdown(
            id='clustering-algorithm',
            options=[
                {'label': 'DBSCAN', 'value': 'dbscan'},
                {'label': 'OPTICS', 'value': 'optics'},
                {'label': 'Spectral Clustering', 'value': 'spectral'},
            ],
            value='dbscan',
            clearable=False
        ),
        html.Div(id='clustering-parameters', style={'marginTop': '10px'}),
        html.Button(
            "Esegui il Clustering",
            id='proceed-button',
            n_clicks=0,
            className="proceed-button",
            style={'marginTop': '10px'}
        ),
        dcc.Loading(
            id="loading-clustering",
            type="circle",
            children=html.Div(id='clustering-output', className='clustering-panel')
        )
    ], id="clustering-section", className="box", style={'display': 'none'}),

    # === BOX 4: Anomaly Detection ===
    html.Div([
        html.H4("Anomaly Detection", className="panel-title"),

        # Nuovo: selezione categoria
        html.Label("Categoria modello:"),
        dcc.Dropdown(
            id='anomaly-category',
            options=[
                {'label': 'Regressione', 'value': 'regression'},
                {'label': 'Distanza', 'value': 'distance'},
            ],
            value='regression',
            clearable=False,
            style={'marginBottom': '10px'}
        ),

        # Esistente: dropdown algoritmi (verrà popolato dinamicamente in futuro)
        html.Label("Seleziona algoritmo di anomaly detection:"),
        dcc.Dropdown(
            id='anomaly-algorithm',
            options=[
                {'label': 'Linear Regression', 'value': 'linreg'},
                {'label': 'Linear Regression + Bagging (Sin/Cos)', 'value': 'linreg_bagging'},
                {'label': 'Random Forest + Bagging', 'value': 'rf_bagging'},
                {'label': 'Gradient Boosting + Bagging', 'value': 'gb_bagging'},
                {'label': 'Extra Trees + Bagging', 'value': 'et_bagging'},
                # Le nuove voci verranno iniettate via callback:
                # {'label': 'Local Outlier Factor (LOF)', 'value': 'lof'}
                # {'label': 'Matrix Profile (STUMP)', 'value': 'matrix_profile'}
            ],
            value='linreg',
            clearable=False
        ),

        # === PARAMETRI DINAMICI MODELLI ===
        html.Div([
            # Gruppo parametri comuni
            html.Div([
                html.H5("Parametri comuni"),
                html.Label("Lunghezza training:"),
                dcc.Input(
                    id='param-train-n',
                    type='number',
                    min=30,
                    step=10,
                    value=180,
                    style={'width': '120px'},
                    placeholder='es. 180'
                ),
                html.Br(),
                html.Label("Dimensione finestra:"),
                dcc.Input(
                    id='param-window-size',
                    type='number',
                    min=3,
                    step=1,
                    value=20,
                    style={'width': '120px'},
                    placeholder='es. 20'
                ),
                html.Br(),
                html.Label("Penalità soglia variabile:"),
                dcc.Input(
                    id='param-pen',
                    type='number',
                    min=1,
                    step=1,
                    value=1,
                    style={'width': '120px'},
                    placeholder='es. 1'
                ),
            ], id='params-common', className='param-group', style={'marginBottom': '15px'}),

            # Gruppo parametri bagging (esistente)
            html.Div([
                html.H5("Bagging"),
                html.Label("Numero modelli:"),
                dcc.Input(
                    id='param-num-models',
                    type='number',
                    min=1,
                    step=1,
                    value=10,
                    style={'width': '120px'},
                    placeholder='es. 10'
                ),
            ], id='params-bagging', className='param-group', style={'display': 'none', 'marginBottom': '15px'}),

            # Random Forest
            html.Div([
                html.H5("Random Forest"),
                html.Label("Numero alberi:"),
                dcc.Input(
                    id='rf-n-estimators',
                    type='number',
                    min=10,
                    step=10,
                    value=100,
                    style={'width': '140px'},
                    placeholder='es. 100'
                ),
                html.Br(),
                html.Label("Profondità massima:"),
                dcc.Input(
                    id='rf-max-depth',
                    type='number',
                    min=1,
                    step=1,
                    value=10,
                    style={'width': '140px'},
                    placeholder='es. 10'
                ),
                html.Br(),
                html.Label("Min campioni per split:"),
                dcc.Input(
                    id='rf-min-split',
                    type='number',
                    min=2,
                    step=1,
                    value=2,
                    style={'width': '140px'},
                    placeholder='es. 2'
                ),
            ], id='params-rf', className='param-group', style={'display': 'none', 'marginBottom': '15px'}),

            # Gradient Boosting
            html.Div([
                html.H5("Gradient Boosting"),
                html.Label("Numero di stadi:"),
                dcc.Input(
                    id='gb-n-estimators',
                    type='number',
                    min=10,
                    step=10,
                    value=100,
                    style={'width': '140px'},
                    placeholder='es. 100'
                ),
                html.Br(),
                html.Label("Learning rate:"),
                dcc.Input(
                    id='gb-learning-rate',
                    type='number',
                    min=0.001,
                    step=0.01,
                    value=0.1,
                    style={'width': '140px'},
                    placeholder='es. 0.1'
                ),
                html.Br(),
                html.Label("Massima profondità:"),
                dcc.Input(
                    id='gb-max-depth',
                    type='number',
                    min=1,
                    step=1,
                    value=3,
                    style={'width': '140px'},
                    placeholder='es. 3'
                ),
            ], id='params-gb', className='param-group', style={'display': 'none', 'marginBottom': '15px'}),

            # Extra Trees
            html.Div([
                html.H5("Extra Trees"),
                html.Label("Numero alberi:"),
                dcc.Input(
                    id='et-n-estimators',
                    type='number',
                    min=10,
                    step=10,
                    value=100,
                    style={'width': '140px'},
                    placeholder='es. 100'
                ),
                html.Br(),
                html.Label("Massima profondità:"),
                dcc.Input(
                    id='et-max-depth',
                    type='number',
                    min=1,
                    step=1,
                    value=10,
                    style={'width': '140px'},
                    placeholder='es. 10'
                ),
                html.Br(),
                html.Label("Min campioni per split:"),
                dcc.Input(
                    id='et-min-split',
                    type='number',
                    min=2,
                    step=1,
                    value=2,
                    style={'width': '140px'},
                    placeholder='es. 2'
                ),
            ], id='params-et', className='param-group', style={'display': 'none', 'marginBottom': '15px'}),

            # Nuovo: Local Outlier Factor
            html.Div([
                html.H5("LOF"),
                html.Label("N neighbors:"),
                dcc.Input(
                    id='lof-n-neighbors',
                    type='number',
                    min=1,
                    step=1,
                    value=20,
                    style={'width': '120px'},
                    placeholder='es. 20'
                ),
                html.Br(),
                html.Label("Contamination:"),
                dcc.Input(
                    id='lof-contamination',
                    type='number',
                    min=0.001,
                    max=0.5,
                    step=0.01,
                    value=0.05,
                    style={'width': '120px'},
                    placeholder='es. 0.05'
                ),
            ], id='params-lof', className='param-group', style={'display': 'none', 'marginBottom': '15px'}),

            # Nuovo: Matrix Profile (solo informativo)
            html.Div([
                html.H5("Matrix Profile"),
                html.P("Nessun parametro aggiuntivo richiesto.", style={'margin': 0, 'fontSize': '13px', 'color': '#555'})
            ], id='params-mp', className='param-group', style={'display': 'none', 'marginBottom': '15px'}),

        ], id='anomaly-parameters', style={'marginTop': '10px'}),

        html.Button(
            "Applica Anomaly Detection",
            id='anomaly-button',
            n_clicks=0,
            className="anomaly-button",
            style={'marginTop': '10px'}
        ),

        dcc.Loading(
            id="loading-anomaly",
            type="circle",
            children=html.Div(id='anomaly-output', className='anomaly-panel')
        )
    ], id="anomaly-section", className="box", style={'display': 'none'}),

])
