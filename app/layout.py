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
        html.Button(
            "Analizza Dati Raw",
            id='analyze-raw-button',
            n_clicks=0,
            className="analyze-button",
            style={'marginTop': '10px'}
        ),
        dcc.Store(id='stored-raw-data')  # Nuovo store per dati raw
    ], className='box'),

    # === BOX 2: Risultati Dati Raw (sempre visibile dopo analisi) ===
    html.Div([
        html.H3("📊 Dati Raw (Phi, Psi)", style={'color': '#7a42ff'}),
        html.Hr(style={'borderColor': '#ddd'}),
        dcc.Loading(
            id="loading-raw",
            type="circle",
            children=html.Div(id='raw-output', className='analysis-panel')
        )
    ], id="raw-section", className="box", style={'display': 'none'}),

    # === BOX 3: Preprocessing Opzionale ===
    html.Div([
        html.H3("🔧 Preprocessing Opzionale", style={'color': '#7a42ff'}),
        html.Hr(style={'borderColor': '#ddd'}),

        html.Div([
            html.Label("Opzioni preprocessing:", style={'fontWeight': 'bold', 'marginBottom': '8px'}),
            dcc.Checklist(
                id='preprocessing-options',
                options=[{'label': ' Applica Sin/Cos', 'value': 'sincos'}],
                value=[],
                labelStyle={'display': 'block', 'marginBottom': '8px'}
            ),

            dcc.Checklist(
                id='pca-toggle',
                options=[{'label': ' Abilita PCA', 'value': 'pca'}],
                value=[],
                labelStyle={'display': 'block', 'marginBottom': '8px'}
            ),

            html.Div([
                html.Label("Numero di componenti principali:", style={'marginTop': '10px'}),
                dcc.Input(
                    id='num-components',
                    type='number',
                    min=1,
                    max=20,
                    step=1,
                    value=3,
                    placeholder='es. 3',
                    style={'width': '120px', 'marginTop': '5px'}
                )
            ], id='num-components-container', style={'display': 'none'}),
        ], style={'marginBottom': '15px'}),

        html.Button(
            "Applica Preprocessing",
            id='apply-preprocessing-button',
            n_clicks=0,
            className="analyze-button",
            style={'marginTop': '10px'}
        ),
        dcc.Store(id='stored-pca-data')  # Store per dati con PCA
    ], id="preprocessing-section", className="box", style={'display': 'none'}),

    # === BOX 4: Risultati PCA (appare solo se PCA attivata) ===
    html.Div([
        html.H3("📈 Risultati PCA", style={'color': '#7a42ff'}),
        html.Hr(style={'borderColor': '#ddd'}),
        dcc.Loading(
            id="loading-pca",
            type="circle",
            children=html.Div(id='pca-output', className='analysis-panel')
        )
    ], id="pca-section", className="box", style={'display': 'none'}),

    # === BOX 5: Clustering ===
    html.Div([
        html.H4("🎯 Clustering", className="panel-title"),
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

    # === BOX 6: Anomaly Detection ===
    html.Div([
        html.H4("⚠️ Anomaly Detection", className="panel-title"),

        html.Label("Categoria modello:"),
        dcc.Dropdown(
            id='anomaly-category',
            options=[
                {'label': 'Regressione', 'value': 'regression'},
                {'label': 'Distanza', 'value': 'distance'},
                {'label': 'Clustering', 'value': 'clustering'},
            ],
            value='regression',
            clearable=False,
            style={'marginBottom': '10px'}
        ),

        html.Label("Seleziona algoritmo di anomaly detection:"),
        dcc.Dropdown(
            id='anomaly-algorithm',
            options=[
                {'label': 'Linear Regression', 'value': 'linreg'},
                {'label': 'Linear Regression + Bagging (Sin/Cos)', 'value': 'linreg_bagging'},
                {'label': 'Random Forest + Bagging', 'value': 'rf_bagging'},
                {'label': 'Gradient Boosting + Bagging', 'value': 'gb_bagging'},
                {'label': 'Extra Trees + Bagging', 'value': 'et_bagging'},
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

            # Gruppo parametri bagging
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

            # LOF
            html.Div([
                html.H5("LOF"),
                html.Label("Numero vicini:"),
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

            # Matrix Profile
            html.Div([
                html.H5("Matrix Profile"),
                html.P("Nessun parametro aggiuntivo richiesto.",
                       style={'margin': 0, 'fontSize': '13px', 'color': '#555'})
            ], id='params-mp', className='param-group', style={'display': 'none', 'marginBottom': '15px'}),

            # DBSCAN
            html.Div([
                html.H5("DBSCAN"),
                html.Label("eps:"),
                dcc.Input(
                    id='dbscan-eps',
                    type='number',
                    min=0,
                    step=0.01,
                    value=0.25,
                    style={'width': '120px'},
                    placeholder='es. 0.25'
                ),
                html.Br(),
                html.Label("Numero min samples:"),
                dcc.Input(
                    id='dbscan-min-samples',
                    type='number',
                    min=1,
                    step=1,
                    value=15,
                    style={'width': '120px'},
                    placeholder='es. 15'
                ),
                html.Br(),
                html.Label("k per kNN (opzionale):"),
                dcc.Input(
                    id='dbscan-knn-k',
                    type='number',
                    min=1,
                    step=1,
                    placeholder='auto',
                    style={'width': '120px'}
                ),
            ], id='params-dbscan-clustering', className='param-group',
                style={'display': 'none', 'marginBottom': '15px'}),

            # OPTICS
            html.Div([
                html.H5("OPTICS"),
                html.Label("Numero min samples:"),
                dcc.Input(
                    id='optics-min-samples',
                    type='number',
                    min=2,
                    step=1,
                    value=5,
                    style={'width': '120px'}
                ),
                html.Br(),
                html.Label("xi:"),
                dcc.Input(
                    id='optics-xi',
                    type='number',
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    value=0.01,
                    style={'width': '120px'}
                ),
                html.Br(),
                html.Label("Dimensione minimo cluster:"),
                dcc.Input(
                    id='optics-min-cluster-size',
                    type='number',
                    min=0.0,
                    step=0.05,
                    value=0.1,
                    style={'width': '120px'}
                ),
            ], id='params-optics-clustering', className='param-group',
                style={'display': 'none', 'marginBottom': '15px'}),

            # K-Means
            html.Div([
                html.H5("K-Means"),
                html.Label("Numero clusters:"),
                dcc.Input(
                    id='kmeans-n-clusters',
                    type='number',
                    min=1,
                    step=1,
                    value=3,
                    style={'width': '120px'}
                ),
                html.Br(),
                html.Label("Random state:"),
                dcc.Input(
                    id='kmeans-random-state',
                    type='number',
                    step=1,
                    value=42,
                    style={'width': '120px'}
                ),
            ], id='params-kmeans-clustering', className='param-group',
                style={'display': 'none', 'marginBottom': '15px'}),

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
