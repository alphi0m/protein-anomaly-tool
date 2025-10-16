from dash import html, dcc

layout = html.Div([

    # === BOX 1: Caricamento file (Side-by-Side) ===
    html.Div([
        html.H1("Protein Anomaly Detector", className="app-title"),
        html.P(
            "Carica una simulazione molecolare per analizzare le dinamiche della proteina.",
            className="description"
        ),

        # Radio buttons per scegliere modalità
        html.Div([
            dcc.RadioItems(
                id='upload-mode',
                options=[
                    {'label': ' Carica file angolari raw (.txt)', 'value': 'raw'},
                    {'label': ' Carica CSV preprocessato (PCA)', 'value': 'preprocessed'}
                ],
                value='raw',
                inline=True,
                className='upload-mode-radio'
            )
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),

        # Contenitore flex per box side-by-side
        html.Div([
            # BOX VIOLA: Upload Raw
            html.Div([
                dcc.Upload(
                    id='upload-files',
                    multiple=True,
                    className='upload-box upload-box-raw',
                    children=html.Div([
                        html.Div([
                            "📂 Trascina file .txt o ",
                            html.A("selezionali", style={'textDecoration': 'underline', 'cursor': 'pointer'})
                        ], style={'fontWeight': 'bold', 'color': '#7a42ff', 'marginBottom': '15px'}),
                        html.Ul(id='file-list', className='file-list-pills')
                    ])
                ),
                html.Button(
                    "🔬 Analizza Dati Raw",
                    id='analyze-raw-button',
                    n_clicks=0,
                    className="analyze-button",
                    style={'marginTop': '15px'}
                )
            ], id='upload-raw-container', className='upload-box-container'),

            # BOX VERDE: Upload CSV
            html.Div([
                dcc.Upload(
                    id='upload-csv',
                    multiple=False,
                    className='upload-box upload-box-csv',
                    children=html.Div([
                        html.Div([
                            "📊 Trascina CSV preprocessato o ",
                            html.A("selezionalo", style={'textDecoration': 'underline', 'cursor': 'pointer'})
                        ], style={'fontWeight': 'bold', 'color': '#10b981', 'marginBottom': '15px'}),
                        html.Div(id='csv-file-name', className='csv-preview')
                    ])
                ),

                html.Div([
                    html.Label("Numero componenti PCA:",
                               style={'fontWeight': 'bold', 'marginTop': '15px', 'color': '#047857'}),
                    dcc.Input(
                        id='csv-num-components',
                        type='number',
                        min=1,
                        max=20,
                        step=1,
                        value=3,
                        placeholder='es. 3',
                        style={'width': '120px', 'marginTop': '5px', 'display': 'block'}
                    )
                ], style={'marginTop': '10px'}),

                html.Button(
                    "📊 Analizza CSV Preprocessato",
                    id='analyze-csv-button',
                    n_clicks=0,
                    className="analyze-button analyze-button-green",
                    style={'marginTop': '15px'}
                )
            ], id='upload-csv-container', className='upload-box-container', style={'display': 'none'})

        ], className='upload-boxes-wrapper'),

        dcc.Store(id='stored-raw-data')
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
        html.H4("🎯 Clustering", className="panel-title", style={'marginBottom': '10px'}),
        html.Hr(style={'borderColor': '#ddd'}),
        html.P("Raggruppa i dati PCA in cluster omogenei",
               style={'fontSize': '13px', 'color': '#666', 'marginBottom': '15px'}),

        html.Label("Seleziona algoritmo:", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='clustering-algorithm',
            options=[
                {'label': '🔵 DBSCAN (Densità)', 'value': 'dbscan'},
                {'label': '🟢 OPTICS (Gerarchia)', 'value': 'optics'},
                {'label': '🟣 Spectral (Grafo)', 'value': 'spectral'},
            ],
            value='dbscan',
            clearable=False
        ),

        html.Div([
            # DBSCAN
            html.Div([
                html.H5("DBSCAN", style={'color': '#7a42ff', 'marginBottom': '10px'}),

                html.Label("🔍 Raggio di ricerca:", style={'fontWeight': 'bold'}),
                html.P("Distanza massima tra punti vicini per formare un cluster",
                       style={'fontSize': '11px', 'color': '#666', 'margin': '2px 0 5px'}),
                dcc.Input(
                    id='clustering-dbscan-eps',
                    type='number',
                    min=0.01,
                    step=0.1,
                    value=0.5,
                    style={'width': '120px'}
                ),

                html.Br(),
                html.Label("👥 Densità minima:", style={'fontWeight': 'bold', 'marginTop': '10px'}),
                html.P("Numero minimo di punti per formare un cluster",
                       style={'fontSize': '11px', 'color': '#666', 'margin': '2px 0 5px'}),
                dcc.Input(
                    id='clustering-dbscan-min-samples',
                    type='number',
                    min=1,
                    step=1,
                    value=15,
                    style={'width': '120px'}
                ),
            ], id='params-clustering-dbscan', className='param-group',
                style={'marginBottom': '15px'}),

            # OPTICS
            html.Div([
                html.H5("OPTICS", style={'color': '#7a42ff', 'marginBottom': '10px'}),

                html.Label("👥 Densità minima:", style={'fontWeight': 'bold'}),
                html.P("Numero minimo di punti per formare un cluster",
                       style={'fontSize': '11px', 'color': '#666', 'margin': '2px 0 5px'}),
                dcc.Input(
                    id='clustering-optics-min-samples',
                    type='number',
                    min=2,
                    step=1,
                    value=5,
                    style={'width': '120px'}
                ),

                html.Br(),
                html.Label("🎯 Sensibilità separazione:", style={'fontWeight': 'bold', 'marginTop': '10px'}),
                html.P("Quanto distinti devono essere i cluster (0.01 = molto sensibile)",
                       style={'fontSize': '11px', 'color': '#666', 'margin': '2px 0 5px'}),
                dcc.Input(
                    id='clustering-optics-xi',  # ✅ CORRETTO!
                    type='number',
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    value=0.01,
                    style={'width': '120px'}
                ),

                html.Br(),
                html.Label("📊 Dimensione minima (%):", style={'fontWeight': 'bold', 'marginTop': '10px'}),
                html.P("Frazione minima del dataset per essere un cluster (0.1 = 10%)",
                       style={'fontSize': '11px', 'color': '#666', 'margin': '2px 0 5px'}),
                dcc.Input(
                    id='clustering-optics-min-cluster-size',
                    type='number',
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    value=0.1,
                    style={'width': '120px'}
                ),
            ], id='params-clustering-optics', className='param-group',
                style={'display': 'none', 'marginBottom': '15px'}),

            # Spectral
            html.Div([
                html.H5("Spectral Clustering", style={'color': '#7a42ff', 'marginBottom': '10px'}),

                html.Label("🎯 Numero di gruppi:", style={'fontWeight': 'bold'}),
                html.P("Quanti cluster vuoi identificare",
                       style={'fontSize': '11px', 'color': '#666', 'margin': '2px 0 5px'}),
                dcc.Input(
                    id='clustering-spectral-n-clusters',
                    type='number',
                    min=2,
                    step=1,
                    value=3,
                    style={'width': '120px'}
                ),

                html.Br(),
                html.Label("🔗 Metodo similarità:", style={'fontWeight': 'bold', 'marginTop': '10px'}),
                html.P("Come calcolare la vicinanza tra punti",
                       style={'fontSize': '11px', 'color': '#666', 'margin': '2px 0 5px'}),
                dcc.Dropdown(
                    id='clustering-spectral-affinity',
                    options=[
                        {'label': 'K vicini più prossimi', 'value': 'nearest_neighbors'},
                        {'label': 'Kernel Gaussiano (RBF)', 'value': 'rbf'},
                    ],
                    value='nearest_neighbors',
                    clearable=False,
                    style={'width': '200px'}
                ),
            ], id='params-clustering-spectral', className='param-group',
                style={'display': 'none', 'marginBottom': '15px'}),

        ], style={'marginTop': '10px'}),

        html.Button(
            "Esegui Clustering",
            id='proceed-button',
            n_clicks=0,
            className="proceed-button",
            style={'marginTop': '15px'}
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
        html.Hr(style={'borderColor': '#ddd'}),
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
                html.Label("Raggio di ricerca (eps):"),
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
                html.Label("Numero minimo samples:"),
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
                html.Label("Numero minimo samples:"),
                dcc.Input(
                    id='optics-min-samples',
                    type='number',
                    min=2,
                    step=1,
                    value=5,
                    style={'width': '120px'}
                ),
                html.Br(),
                html.Label("Sensibilità separazione (xi):"),
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
                html.Label("Dimensione minima cluster:"),
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
