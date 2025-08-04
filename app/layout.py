from dash import html, dcc

layout = html.Div([
    html.Div([
        html.H1("Protein Anomaly Detector", className="app-title"),
        html.P("Carica una simulazione molecolare per analizzare le dinamiche della proteina.",
               className="description"),

        dcc.Upload(
            id='upload-file',
            children=html.Div([
                'Trascina un file qui o ',
                html.A('selezionalo dal tuo computer')
            ]),
            style={
                'width': '100%',
                'height': '100px',
                'lineHeight': '100px',
                'borderWidth': '2px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'marginBottom': '20px'
            },
            multiple=False
        ),

    html.Div([
        html.H4("Opzioni di analisi", className="panel-title"),

        dcc.Checklist(
            id='analysis-options',
        options=[
                {'label': 'Applica Sin/Cos', 'value': 'anomaly'},
                {'label': 'Modifica numero di componenti principali', 'value': 'plot'}
            ],
            value=[],  # default: nessuna selezionata
            labelStyle={'display': 'block'},  # una sotto l'altra
            inputStyle={"margin-right": "10px"}
        )
    ], className="options-panel"),

        html.Button("Analizza", id='analyze-button', n_clicks=0, className="analyze-button"),

        html.Div(id='analysis-output', className='output-section'),

    ], className='main-container')
])