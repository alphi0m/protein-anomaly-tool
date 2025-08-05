from dash import Dash
from app.layout import layout
from app.callbacks import register_callbacks

app = Dash(__name__, external_stylesheets=[
    "https://fonts.googleapis.com/css2?family=Lato&display=swap"
])
app.layout = layout

register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)