import base64
import io
import pandas as pd
import plotly.express as px
import openai
from flask import Flask
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State


openai.api_key = "your-api-key-here"

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container(
    [
        html.H2("Dataset Uploader and Analyzer", className="text-center my-4"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drop a file here or ", html.A("select one")]),
            style={
                "height": "60px", "lineHeight": "60px", "borderWidth": "1px",
                "borderStyle": "dashed", "borderRadius": "5px", "textAlign": "center",
                "margin-bottom": "20px",
            },
        ),
        html.Div(id="output-data-upload"),
        html.Hr(),
        html.Div(id="openai-report", style={"padding": "20px", "background": "#F7F7F7", "border-radius": "10px"}),
        html.Hr(),
        dbc.Row(id="stat-cards", className="mb-4"),
    ],
    fluid=True
)

# File parsing
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith('.csv'):
            return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith(('.xls', '.xlsx')):
            return pd.read_excel(io.BytesIO(decoded))
    except Exception:
        return None
    return None

# Statistical cards
def generate_stat_cards(df):
    return [
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Total Rows", className="card-title"),
            html.P(f"{len(df)}", className="card-text")
        ]), color="primary", inverse=True, className="rounded shadow-sm"), width=4),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Total Columns", className="card-title"),
            html.P(f"{len(df.columns)}", className="card-text")
        ]), color="success", inverse=True, className="rounded shadow-sm"), width=4),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5(f"Unique {df.columns[0]}", className="card-title"),
            html.P(f"{df[df.columns[0]].nunique()}", className="card-text")
        ]), color="warning", inverse=True, className="rounded shadow-sm"), width=4),
    ]

# Dashboard visualizations
def generate_dashboard(df):
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=px.histogram(df, x=df.columns[0])), width=6),
            dbc.Col(dcc.Graph(figure=px.scatter(df, x=df.columns[0], y=df.columns[1])), width=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=px.box(df, x=df.columns[0])), width=6),
            dbc.Col(dcc.Graph(figure=px.line(df, x=df.columns[0], y=df.columns[1])), width=6)
        ])
    ])

# OpenAI-based summary
def generate_openai_report(df):
    prompt = f"Summarize the dataset in English and explain key insights from the data: {df.describe()}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a data analysis expert."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        temperature=0.5,
    )
    summary = response['choices'][0]['message']['content']
    return [html.P(line.strip()) for line in summary.split("\n") if line.strip()]

# Upload callback
@app.callback(
    [Output("output-data-upload", "children"), Output("stat-cards", "children")],
    [Input("upload-data", "contents")],
    [State("upload-data", "filename")]
)
def update_output(contents, filename):
    if contents:
        df = parse_contents(contents, filename)
        if df is not None:
            return generate_dashboard(df), generate_stat_cards(df)
        return html.Div(["An error occurred while processing the file."]), []
    return html.Div(), []

# OpenAI summary callback
@app.callback(
    Output("openai-report", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def display_openai_summary(contents, filename):
    if contents:
        df = parse_contents(contents, filename)
        if df is not None:
            return generate_openai_report(df)
    return html.Div()

# Run app
if __name__ == "__main__":
    app.run_server(debug=True)
