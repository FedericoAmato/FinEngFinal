# Fetches and displays a basic candlestick app.

# Example for making a basic candlestick plot, changing an attribute ("Title"), and displaying in Dash
import dash
import pandas as pd
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html

#1) Read data from a csv
z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')

#2) Create Figure
fig = go.Figure(data=[go.Surface(z=z_data.values)])

#3) Figure layout
fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

# 4) Create a Dash app
app = dash.Dash(__name__)

# 5) Define a very simple layout -- just a plot inside a div. No inputs or outputs because the figure doesn't change.
app.layout = html.Div([dcc.Graph(id='3d-graph', figure=fig)])

# Run it!
if __name__ == '__main__':
    app.run_server(debug=True)
