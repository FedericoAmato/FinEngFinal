# Fetches and displays a basic candlestick app.

# Example for making a basic candlestick plot, changing an attribute ("Title"), and displaying in Dash
import dash
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html

import requests
from bs4 import BeautifulSoup
import pandas as pd

#1)

URL = 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yieldYear&year=2021'
page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')

table_html = soup.findAll('table', {'class': 't-chart'})

df = pd.read_html(str(table_html))[0]

#2) Create Figure
fig = go.Figure(data=[go.Surface(z=df)])

#3) Figure layout
fig.update_layout(title='Bond Yields, 2021', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

# 4) Create a Dash app
app = dash.Dash(__name__)

# 5) Define a very simple layout -- just a plot inside a div. No inputs or outputs because the figure doesn't change.
app.layout = html.Div([dcc.Graph(id='3d-graph', figure=fig)])

# Run it!
if __name__ == '__main__':
    app.run_server(debug=True)
