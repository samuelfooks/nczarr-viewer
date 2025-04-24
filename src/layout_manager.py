from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash
class LayoutManager:
    def __init__(self, app, ds, dataseturl):
        self.app = app
        self.ds = ds
        self.dataseturl = dataseturl

    def setup_layout(self, return_layout=False):
        layout = html.Div([
            html.H1("Zarr NetCDF Data Viewer"),
            html.Div([
                dcc.Dropdown(id='variable-dropdown'),
                html.Div(id='dimension-checklist-container')
            ]),
            html.Div([
                html.Div(id='dimension-dropdowns-container'),
                html.Div(id='selection-display')
            ]),
            html.Div([
                html.Div(id='map-container', children=[html.Img(id='map')]),
                html.Div(id='data-array-display')
            ]),
            html.Button('Max/Min/Mean/Med/STDEV', id='show-data-button', n_clicks=0),
            html.Button('Show Plot', id='show-plot-button', n_clicks=0),
            html.Button('Reset', id='reset-button', n_clicks=0),
            html.Div(id='dataset-info-container'),
            dcc.Store(id='selected-dimensions-store')
        ])

        if return_layout:
            return layout
        self.app.layout = layout

        from dash import Output, Input

class ResetFunctionality:
    def __init__(self, app, ds):
        self.app = app
        self.ds = ds

    def setup_callbacks(self):
        @self.app.callback(
            Output('selected-dimensions-store', 'data', allow_duplicate=True),
            Output('variable-dropdown', 'value'),
            Output('dimension-dropdowns-container', 'children', allow_duplicate=True),
            Output('data-array-display', 'children', allow_duplicate=True),
            Output('map-container', 'children', allow_duplicate=True),
            Input('reset-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def reset_store(n_clicks):
            if n_clicks > 0:
                return {}, [], [], 'No data selected.', ""
            return dash.no_update