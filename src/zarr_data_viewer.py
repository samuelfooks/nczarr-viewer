import dash
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import xarray as xr
import signal
import os

from variables import VariableSelection
from dimension import DimensionSelection
from data import DataDisplay, DataRetriever, DataPlot
from layout_manager import LayoutManager, ResetFunctionality

class TimeoutException(Exception):
    pass

class ZarrDataViewerApp:
    def __init__(self):
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.config.suppress_callback_exceptions = True
        self.ds = None
        self.dataset_engine = None
        self.dataseturl = None

        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("Zarr/NetCDF Data Viewer", className="text-center mb-4"), width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Load Dataset"),
                        dbc.CardBody([
                            dcc.Input(id='dataset-url-input', type='text', placeholder='Enter dataset URL or path', style={'width': '100%'}),
                            html.Br(),
                            html.Br(),
                            dbc.Button('Load Dataset', id='load-dataset-button', color='primary', n_clicks=0, className='mb-2'),
                            html.Div(id='load-status'),
                        ])
                    ], className='mb-3'),
                    dbc.Card([
                        dbc.CardHeader("Variable Selection"),
                        dbc.CardBody([
                            dcc.Dropdown(id='variable-dropdown'),
                        ])
                    ], className='mb-3'),
                    dbc.Card([
                        dbc.CardHeader("Dimension Selection"),
                        dbc.CardBody([
                            html.Div(id='dimension-checklist-container'),
                            html.Div(id='dimension-dropdowns-container'),
                        ])
                    ]),
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Quick Statistics"),
                        dbc.CardBody([
                            html.Div([
                                html.Label("Data Filter (Min/Max):", className="mb-1"),
                                dbc.Input(id='data-filter-min', type='number', placeholder='Min value', style={'width': '45%', 'display': 'inline-block', 'marginRight': '10px'}),
                                dbc.Input(id='data-filter-max', type='number', placeholder='Max value', style={'width': '45%', 'display': 'inline-block'}),
                            ], className='mb-2'),
                            dcc.Loading(html.Div(id='data-array-display'), type='circle'),
                            dbc.Button('Max/Min/Mean/Med/STDEV', id='show-data-button', color='info', n_clicks=0, className='mt-2'),
                        ])
                    ], className='mb-3'),
                    dbc.Card([
                        dbc.CardHeader("Map Plot"),
                        dbc.CardBody([
                            html.Div([
                                html.Label("Color Scale (Min/Max):", className="mb-1"),
                                dbc.Input(id='color-min', type='number', placeholder='Color min', style={'width': '45%', 'display': 'inline-block', 'marginRight': '10px'}),
                                dbc.Input(id='color-max', type='number', placeholder='Color max', style={'width': '45%', 'display': 'inline-block'}),
                            ], className='mb-2'),
                            dcc.Loading(html.Div([
                                html.Div(id='map-container', children=[html.Img(id='map', style={'width': '100%', 'height': 'auto'})]),
                            ]), type='circle'),
                            dbc.Button('Show Plot', id='show-plot-button', color='success', n_clicks=0, className='mt-2'),
                        ])
                    ]),
                    dbc.Button('Reset', id='reset-button', color='secondary', n_clicks=0, className='mt-3'),
                    html.Div(id='dataset-info-container', className='mt-3'),
                ], width=8),
            ]),
            dcc.Store(id='selected-dimensions-store'),
            html.Div(id='debug-output')
        ], fluid=True)

        # Register all callbacks at startup, using a getter for self.ds
        self.variable_selection = VariableSelection(self.app, lambda: self.ds)
        self.variable_selection.setup_callbacks()
        self.dimension_selection = DimensionSelection(self.app, lambda: self.ds)
        self.dimension_selection.setup_callbacks()
        self.data_display = DataDisplay(self.app, lambda: self.ds, lambda: self.dataseturl, lambda: self.dataset_engine)
        self.data_display.setup_callbacks()
        self.data_plot = DataPlot(self.app, lambda: self.ds, self.dimension_selection, lambda: self.dataseturl, lambda: self.dataset_engine)
        self.data_plot.setup_callbacks()
        self.reset_functionality = ResetFunctionality(self.app, lambda: self.ds)
        self.reset_functionality.setup_callbacks()

        self.setup_callbacks()

    def setup_callbacks(self):
        @self.app.callback(
            Output('load-status', 'children'),
            Input('load-dataset-button', 'n_clicks'),
            State('dataset-url-input', 'value'),
            prevent_initial_call=True
        )
        def load_dataset(n_clicks, url):
            if not url:
                return "Please provide a valid dataset URL or path."
            self.dataseturl = url
            self.ds, self.dataset_engine = self.read_dataset_metadata(url)
            if self.ds is None:
                return f"Failed to load dataset from {url}"
            # Optionally update dataset-info-container or other cards here
            return f"Successfully loaded dataset from {url}"

    def read_dataset_metadata(self, dataseturl):
        try:
            if '.nc' in dataseturl:
                engine = 'netcdf4'
            elif '.zarr' in dataseturl:
                engine = 'zarr'
            else:
                raise ValueError("Unsupported file format")

            print(f"Opening dataset {dataseturl} using engine {engine}")
            ds = xr.open_dataset(dataseturl, engine=engine, decode_timedelta=True)
            return ds, engine
        except Exception as e:
            print(f"Error opening dataset: {e}")
            return None, None

    def run(self):
        self.app.run(debug=True, host='0.0.0.0')

if __name__ == '__main__':
    app = ZarrDataViewerApp()
    app.run()
