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
            # Top row: Load Dataset and Variable Selection side by side
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Load Dataset"),
                        dbc.CardBody([
                            dcc.Input(id='dataset-url-input', type='text', placeholder='Enter dataset URL or path', style={'width': '100%'}),
                            html.Br(),
                            html.Br(),
                            dbc.Button('Load Dataset', id='load-dataset-button', color='primary', n_clicks=0, className='mb-2'),
                            dcc.Loading(html.Div(id='load-status'), type='default'),
                        ])
                    ], className='mb-3'),
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Variable Selection"),
                        dbc.CardBody([
                            dcc.Dropdown(id='variable-dropdown', style={'width': '100%'}),
                        ])
                    ], className='mb-3'),
                ], width=6),
            ]),
            # Second row: Dimension Selection
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Dimension Selection"),
                        dbc.CardBody([
                            html.Div(id='dimension-checklist-container'),
                            html.Div(id='dimension-dropdowns-container'),
                        ])
                    ], className='mb-3'),
                ], width=12),
            ]),
            # Third row: Select Data (filter and button)
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Select Data"),
                        dbc.CardBody([
                            html.Div([
                                html.Label("Data Filter (Min/Max):", className="mb-1"),
                                dbc.Input(id='data-filter-min', type='number', placeholder='Min value', style={'width': '45%', 'display': 'inline-block', 'marginRight': '10px'}),
                                dbc.Input(id='data-filter-max', type='number', placeholder='Max value', style={'width': '45%', 'display': 'inline-block'}),
                            ], className='mb-2'),
                            dbc.Button('Show Data Quick Stats (Max/Min/Med/Stdev)', id='show-data-button', color='info', n_clicks=0, className='mt-2'),
                        ])
                    ], className='mb-3'),
                ], width=12),
            ]),
            # Fourth row: Quick Stats
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Quick Stats (Max/Min/Mean/Med/STDEV)"),
                        dbc.CardBody([
                            dcc.Loading(html.Div(id='data-array-display'), type='circle'),
                        ])
                    ], className='mb-3'),
                ], width=12),
            ]),
            # Fifth row: Plot Selected Data
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Plot Selected Data"),
                        dbc.CardBody([
                            # html.Div([
                            #     html.Label("Color Scale (Min/Max):", className="mb-1"),
                            #     dbc.Input(id='color-min', type='number', placeholder='Color min', style={'width': '45%', 'display': 'inline-block', 'marginRight': '10px'}),
                            #     dbc.Input(id='color-max', type='number', placeholder='Color max', style={'width': '45%', 'display': 'inline-block'}),
                            # ], className='mb-2'),
                            dcc.Loading(html.Div([
                                html.Div(id='map-container', children=[html.Img(id='map', style={'width': '100%', 'height': 'auto'})]),
                            ]), type='circle'),
                            dbc.Button('Show Plot', id='show-plot-button', color='success', n_clicks=0, className='mt-2'),
                        ])
                    ]),
                    dbc.Button('Reset', id='reset-button', color='secondary', n_clicks=0, className='mt-3'),
                ], width=12),
            ]),
            # Metadata row: full width at the bottom
            dbc.Row([
                dbc.Col([
                    html.Div(id='dataset-info-container', className='mt-3'),
                ], width=12),
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

    def get_metadata_summary(self):
        if self.ds is None:
            return "No dataset loaded."
        try:
            dims = self.ds.dims.items()
            vars = self.ds.variables.keys()
            attrs = self.ds.attrs.items()
            from dash import callback_context
            selected_var = None
            try:
                ctx = callback_context
                if ctx and ctx.states and 'variable-dropdown.value' in ctx.states:
                    selected_var = ctx.states['variable-dropdown.value']
            except Exception:
                pass
            var_proj_info = []
            if selected_var and selected_var in self.ds.data_vars:
                var_attrs = self.ds[selected_var].attrs
                if var_attrs:
                    var_proj_info.append(html.Tr([
                        html.Td("Variable Attributes:"),
                        html.Td(html.Ul([
                            html.Li([
                                html.B(str(k)), ": ", str(v)
                            ]) for k, v in var_attrs.items()
                        ]))
                    ]))
            return html.Div([
                html.H5("Dataset Metadata", style={"marginTop": "10px"}),
                html.Table([
                    html.Tbody([
                        html.Tr([
                            html.Td("Dimensions:"),
                            html.Td(html.Ul([html.Li(f"{k}: {v}") for k, v in dims]))
                        ]),
                        html.Tr([
                            html.Td("Variables:"),
                            html.Td(html.Ul([html.Li(var) for var in vars]))
                        ]),
                        html.Tr([
                            html.Td("Attributes:"),
                            html.Td(html.Ul([
                                html.Li([
                                    html.B(str(k)), ": ", str(v)
                                ]) for k, v in attrs
                            ]))
                        ]),
                        *var_proj_info
                    ])
                ], style={"width": "100%", "fontSize": "13px", "color": "#222", "background": "#fff", "borderRadius": "6px", "padding": "8px"})
            ], className="dataset-info-container")
        except Exception as e:
            return f"Error reading metadata: {e}"

    def setup_callbacks(self):
        @self.app.callback(
            [Output('load-status', 'children'),
             Output('dataset-info-container', 'children')],
            Input('load-dataset-button', 'n_clicks'),
            State('dataset-url-input', 'value'),
            prevent_initial_call=True
        )
        def load_dataset(n_clicks, url):
            if not url:
                return "Please provide a valid dataset URL or path.", dash.no_update
            # Show loading spinner automatically via dcc.Loading
            self.dataseturl = url
            self.ds, self.dataset_engine = self.read_dataset_metadata(url)
            if self.ds is None:
                return f"Failed to load dataset from {url}", dash.no_update
            return f"Successfully loaded dataset from {url}", self.get_metadata_summary()

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

