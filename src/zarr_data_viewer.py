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
            # Add a card for variable metadata display
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Variable Metadata"),
                        dbc.CardBody([
                            html.Div(id='variable-metadata-container')
                        ])
                    ], className='mb-3'),
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
        self.update_variable_dropdown()
        self.setup_variable_metadata_callback()

    def update_variable_dropdown(self):
        # Update the variable dropdown to list all data variables
        if self.ds is not None:
            from dash import callback_context
            options = [
                {"label": f"{var} ({self.ds[var].attrs.get('long_name', var)})", "value": var}
                for var in self.ds.data_vars.keys()
            ]
            self.app.callback_map['variable-dropdown.value']['inputs'][0]['options'] = options

    def get_metadata_summary(self):
        if self.ds is None:
            return "No dataset loaded."
        try:
            vars = self.ds.data_vars.keys()
            dims = self.ds.dims.items()
            attrs = self.ds.attrs.items()
            # Build a table of variables with their dimensions and CF attributes
            var_rows = []
            for var in vars:
                v = self.ds[var]
                dims_str = ', '.join([f"{d} ({v.sizes[d]})" for d in v.dims])
                long_name = v.attrs.get('long_name', '-')
                standard_name = v.attrs.get('standard_name', '-')
                grid_mapping = v.attrs.get('grid_mapping', '-')
                var_rows.append(html.Tr([
                    html.Td(var),
                    html.Td(dims_str),
                    html.Td(long_name),
                    html.Td(standard_name),
                    html.Td(grid_mapping)
                ]))
            # If grid mapping variable exists, show its attributes
            grid_mapping_info = []
            for var in vars:
                v = self.ds[var]
                grid_mapping_name = v.attrs.get('grid_mapping')
                if grid_mapping_name and grid_mapping_name in self.ds:
                    gm_var = self.ds[grid_mapping_name]
                    grid_mapping_info.append(html.Div([
                        html.H6(f"Grid Mapping: {grid_mapping_name}"),
                        html.Ul([
                            html.Li(f"{k}: {v}") for k, v in gm_var.attrs.items()
                        ])
                    ]))
            # Make dataset metadata div much neater and more readable
            return html.Div([
                html.H5("Dataset Metadata (CF Conventions)", style={"marginTop": "10px", "color": "#222"}),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Variable"),
                        html.Th("Dimensions"),
                        html.Th("long_name"),
                        html.Th("standard_name"),
                        html.Th("grid_mapping")
                    ])),
                    html.Tbody(var_rows)
                ], id="dataset-info-table", style={"width": "100%", "fontSize": "13px", "color": "#222", "background": "#fff", "borderRadius": "6px", "padding": "8px", "marginBottom": "16px"}),
                html.H6("Global Attributes", style={"color": "#222", "marginTop": "12px"}),
                html.Table([
                    html.Tbody([
                        html.Tr([
                            html.Td(html.B(str(k)), style={"paddingRight": "8px"}),
                            html.Td(str(v))
                        ]) for k, v in attrs
                    ])
                ], style={"fontSize": "13px", "background": "#f8f9fa", "borderRadius": "6px", "padding": "8px", "width": "auto"}),
                *grid_mapping_info
            ], className="dataset-info-container", style={"background": "#fff", "color": "#222", "borderRadius": "8px", "padding": "16px", "marginBottom": "16px", "boxShadow": "0 2px 8px rgba(0,0,0,0.04)"})
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

    def setup_variable_metadata_callback(self):
        @self.app.callback(
            Output('variable-metadata-container', 'children'),
            Input('variable-dropdown', 'value')
        )
        def update_variable_metadata(selected_var):
            ds = self.ds
            if ds is None or not selected_var or selected_var not in ds:
                return "No variable selected."
            v = ds[selected_var]
            # Attributes (show all, including long_name, units, etc.)
            attr_items = []
            for k, vv in v.attrs.items():
                attr_items.append(html.Tr([
                    html.Td(str(k), style={"fontWeight": "bold", "paddingRight": "8px"}),
                    html.Td(str(vv))
                ]))
            attr_table = html.Table([
                html.Tbody(attr_items)
            ], style={"fontSize": "14px", "marginBottom": "12px", "background": "#f8f9fa", "borderRadius": "6px", "padding": "8px", "width": "auto"})
            # Dimensions (use .sizes for mapping from name to length)
            dims = v.dims
            dim_items = [html.Li(f"{d}: {v.sizes[d]}") for d in dims]
            # Coordinates (show preview values, handle loading errors, only load a small slice)
            coord_items = []
            for c in v.coords:
                try:
                    arr = v.coords[c].isel({d: slice(0, 5) for d in v.coords[c].dims}).values
                    preview = ', '.join([str(arr[i]) for i in range(min(5, arr.size))])
                    if v.coords[c].sizes and list(v.coords[c].sizes.values())[0] > 5:
                        preview += ', ...'
                    coord_items.append(html.Li([
                        html.B(c), f": [{preview}] (size={v.coords[c].sizes[list(v.coords[c].sizes.keys())[0]]})"
                    ]))
                except Exception as e:
                    coord_items.append(html.Li([
                        html.B(c), f": [Error loading values: {e}]"
                    ]))
            # Grid mapping
            grid_mapping_name = v.attrs.get('grid_mapping')
            grid_mapping_info = None
            if grid_mapping_name and grid_mapping_name in ds:
                gm_var = ds[grid_mapping_name]
                grid_mapping_info = html.Div([
                    html.H6(f"Grid Mapping: {grid_mapping_name}"),
                    html.Ul([
                        html.Li(f"{k}: {vv}") for k, vv in gm_var.attrs.items()
                    ])
                ])
            return html.Div([
                html.H5(f"Variable: {selected_var}"),
                html.H6("Attributes:"),
                attr_table,
                html.H6("Dimensions:"),
                html.Ul(dim_items),
                html.H6("Coordinates (preview):"),
                html.Ul(coord_items),
                grid_mapping_info if grid_mapping_info else None
            ], style={"background": "#fff", "color": "#222", "borderRadius": "8px", "padding": "16px", "marginBottom": "16px", "boxShadow": "0 2px 8px rgba(0,0,0,0.04)"})

    def run(self):
        self.app.run(debug=True, host='0.0.0.0')

if __name__ == '__main__':
    app = ZarrDataViewerApp()
    app.run()

