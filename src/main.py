import dash
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# Setup module for initialization tasks
from setup import setup_cartopy_data

from variables import VariableSelection
from dimension import DimensionSelection
from data import DataManager, DatasetLoader


class TimeoutException(Exception):
    pass


class ViewerApp:
    def __init__(self):
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.config.suppress_callback_exceptions = True
        self.ds = None
        self.dataset_engine = None
        self.dataseturl = None
        self.dataset_loader = DatasetLoader()
        setup_cartopy_data()
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("NCZarr Viewer",
                        className="text-center mb-4"), width=12)
            ]),
            # Top row: Load Dataset and Variable Selection side by side
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Load Dataset"),
                        dbc.CardBody([
                            html.Div([
                                html.Label("Choose from predefined datasets:",
                                           className="mb-2",
                                           style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id='dataset-dropdown',
                                    options=[
                                        {
                                            "label": ("https://s3.waw3-1.cloudferro.com/"
                                                      "emodnet/emodnet_seabed_habitats/12548/"
                                                      "EUSeaMap_2023.zarr"),
                                            "value": ("https://s3.waw3-1.cloudferro.com/"
                                                      "emodnet/emodnet_seabed_habitats/12548/"
                                                      "EUSeaMap_2023.zarr")
                                        },
                                        {
                                            "label": ("https://s3.waw3-1.cloudferro.com/"
                                                      "emodnet/emodnet_arco/emodnet_chemistry/"
                                                      "water_body_dissolved_inorganic_nitrogen/"
                                                      "water_body_dissolved_inorganic_nitrogen_"
                                                      "masked_using_relative_error_threshold_0.5_"
                                                      "baltic_sea/Water_body_dissolved_inorganic_"
                                                      "nitrogen.4Danl.zarr"),
                                            "value": ("https://s3.waw3-1.cloudferro.com/"
                                                      "emodnet/emodnet_arco/emodnet_chemistry/"
                                                      "water_body_dissolved_inorganic_nitrogen/"
                                                      "water_body_dissolved_inorganic_nitrogen_"
                                                      "masked_using_relative_error_threshold_0.5_"
                                                      "baltic_sea/Water_body_dissolved_inorganic_"
                                                      "nitrogen.4Danl.zarr")
                                        },
                                        {
                                            "label": ("https://s3.waw3-1.cloudferro.com/"
                                                      "emodnet/emodnet_geology/12495/"
                                                      "EMODnet_Seabed_Substrate_1M.zarr"),
                                            "value": ("https://s3.waw3-1.cloudferro.com/"
                                                      "emodnet/emodnet_geology/12495/"
                                                      "EMODnet_Seabed_Substrate_1M.zarr")
                                        },
                                        {
                                            "label": ("https://s3.waw3-1.cloudferro.com/mdl-arco-geo-001/"
                                                      "arco/ARCTIC_MULTIYEAR_BGC_002_005/"
                                                      "cmems_mod_arc_bgc_my_ecosmo_P1M_202105/"
                                                      "geoChunked.zarr"),
                                            "value": ("https://s3.waw3-1.cloudferro.com/mdl-arco-geo-001/"
                                                      "arco/ARCTIC_MULTIYEAR_BGC_002_005/"
                                                      "cmems_mod_arc_bgc_my_ecosmo_P1M_202105/"
                                                      "geoChunked.zarr")
                                        }
                                    ],
                                    placeholder='Select a dataset from the list',
                                    style={'width': '100%'},
                                    clearable=True,
                                    searchable=True
                                ),
                                html.Br(),
                                html.Label("Or enter a custom URL:",
                                           className="mb-2",
                                           style={"fontWeight": "bold"}),
                                dcc.Input(
                                    id='dataset-url-input',
                                    type='text',
                                    placeholder='Enter custom dataset URL or path',
                                    style={'width': '100%'}
                                ),
                                html.Div([
                                    html.Small([
                                        html.I("💡 Tip: For netcdf files on s3 storage, add ",
                                               style={"color": "#6c757d"}),
                                        html.Code("#mode=bytes", style={"backgroundColor": "#f8f9fa", "padding": "2px 4px", "borderRadius": "3px"}),
                                        html.I(" at the end of the URL to ensure proper data access.",
                                               style={"color": "#6c757d"})
                                    ], style={"marginTop": "5px", "fontSize": "12px"})
                                ]),
                                html.Br(),
                                html.Br(),
                                html.Label("Backend Selection:",
                                           className="mb-2",
                                           style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id='backend-dropdown',
                                    options=[
                                        {"label": "Auto-detect", "value": "auto"},
                                        {"label": "xarray", "value": "xarray"},
                                        {"label": "Copernicus Marine",
                                         "value": "copernicusmarine"},
                                    ],
                                    value="auto",
                                    style={'width': '100%'}
                                ),
                                html.Br(),
                                html.Label("Backend Configuration (JSON):",
                                           className="mb-2",
                                           style={"fontWeight": "bold"}),
                                html.Div([
                                    html.A("How to use kwargs",
                                           href="#",
                                           id="help-link",
                                           style={"color": "#007bff", "textDecoration": "underline"}),
                                    html.Span(" - Click for examples",
                                              className="text-muted",
                                              style={"fontSize": "12px"})
                                ], className="mb-2"),
                                dcc.Textarea(
                                    id='additional-params-input',
                                    placeholder='{"backend": "xarray", "engine": "zarr", "chunks": {"time": 1}}',
                                    style={'width': '100%', 'height': '80px'}
                                ),
                                # Help modal content (hidden by default)
                                dbc.Modal([
                                    dbc.ModalHeader(dbc.ModalTitle(
                                        "How to use Backend Configuration")),
                                    dbc.ModalBody([
                                        html.H6("xarray Backend Examples:"),
                                        html.Pre('''
{
  "backend": "xarray",
  "engine": "zarr",
  "chunks": {"time": 1},
  "decode_timedelta": true
}

{
  "backend": "xarray",
  "engine": "netcdf4",
  "chunks": {"time": 1, "lat": 100, "lon": 100},
  "decode_timedelta": false
}

{
  "backend": "xarray",
  "engine": "h5netcdf",
  "chunks": {"time": 1},
  "decode_cf": true,
  "mask_and_scale": true
}
                                         ''', style={"backgroundColor": "#f8f9fa", "padding": "10px", "borderRadius": "4px"}),

                                        html.H6(
                                            "Copernicus Marine Examples:"),
                                        html.Pre('''
{
  "backend": "copernicusmarine",
  "engine": "copernicusmarinetoolbox",
  "username": "your_username",
  "password": "your_password",
  "dataset_id": "dataset_identifier"
}

{
  "backend": "copernicusmarine",
  "engine": "custom_open_zarr.open_zarr",
  "chunks": {"time": 1}
}
                                         ''', style={"backgroundColor": "#f8f9fa", "padding": "10px", "borderRadius": "4px"}),

                                        html.H6("Notes:"),
                                        html.Ul([
                                            html.Li(
                                                "The 'backend' field in JSON overrides the dropdown selection"),
                                            html.Li(
                                                "For xarray: engine can be 'zarr', 'netcdf4'"),
                                            html.Li(
                                                "For Copernicus Marine: engine can be 'copernicusmarinetoolbox' or 'custom_open_zarr.open_zarr'"),
                                            html.Li(
                                                "Additional parameters like chunks, decode_timedelta, s3 credentials can be included"),
                                            html.Li(
                                                "All xarray parameters (chunks, decode_timedelta, etc.) can be passed through JSON"),
                                            html.Li(
                                                "Common xarray params: chunks, decode_cf, decode_timedelta, mask_and_scale, storage_options")
                                        ])
                                    ]),
                                    dbc.ModalFooter(
                                        dbc.Button(
                                            "Close", id="close-help-modal", className="ms-auto")
                                    )
                                ], id="help-modal", is_open=False)
                            ]),
                            html.Br(),
                            html.Br(),
                            dbc.Button('Load Dataset', id='load-dataset-button',
                                       color='primary', n_clicks=0, className='mb-2'),
                            dcc.Loading(html.Div(id='load-status'),
                                        type='default'),
                        ])
                    ], className='mb-3'),
                ], width=10),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Variable Selection"),
                        dbc.CardBody([
                            dcc.Dropdown(id='variable-dropdown',
                                         style={'width': '100%'}),
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
                                html.Label("Data Filter (Min/Max):",
                                           className="mb-1"),
                                dbc.Input(id='data-filter-min', type='number', placeholder='Min value', style={
                                          'width': '45%', 'display': 'inline-block', 'marginRight': '10px'}),
                                dbc.Input(id='data-filter-max', type='number', placeholder='Max value', style={
                                          'width': '45%', 'display': 'inline-block'}),
                            ], className='mb-2'),
                            dbc.Button('Show Data Quick Stats (Max/Min/Med/Stdev)',
                                       id='show-data-button', color='info', n_clicks=0, className='mt-2'),
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
                            dcc.Loading(
                                html.Div(id='data-array-display'), type='circle'),
                        ])
                    ], className='mb-3'),
                ], width=12),
            ]),
            # Fifth row: Plot Selected Data (full width)
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Plot Selected Data"),
                        dbc.CardBody([
                            dcc.Loading(html.Div([
                                html.Div(
                                    id='map-container', children=[html.Img(id='map', style={'width': '100%', 'height': 'auto'})]),
                            ]), type='circle'),

                            dbc.Button('Extract Image', id='extract-plot-button',
                                       color='success', n_clicks=0, className='mt-2'),
                        ])
                    ]),
                ], width=12),
            ]),
            # Sixth row: Raster Image Display (full width)
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Generated Raster Image"),
                        dbc.CardBody([
                            html.Div(id='raster-container', children=[
                                html.P("Click 'Extract Image' to generate a raster image",
                                       className="text-muted text-center")
                            ])
                        ])
                    ]),
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
            # Debug and logging row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            "Debug Output & Logs",
                            dbc.Button("Clear Logs", id="clear-logs-button",
                                       color="warning", size="sm", className="float-end")
                        ]),
                        dbc.CardBody([
                            html.Div(id='debug-output-container'),
                            html.Div(id='logs-container', style={
                                'backgroundColor': '#f8f9fa',
                                'border': '1px solid #dee2e6',
                                'borderRadius': '4px',
                                'padding': '10px',
                                'fontFamily': 'monospace',
                                'fontSize': '12px',
                                'maxHeight': '300px',
                                'overflowY': 'auto',
                                'whiteSpace': 'pre-wrap'
                            })
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
        self.dimension_selection = DimensionSelection(
            self.app, lambda: self.ds)
        self.dimension_selection.setup_callbacks()

        # Use the new unified DataManager for all data operations
        self.data_manager = DataManager(self.app, lambda: self.ds)
        self.data_manager.setup_callbacks()

        self.setup_callbacks()
        self.update_variable_dropdown()
        self.setup_variable_metadata_callback()
        self.setup_help_modal_callback()

    def update_variable_dropdown(self):
        # Update the variable dropdown to list all data variables
        if self.ds is not None:
            from dash import callback_context
            options = [
                {"label": f"{var} ({self.ds[var].attrs.get('long_name', var)})",
                 "value": var}
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
                html.H5("Dataset Metadata (CF Conventions)", style={
                        "marginTop": "10px", "color": "#222"}),
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
                html.H6("Global Attributes", style={
                        "color": "#222", "marginTop": "12px"}),
                html.Table([
                    html.Tbody([
                        html.Tr([
                            html.Td(html.B(str(k)), style={
                                    "paddingRight": "8px"}),
                            html.Td(str(v))
                        ]) for k, v in attrs
                    ])
                ], style={"fontSize": "13px", "background": "#f8f9fa", "borderRadius": "6px", "padding": "8px", "width": "auto"}),
                *grid_mapping_info
            ], className="dataset-info-container", style={"background": "#fff", "color": "#222", "borderRadius": "8px", "padding": "16px", "marginBottom": "16px", "boxShadow": "0 2px 8px rgba(0,0,0,0.04)"})
        except Exception as e:
            return f"Error reading metadata: {e}"

    def setup_help_modal_callback(self):
        @self.app.callback(
            Output("help-modal", "is_open"),
            [Input("help-link", "n_clicks"),
             Input("close-help-modal", "n_clicks")],
            [State("help-modal", "is_open")],
            prevent_initial_call=True
        )
        def toggle_help_modal(n1, n2, is_open):
            if n1 or n2:
                return not is_open
            return is_open

    def setup_callbacks(self):
        @self.app.callback(
            [Output('load-status', 'children'),
             Output('dataset-info-container', 'children'),
             Output('logs-container', 'children')],
            [Input('load-dataset-button', 'n_clicks'),
             Input('clear-logs-button', 'n_clicks')],
            [State('dataset-dropdown', 'value'),
             State('dataset-url-input', 'value'),
             State('backend-dropdown', 'value'),
             State('additional-params-input', 'value')],
            prevent_initial_call=True
        )
        def handle_dataset_operations(load_clicks, clear_clicks, dropdown_url, custom_url, backend, additional_params):
            # Determine which button was clicked
            ctx = callback_context
            if ctx.triggered:
                triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
            else:
                triggered_id = None

            if triggered_id == 'clear-logs-button':
                # Clear logs operation
                self.dataset_loader.clear_logs()
                return dash.no_update, dash.no_update, "Logs cleared."

            elif triggered_id == 'load-dataset-button':
                # Load dataset operation
                # Use dropdown selection if available, otherwise use custom input
                url = dropdown_url if dropdown_url else custom_url

                if not url:
                    return "Please select a dataset from the list or enter a custom URL.", dash.no_update, dash.no_update

                # Parse additional parameters
                kwargs = {}
                if additional_params and additional_params.strip():
                    try:
                        import json
                        kwargs = json.loads(additional_params)
                    except json.JSONDecodeError as e:
                        return f"Invalid JSON in backend args: {e}", dash.no_update, dash.no_update

                # Parse JSON configuration
                config = {}
                if additional_params and additional_params.strip():
                    try:
                        import json
                        config = json.loads(additional_params)
                    except json.JSONDecodeError as e:
                        return f"Invalid JSON in backend args: {e}", dash.no_update, dash.no_update

                # If no backend specified in JSON, use dropdown selection
                if 'backend' not in config and backend != "auto":
                    config['backend'] = backend

                # Show loading spinner automatically via dcc.Loading
                self.dataseturl = url

                # Load dataset with enhanced error handling
                result = self.dataset_loader.load_dataset(
                    url, **config)

                if isinstance(result, tuple) and len(result) == 2:
                    self.ds, self.dataset_engine = result
                    if self.ds is None:
                        error_msg = f"Failed to load dataset from {url}. Error: {self.dataset_engine}"
                        return error_msg, dash.no_update, self.dataset_loader.get_logs()
                    else:
                        success_msg = f"Successfully loaded dataset from {url} using {self.dataset_engine}"
                        return success_msg, self.get_metadata_summary(), self.dataset_loader.get_logs()
                else:
                    return f"Unexpected result from dataset loader: {result}", dash.no_update, self.dataset_loader.get_logs()

            # Default case - shouldn't happen
            return dash.no_update, dash.no_update, dash.no_update

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
                    html.Td(str(k), style={
                            "fontWeight": "bold", "paddingRight": "8px"}),
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
                    arr = v.coords[c].isel({d: slice(0, 5)
                                           for d in v.coords[c].dims}).values
                    preview = ', '.join([str(arr[i])
                                        for i in range(min(5, arr.size))])
                    if v.coords[c].sizes and list(v.coords[c].sizes.values())[0] > 5:
                        preview += ', ...'
                    coord_items.append(html.Li([
                        html.B(
                            c), f": [{preview}] (size={v.coords[c].sizes[list(v.coords[c].sizes.keys())[0]]})"
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
    app = ViewerApp()
    app.run()
