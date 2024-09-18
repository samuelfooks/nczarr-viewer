import dash
from dash import Dash, html, dcc, Input, Output, State, ALL, callback_context, dcc, MATCH, dash_table
import dash_bootstrap_components as dbc
from plotly import graph_objects as go
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
import io
import base64
import json
import os
import leafmap.foliumap as leafmap
import folium
import argparse
from copernicusmarine.core_functions import custom_open_zarr
import signal
# Main App Class
class TimeoutException(Exception):
    pass
class ZarrDataViewerApp:
    def __init__(self, dataseturl):
        self.dataseturl = dataseturl
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.config.suppress_callback_exceptions = True
        self.ds = None  # Initialize the dataset to None
        self.dataset_engine = None  # Initialize the engine to None

        # Try to read the dataset
        self.ds, self.dataset_engine = self.read_dataset_metadata(dataseturl)
        
        # Set up the layout depending on whether the dataset was successfully loaded
        self.layout_manager = LayoutManager(self.app, self.ds, dataseturl)
        self.layout_manager.setup_layout()

        # If dataset was successfully loaded, initialize other components
        if self.ds is not None:
            self.variable_selection = VariableSelection(self.app, self.ds)
            self.dimension_selection = DimensionSelection(self.app, self.ds)
            self.data_display = DataDisplay(self.app, self.ds, self.dataseturl, self.dataset_engine)
            self.data_plot = DataPlot(self.app, self.ds, self.dimension_selection, self.dataseturl, self.dataset_engine)
            self.reset_functionality = ResetFunctionality(self.app, self.ds)

            # Set up the callbacks
            self.variable_selection.setup_callbacks()
            self.dimension_selection.setup_callbacks()
            self.data_display.setup_callbacks()
            self.data_plot.setup_callbacks()
            self.reset_functionality.setup_callbacks()

    def timeout_handler(signum, frame):
        raise TimeoutException
    
    def read_dataset_metadata(self, dataseturl):
        # Set the timeout handler
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(10)  # Set the timeout to 10 seconds

        try:
            if '.nc' in dataseturl:
                dataset_engine = 'netcdf4'
            elif '.zarr' in dataseturl:
                dataset_engine = 'zarr'
            else:
                raise ValueError("Unsupported file format")

            print(f"Trying to open dataset {dataseturl} with engine {dataset_engine}")
            ds = xr.open_dataset(dataseturl, engine=dataset_engine)
            print(f"Successfully opened dataset {dataseturl} with engine {dataset_engine}")
            return ds, dataset_engine
        except TimeoutException:
            print(f"Timeout occurred while trying to open dataset {dataseturl}")
        except Exception as e:
            print(f"Failed to open file {dataseturl}: {e}")
        finally:
            signal.alarm(0)  # Disable the alarm

        return None

    def run(self):
        self.app.run_server(debug=True, host='0.0.0.0')

class LayoutManager:
    def __init__(self, app, ds, dataseturl):
        self.dataseturl = dataseturl
        self.ds = ds
        self.app = app

    def setup_layout(self):
        if self.ds is None:
            # Display an error message if dataset failed to load
            dataset_info = "Error: Failed to load dataset."
            dataset_url = f"Dataset URL: {self.dataseturl}"
            self.app.layout = html.Div([
                html.H1("Zarr NetCDF Data Viewer"),
                html.Div(id='dataset-info-container', children=[
                    html.Div(dataset_info),
                    html.Div(dataset_url)
                ])
            ])
        else:
            # Display dataset info and URL if the dataset loaded successfully
            dataset_info = str(self.ds)
            dataset_url = f"Dataset URL: {self.dataseturl}"
            self.app.layout = html.Div([
                html.H1("Zarr NetCDF Data Viewer"),
                
                html.Div([
                    html.Div([
                        dcc.Dropdown(
                            id='variable-dropdown',
                            options=[{'label': var, 'value': var} for var in self.ds.data_vars],
                            placeholder="Select a variable"
                        ),
                        html.Div(id='dimension-checklist-container')
                    ], className='left-container'),
                    html.Div([
                        html.Div(id='dimension-dropdowns-container'),
                        html.Div(id='selection-display')
                    ], className='right-container')
                ], className='top-container'),
                html.Div([
                    html.Div(id='map-container', className='map-container', children=[html.Img(id='map')]),
                    html.Div(id='data-array-display')
                ], className='bottom-container'),
                html.Button('Max/Min/Mean/Med/STDEV', id='show-data-button', n_clicks=0),
                html.Button('Show Plot', id='show-plot-button', n_clicks=0),
                html.Button('Reset', id='reset-button', n_clicks=0),
                html.Div(id='dataset-info-container', children=[
                    self.add_dataset_info(),
                    # self.add_dataset_url()
                ]),
                dcc.Store(id='selected-dimensions-store')
            ])
    
    def add_dataset_info(self):
        """
        Method to add dataset information to the layout.
        """
        if self.ds is None:
            return html.Div("Error: Dataset could not be loaded.", className="dataset-error")

        # Create a table to display dataset information
        dataset_info_table = html.Table([
            html.Tbody([
                html.Tr([html.Td("Dimensions:"), html.Td(str(list(self.ds.dims)))]),
                # html.Tr([html.Td("Coordinates:"), html.Td(str(self.ds.coords))]),
                html.Tr([html.Td("Data Variables:"), html.Td(str(list(self.ds.data_vars)))]),
                html.Tr([html.Td("Attributes:"), html.Td(str(self.ds.attrs))])
            ])
        ], id="dataset-info-table")

        dataset_url_div = html.Div([
            html.H3("Dataset URL"),
            html.P(self.dataseturl)
        ], id="dataset-url-div")

        return html.Div([dataset_url_div, dataset_info_table], className="dataset-info-container")

# VariableSelection Class
class VariableSelection:
    def __init__(self, app, ds):
        self.app = app
        self.ds = ds

    def setup_callbacks(self):
        @self.app.callback(
            Output('variable-dropdown', 'options'),
            Input('variable-dropdown', 'value')
        )
        def update_variable_options(selected_var):
            options = [{'label': var, 'value': var} for var in self.ds.data_vars]
            return options

# DimensionSelection Class
class DimensionSelection:
    def __init__(self, app, ds):
        self.app = app
        self.ds = ds
        self.dim_selections = {}

    def setup_callbacks(self):
        @self.app.callback(
            Output('dimension-checklist-container', 'children'),
            Input('variable-dropdown', 'value')
        )
        def update_dimension_checklist(selected_var):
            if selected_var:
                return self.generate_dimension_checklist(selected_var)
            return ""

        @self.app.callback(
            Output('dimension-dropdowns-container', 'children'),
            Input('dimension-checklist', 'value'),
            State('variable-dropdown', 'value')
        )
        def update_dimension_controls(selected_dims, selected_var):
            if selected_var and selected_dims:
                return self.generate_dimension_controls(selected_dims, selected_var)
            return ""

        @self.app.callback(
            Output('selected-dimensions-store', 'data'),
            [Input({'type': 'dimension-slider', 'index': ALL}, 'value'),
             Input({'type': 'dimension-dropdown', 'index': ALL}, 'value')],
            State('variable-dropdown', 'value')
        )
        def store_selected_dimensions(slider_values, dropdown_values, selected_var):
            selected_dims = self.store_user_selection(selected_var, slider_values, dropdown_values)
            self.dim_selections = selected_dims  # Update dim_selections
            return selected_dims

        @self.app.callback(
            Output({'type': 'slider-output', 'index': MATCH}, 'children'),
            Input({'type': 'dimension-slider', 'index': MATCH}, 'value'),
            State('variable-dropdown', 'value')
        )
        def update_slider_output(slider_value, selected_var):
            if selected_var is None or slider_value is None:
                return ""
            
            try:
                dim = callback_context.triggered[0]['prop_id'].split('.')[0]
                if not dim:
                    return "Make a selection"
                dim_dict = json.loads(dim.replace("'", "\""))
                dim_index = dim_dict['index']
                dim_values = self.ds[selected_var][dim_index].values
                 # Sort the dimension values
                sorted_dim_values = sorted(dim_values)
                start_idx, end_idx = slider_value
                if start_idx < 0 or end_idx >= len(dim_values):
                    return "Index out of bounds"

                start_value = f"{sorted_dim_values[start_idx]:.4f}"
                end_value = f"{sorted_dim_values[end_idx]:.4f}"
                return f"Selected range: {start_value} to {end_value}"
            except Exception as e:
                print(f"Error: {e}")
                return "Error updating slider output"
        
    def generate_dimension_checklist(self, selected_var):
        if selected_var is None:
            return []
        
        dimensions = self.ds[selected_var].dims
        lat_present = any('lat' in dim.lower() for dim in dimensions)
        lon_present = any('lon' in dim.lower() for dim in dimensions)
        
        if not lat_present or not lon_present:
            return html.Div("Error: Latitude and/or Longitude dimensions are not present in the selected variable.")
        
        return html.Div([
            html.Label("Select dimensions to filter:"),
            dcc.Checklist(
                id='dimension-checklist',
                options=[{'label': dim, 'value': dim} for dim in dimensions],
                value=[dim for dim in dimensions if 'lat' in dim.lower() or 'lon' in dim.lower()]
            )
        ])

    def generate_dimension_controls(self, selected_dims, selected_var):
        if selected_var is None or selected_dims is None:
            return []

        dimension_controls = []
        for dim in selected_dims:
            if 'lat' in dim.lower() or 'lon' in dim.lower():
                dimension_controls.append(self.create_range_slider(dim, selected_var))
            else:
                dimension_controls.append(self.create_dropdown(dim, selected_var))
        return dimension_controls

    def create_range_slider(self, dim, selected_var):
        dim_values = self.ds[selected_var][dim].values
        sorted_dim_values = sorted(dim_values)
        min_val = 0
        max_val = len(sorted_dim_values) - 1
        range_25 = int(0.25 * max_val)
        range_75 = int(0.75 * max_val)

        # Create marks at regular intervals
        step = max(1, len(sorted_dim_values) // 10)  # Adjust the step as needed
        marks = {i: f"{sorted_dim_values[i]:.4f}" for i in range(0, len(sorted_dim_values), step)}

        return html.Div([
            html.Label(f'Select {dim} range'),
            dcc.RangeSlider(
                id={'type': 'dimension-slider', 'index': dim},
                min=min_val,
                max=max_val,
                value=[range_25, range_75],
                marks=marks,
                step=1
            ),
            html.Div(id={'type': 'slider-output', 'index': dim})
        ])

    def create_dropdown(self, dim, selected_var):
        return html.Div([
            html.Label(f'Select {dim}'),
            dcc.Dropdown(
                id={'type': 'dimension-dropdown', 'index': dim},
                options=[{'label': str(val), 'value': idx} for idx, val in enumerate(self.ds[selected_var][dim].values)],
                placeholder=f"Select {dim}"
            )
        ])
    def store_user_selection(self, selected_var, slider_values, dropdown_values):
        selected_dims = {}
        ctx = callback_context
        flattened_inputs = [item for sublist in ctx.inputs_list for item in sublist]
        slider_inputs = [input for input in flattened_inputs if 'dimension-slider' in input['id']['type']]
        dropdown_inputs = [input for input in flattened_inputs if 'dimension-dropdown' in input['id']['type']]

        # Process slider values
        for slider_val, slider_input in zip(slider_values, slider_inputs):
            if slider_input:
                dimension_name = slider_input['id']['index']
                if slider_val and len(slider_val) > 0:
                    sorted_dim_values = sorted(self.ds[selected_var][dimension_name].values)
                    start_idx = slider_val[0]
                    end_idx = slider_val[1]
                    selected_dims[dimension_name] = [start_idx, end_idx]

        # Process dropdown values
        for dropdown_val, dropdown_input in zip(dropdown_values, dropdown_inputs):
            if dropdown_input:
                dimension_name = dropdown_input['id']['index']
                if dropdown_val:
                    selected_dims[dimension_name] = dropdown_val
        print(f"Selected dimensions: {selected_dims}")
        return selected_dims

class DataRetriever:
    def __init__(self, selected_var, user_selection, dataseturl, dataset_engine):
        self.dataseturl = dataseturl
        self.dataset_engine = dataset_engine
        self.user_selection = user_selection
        self.selected_var = selected_var
    
    def open_standard_file(self, dataseturl, selected_var, user_selection, dataset_engine):
        try:
            ds = xr.open_dataset(dataseturl, engine=dataset_engine)
        except Exception as e:
            print(f"Failed to open file {dataseturl} engine {dataset_engine}: {e}")

        selected_array = ds[selected_var].sel(**user_selection)
        values = selected_array.compute()
        return values
    

    def open_cmems_file(self, file, selected_var, user_selection):
        username = 'sfooks'
        ds = custom_open_zarr.open_zarr(
            file,
            copernicus_marine_username=username
        )
        selected_array = ds[selected_var].sel(**user_selection)
        values = selected_array.compute()
        return values
 

    def retrieve_data_using_dimension_selections(self):
        try:
            data = self.open_standard_file(self.dataseturl, self.selected_var, self.user_selection, self.dataset_engine)
            return data
        except Exception as e:
            print(f"Failed to open file {self.dataseturl}: {e} trying custom open")
            try:
                
                data = self.open_cmems_file(self.dataseturl, self.selected_var, self.user_selection)
                print(f"Successfully retrieved data using custom open")
                return data
            except Exception as e:
                print(f"Failed to open file {self.dataseturl}: {e} using custom open")
                return None    

# DataDisplay Class
class DataDisplay:
    def __init__(self, app, ds, dataseturl, dataset_engine):
        self.app = app
        self.ds = ds
        self.dataset_engine = dataset_engine
        self.dim_select = DimensionSelection(app, ds)
        self.dataseturl = dataseturl
    
    def setup_callbacks(self):
        @self.app.callback(
            Output('data-array-display', 'children'),
            Input('show-data-button', 'n_clicks'),
            State('variable-dropdown', 'value'),
            State('selected-dimensions-store', 'data')
        )
    
        def display_data(n_clicks, selected_var, selected_dims):
            print('displaying data')
            if n_clicks > 0 and selected_var:
                try:
                    selection = {}
                    for dim, value in selected_dims.items():
                        if isinstance(value, list):
                            selection[dim] = self.ds[selected_var][dim].values[value[0]:value[1]]
                        elif 'lat' in dim.lower() or 'lon' in dim.lower():
                            selection[dim]= slice(value[0], value[1])
                        elif isinstance(value, int):
                            selection[dim] = self.ds[selected_var][dim].values[value]
                    data_retriever = DataRetriever(selected_var, selection, self.dataseturl, self.dataset_engine)
                    selected_data = data_retriever.retrieve_data_using_dimension_selections()

                    print(selected_data)
                    array_values = selected_data.values
                    max_value = float(np.nanmax(array_values))
                    min_value = float(np.nanmin(array_values))
                    mean_value = float(np.nanmean(array_values))
                    median_value = float(np.nanmedian(array_values))
                    std_value = float(np.nanstd(array_values))

                    return html.Div([
                        html.H4("Selected Data Array"),
                        html.P(f"Max Value: {max_value}"),
                        html.P(f"Min Value: {min_value}"),
                        html.P(f"Mean Value: {mean_value}"),
                        html.P(f"Median Value: {median_value}"),
                        html.P(f"Standard Deviation: {std_value}")
                    ])
                except Exception as e:
                    return html.Div([
                        html.H4("Error selecting data"),
                        html.P(str(e))
                    ])
            return html.Div("Show Max/Min/Mean/Med/STDEV")

# DataPlot Class
class DataPlot:
    def __init__(self, app, ds, dim_select, dataseturl, dataset_engine):
        self.app = app
        self.ds = ds
        self.dim_select = dim_select
        self.dataseturl = dataseturl
        self.dataset_engine = dataset_engine

    def setup_callbacks(self):
        @self.app.callback(
            [Output('map', 'src'),
             Output('map-container', 'style')],
            Input('show-plot-button', 'n_clicks'),
            State('variable-dropdown', 'value'),
            State('selected-dimensions-store', 'data')
        )
        def display_plot(n_clicks, selected_var, selected_dims):
            if n_clicks > 0 and selected_var:
                map_src = self.plot_selected_data(selected_var, selected_dims)
                if map_src:
                    return map_src, {'display': 'block'}
            return "", {'display': 'none'}

    def plot_selected_data(self, selected_var, selected_dims):
        if selected_var is None:
            return ""
        try:
            selection = {}
            for dim, value in selected_dims.items():
                    if isinstance(value, list):
                        selection[dim] = self.ds[selected_var][dim].values[value[0]:value[1]]
                    elif 'lat' in dim.lower() or 'lon' in dim.lower():
                        selection[dim]= slice(value[0], value[1])
                    elif isinstance(value, int):
                        selection[dim] = self.ds[selected_var][dim].values[value]
            data_retriever = DataRetriever(selected_var, selection, self.dataseturl, self.dataset_engine)
            
            selected_data = data_retriever.retrieve_data_using_dimension_selections()
            for dim in selected_dims:
                if 'lon' in dim.lower():
                    lons = selected_data[dim].values
                if 'lat' in dim.lower():
                    lats = selected_data[dim].values
            extent = [lons.min(), lons.max(), lats.min(), lats.max()]

            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
            ax.coastlines()
            ax.set_extent(extent)
            lon, lat = np.meshgrid(lons, lats)
            img = ax.pcolormesh(lon, lat, selected_data.values, transform=ccrs.PlateCarree(), shading='auto')
            
            ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
            ax.add_feature(cartopy.feature.LAND, edgecolor='black')
            ax.add_feature(cartopy.feature.OCEAN)
            ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
            ax.add_feature(cartopy.feature.RIVERS)
            cbar = plt.colorbar(img, ax=ax, orientation='vertical', label=selected_var, shrink=0.8)
            cbar.set_label(selected_var, fontsize=8)

            dim_ranges = []
            for dim, value in selected_dims.items():
                if 'lon' in dim.lower():
                    dim_ranges.append(f"Lon: {lons.min():.4f} to {lons.max():.4f}")
                elif 'lat' in dim.lower():
                    dim_ranges.append(f"Lat: {lats.min():.4f} to {lats.max():.4f}")
                else:
                    dim_ranges.append(f"{dim}: {self.ds[dim].values[value]}")

            title_str = f"{selected_var}\n" + "\n".join(dim_ranges)
            plt.title(title_str, fontsize=10, loc='left')
            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close(fig)
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"Error during plotting: {e}")
            return ""
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
                # Return an empty dictionary to clear the dimensions store
                return (
                    {},
                    [],
                    [],
                    'No data selected.',
                    ""
                )
            return dash.no_update
def is_url(datasetPath):
    return datasetPath.startswith('http://') or datasetPath.startswith('https://')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Zarr Data Viewer App with a dataset URL or local file path.')
    parser.add_argument('datasetPath', type=str, help='The URL or local file path of the dataset to load')
    args = parser.parse_args()

    if is_url(args.datasetPath):
        file_path = args.datasetPath
    else:
        if not os.path.isfile(args.datasetPath):
            raise FileNotFoundError(f"The file {args.datasetPath} does not exist.")
        file_path = args.datasetPath

    app = ZarrDataViewerApp(file_path)
    app.run()


