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

        self.app.layout = html.Div([
            html.H1("Zarr/NetCDF Viewer"),
            dcc.Input(id='dataset-url-input', type='text', placeholder='Enter dataset URL or path', style={'width': '80%'}),
            html.Button('Load Dataset', id='load-dataset-button', n_clicks=0),
            html.Div(id='load-status'),
            html.Div(id='main-app-container'),
            # Predefine latent lat/lon dropdowns to avoid State errors
            dcc.Dropdown(id='lat-dim-dropdown', style={'display': 'none'}),
            dcc.Dropdown(id='lon-dim-dropdown', style={'display': 'none'})
        ])

        self.setup_callbacks()

    def setup_callbacks(self):
        @self.app.callback(
            Output('main-app-container', 'children'),
            Output('load-status', 'children'),
            Input('load-dataset-button', 'n_clicks'),
            State('dataset-url-input', 'value'),
            prevent_initial_call=True
        )
        def load_dataset(n_clicks, url):
            if not url:
                return "", "Please provide a valid dataset URL or path."

            self.dataseturl = url
            self.ds, self.dataset_engine = self.read_dataset_metadata(url)

            if self.ds is None:
                return "", f"Failed to load dataset from {url}"

            self.variable_selection = VariableSelection(self.app, self.ds)
            self.dimension_selection = DimensionSelection(self.app, self.ds)
            self.data_display = DataDisplay(self.app, self.ds, self.dataseturl, self.dataset_engine)
            self.data_plot = DataPlot(self.app, self.ds, self.dimension_selection, self.dataseturl, self.dataset_engine)
            self.reset_functionality = ResetFunctionality(self.app, self.ds)

            self.variable_selection.setup_callbacks()
            self.dimension_selection.setup_callbacks()
            self.data_display.setup_callbacks()
            self.data_plot.setup_callbacks()
            self.reset_functionality.setup_callbacks()

            layout_manager = LayoutManager(self.app, self.ds, self.dataseturl)
            return layout_manager.setup_layout(return_layout=True), f"Successfully loaded dataset from {url}"

    def read_dataset_metadata(self, dataseturl):
        try:
            if '.nc' in dataseturl:
                engine = 'netcdf4'
            elif '.zarr' in dataseturl:
                engine = 'zarr'
            else:
                raise ValueError("Unsupported file format")

            print(f"Opening dataset {dataseturl} using engine {engine}")
            ds = xr.open_dataset(dataseturl, engine=engine)
            return ds, engine
        except Exception as e:
            print(f"Error opening dataset: {e}")
            return None, None

    def run(self):
        self.app.run_server(debug=True, host='0.0.0.0')

if __name__ == '__main__':
    app = ZarrDataViewerApp()
    app.run()
