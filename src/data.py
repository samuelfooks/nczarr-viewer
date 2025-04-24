from dash import Output, Input, State, html
import numpy as np

class DataDisplay:
    def __init__(self, app, ds, dataseturl, dataset_engine):
        self.app = app
        self.ds = ds
        self.dataseturl = dataseturl
        self.dataset_engine = dataset_engine

    def setup_callbacks(self):
        @self.app.callback(
            Output('data-array-display', 'children'),
            Input('show-data-button', 'n_clicks'),
            State('variable-dropdown', 'value'),
            State('selected-dimensions-store', 'data')
        )
        def display_data(n_clicks, selected_var, selected_dims):
            if n_clicks > 0 and selected_var:
                try:
                    selection = {dim: self.ds[selected_var][dim].values[val[0]:val[1]] if isinstance(val, list) else self.ds[selected_var][dim].values[val] for dim, val in selected_dims.items()}
                    data_retriever = DataRetriever(selected_var, selection, self.dataseturl, self.dataset_engine)
                    selected_data = data_retriever.retrieve_data_using_dimension_selections()
                    array_values = selected_data.values
                    return html.Div([
                        html.P(f"Max: {np.nanmax(array_values)}"),
                        html.P(f"Min: {np.nanmin(array_values)}"),
                        html.P(f"Mean: {np.nanmean(array_values)}"),
                        html.P(f"Median: {np.nanmedian(array_values)}"),
                        html.P(f"Std: {np.nanstd(array_values)}")
                    ])
                except Exception as e:
                    return html.Div(["Error displaying data: ", str(e)])
            return html.Div("Show Max/Min/Mean/Med/STDEV")


# data_plot.py
from dash import Output, Input, State
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
import io
import base64
import numpy as np

class DataPlot:
    def __init__(self, app, ds, dim_select, dataseturl, dataset_engine):
        self.app = app
        self.ds = ds
        self.dim_select = dim_select
        self.dataseturl = dataseturl
        self.dataset_engine = dataset_engine

    def setup_callbacks(self):
        @self.app.callback(
            Output('map', 'src'),
            Output('map-container', 'style'),
            Input('show-plot-button', 'n_clicks'),
            State('variable-dropdown', 'value'),
            State('selected-dimensions-store', 'data'),
            State('lat-dim-dropdown', 'value'),
            State('lon-dim-dropdown', 'value')
        )
        def display_plot(n_clicks, selected_var, selected_dims, lat_dim, lon_dim):
            if n_clicks > 0 and selected_var and lat_dim and lon_dim:
                try:
                    selection = {dim: self.ds[selected_var][dim].values[val[0]:val[1]] if isinstance(val, list) else self.ds[selected_var][dim].values[val] for dim, val in selected_dims.items()}
                    data_retriever = DataRetriever(selected_var, selection, self.dataseturl, self.dataset_engine)
                    selected_data = data_retriever.retrieve_data_using_dimension_selections()

                    lons = selected_data[lon_dim].values
                    lats = selected_data[lat_dim].values
                    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

                    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
                    ax.coastlines()
                    ax.set_extent(extent)
                    lon, lat = np.meshgrid(lons, lats)
                    img = ax.pcolormesh(lon, lat, selected_data.values, transform=ccrs.PlateCarree(), shading='auto')
                    ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
                    cbar = plt.colorbar(img, ax=ax, orientation='vertical')

                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    plt.close(fig)
                    buffer.seek(0)
                    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    return f"data:image/png;base64,{img_str}", {'display': 'block'}
                except Exception as e:
                    print(f"Plot error: {e}")
            return "", {'display': 'none'}


# data_retriever.py
import xarray as xr
from copernicusmarine.core_functions import custom_open_zarr

class DataRetriever:
    def __init__(self, selected_var, user_selection, dataseturl, dataset_engine):
        self.selected_var = selected_var
        self.user_selection = user_selection
        self.dataseturl = dataseturl
        self.dataset_engine = dataset_engine

    def retrieve_data_using_dimension_selections(self):
        try:
            ds = xr.open_dataset(self.dataseturl, engine=self.dataset_engine)
            return ds[self.selected_var].sel(**self.user_selection).compute()
        except Exception:
            try:
                ds = custom_open_zarr.open_zarr(self.dataseturl, copernicus_marine_username='sfooks')
                return ds[self.selected_var].sel(**self.user_selection).compute()
            except Exception as e:
                print(f"Data retrieval error: {e}")
                return None
        return None