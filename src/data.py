from dash import Output, Input, State, html, dcc
import numpy as np
import plotly.graph_objs as go

class DataDisplay:
    def __init__(self, app, ds_getter, dataseturl_getter, dataset_engine_getter):
        self.app = app
        self.ds_getter = ds_getter
        self.dataseturl_getter = dataseturl_getter
        self.dataset_engine_getter = dataset_engine_getter

    def setup_callbacks(self):
        @self.app.callback(
            Output('data-array-display', 'children'),
            Input('show-data-button', 'n_clicks'),
            State('variable-dropdown', 'value'),
            State('selected-dimensions-store', 'data'),
            State('data-filter-min', 'value'),
            State('data-filter-max', 'value')
        )
        def display_data(n_clicks, selected_var, selected_dims, filter_min, filter_max):
            ds = self.ds_getter()
            dataseturl = self.dataseturl_getter()
            dataset_engine = self.dataset_engine_getter()
            if n_clicks > 0 and selected_var and ds is not None:
                try:
                    if not selected_dims or not hasattr(selected_dims, 'items'):
                        return html.Div(["No dimension selections made."])
                    selection = {dim: ds[selected_var][dim].values[val[0]:val[1]] if isinstance(val, list) else ds[selected_var][dim].values[val] for dim, val in selected_dims.items()}
                    data_retriever = DataRetriever(selected_var, selection, dataseturl, dataset_engine)
                    selected_data = data_retriever.retrieve_data_using_dimension_selections()
                    array_values = selected_data.values
                    # Convert timedelta64 or datetime64 to float for stats
                    if np.issubdtype(array_values.dtype, np.timedelta64):
                        array_values = array_values.astype('timedelta64[h]').astype(float)
                    elif np.issubdtype(array_values.dtype, np.datetime64):
                        array_values = (array_values - array_values.min()).astype('timedelta64[D]').astype(float)
                    # Apply filter if set
                    if filter_min is not None:
                        array_values = np.where(array_values < filter_min, np.nan, array_values)
                    if filter_max is not None:
                        array_values = np.where(array_values > filter_max, np.nan, array_values)
                    max_value = float(np.nanmax(array_values))
                    min_value = float(np.nanmin(array_values))
                    mean_value = float(np.nanmean(array_values))
                    median_value = float(np.nanmedian(array_values))
                    std_value = float(np.nanstd(array_values))
                    return html.Div([
                        html.P(f"Max: {max_value}"),
                        html.P(f"Min: {min_value}"),
                        html.P(f"Mean: {mean_value}"),
                        html.P(f"Median: {median_value}"),
                        html.P(f"Std: {std_value}")
                    ])
                except Exception as e:
                    return html.Div(["Error displaying data: ", str(e)])
            return html.Div("Show Max/Min/Mean/Med/STDEV")


# data_plot.py
from dash import Output, Input, State
import numpy as np
import plotly.graph_objs as go

class DataPlot:
    def __init__(self, app, ds_getter, dim_select, dataseturl_getter, dataset_engine_getter):
        self.app = app
        self.ds_getter = ds_getter
        self.dim_select = dim_select
        self.dataseturl_getter = dataseturl_getter
        self.dataset_engine_getter = dataset_engine_getter

    def setup_callbacks(self):
        @self.app.callback(
            Output('map-container', 'children'),
            Output('map-container', 'style'),
            Input('show-plot-button', 'n_clicks'),
            State('variable-dropdown', 'value'),
            State('selected-dimensions-store', 'data'),
            State('data-filter-min', 'value'),
            State('data-filter-max', 'value')
        )
        def display_plot(n_clicks, selected_var, selected_dims, filter_min, filter_max):
            ds = self.ds_getter()

            dataseturl = self.dataseturl_getter()
            dataset_engine = self.dataset_engine_getter()
            print('plotting')
            if n_clicks > 0 and selected_var and selected_dims and ds is not None:
                print('plotting 2')
                try:
                    # Infer lat/lon/x/y from selected_dims or ds[selected_var].dims
                    lat_dim = None
                    lon_dim = None
                    print(f"selected_dims: {selected_dims}")
                    for dim in selected_dims:
                        if 'lat' in dim.lower() or 'y' in dim.lower():
                            lat_dim = dim
                            print(f"got lat dim {lat_dim}")
                        if 'lon' in dim.lower() or 'x' in dim.lower():
                            lon_dim = dim
                            print(f"got lon dim {lon_dim}")
                    if not lat_dim or not lon_dim:
                        print(f"found no lat lon selected_dims: {selected_dims}")
                        return html.Div(["No valid latitude/longitude dimensions for plotting."]), {'display': 'none'}
                    print(f"got dims lat/lon {lat_dim}/{lon_dim}")
                    selection = {}
                    for dim, val in selected_dims.items():
                        if ('lat' in dim.lower() or 'y' in dim.lower() or 'lon' in dim.lower() or 'x' in dim.lower()) and isinstance(val, list):
                            selection[dim] = ds[selected_var][dim].values[val[0]:val[1]]
                        else:
                            # For dropdowns (like depth), val is an int index

                            selection[dim] = ds[selected_var][dim].values[val] if isinstance(val, int) else val
                    data_retriever = DataRetriever(selected_var, selection, dataseturl, dataset_engine)
                    print(f"DataRetriever: {data_retriever}")
                    selected_data = data_retriever.retrieve_data_using_dimension_selections()
                    lons = selected_data[lon_dim].values
                    lats = selected_data[lat_dim].values
                    data_values = np.array(selected_data.values)
                    # Convert timedelta64 or datetime64 to float for plotting
                    if np.issubdtype(data_values.dtype, np.timedelta64):
                        data_values = data_values.astype('timedelta64[h]').astype(float)
                    elif np.issubdtype(data_values.dtype, np.datetime64):
                        data_values = (data_values - data_values.min()).astype('timedelta64[D]').astype(float)
                    # Apply filter if set
                    if filter_min is not None:
                        data_values = np.where(data_values < filter_min, np.nan, data_values)
                    if filter_max is not None:
                        data_values = np.where(data_values > filter_max, np.nan, data_values)
                    # Remove NaN/Inf from lons/lats and data_values
                    lons = np.ravel(lons)
                    lats = np.ravel(lats)
                    data_values = np.where(np.isfinite(data_values), data_values, np.nan)
                    if lons.size == 0 or lats.size == 0 or np.all(np.isnan(data_values)):
                        return html.Div(["No valid longitude/latitude or data values for plotting."]), {'display': 'none'}
                    # If data_values is 2D, ensure it matches meshgrid shape
                    if data_values.ndim == 2 and (data_values.shape != (lats.size, lons.size)):
                        try:
                            data_values = np.squeeze(data_values)
                            if data_values.shape != (lats.size, lons.size):
                                return html.Div(["Data shape does not match lat/lon selections for plotting."]), {'display': 'none'}
                        except Exception:
                            return html.Div(["Data shape error for plotting."]), {'display': 'none'}
                    # Use Plotly for fast, interactive plotting
                    fig = go.Figure(go.Heatmap(
                        z=data_values,
                        x=lons,
                        y=lats,
                        colorscale='Viridis',
                        colorbar=dict(title=selected_var),
                        zmin=np.nanmin(data_values),
                        zmax=np.nanmax(data_values),
                        hoverongaps=False
                    ))
                    fig.update_layout(
                        title=f"{selected_var} Array Plot",
                        xaxis_title=lon_dim,
                        yaxis_title=lat_dim,
                        height=600,
                        margin=dict(l=0, r=0, t=40, b=0),
                        template="plotly_white"
                    )
                    return dcc.Graph(figure=fig, config={"displayModeBar": True, "scrollZoom": True}), {'display': 'block'}
                except Exception as e:
                    print(f"Plot error: {e}")
                    return html.Div([f"Plot error: {e}"]), {'display': 'none'}
            return html.Div("No plot."), {'display': 'none'}


# data_retriever.py
class DataRetriever:
    def __init__(self, selected_var, user_selection, dataseturl, dataset_engine):
        self.selected_var = selected_var
        self.user_selection = user_selection
        self.dataseturl = dataseturl
        self.dataset_engine = dataset_engine

    def retrieve_data_using_dimension_selections(self):
        """
        Efficiently retrieve a sliced DataArray using integer-based selections (isel) when possible.
        Only call .compute() after all slicing, and return the DataArray (not .values).
        """
        try:
            import xarray as xr
            ds = xr.open_dataset(self.dataseturl, engine=self.dataset_engine)
            var = ds[self.selected_var]
            isel_dict = {}
            sel_dict = {}
            for dim, val in self.user_selection.items():
                print(f"Dimension: {dim}, Value: {val}")
                # If val is a list of two ints, treat as a range of indices (from slider)
                if isinstance(val, list) and len(val) == 2 and all(isinstance(v, int) for v in val):
                    isel_dict[dim] = slice(val[0], val[1])
                # If val is a single int, treat as a single index
                elif isinstance(val, int):
                    isel_dict[dim] = val
                # Otherwise, treat as a value-based selection (e.g., for time)
                else:
                    sel_dict[dim] = val
            if isel_dict:
                var = var.isel(**isel_dict)
            if sel_dict:
                var = var.sel(**sel_dict)
            return var.compute()  # Only compute after all slicing
        except Exception:
            try:
                from copernicusmarine.core_functions import custom_open_zarr
                ds = custom_open_zarr.open_zarr(self.dataseturl, copernicus_marine_username='sfooks')
                var = ds[self.selected_var]
                isel_dict = {}
                sel_dict = {}
                for dim, val in self.user_selection.items():
                    if isinstance(val, list) and len(val) == 2 and all(isinstance(v, int) for v in val):
                        isel_dict[dim] = slice(val[0], val[1])
                    elif isinstance(val, int):
                        isel_dict[dim] = val
                    else:
                        sel_dict[dim] = val
                if isel_dict:
                    var = var.isel(**isel_dict)
                if sel_dict:
                    var = var.sel(**sel_dict)
                return var.compute()
            except Exception as e:
                print(f"Data retrieval error: {e}")
                return None
        return None