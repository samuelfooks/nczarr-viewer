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
                    selection = {dim: ds[selected_var][dim].values[val[0]:val[1]] if isinstance(
                        val, list) else ds[selected_var][dim].values[val] for dim, val in selected_dims.items()}
                    data_retriever = DataRetriever(
                        selected_var, selection, dataseturl, dataset_engine)
                    selected_data = data_retriever.retrieve_data_using_dimension_selections()
                    array_values = selected_data.values
                    # Convert timedelta64 or datetime64 to float for stats
                    if np.issubdtype(array_values.dtype, np.timedelta64):
                        array_values = array_values.astype(
                            'timedelta64[h]').astype(float)
                    elif np.issubdtype(array_values.dtype, np.datetime64):
                        array_values = (
                            array_values - array_values.min()).astype('timedelta64[D]').astype(float)
                    # Apply filter if set
                    if filter_min is not None:
                        array_values = np.where(
                            array_values < filter_min, np.nan, array_values)
                    if filter_max is not None:
                        array_values = np.where(
                            array_values > filter_max, np.nan, array_values)
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


class DataPlot:
    def __init__(self, app, ds_getter, dim_select, dataseturl_getter, dataset_engine_getter):
        self.app = app
        self.ds_getter = ds_getter
        self.dim_select = dim_select
        self.dataseturl_getter = dataseturl_getter
        self.dataset_engine_getter = dataset_engine_getter

    def setup_callbacks(self):
        print("Setting up DataPlot callbacks...")

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
            print(f"=== PLOT CALLBACK TRIGGERED ===")
            print(f"n_clicks: {n_clicks}")
            print(f"selected_var: {selected_var}")
            print(f"selected_dims: {selected_dims}")
            print(f"filter_min: {filter_min}")
            print(f"filter_max: {filter_max}")

            ds = self.ds_getter()
            print(f"ds is None: {ds is None}")

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
                        print(
                            f"found no lat lon selected_dims: {selected_dims}")
                        return html.Div(["No valid latitude/longitude dimensions for plotting."]), {'display': 'none'}
                    print(f"got dims lat/lon {lat_dim}/{lon_dim}")
                    selection = {}
                    for dim, val in selected_dims.items():
                        if ('lat' in dim.lower() or 'y' in dim.lower() or 'lon' in dim.lower() or 'x' in dim.lower()) and isinstance(val, list):
                            selection[dim] = ds[selected_var][dim].values[val[0]:val[1]]
                        else:
                            # For dropdowns (like depth), val is an int index

                            selection[dim] = ds[selected_var][dim].values[val] if isinstance(
                                val, int) else val
                    data_retriever = DataRetriever(
                        selected_var, selection, dataseturl, dataset_engine)
                    print(f"DataRetriever: {data_retriever}")
                    selected_data = data_retriever.retrieve_data_using_dimension_selections()

                    # Get coordinate values and convert to WGS84 if needed
                    lons = selected_data[lon_dim].values
                    lats = selected_data[lat_dim].values

                    # Check if we need to convert coordinates to WGS84
                    try:
                        # Look for grid mapping information
                        grid_mapping = selected_data.attrs.get('grid_mapping')
                        if grid_mapping and grid_mapping in selected_data.coords:
                            gm_var = selected_data.coords[grid_mapping]
                            try:
                                # Try to construct a pyproj CRS from grid mapping attributes
                                import pyproj
                                crs = pyproj.CRS.from_cf(gm_var.attrs)
                                # If not already WGS84
                                if crs is not None and crs != pyproj.CRS(4326):
                                    print(
                                        f"Converting coordinates from {crs} to WGS84")
                                    transformer = pyproj.Transformer.from_crs(
                                        crs, 4326, always_xy=True)

                                    # Create coordinate grids for transformation
                                    lon_grid, lat_grid = np.meshgrid(
                                        lons, lats)
                                    lon_wgs84, lat_wgs84 = transformer.transform(
                                        lon_grid, lat_grid)

                                    # Update coordinate values
                                    lons = lon_wgs84
                                    lats = lat_wgs84
                                    print(
                                        f"Converted coordinates to WGS84: lon range {lons.min():.4f} to {lons.max():.4f}, lat range {lats.min():.4f} to {lats.max():.4f}")
                            except Exception as e:
                                print(
                                    f"Coordinate conversion error: {e}, using original coordinates")
                        else:
                            # Check if coordinates already look like degrees
                            if np.all(np.abs(lons) <= 180) and np.all(np.abs(lats) <= 90):
                                print(
                                    "Coordinates appear to already be in WGS84 degrees")
                            else:
                                print(
                                    "No grid mapping found, coordinates may need conversion")
                    except Exception as e:
                        print(f"Error checking coordinate system: {e}")

                    data_values = np.array(selected_data.values)
                    # Convert timedelta64 or datetime64 to float for plotting
                    if np.issubdtype(data_values.dtype, np.timedelta64):
                        data_values = data_values.astype(
                            'timedelta64[h]').astype(float)
                    elif np.issubdtype(data_values.dtype, np.datetime64):
                        data_values = (data_values - data_values.min()
                                       ).astype('timedelta64[D]').astype(float)
                    # Apply filter if set
                    if filter_min is not None:
                        data_values = np.where(
                            data_values < filter_min, np.nan, data_values)
                    if filter_max is not None:
                        data_values = np.where(
                            data_values > filter_max, np.nan, data_values)
                    # Debug: Print data shapes and values
                    print(f"Original data shape: {data_values.shape}")
                    print(
                        f"Original lons shape: {lons.shape}, range: {lons.min():.4f} to {lons.max():.4f}")
                    print(
                        f"Original lats shape: {lats.shape}, range: {lats.min():.4f} to {lats.max():.4f}")
                    print(
                        f"Data values range: {np.nanmin(data_values):.4f} to {np.nanmax(data_values):.4f}")

                    # Remove NaN/Inf from lons/lats and data_values
                    lons = np.ravel(lons)
                    lats = np.ravel(lats)
                    data_values = np.where(np.isfinite(
                        data_values), data_values, np.nan)

                    print(
                        f"After ravel - lons: {lons.shape}, lats: {lats.shape}, data: {data_values.shape}")

                    if lons.size == 0 or lats.size == 0 or np.all(np.isnan(data_values)):
                        return html.Div(["No valid longitude/latitude or data values for plotting."]), {'display': 'none'}

                    # Handle different data dimensions properly
                    print(
                        f"Processing data with {data_values.ndim}D shape: {data_values.shape}")

                    # First, try to squeeze out single dimensions
                    original_shape = data_values.shape
                    data_values = np.squeeze(data_values)
                    print(
                        f"After squeeze: {data_values.shape} (was {original_shape})")

                    if data_values.ndim == 1:
                        # 1D data - create scatter plot
                        print("1D data detected, creating scatter plot")
                        if lons.size == data_values.size:
                            # Data points correspond to longitudes
                            x_coords = lons
                            y_coords = data_values
                        elif lats.size == data_values.size:
                            # Data points correspond to latitudes
                            x_coords = lats
                            y_coords = data_values
                        else:
                            return html.Div([f"1D data size {data_values.size} doesn't match lat size {lats.size} or lon size {lons.size}"]), {'display': 'none'}
                    elif data_values.ndim == 2:
                        # 2D data - ensure proper alignment for heatmap
                        print(f"2D data shape: {data_values.shape}")
                        expected_shape = (lats.size, lons.size)
                        print(f"Expected shape: {expected_shape}")

                        if data_values.shape == expected_shape:
                            print("Data shape matches lat/lon grid - good!")
                        elif data_values.shape == (lons.size, lats.size):
                            print("Data shape is transposed - fixing...")
                            data_values = data_values.T
                        else:
                            return html.Div([f"Data shape {data_values.shape} does not match expected {expected_shape} for plotting."]), {'display': 'none'}
                    else:
                        return html.Div([f"Cannot plot {data_values.ndim}D data after squeeze. Shape: {data_values.shape}"]), {'display': 'none'}

                    print(
                        f"Final shapes - lons: {lons.shape}, lats: {lats.shape}, data: {data_values.shape}")
                    # Use Plotly for fast, interactive plotting with proper geographic layout
                    if data_values.ndim == 1:
                        # 1D data - create a simple scatter plot
                        print("Creating 1D scatter plot")
                        fig = go.Figure(go.Scatter(
                            x=x_coords,
                            y=y_coords,
                            mode='lines+markers',
                            name=selected_var,
                            line=dict(color='blue', width=2),
                            marker=dict(size=4)
                        ))
                        fig.update_layout(
                            title=f"{selected_var} Line Plot",
                            xaxis_title="Coordinate",
                            yaxis_title=f"{selected_var}",
                            height=600
                        )
                    elif data_values.ndim == 2:
                        # 2D data - use heatmap
                        print("Creating 2D heatmap")
                        print(
                            f"Heatmap input: z={data_values.shape}, x={lons.shape}, y={lats.shape}")
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

                        # Update layout with proper geographic projection and labels
                        fig.update_layout(
                            title=f"{selected_var} Array Plot",
                            xaxis_title=f"{lon_dim} (Longitude)",
                            yaxis_title=f"{lat_dim} (Latitude)",
                            height=600,
                            margin=dict(l=0, r=0, t=40, b=0),
                            template="plotly_white"
                        )

                        # Update axes to show proper geographic coordinates with auto-zoom
                        fig.update_xaxes(
                            tickformat='.4f',
                            tickprefix='',
                            ticksuffix='°',
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray',
                            range=[lons.min(), lons.max()],
                            constrain='domain'
                        )
                        fig.update_yaxes(
                            tickformat='.4f',
                            tickprefix='',
                            ticksuffix='°',
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray',
                            range=[lats.min(), lats.max()],
                            constrain='domain'
                        )

                        # Add some padding around the data bounds for better visualization
                        lon_padding = (lons.max() - lons.min()) * 0.05
                        lat_padding = (lats.max() - lats.min()) * 0.05

                        fig.update_layout(
                            xaxis=dict(
                                range=[lons.min() - lon_padding,
                                       lons.max() + lon_padding]
                            ),
                            yaxis=dict(
                                range=[lats.min() - lat_padding,
                                       lats.max() + lat_padding]
                            )
                        )
                    else:
                        return html.Div([f"Unexpected data dimensions: {data_values.ndim}"]), {'display': 'none'}

                    print("Plot created successfully, returning to browser")
                    return dcc.Graph(figure=fig, config={"displayModeBar": True, "scrollZoom": True}), {'display': 'block'}
                except Exception as e:
                    print(f"Plot error: {e}")
                    return html.Div([f"Plot error: {e}"]), {'display': 'none'}
            print("No plot conditions met - returning empty plot")
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
                ds = custom_open_zarr.open_zarr(
                    self.dataseturl, copernicus_marine_username='sfooks')
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
