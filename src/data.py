from dash import Output, Input, State, html, dcc
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
import xarray as xr
import traceback
import logging
from io import StringIO
import sys
import requests


class DatasetLoader:
    """Handles dataset loading with multiple backends and error handling"""

    def __init__(self):
        self.available_backends = self._detect_backends()
        self.log_output = StringIO()
        self.setup_logging()

    def _detect_backends(self):
        """Detect available backends for dataset loading"""
        backends = {
            'xarray': {
                'engines': ['netcdf4', 'zarr'],
                'description': 'Standard xarray engines'
            }
        }

        # Check for specialized backends
        try:
            import copernicusmarine  # noqa: F401
            backends['copernicusmarine'] = {
                'engines': ['default', 'custom_open_zarr'],
                'description': 'Copernicus Marine Service backend'
            }
        except ImportError:
            pass

        try:
            import pydap  # noqa: F401
            backends['pydap'] = {
                'engines': ['pydap'],
                'description': 'OPeNDAP backend'
            }
        except ImportError:
            pass

        try:
            import rasterio  # noqa: F401
            backends['rasterio'] = {
                'engines': ['rasterio'],
                'description': 'Raster backend'
            }
        except ImportError:
            pass

        return backends

    def setup_logging(self):
        """Setup logging to capture output"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(self.log_output),
                logging.StreamHandler(sys.stdout)  # Also show in terminal
            ]
        )
        self.logger = logging.getLogger(__name__)

    def clear_logs(self):
        """Clear the log output buffer"""
        self.log_output.truncate(0)
        self.log_output.seek(0)

    def get_logs(self):
        """Get the current log output"""
        return self.log_output.getvalue()

    def load_dataset(self, url, backend='xarray', engine='auto', **kwargs):
        """
        Load dataset with specified backend and engine

        Args:
            url: Dataset URL or path
            backend: Backend to use ('xarray', 'copernicusmarine', etc.)
            engine: Engine to use with the backend
            **kwargs: Additional arguments passed to the dataset loader
        """
        self.clear_logs()
        self.logger.info(f"Attempting to load dataset: {url}")
        self.logger.info(f"Backend: {backend}, Engine: {engine}")
        self.logger.info(f"Additional kwargs: {kwargs}")

        try:
            if backend == 'xarray':
                return self._load_with_xarray(url, engine, **kwargs)
            elif backend == 'copernicusmarine':
                return self._load_with_copernicusmarine(url, engine, **kwargs)
            else:
                raise ValueError(f"Unknown backend: {backend}")

        except Exception as e:
            self.logger.error(f"Failed to load dataset: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None, str(e)

    def _load_with_xarray(self, url, engine='auto', **kwargs):
        """Load dataset using xarray"""
        self.logger.info(f"Loading with xarray, engine: {engine}")

        if engine == 'auto':
            # Auto-detect engine based on URL
            if '.nc' in url or '.nc4' in url:
                engine = 'netcdf4'
            elif '.zarr' in url:
                engine = 'zarr'
            else:
                engine = 'netcdf4'  # Default

        self.logger.info(f"Selected engine: {engine}")

        # Try to open with selected engine
        try:
            # Filter out only backend and engine from kwargs to avoid conflicts
            # All other xarray parameters (including decode_timedelta) are passed through
            xarray_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ['engine', 'backend']
            }

            self.logger.info(f"Passing kwargs to xarray: {xarray_kwargs}")

            ds = xr.open_dataset(
                url, engine=engine, **xarray_kwargs
            )
            self.logger.info(f"Successfully opened dataset with {engine}")
            self.logger.info(f"Dataset shape: {dict(ds.dims)}")
            self.logger.info(f"Data variables: {list(ds.data_vars.keys())}")
            return ds, engine
        except Exception as e:
            self.logger.warning(f"Failed with engine {engine}: {e}")

            # Try alternative engines if the first one fails
            alternative_engines = [
                'netcdf4', 'zarr'
            ]
            if engine in alternative_engines:
                alternative_engines.remove(engine)

            for alt_engine in alternative_engines:
                try:
                    self.logger.info(
                        f"Trying alternative engine: {alt_engine}"
                    )
                    ds = xr.open_dataset(
                        url, engine=alt_engine, **xarray_kwargs
                    )
                    self.logger.info(
                        f"Successfully opened with alternative engine: {alt_engine}"
                    )
                    return ds, alt_engine
                except Exception as alt_e:
                    self.logger.warning(
                        f"Alternative engine {alt_engine} failed: {alt_e}"
                    )

            # Provide helpful error message with tips
            error_msg = f"All engines failed. Last error: {e}"
            
            # Add specific help for common issues
            if "netcdf4" in str(e).lower() or "netcdf" in str(e).lower():
                error_msg += "\n\n💡 NetCDF loading failed. Try these solutions:"
                error_msg += "\n• For netcdf files on s3 storage: Add #mode=bytes at the end of the URL"
                error_msg += "\n• Check if the file is corrupted or incomplete"
            
            raise Exception(error_msg)

    def _load_with_copernicusmarine(self, url, engine='default', **kwargs):
        """Load dataset using copernicusmarine backend"""
        self.logger.info(f"Loading with copernicusmarine, engine: {engine}")

        # For custom_open_zarr engine, we don't need credentials
        if engine == 'custom_open_zarr.open_zarr':
            try:
                from copernicusmarine.core_functions import custom_open_zarr
                # Try to open the store and then open it as a dataset
                print(
                    f"opening dataset with custom_open_zarr.open_zarr and {url}")

                # The custom_open_zarr.open_zarr function accepts:
                # - dataset_url (positional)
                # - copernicus_marine_username (optional)
                # - **kwargs (which get passed to xarray.open_zarr)

                # Filter out copernicusmarine-specific parameters
                zarr_kwargs = {k: v for k, v in kwargs.items()
                               if k not in ['username', 'password', 'dataset_id']}

                # Set S3 client configuration through environment variables
                # This is more reliable than passing through storage_options
                import os
                os.environ.setdefault(
                    'AWS_MAX_POOL_CONNECTIONS', '20')  # Increased from 10
                os.environ.setdefault('AWS_RETRY_MODE', 'adaptive')
                os.environ.setdefault('AWS_MAX_ATTEMPTS', '5')  # Increased
                # Increased from 60 to 120
                os.environ.setdefault('AWS_CONNECT_TIMEOUT', '120')
                # Increased from 120 to 300
                os.environ.setdefault('AWS_READ_TIMEOUT', '300')
                os.environ.setdefault('AWS_S3_ADDRESSING_STYLE', 'virtual')

                # Debug: Print current environment variables
                print(
                    f"DEBUG: AWS_MAX_POOL_CONNECTIONS = {os.environ.get('AWS_MAX_POOL_CONNECTIONS')}")
                print(
                    f"DEBUG: AWS_RETRY_MODE = {os.environ.get('AWS_RETRY_MODE')}")
                print(
                    f"DEBUG: AWS_MAX_ATTEMPTS = {os.environ.get('AWS_MAX_ATTEMPTS')}")

                # Pass user kwargs to custom_open_zarr.open_zarr
                # The environment variables will configure the S3 client
                ds = custom_open_zarr.open_zarr(url, **zarr_kwargs)

                # Debug: Check if dataset has any S3-related attributes
                print(f"DEBUG: Dataset type: {type(ds)}")
                if hasattr(ds, '_file_obj'):
                    print(f"DEBUG: Dataset has _file_obj: {ds._file_obj}")
                if hasattr(ds, 'encoding'):
                    print(f"DEBUG: Dataset encoding: {ds.encoding}")

                    # Debug: Try to access a small piece of data to trigger S3 operations
                # This will help us see what connection parameters are being used
                try:
                    if hasattr(ds, 'data_vars') and list(ds.data_vars.keys()):
                        var_name = list(ds.data_vars.keys())[0]
                        print(
                            f"DEBUG: Attempting to access variable: {var_name}")
                        # Get just the first element to minimize data transfer
                        sample_data = ds[var_name].isel(
                            {dim: 0 for dim in ds[var_name].dims})
                        print(
                            f"DEBUG: Successfully accessed sample data: {sample_data.shape}")
                        print(
                            f"DEBUG: This should have triggered S3 operations with our connection settings")
                    else:
                        print("DEBUG: No data variables found in dataset")
                except Exception as e:
                    print(f"DEBUG: Error during data access: {e}")
                    print(f"DEBUG: Error type: {type(e)}")

                print(f"DEBUG: Dataset loaded successfully")
                print(f"dataset: {ds}")

                self.logger.info("Successfully opened with custom_open_zarr")
                return ds, f"copernicusmarine_custom_open_zarr"
            except ImportError:
                raise Exception("copernicusmarine not available")
            except Exception as e:
                raise Exception(f"custom_open_zarr failed: {e}")

        # For other engines, extract copernicusmarine-specific parameters
        username = kwargs.get('username')
        password = kwargs.get('password')
        dataset_id = kwargs.get('dataset_id')

        if not username or not password:
            raise Exception(
                "copernicusmarine requires 'username' and 'password' in backend args")

        if not dataset_id:
            # If no dataset_id provided, try to use the URL as dataset_id
            dataset_id = url
            self.logger.info(
                f"No dataset_id provided, using URL as dataset_id: {dataset_id}")

        self.logger.info(f"Using dataset_id: {dataset_id}")
        self.logger.info(f"Username: {username}")

        try:
            import copernicusmarine

            # Filter out copernicusmarine-specific parameters to avoid duplicates
            filtered_kwargs = {k: v for k, v in kwargs.items()
                               if k not in ['username', 'password', 'dataset_id']}

            if engine == 'default':
                # Use copernicusmarine.open_dataset with dataset_id
                ds = copernicusmarine.open_dataset(
                    dataset_id, username=username, password=password, **filtered_kwargs
                )
            else:
                raise ValueError(f"Unknown copernicusmarine engine: {engine}")

            self.logger.info("Successfully opened with copernicusmarine")
            return ds, f"copernicusmarine_{engine}"

        except ImportError:
            raise Exception("copernicusmarine not available")
        except Exception as e:
            raise Exception(f"copernicusmarine failed: {e}")


class DataManager:
    """
    Unified data management class that handles subsetting, statistics, and plotting.
    This consolidates the previously scattered functionality into a clean, simple API.
    """

    def __init__(self, app, dataset_getter):
        self.app = app
        # Function that returns the current dataset
        self.dataset_getter = dataset_getter

    def setup_callbacks(self):
        """Setup all data-related callbacks in one place"""

        # Callback for quick statistics
        @self.app.callback(
            Output('data-array-display', 'children'),
            Input('show-data-button', 'n_clicks'),
            State('variable-dropdown', 'value'),
            State('selected-dimensions-store', 'data'),
            State('data-filter-min', 'value'),
            State('data-filter-max', 'value'),
            prevent_initial_call=True
        )
        def show_quick_stats(n_clicks, selected_var, selected_dims, filter_min, filter_max):
            """Display quick statistics for the selected data"""
            if n_clicks is None or n_clicks == 0:
                return "Click 'Show Data Quick Stats' to see statistics"

            if not selected_var or not selected_dims:
                return "Please select a variable and dimensions first"

            try:
                # Get subsetted data
                subsetted_data = self._get_subsetted_data(
                    selected_var, selected_dims)
                if subsetted_data is None:
                    return "Error: Could not subset data"

                # Calculate and display statistics
                stats = self._calculate_statistics(
                    subsetted_data, filter_min, filter_max)
                if stats is None:
                    return "Error: Could not calculate statistics"

                return self._format_statistics_display(stats)

            except Exception as e:
                return f"Error: {str(e)}"



        # Callback for extracting image and showing in separate container
        @self.app.callback(
            Output('raster-container', 'children'),
            Input('extract-plot-button', 'n_clicks'),
            State('variable-dropdown', 'value'),
            State('selected-dimensions-store', 'data'),
            State('data-filter-min', 'value'),
            State('data-filter-max', 'value'),
            prevent_initial_call=True
        )
        def extract_image(n_clicks, selected_var, selected_dims, filter_min, filter_max):
            """Generate raster image and display in sidebar"""
            print(f"=== EXTRACT IMAGE CALLBACK TRIGGERED ===")
            print(f"n_clicks: {n_clicks}")
            print(f"selected_var: {selected_var}")
            print(f"selected_dims: {selected_dims}")
            print(f"filter_min: {filter_min}")
            print(f"filter_max: {filter_max}")

            if n_clicks is None or n_clicks == 0:
                print("No clicks detected, returning empty")
                return []

            if not selected_var or not selected_dims:
                print("Missing variable or dimensions, returning message")
                return [html.P("Please select a variable and dimensions first", className="text-muted text-center")]

            try:
                print("Getting subsetted data...")
                # Get subsetted data
                subsetted_data = self._get_subsetted_data(
                    selected_var, selected_dims)
                if subsetted_data is None:
                    return [html.P("Error: Could not subset data", className="text-danger text-center")]

                print("Finding spatial dimensions...")
                # Find spatial dimensions
                lat_dim = None
                lon_dim = None
                for dim in subsetted_data.dims:
                    dim_lower = dim.lower()
                    if 'lat' in dim_lower or 'y' in dim_lower:
                        lat_dim = dim
                    elif 'lon' in dim_lower or 'x' in dim_lower:
                        lon_dim = dim

                if not lat_dim or not lon_dim:
                    return [html.P("Error: No spatial dimensions found", className="text-danger text-center")]

                print(
                    f"Creating raster image with lat_dim={lat_dim}, lon_dim={lon_dim}")
                # Create raster image
                image_path = self.create_raster_image(
                    subsetted_data, selected_var, lat_dim, lon_dim)

                print("Storing data for overlay...")
                # Store the current variable and image data for overlay button
                self.current_raster_var = selected_var
                self.current_raster_data = subsetted_data
                self.current_lat_dim = lat_dim
                self.current_lon_dim = lon_dim
                self.current_raster_image = image_path  # This is now base64 data

                print("Creating full-width raster display...")
                # Create container with base64 image for full-width display
                container_content = [
                    html.Div([
                        html.H4(f"📊 {selected_var} - Raster Analysis", 
                               className="text-center mb-3 text-primary"),
                        html.Img(
                            src=image_path,  # This is now base64 data
                            style={
                                'width': '100%', 
                                'height': 'auto',
                                'maxWidth': '1200px',
                                'display': 'block',
                                'margin': '0 auto',
                                'borderRadius': '8px',
                                'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
                            },
                            className="raster-image"
                        ),
                        html.Div([
                            html.P("✅ Image generated successfully!", 
                                   className="text-success text-center mt-3 mb-2", 
                                   style={'fontSize': '14px', 'fontWeight': 'bold'}),
                            dbc.Button('🗺️ Overlay on World Map', 
                                       id='overlay-button',
                                       color='primary', 
                                       size='lg', 
                                       className='mx-auto d-block')
                        ], className="text-center")
                    ])
                ]

                print("Returning container content successfully!")
                return container_content

            except Exception as e:
                print(f"Error in extract_image: {str(e)}")
                print(f"Exception type: {type(e)}")
                import traceback
                traceback.print_exc()
                return [html.P(f"Error: {str(e)}", className="text-danger text-center")]

        # Callback for overlaying image on world map
        @self.app.callback(
            [Output('map-container', 'children', allow_duplicate=True),
             Output('map-container', 'style', allow_duplicate=True)],
            Input('overlay-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def overlay_on_map(n_clicks):
            """Overlay the generated raster image on the world map"""
            print(f"=== OVERLAY CALLBACK TRIGGERED ===")
            print(f"n_clicks: {n_clicks}")
            
            if n_clicks is None or n_clicks == 0:
                print("No clicks detected")
                return "Click 'Overlay on Map' to see the result", {'display': 'none'}

            try:
                print("Checking for stored raster data...")
                if not hasattr(self, 'current_raster_var'):
                    print("No current_raster_var found")
                    return "Error: No raster image generated yet", {'display': 'none'}

                # Get the stored data
                selected_var = self.current_raster_var
                subsetted_data = self.current_raster_data
                lat_dim = self.current_lat_dim
                lon_dim = self.current_lon_dim
                
                print(f"Stored data - var: {selected_var}, lat_dim: {lat_dim}, lon_dim: {lon_dim}")
                print(f"Subsetted data shape: {dict(subsetted_data.sizes)}")

                # Get the stored base64 image data
                if not hasattr(self, 'current_raster_image'):
                    print("No current_raster_image found")
                    return "Error: No raster image data found", {'display': 'none'}

                image_src = self.current_raster_image
                print(f"Image source type: {type(image_src)}")
                print(f"Image source length: {len(image_src) if isinstance(image_src, str) else 'N/A'}")

                print("Creating world map with overlay...")
                # Create world map with overlay
                overlay_figure = self._create_world_map_with_overlay(
                    selected_var, subsetted_data, lat_dim, lon_dim)

                if overlay_figure is None:
                    print("Overlay figure creation failed")
                    return "Error: Could not create overlay", {'display': 'none'}

                print("Overlay figure created successfully, returning...")
                
                # Return the Plotly figure in a Graph component
                return dcc.Graph(figure=overlay_figure, config={"displayModeBar": True, "scrollZoom": True}), {'display': 'block'}

            except Exception as e:
                print(f"Error in overlay_on_map: {str(e)}")
                print(f"Exception type: {type(e)}")
                import traceback
                traceback.print_exc()
                return f"Error: {str(e)}", {'display': 'none'}

    def _get_subsetted_data(self, selected_var, selected_dims):
        """Get subsetted data based on user selections"""
        try:
            dataset = self.dataset_getter()
            if dataset is None:
                return None

            variable_data = dataset[selected_var]

            # Apply dimension selections
            if selected_dims:
                # Handle different types of selections
                isel_dict = {}  # For integer-based indexing
                sel_dict = {}   # For label-based selection

                for dim, val in selected_dims.items():
                    print(f"dim: {dim}, val: {val}")
                    # Debug: show coordinate values and their order
                    if dim in variable_data.coords:
                        coords_vals = variable_data.coords[dim].values
                        print(f"  {dim} coords: {coords_vals[:5]}... (length: {len(coords_vals)}, ascending: {coords_vals[0] < coords_vals[-1]})")
                    if isinstance(val, tuple):
                        if len(val) == 2:
                            # Range selection (start, end) - use slice for array subsetting
                            start_val, end_val = val
                            # Find indices for the start and end values
                            dim_coords = variable_data.coords[dim].values

                            # Handle different coordinate types
                            if np.issubdtype(dim_coords.dtype, np.datetime64):
                                # For datetime coordinates, convert to numpy datetime64 for comparison
                                if isinstance(start_val, str):
                                    start_val = np.datetime64(start_val)
                                if isinstance(end_val, str):
                                    end_val = np.datetime64(end_val)

                            # Find the range of coordinates that fall within the user's selection
                            # This handles both ascending and descending coordinate arrays correctly
                            min_val = min(start_val, end_val)
                            max_val = max(start_val, end_val)
                            
                            # Find indices where coordinates fall within the range
                            valid_mask = (dim_coords >= min_val) & (dim_coords <= max_val)
                            valid_indices = np.where(valid_mask)[0]
                            
                            if len(valid_indices) > 0:
                                start_idx = int(valid_indices[0])
                                end_idx = int(valid_indices[-1])
                                
                                # Create slice - this preserves the original coordinate order
                                isel_dict[dim] = slice(start_idx, end_idx + 1)
                                print(f"  Created slice for {dim}: {start_idx}:{end_idx + 1} (from values {min_val} to {max_val})")
                            else:
                                # Fallback: use searchsorted approach
                                start_idx = np.searchsorted(dim_coords, min_val)
                                end_idx = np.searchsorted(dim_coords, max_val)
                                
                                # Ensure we don't go out of bounds
                                start_idx = max(0, min(start_idx, len(dim_coords) - 1))
                                end_idx = max(0, min(end_idx, len(dim_coords) - 1))
                                # Convert numpy types to Python types for slice
                                start_idx = int(start_idx)
                                end_idx = int(end_idx)
                                
                                # Ensure slice indices are in correct order for Python slicing
                                if start_idx > end_idx:
                                    start_idx, end_idx = end_idx, start_idx
                                
                                # Create slice
                                isel_dict[dim] = slice(start_idx, end_idx + 1)
                                print(f"  Created slice for {dim}: {start_idx}:{end_idx + 1} (from values {min_val} to {max_val})")
                        elif len(val) == 1:
                            # Single selection (val,) - use exact value selection
                            sel_dict[dim] = val[0]
                    elif isinstance(val, list):
                        # Handle list format (fallback for old dimension selection)
                        if len(val) == 2:
                            # Range selection [start, end] - convert to tuple format
                            start_val, end_val = val
                            # Handle timestamp conversion for time dimension
                            if dim.lower() in ['time', 'date'] and isinstance(start_val, (int, float)):
                                # Convert nanosecond timestamp to datetime
                                start_val = np.datetime64(start_val, 'ns')
                                end_val = np.datetime64(end_val, 'ns')

                            # Find indices for the start and end values
                            dim_coords = variable_data.coords[dim].values
                            
                            # Find the range of coordinates that fall within the user's selection
                            # This handles both ascending and descending coordinate arrays correctly
                            min_val = min(start_val, end_val)
                            max_val = max(start_val, end_val)
                            
                            # Find indices where coordinates fall within the range
                            valid_mask = (dim_coords >= min_val) & (dim_coords <= max_val)
                            valid_indices = np.where(valid_mask)[0]
                            
                            if len(valid_indices) > 0:
                                start_idx = int(valid_indices[0])
                                end_idx = int(valid_indices[-1])
                                
                                # Create slice - this preserves the original coordinate order
                                isel_dict[dim] = slice(start_idx, end_idx + 1)
                                print(f"  Created slice for {dim}: {start_idx}:{end_idx + 1} (from values {min_val} to {max_val})")
                            else:
                                # Fallback: use searchsorted approach
                                start_idx = np.searchsorted(dim_coords, min_val)
                                end_idx = np.searchsorted(dim_coords, max_val)
                                
                                # Ensure we don't go out of bounds
                                start_idx = max(0, min(start_idx, len(dim_coords) - 1))
                                end_idx = max(0, min(end_idx, len(dim_coords) - 1))
                                # Convert numpy types to Python types for slice
                                start_idx = int(start_idx)
                                end_idx = int(end_idx)
                                
                                # Ensure slice indices are in correct order for Python slicing
                                if start_idx > end_idx:
                                    start_idx, end_idx = end_idx, start_idx
                                
                                # Create slice
                                isel_dict[dim] = slice(start_idx, end_idx + 1)
                                print(f"  Created slice for {dim}: {start_idx}:{end_idx + 1} (from values {min_val} to {max_val})")
                        elif len(val) == 1:
                            # Single selection [val] - convert to tuple format
                            single_val = val[0]
                            # Handle timestamp conversion for time dimension
                            if dim.lower() in ['time', 'date'] and isinstance(single_val, (int, float)):
                                single_val = np.datetime64(single_val, 'ns')
                            sel_dict[dim] = single_val
                    elif isinstance(val, (int, float)):
                        # Direct value selection
                        sel_dict[dim] = val
                    else:
                        # Fallback for other types
                        sel_dict[dim] = val

                # Apply integer-based selections first
                if isel_dict:
                    print(f"Applying integer-based selections: {isel_dict}")
                    selected_data = variable_data.isel(**isel_dict)
                else:
                    selected_data = variable_data

                # Apply label-based selections
                if sel_dict:
                    print(f"Applying label-based selections: {sel_dict}")
                    selected_data = selected_data.sel(**sel_dict)
            elif selected_var and not selected_dims:
                selected_data = variable_data

            print(f"selected_data shape: {dict(selected_data.sizes)}")

            return selected_data

        except Exception as e:
            print(f"Error subsetting data: {e}")
            return None

    def _calculate_statistics(self, data_array, filter_min=None, filter_max=None):
        """Calculate basic statistics from a data array"""
        try:
            if data_array is None:
                return None

            # Get values and convert to numpy array
            values = np.array(data_array.values)

            # Convert timedelta64 or datetime64 to float for stats
            if np.issubdtype(values.dtype, np.timedelta64):
                values = values.astype('timedelta64[h]').astype(float)
            elif np.issubdtype(values.dtype, np.datetime64):
                values = (values - values.min()
                          ).astype('timedelta64[D]').astype(float)

            # Apply filters if set
            if filter_min is not None:
                values = np.where(values < filter_min, np.nan, values)
            if filter_max is not None:
                values = np.where(values > filter_max, np.nan, values)

            # Calculate statistics
            stats = {
                'min': float(np.nanmin(values)),
                'max': float(np.nanmax(values)),
                'mean': float(np.nanmean(values)),
                'median': float(np.nanmedian(values)),
                'std': float(np.nanstd(values)),
                'count': int(np.sum(np.isfinite(values))),
                'total': int(values.size)
            }

            return stats

        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return None

    def _format_statistics_display(self, stats):
        """Format statistics for display"""
        if stats is None:
            return "No statistics available"

        return html.Div([
            html.H6("Data Statistics", className="mb-3"),
            html.Div([
                html.Div([
                    html.Strong("Min: "), f"{stats['min']:.4g}",
                ], className="me-4"),
                html.Div([
                    html.Strong("Max: "), f"{stats['max']:.4g}",
                ], className="me-4"),
                html.Div([
                    html.Strong("Mean: "), f"{stats['mean']:.4g}",
                ], className="me-4"),
                html.Div([
                    html.Strong("Median: "), f"{stats['median']:.4g}",
                ], className="me-4"),
                html.Div([
                    html.Strong("Std: "), f"{stats['std']:.4g}",
                ], className="me-4"),
            ], className="d-flex flex-wrap"),
            html.Div([
                html.Small(
                    f"Valid values: {stats['count']} / {stats['total']}"),
            ], className="mt-2 text-muted")
        ])

    def _create_plot(self, data_array, variable_name, filter_min=None, filter_max=None):
        """Create a plot from the data array"""
        try:
            if data_array is None:
                return None

            # Get coordinate information
            coords = list(data_array.coords.keys())
            dims = list(data_array.dims)

            print(f"coords: {coords}")
            print(f"dims: {dims}")
            print(f"data_array: {data_array}")
            print(f"variable_name: {variable_name}")
            print(f"filter_min: {filter_min}")
            print(f"filter_max: {filter_max}")

            # Find spatial dimensions
            lat_dim = None
            lon_dim = None

            for dim in dims:
                dim_lower = dim.lower()
                if 'lat' in dim_lower or 'y' in dim_lower:
                    lat_dim = dim
                elif 'lon' in dim_lower or 'x' in dim_lower:
                    lon_dim = dim

            # Get data values
            values = np.array(data_array.values)

            # Convert timedelta64 or datetime64 to float for plotting
            if np.issubdtype(values.dtype, np.timedelta64):
                values = values.astype('timedelta64[h]').astype(float)
            elif np.issubdtype(values.dtype, np.datetime64):
                values = (values - values.min()
                          ).astype('timedelta64[D]').astype(float)

            # Apply filters
            if filter_min is not None:
                values = np.where(values < filter_min, np.nan, values)
            if filter_max is not None:
                values = np.where(values > filter_max, np.nan, values)

            # Handle different data dimensions
            values = np.squeeze(values)

            if values.ndim == 1:
                # 1D data - create line plot
                return self._create_1d_plot(values, data_array, variable_name, coords)
            elif values.ndim == 2 and lat_dim and lon_dim:
                # 2D spatial data - create heatmap
                return self._create_2d_heatmap(values, data_array, variable_name, lat_dim, lon_dim)
            else:
                # Fallback for other cases
                return self._create_fallback_plot(values, variable_name)

        except Exception as e:
            print(f"Error creating plot: {e}")
            return None

    def _create_1d_plot(self, values, data_array, variable_name, coords):
        """Create a 1D line plot"""
        if len(coords) == 0:
            return None

        # Use the first coordinate for x-axis
        x_coord = coords[0]
        x_values = data_array.coords[x_coord].values

        # Convert datetime64 to string for plotting if needed
        if np.issubdtype(x_values.dtype, np.datetime64):
            x_values = [str(x) for x in x_values]
        else:
            x_values = x_values.tolist()

        # Ensure values is also a list
        y_values = values.tolist()

        fig = go.Figure(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            name=variable_name,
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))

        fig.update_layout(
            title=f"{variable_name} vs {x_coord}",
            xaxis_title=x_coord,
            yaxis_title=variable_name,
            height=500,
            template="plotly_white"
        )

        return fig

    def _create_2d_heatmap(self, values, data_array, variable_name, lat_dim, lon_dim):
        """Create a 2D heatmap using Plotly"""
        print("Creating 2D heatmap...")

        lats = data_array.coords[lat_dim].values
        lons = data_array.coords[lon_dim].values

        # Ensure proper alignment
        if values.shape != (lats.size, lons.size):
            if values.shape == (lons.size, lats.size):
                values = values.T
            else:
                values = values.reshape(lats.size, lons.size)

        # For very large datasets, downsample for performance
        max_resolution = 300  # Keep it manageable
        if lats.size > max_resolution or lons.size > max_resolution:
            lat_factor = max(1, lats.size // max_resolution)
            lon_factor = max(1, lons.size // max_resolution)

            values_downsampled = values[::lat_factor, ::lon_factor]
            lats_downsampled = lats[::lat_factor]
            lons_downsampled = lons[::lon_factor]

            print(
                f"Downsampled from {values.shape} to {values_downsampled.shape}")
        else:
            values_downsampled = values
            lats_downsampled = lats
            lons_downsampled = lons

        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=values_downsampled,
            x=lons_downsampled,
            y=lats_downsampled,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='<b>%{y:.3f}°N, %{x:.3f}°E</b><br>' +
            f'{variable_name}: %{{z:.3g}}<br>' +
            '<extra></extra>',
            showscale=True,
            colorbar=dict(
                title=f"{variable_name}",
                x=1.02,
                len=0.8
            )
        ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{variable_name} Heatmap",
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                y=0.95
            ),
            height=700,
            width=None,
            margin=dict(l=0, r=0, t=80, b=0),
            showlegend=False
        )

        print("2D heatmap created successfully!")
        return fig



    def create_raster_image(self, data_array, variable_name, lat_dim, lon_dim):
        """Create a raster image from the data array and save it"""
        print("Creating raster image...")

        import matplotlib.pyplot as plt
        import os

        lats = data_array.coords[lat_dim].values
        lons = data_array.coords[lon_dim].values
        values = data_array.values

        print(f"Latitude range: {lats.min()} to {lats.max()}")
        print(f"Longitude range: {lons.min()} to {lons.max()}")
        print(f"Values shape: {values.shape}")

        # Ensure proper alignment
        if values.shape != (lats.size, lons.size):
            if values.shape == (lons.size, lats.size):
                values = values.T
            else:
                values = values.reshape(lats.size, lons.size)

        # For large datasets, downsample for performance but keep higher resolution
        max_resolution = 2000  # Increased from 800 for much better quality
        if lats.size > max_resolution or lons.size > max_resolution:
            lat_factor = max(1, lats.size // max_resolution)
            lon_factor = max(1, lons.size // max_resolution)

            values_downsampled = values[::lat_factor, ::lon_factor]
            lats_downsampled = lats[::lat_factor]
            lons_downsampled = lons[::lon_factor]

            print(
                f"Downsampled from {values.shape} to {values_downsampled.shape}")
        else:
            values_downsampled = values
            lats_downsampled = lats
            lons_downsampled = lons

        # Handle latitude orientation - ensure north is at the top
        # If latitude coordinates are ascending (-90 to 90), flip the data vertically
        # so that north (90) appears at the top of the image
        if lats_downsampled[0] < lats_downsampled[-1]:  # Ascending order
            values_display = np.flipud(values_downsampled)
            lat_extent = [lats_downsampled.max(), lats_downsampled.min()]  # Reverse extent
            lats_for_mesh = np.flipud(lats_downsampled)  # Flip coords to match flipped data
            print("Latitude coordinates are ascending, flipping data vertically for correct display")
        else:  # Descending order (90 to -90)
            values_display = values_downsampled
            lat_extent = [lats_downsampled.min(), lats_downsampled.max()]
            lats_for_mesh = lats_downsampled  # Use coords as-is
            print("Latitude coordinates are descending, using data as-is")

        # Create the plot with cartopy for better geographic visualization
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        # Determine appropriate projection based on data extent
        lon_min, lon_max = lons_downsampled.min(), lons_downsampled.max()
        lat_min, lat_max = lat_extent[0], lat_extent[1]
        
        # Use Plate Carree for global data, or appropriate regional projection
        if lon_max - lon_min > 300:  # Global or near-global data
            projection = ccrs.PlateCarree()
        else:  # Regional data
            projection = ccrs.PlateCarree()
        
        fig = plt.figure(figsize=(16, 12))
        ax = plt.axes(projection=projection)
        
        # Set map extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add natural earth features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        
        # Plot the data using pcolormesh for better geographic accuracy
        # Use the appropriate latitude coordinates based on whether data was flipped
        lons_mesh, lats_mesh = np.meshgrid(lons_downsampled, lats_for_mesh)
        mesh = ax.pcolormesh(lons_mesh, lats_mesh, values_display, 
                           transform=ccrs.PlateCarree(), 
                           cmap='viridis', shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(variable_name, fontsize=14)
        
        # Add gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Set title
        ax.set_title(f"{variable_name} Raster", fontsize=16, fontweight='bold', pad=20)

        # Create a temporary directory for images (not in assets)
        import tempfile
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp_images')
        os.makedirs(temp_dir, exist_ok=True)

        # Save the image to temp directory with higher DPI
        image_path = os.path.join(temp_dir, f'raster_{variable_name}.png')
        plt.savefig(image_path, dpi=300, bbox_inches='tight', pad_inches=0.1, 
                   facecolor='white', edgecolor='none')
        plt.close()

        # Convert to base64 for display in the web app
        import base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode()
        
        image_src = f"data:image/png;base64,{image_data}"
        
        print("Raster image converted to base64 successfully")
        return image_src

    def _create_world_map_with_overlay(self, variable_name, data_array, lat_dim, lon_dim):
        """Create a world map with data overlay on a Mapbox 3D globe"""
        print("Creating world map with overlay...")
        return self._create_mapbox_globe(variable_name, data_array, lat_dim, lon_dim)

        # Sample the data for better performance while maintaining quality
        # Use the same step size for both dimensions to ensure matching shapes
        max_dim_size = max(lats.size, lons.size)
        sample_step = max(1, max_dim_size // 100)
        print(f"Sampling data with step {sample_step} for 3D globe visualization")
        
        lats_sampled = lats[::sample_step]
        lons_sampled = lons[::sample_step]
        values_sampled = data_array.values[::sample_step, ::sample_step]
        
        print(f"Sampled shapes - lats: {lats_sampled.shape}, lons: {lons_sampled.shape}, values: {values_sampled.shape}")
        
        # Create a 3D scatter plot that will appear on the globe surface
        # Convert lat/lon to 3D coordinates on a unit sphere
        lats_rad = np.radians(lats_sampled)
        lons_rad = np.radians(lons_sampled)
        
        # Create meshgrid to ensure proper broadcasting
        lats_mesh, lons_mesh = np.meshgrid(lats_rad, lons_rad, indexing='ij')
        
        # 3D coordinates on unit sphere (radius = 1)
        radius = 1.0
        x = radius * np.cos(lats_mesh) * np.cos(lons_mesh)
        y = radius * np.cos(lats_mesh) * np.sin(lons_mesh)
        z = radius * np.sin(lats_mesh)
        
        # Flatten arrays for scatter plot
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        values_flat = values_sampled.flatten()
        
        # Filter out NaN values
        valid_mask = ~np.isnan(values_flat)
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        z_valid = z_flat[valid_mask]
        values_valid = values_flat[valid_mask]
        
        print(f"Valid 3D data points: {len(values_valid)}")
        
        # First, add the Earth globe surface
        print("Adding Earth globe surface...")
        
        # Create a basic Earth sphere with landmasses
        # Generate a sphere with more points for better appearance
        phi = np.linspace(0, 2*np.pi, 100)
        theta = np.linspace(-np.pi/2, np.pi/2, 50)
        phi_mesh, theta_mesh = np.meshgrid(phi, theta)
        
        # Convert to Cartesian coordinates
        earth_radius = 0.98  # Slightly smaller than data points
        x_earth = earth_radius * np.cos(theta_mesh) * np.cos(phi_mesh)
        y_earth = earth_radius * np.cos(theta_mesh) * np.sin(phi_mesh)
        z_earth = earth_radius * np.sin(theta_mesh)
        
        # Add the Earth surface
        fig.add_trace(go.Surface(
            x=x_earth,
            y=y_earth,
            z=z_earth,
            colorscale='Earth',
            opacity=0.8,
            showscale=False,
            name='Earth Surface'
        ))
        
        # Create the 3D scatter plot on the globe
        fig.add_trace(go.Scatter3d(
            x=x_valid,
            y=y_valid,
            z=z_valid,
            mode='markers',
            marker=dict(
                size=2.0,  # Slightly larger for better visibility
                color=values_valid,
                colorscale='viridis',
                opacity=0.9,
                showscale=True,
                colorbar=dict(
                    title=variable_name,
                    title_side="right",
                    thickness=15,
                    len=0.6
                )
            ),
            text=[f"{variable_name}: {val:.3f}" for val in values_valid],
            hoverinfo='text',
            name=variable_name
        ))

        # Update layout for 3D globe
        fig.update_layout(
            title=dict(
                text=f"🌍 {variable_name} - 3D Globe Overlay",
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                y=0.95
            ),
            height=800,
            width=None,
            scene=dict(
                xaxis=dict(
                    title="",
                    showgrid=False,
                    showticklabels=False,
                    range=[-1.2, 1.2]
                ),
                yaxis=dict(
                    title="",
                    showgrid=False,
                    showticklabels=False,
                    range=[-1.2, 1.2]
                ),
                zaxis=dict(
                    title="",
                    showgrid=False,
                    showticklabels=False,
                    range=[-1.2, 1.2]
                ),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                # Add annotations for better orientation
                annotations=[
                    dict(
                        x=0,
                        y=0,
                        z=1.1,
                        text="North Pole",
                        showarrow=False,
                        font=dict(size=12, color="black")
                    ),
                    dict(
                        x=0,
                        y=0,
                        z=-1.1,
                        text="South Pole",
                        showarrow=False,
                        font=dict(size=12, color="black")
                    )
                ]
            ),
            margin=dict(l=0, r=0, t=80, b=0),
            showlegend=False
        )
        
        # Add a note about progressive rendering
        fig.add_annotation(
            text="💡 Tip: Zoom in to see more detail. The globe shows sampled data for performance.",
            xref="paper", yref="paper",
            x=0, y=0,
            xanchor='left', yanchor='bottom',
            showarrow=False,
            font=dict(size=10, color="gray"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )

        print("3D globe with raster overlay created successfully!")
        print("Note: Future enhancement - progressive rendering will show more detail on zoom")
        return fig

    def _create_fallback_plot(self, values, variable_name):
        """Create a fallback plot for unexpected data shapes"""
        fig = go.Figure(go.Scatter(
            y=values.flatten(),
            mode='lines',
            name=variable_name
        ))

        fig.update_layout(
            title=f"{variable_name} Data Plot",
            yaxis_title=variable_name,
            height=400,
            template="plotly_white"
        )

        return fig

    def _create_improved_globe(self, variable_name, data_array, lat_dim, lon_dim):
        """Create an improved world map with deck.gl globe visualization"""
        print("Creating improved world map with deck.gl globe...")
        print(f"Variable: {variable_name}")
        print(f"Lat dim: {lat_dim}, Lon dim: {lon_dim}")
        print(f"Data array shape: {dict(data_array.sizes)}")

        # Check if required packages are available
        try:
            import pydeck as pdk
            import geopandas as gpd
            import json
        except ImportError as e:
            print(f"Required packages not available: {e}")
            print("Falling back to basic 3D globe...")
            return self._create_fallback_3d_globe(variable_name, data_array, lat_dim, lon_dim)

        # Create the deck.gl view state
        view_state = pdk.ViewState(
            latitude=0,
            longitude=0,
            zoom=1,
            pitch=0,
            bearing=0
        )

        # Create basic layers first (fast loading)
        layers = []

        # 1. Base Earth layer with natural colors
        earth_layer = pdk.Layer(
            "GeoJsonLayer",
            data=self._get_earth_geojson(),
            stroked=False,
            filled=True,
            get_fill_color=[200, 200, 200, 180],  # Light gray for land
            get_line_color=[0, 0, 0, 0],
            pickable=False,
            visible=True
        )
        layers.append(earth_layer)

        # 2. Ocean layer
        ocean_layer = pdk.Layer(
            "GeoJsonLayer",
            data=self._get_ocean_geojson(),
            stroked=False,
            filled=True,
            get_fill_color=[100, 150, 255, 120],  # Blue for water
            get_line_color=[0, 0, 0, 0],
            pickable=False,
            visible=True
        )
        layers.append(ocean_layer)

        # 3. Country boundaries - removed for now to focus on core functionality

        # 4. Add initial data overlay (coarse for fast loading)
        data_layer = self._add_data_overlay_progressive(variable_name, data_array, lat_dim, lon_dim)
        if data_layer:
            layers.append(data_layer)
            print("Added initial data overlay layer")

        # Create the basic deck.gl deck first (fast)
        basic_deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/satellite-v9',
            height=800
        )

        # Convert to HTML component for Dash
        basic_deck_html = basic_deck.to_html()
        
        # Create a Dash component that can be embedded
        globe_component = html.Div([
            html.H3(f"🌍 {variable_name} - Interactive Globe", 
                    style={'textAlign': 'center', 'marginBottom': '20px'}),
            html.Div([
                html.Iframe(
                    srcDoc=basic_deck_html,
                    width='100%',
                    height='800px',
                    style={'border': 'none', 'borderRadius': '10px'}
                )
            ], style={'textAlign': 'center'}),
            html.Div([
                html.P("💡 Tip: Use mouse wheel to zoom, drag to rotate, and right-click to pan.",
                       style={'textAlign': 'center', 'color': 'gray', 'fontSize': '14px'})
            ], style={'marginTop': '10px'})
        ])

        print("Basic deck.gl globe created successfully!")
        return globe_component

    def _add_data_overlay_progressive(self, variable_name, data_array, lat_dim, lon_dim):
        """Add data overlay progressively to the existing globe"""
        print("Adding data overlay progressively...")
        
        lats = data_array.coords[lat_dim].values
        lons = data_array.coords[lon_dim].values

        # Get the geographic bounds
        lat_min, lat_max = lats.min(), lats.max()
        lon_min, lon_max = lons.min(), lons.max()
        
        print(f"Geographic bounds: lat [{lat_min:.4f}, {lat_max:.4f}], lon [{lon_min:.4f}, {lon_max:.4f}]")

        # Start with very coarse sampling for initial view
        max_dim_size = max(lats.size, lons.size)
        initial_sample_step = max(1, max_dim_size // 50)  # Very coarse for fast initial load
        print(f"Initial sampling with step {initial_sample_step} for fast loading")
        
        lats_sampled = lats[::initial_sample_step]
        lons_sampled = lons[::initial_sample_step]
        values_sampled = data_array.values[::initial_sample_step, ::initial_sample_step]
        
        # Create initial data points
        data_points = []
        for i in range(lats_sampled.shape[0]):
            for j in range(lons_sampled.shape[0]):
                if not np.isnan(values_sampled[i, j]):
                    data_points.append({
                        'latitude': float(lats_sampled[i]),
                        'longitude': float(lons_sampled[j]),
                        'value': float(values_sampled[i, j])
                    })

        print(f"Created {len(data_points)} initial data points")
        
        # Create a simple data overlay layer
        if data_points:
            try:
                import pydeck as pdk
                
                data_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=data_points,
                    get_position=['longitude', 'latitude'],
                    get_color='value',
                    get_radius=20000,  # 20km radius for coarse view
                    pickable=False,
                    opacity=0.8,
                    stroked=False,
                    filled=True,
                    radius_scale=3,
                    radius_min_pixels=3,
                    radius_max_pixels=8,
                    color_range=[[0, 0, 255], [0, 255, 0], [255, 0, 0]],  # Blue to Green to Red
                    color_domain=[min(p['value'] for p in data_points), max(p['value'] for p in data_points)]
                )
                
                print("Data overlay layer created successfully!")
                return data_layer
                
            except Exception as e:
                print(f"Failed to create data overlay: {e}")
                return None
        
        return None

    def _create_coastline_outline(self, continent_type, radius):
        """Create realistic continent outlines using actual geographic shapes"""
        try:
            if continent_type == 'north_america':
                # North America - more realistic shape
                lons = [-140, -130, -120, -110, -100, -90, -80, -70, -60, -50, -60, -70, -80, -90, -100, -110, -120, -130, -140]
                lats = [60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 15, 20, 25, 30, 35, 40, 45, 50]
                
            elif continent_type == 'europe_asia':
                # Europe/Asia - more realistic shape
                lons = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
                lats = [70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20]
                
            elif continent_type == 'africa':
                # Africa - more realistic shape
                lons = [-20, -10, 0, 10, 20, 30, 40, 50, 40, 30, 20, 10, 0, -10, -20]
                lats = [35, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30, -35]
                
            elif continent_type == 'south_america':
                # South America - more realistic shape
                lons = [-80, -70, -60, -50, -40, -30, -20, -10, -20, -30, -40, -50, -60, -70, -80]
                lats = [10, 5, 0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50, -55, -60]
                
            elif continent_type == 'australia':
                # Australia - more realistic shape
                lons = [110, 120, 130, 140, 150, 160, 150, 140, 130, 120, 110]
                lats = [-10, -15, -20, -25, -30, -35, -40, -35, -30, -25, -20]
                
            else:
                return None
            
            # Convert to radians
            lons_rad = np.radians(lons)
            lats_rad = np.radians(lats)
            
            # Convert to 3D coordinates
            x_coast = radius * np.cos(lats_rad) * np.cos(lons_rad)
            y_coast = radius * np.cos(lats_rad) * np.sin(lons_rad)
            z_coast = radius * np.sin(lats_rad)
            
            # Create a line trace for the coastline
            return go.Scatter3d(
                x=x_coast,
                y=y_coast,
                z=z_coast,
                mode='lines',
                line=dict(
                    color='rgb(100, 150, 50)',  # Green land color
                    width=2
                ),
                opacity=0.9,
                showlegend=False,
                hoverinfo='skip'
            )
        except Exception as e:
            print(f"Could not create coastline outline: {e}")
            return None

    def _create_fallback_3d_globe(self, variable_name, data_array, lat_dim, lon_dim):
        """Fallback to basic 3D globe if deck.gl is not available"""
        print("Creating fallback 3D globe...")
        
        lats = data_array.coords[lat_dim].values
        lons = data_array.coords[lon_dim].values

        # Sample the data for better performance
        max_dim_size = max(lats.size, lons.size)
        sample_step = max(1, max_dim_size // 100)
        
        lats_sampled = lats[::sample_step]
        lons_sampled = lons[::sample_step]
        values_sampled = data_array.values[::sample_step, ::sample_step]
        
        # Create a 3D scatter plot on a sphere
        lats_rad = np.radians(lats_sampled)
        lons_rad = np.radians(lons_sampled)
        
        lats_mesh, lons_mesh = np.meshgrid(lats_rad, lons_rad, indexing='ij')
        
        radius = 1.0
        x = radius * np.cos(lats_mesh) * np.cos(lons_mesh)
        y = radius * np.cos(lats_mesh) * np.sin(lons_mesh)
        z = radius * np.sin(lats_mesh)
        
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        values_flat = values_sampled.flatten()
        
        valid_mask = ~np.isnan(values_flat)
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        z_valid = z_flat[valid_mask]
        values_valid = values_flat[valid_mask]
        
        fig = go.Figure()
        
        # Add Earth surface with realistic oceans and land
        phi = np.linspace(0, 2*np.pi, 200)
        theta = np.linspace(-np.pi/2, np.pi/2, 100)
        phi_mesh, theta_mesh = np.meshgrid(phi, theta)
        
        earth_radius = 0.98
        x_earth = earth_radius * np.cos(theta_mesh) * np.cos(phi_mesh)
        y_earth = earth_radius * np.cos(theta_mesh) * np.sin(phi_mesh)
        z_earth = earth_radius * np.sin(theta_mesh)
        
        # Create a fast, realistic Earth using simple vector overlays
        # Start with a clean blue ocean base
        fig.add_trace(go.Surface(
            x=x_earth,
            y=y_earth,
            z=z_earth,
            colorscale='blues',  # Simple blue ocean
            opacity=0.9,
            showscale=False,
            name='Ocean Base'
        ))
        
        # Add continent outlines using realistic geographic shapes
        # North America coastline
        na_coast = self._create_coastline_outline('north_america', 0.99)
        if na_coast:
            fig.add_trace(na_coast)
        
        # Europe/Asia coastline
        eu_coast = self._create_coastline_outline('europe_asia', 0.99)
        if eu_coast:
            fig.add_trace(eu_coast)
        
        # Africa coastline
        af_coast = self._create_coastline_outline('africa', 0.99)
        if af_coast:
            fig.add_trace(af_coast)
        
        # South America coastline
        sa_coast = self._create_coastline_outline('south_america', 0.99)
        if sa_coast:
            fig.add_trace(sa_coast)
        
        # Australia
        au_coast = self._create_coastline_outline('australia', 0.99)
        if au_coast:
            fig.add_trace(au_coast)
        
        # Add data points
        if len(values_valid) > 0:
            fig.add_trace(go.Scatter3d(
                x=x_valid,
                y=y_valid,
                z=z_valid,
                mode='markers',
                marker=dict(
                    size=3.0,
                    color=values_valid,
                    colorscale='viridis',
                    opacity=0.9,
                    showscale=True,
                    colorbar=dict(
                        title=variable_name,
                        title_side="right",
                        thickness=15,
                        len=0.6
                    )
                ),
                text=[f"{variable_name}: {val:.3f}" for val in values_valid],
                hoverinfo='text',
                name=variable_name
            ))
        
        fig.update_layout(
            title=dict(
                text=f"🌍 {variable_name} - Enhanced 3D Globe",
                font=dict(size=20, color='#2c3e50'),
                x=0.5,
                y=0.95
            ),
            height=800,
            width=None,
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, range=[-1.2, 1.2]),
                yaxis=dict(showgrid=False, showticklabels=False, range=[-1.2, 1.2]),
                zaxis=dict(showgrid=False, showticklabels=False, range=[-1.2, 1.2]),
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, t=80, b=0),
            showlegend=False
        )
        
        return fig

    def _create_mapbox_globe(self, variable_name, data_array, lat_dim, lon_dim):
        """Create a Mapbox 3D globe with data overlay"""
        print("Creating Mapbox 3D globe...")
        
        # Load Mapbox API token from environment file
        try:
            import os
            from dotenv import load_dotenv
            
            # Load the Mapbox API token from the credentials file
            creds_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'creds', 'mapboxapi.env')
            load_dotenv(creds_path)
            
            mapbox_token = os.getenv('MAPBOX_API_TOKEN')
            if not mapbox_token:
                print("Warning: MAPBOX_API_TOKEN not found, falling back to basic globe")
                return self._create_fallback_3d_globe(variable_name, data_array, lat_dim, lon_dim)
                
            print("Mapbox API token loaded successfully")
            
        except Exception as e:
            print(f"Error loading Mapbox token: {e}, falling back to basic globe")
            return self._create_fallback_3d_globe(variable_name, data_array, lat_dim, lon_dim)
        
        lats = data_array.coords[lat_dim].values
        lons = data_array.coords[lon_dim].values
        
        # Get geographic bounds
        lat_min, lat_max = lats.min(), lats.max()
        lon_min, lon_max = lons.min(), lons.max()
        
        # Sample data for performance
        max_dim_size = max(lats.size, lons.size)
        sample_step = max(1, max_dim_size // 100)
        
        lats_sampled = lats[::sample_step]
        lons_sampled = lons[::sample_step]
        values_sampled = data_array.values[::sample_step, ::sample_step]
        
        # Create data points for overlay
        data_points = []
        for i in range(lats_sampled.shape[0]):
            for j in range(lons_sampled.shape[0]):
                if not np.isnan(values_sampled[i, j]):
                    data_points.append({
                        'lat': float(lats_sampled[i]),
                        'lon': float(lons_sampled[j]),
                        'value': float(values_sampled[i, j])
                    })
        
        # Create a 3D globe figure using Plotly's built-in 3D projection
        fig = go.Figure()
        
        # Add the data as a 3D scatter plot on a sphere
        if data_points:
            # Convert lat/lon to 3D coordinates on a sphere
            radius = 1.0
            x_coords = []
            y_coords = []
            z_coords = []
            values = []
            
            for point in data_points:
                lat_rad = np.radians(point['lat'])
                lon_rad = np.radians(point['lon'])
                
                x = radius * np.cos(lat_rad) * np.cos(lon_rad)
                y = radius * np.cos(lat_rad) * np.sin(lon_rad)
                z = radius * np.sin(lat_rad)
                
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)
                values.append(point['value'])
            
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers',
                marker=dict(
                    size=4,
                    color=values,
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title=variable_name)
                ),
                text=[f"{variable_name}: {val:.3f}" for val in values],
                hoverinfo='text',
                name=variable_name
            ))
        
        # Add a base Earth sphere with Mapbox satellite texture
        # Create a sphere surface
        phi = np.linspace(0, 2*np.pi, 100)
        theta = np.linspace(-np.pi/2, np.pi/2, 50)
        phi_mesh, theta_mesh = np.meshgrid(phi, theta)
        
        earth_radius = 0.98
        x_earth = earth_radius * np.cos(theta_mesh) * np.cos(phi_mesh)
        y_earth = earth_radius * np.cos(theta_mesh) * np.sin(phi_mesh)
        z_earth = earth_radius * np.sin(theta_mesh)
        
        # Add Earth surface with realistic colors
        fig.add_trace(go.Surface(
            x=x_earth,
            y=y_earth,
            z=z_earth,
            colorscale='earth',  # Use Plotly's built-in Earth colorscale
            opacity=0.8,
            showscale=False,
            name='Earth Surface'
        ))
        
        # Configure the 3D scene for globe view
        fig.update_layout(
            title=f"🌍 {variable_name} - Globe",
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, range=[-1.2, 1.2]),
                yaxis=dict(showgrid=False, showticklabels=False, range=[-1.2, 1.2]),
                zaxis=dict(showgrid=False, showticklabels=False, range=[-1.2, 1.2]),
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=800,
            margin=dict(l=0, r=0, t=80, b=0),
            showlegend=False
        )
        
        print("Mapbox 3D globe created successfully!")
        return fig

    def _get_earth_geojson(self):
        """Get realistic Earth landmasses GeoJSON"""
        # More realistic continent boundaries
        earth_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "North America"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-170, 50], [-60, 50], [-60, 25], [-80, 15], [-170, 15], [-170, 50]
                        ]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "South America"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-80, 15], [-35, 15], [-35, -55], [-80, -55], [-80, 15]
                        ]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "Europe"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-10, 70], [40, 70], [40, 35], [-10, 35], [-10, 70]
                        ]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "Africa"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-20, 35], [50, 35], [50, -35], [-20, -35], [-20, 35]
                        ]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "Asia"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [40, 70], [180, 70], [180, 15], [100, 15], [40, 15], [40, 70]
                        ]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "Australia"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [110, -10], [155, -10], [155, -45], [110, -45], [110, -10]
                        ]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "Antarctica"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-180, -60], [180, -60], [180, -90], [-180, -90], [-180, -60]
                        ]]
                    }
                }
            ]
        }
        return earth_data

    def _get_ocean_geojson(self):
        """Get realistic ocean GeoJSON with major ocean basins"""
        ocean_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "Pacific Ocean"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [100, 70], [180, 70], [180, -60], [100, -60], [100, 70]
                        ]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "Atlantic Ocean"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-80, 70], [20, 70], [20, -60], [-80, -60], [-80, 70]
                        ]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "Indian Ocean"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [20, 35], [100, 35], [100, -60], [20, -60], [20, 35]
                        ]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "Arctic Ocean"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-180, 70], [180, 70], [180, 90], [-180, 90], [-180, 70]
                        ]]
                    }
                }
            ]
        }
        return ocean_data

    def _get_countries_geojson(self):
        """Get country boundaries GeoJSON from Natural Earth data"""
        try:
            # Try to fetch from Natural Earth (free, public domain)
            url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
            # Use the global requests import
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to fetch countries data: {response.status_code}")
                return self._get_simple_countries_geojson()
        except Exception as e:
            print(f"Error fetching countries data: {e}")
            return self._get_simple_countries_geojson()

    def _get_simple_countries_geojson(self):
        """Fallback simple country boundaries"""
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "Continents"},
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": [
                            # Simplified continent boundaries
                            [[[-180, -60], [-60, -60], [-60, 0], [-180, 0], [-180, -60]]],  # South America
                            [[[-180, 0], [-60, 0], [-60, 60], [-180, 60], [-180, 0]]],      # North America
                            [[[-60, 0], [60, 0], [60, 60], [-60, 60], [-60, 0]]],           # Europe/Asia
                            [[[60, 0], [180, 0], [180, 60], [60, 60], [60, 0]]],            # Asia
                            [[[-60, -60], [60, -60], [60, 0], [-60, 0], [-60, -60]]],       # Africa
                            [[[60, -60], [180, -60], [180, 0], [60, 0], [60, -60]]]         # Australia
                        ]
                    }
                }
            ]
        }

    def _create_enhanced_earth_texture(self):
        """Create an enhanced Earth texture with realistic land/water patterns"""
        # This creates a more realistic Earth appearance with elevation-based coloring
        import numpy as np
        
        # Create a high-resolution grid for the Earth surface
        phi = np.linspace(0, 2*np.pi, 360)  # Longitude
        theta = np.linspace(-np.pi/2, np.pi/2, 180)  # Latitude
        
        phi_mesh, theta_mesh = np.meshgrid(phi, theta)
        
        # Convert to degrees for easier calculations
        lat_deg = np.degrees(theta_mesh)
        lon_deg = np.degrees(phi_mesh)
        
        # Create elevation-based coloring
        # Simulate continents and oceans based on latitude/longitude patterns
        elevation = np.zeros_like(lat_deg)
        
        # North America (rough approximation)
        na_mask = (lon_deg >= -170) & (lon_deg <= -50) & (lat_deg >= 15) & (lat_deg <= 70)
        elevation[na_mask] = 0.3  # Land elevation
        
        # South America
        sa_mask = (lon_deg >= -80) & (lon_deg <= -35) & (lat_deg >= -55) & (lat_deg <= 15)
        elevation[sa_mask] = 0.3
        
        # Europe
        eu_mask = (lon_deg >= -10) & (lon_deg <= 40) & (lat_deg >= 35) & (lat_deg <= 70)
        elevation[eu_mask] = 0.3
        
        # Africa
        af_mask = (lon_deg >= -20) & (lon_deg <= 50) & (lat_deg >= -35) & (lat_deg <= 35)
        elevation[af_mask] = 0.3
        
        # Asia
        asia_mask = (lon_deg >= 40) & (lon_deg <= 180) & (lat_deg >= 15) & (lat_deg <= 70)
        elevation[asia_mask] = 0.3
        
        # Australia
        aus_mask = (lon_deg >= 110) & (lon_deg <= 155) & (lat_deg >= -45) & (lat_deg <= -10)
        elevation[aus_mask] = 0.3
        
        # Antarctica
        ant_mask = (lat_deg <= -60)
        elevation[ant_mask] = 0.4  # Higher elevation for ice
        
        # Add some noise for more realistic appearance
        np.random.seed(42)  # For reproducible results
        noise = np.random.normal(0, 0.05, elevation.shape)
        elevation += noise
        elevation = np.clip(elevation, 0, 1)
        
        return phi_mesh, theta_mesh, elevation


# Legacy classes for backward compatibility (can be removed later)
class DataQuickStats:
    """Legacy class - use DataManager instead"""

    def __init__(self, app, ds_getter, dataseturl_getter, dataset_engine_getter):
        self.data_manager = DataManager(app, ds_getter)

    def setup_callbacks(self):
        self.data_manager.setup_callbacks()


class DataSubsetter:
    """Legacy class - use DataManager instead"""

    def __init__(self, dataset):
        self.dataset = dataset

    def subset_data(self, selected_var, user_selection, compute=True):
        # This functionality is now in DataManager._get_subsetted_data
        pass


class DataPlot:
    """Legacy class - use DataManager instead"""

    def __init__(self, app, data_array, dimension_selection, dataseturl_getter, dataset_engine_getter):
        self.data_manager = DataManager(app, lambda: data_array)

    def setup_callbacks(self):
        self.data_manager.setup_callbacks()
