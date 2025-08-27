# dimension_selection.py
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State, ALL, MATCH
import pyproj
import json
import numpy as np


class DimensionSelection:
    def __init__(self, app, ds_getter):
        self.app = app
        self.ds_getter = ds_getter
        self.dim_selections = {}

    def setup_callbacks(self):
        @self.app.callback(
            Output('dimension-checklist-container', 'children'),
            Input('variable-dropdown', 'value')
        )
        def update_dimension_checklist(selected_var):
            ds = self.ds_getter()
            if selected_var and ds is not None:
                return self.generate_dimension_checklist(ds, selected_var)
            return ""

        @self.app.callback(
            Output('dimension-dropdowns-container', 'children'),
            Input('dimension-checklist', 'value'),
            State('variable-dropdown', 'value')
        )
        def update_dimension_controls(selected_dims, selected_var):
            ds = self.ds_getter()
            if selected_var and selected_dims and ds is not None:
                return self.generate_dimension_controls(ds, selected_dims, selected_var)
            return ""

        @self.app.callback(
            Output('selected-dimensions-store', 'data'),
            Input({'type': 'dimension-slider', 'index': ALL}, 'value'),
            Input({'type': 'dimension-dropdown', 'index': ALL}, 'value'),
            Input('dimension-checklist', 'value'),
            State('variable-dropdown', 'value'),
            State({'type': 'dimension-slider', 'index': ALL}, 'id'),
            State({'type': 'dimension-dropdown', 'index': ALL}, 'id'),
            prevent_initial_call=True
        )
        def store_selected_dimensions(slider_values, dropdown_values,
                                      checked_dims, selected_var,
                                      slider_ids, dropdown_ids):
            ds = self.ds_getter()
            if not checked_dims or ds is None:
                return {}

            selected_dims = {}

            # Process slider values (range selections)
            for i, slider_val in enumerate(slider_values):
                if slider_val is not None and i < len(slider_ids):
                    dim = slider_ids[i]['index']
                    if dim in checked_dims:
                        # Convert slider indices to actual dimension values
                        dim_values = ds[selected_var][dim].values
                        if (isinstance(slider_val, list) and
                                len(slider_val) == 2):
                            # Range selection - return tuple of (start, end) values
                            start_idx, end_idx = slider_val
                            start_val = dim_values[start_idx]
                            end_val = dim_values[end_idx]
                            # Convert to proper data type if needed
                            if isinstance(start_val, np.datetime64):
                                start_val = start_val.item()
                            if isinstance(end_val, np.datetime64):
                                end_val = end_val.item()
                            selected_dims[dim] = (start_val, end_val)
                        elif isinstance(slider_val, (int, float)):
                            # Single value selection - return tuple with single value
                            idx = int(slider_val)
                            val = dim_values[idx]
                            # Convert to proper data type if needed
                            if isinstance(val, np.datetime64):
                                val = val.item()
                            selected_dims[dim] = (val,)

            # Process dropdown values (single selections)
            for i, dropdown_val in enumerate(dropdown_values):
                if dropdown_val is not None and i < len(dropdown_ids):
                    dim = dropdown_ids[i]['index']
                    if (dim in checked_dims and
                            dim not in selected_dims):  # Don't override slider selections
                        # Convert dropdown index to actual dimension value
                        dim_values = ds[selected_var][dim].values
                        idx = dropdown_val
                        val = dim_values[idx]
                        # Convert to proper data type if needed
                        if isinstance(val, np.datetime64):
                            val = val.item()
                        # Single selection - return tuple with single value
                        selected_dims[dim] = (val,)

            return selected_dims

        @self.app.callback(
            Output({'type': 'dim-control-widget', 'index': MATCH}, 'children'),
            Input({'type': 'dim-control-type', 'index': MATCH}, 'value'),
            State('variable-dropdown', 'value'),
            State('dimension-checklist', 'value'),
            State({'type': 'dim-control-type', 'index': MATCH}, 'id'),
        )
        def render_dim_control(control_type, selected_var, checked_dims, id_dict):
            ds = self.ds_getter()
            dim = id_dict['index']
            if not selected_var or not checked_dims or ds is None:
                return None
            if control_type == 'slider':
                return self.create_range_slider(ds, dim, selected_var)
            else:
                return self.create_dropdown(ds, dim, selected_var)

    def generate_dimension_checklist(self, ds, selected_var):
        if selected_var is None:
            return []
        # Use .sizes for mapping from dimension name to length (future-proof)
        dimensions = list(ds[selected_var].sizes.keys())
        return html.Div([
            html.Label("Select dimensions to filter:", className="mb-2"),
            dcc.Checklist(
                id='dimension-checklist',
                options=[{'label': dim, 'value': dim} for dim in dimensions],
                value=dimensions,
                className="mb-3"
            )
        ])

    def generate_dimension_controls(self, ds, selected_dims, selected_var):
        if selected_var is None or selected_dims is None:
            return []

        dimension_controls = []
        for dim in selected_dims:
            dim_lower = dim.lower()
            # Determine default control type based on dimension characteristics
            default_control = self._get_default_control_type(
                ds, selected_var, dim)

            controls = [
                html.Label(f"{dim} control type:", className="mb-2"),
                dcc.RadioItems(
                    id={'type': 'dim-control-type', 'index': dim},
                    options=[
                        {'label': 'Slider (Range)', 'value': 'slider'},
                        {'label': 'Dropdown (Single)', 'value': 'dropdown'}
                    ],
                    value=default_control,
                    inline=True,
                    className="mb-3"
                ),
                html.Div(id={'type': 'dim-control-widget', 'index': dim})
            ]
            dimension_controls.append(
                html.Div(controls, className="mb-4 p-3 border rounded")
            )
        return dimension_controls

    def _get_default_control_type(self, ds, selected_var, dim):
        """Determine the default control type for a dimension"""
        dim_lower = dim.lower()
        dim_size = ds[selected_var][dim].size

        # Spatial dimensions typically work better with sliders
        if any(key in dim_lower for key in ['lat', 'lon', 'x', 'y']):
            return 'slider'
        # Time dimensions often work better with sliders
        elif any(key in dim_lower for key in ['time', 'date']):
            return 'slider'
        # Elevation/depth can work with either, default to slider for large ranges
        elif any(key in dim_lower for key in ['depth', 'elevation', 'height', 'level']):
            return 'slider' if dim_size > 5 else 'dropdown'
        # For other dimensions, use dropdown if small, slider if large
        else:
            return 'slider' if dim_size > 10 else 'dropdown'

    def create_range_slider(self, ds, dim, selected_var):
        dim_values = ds[selected_var][dim].values
        sorted_dim_values = sorted(dim_values)
        min_idx = 0
        max_idx = len(sorted_dim_values) - 1

        # Set default range to middle 50% of the data
        range_25 = int(0.25 * max_idx)
        range_75 = int(0.75 * max_idx)
        step = max(1, len(sorted_dim_values) // 20)  # More granular steps

        # Create marks for better visualization
        marks = self._create_slider_marks(
            ds, selected_var, dim, sorted_dim_values, step)

        return html.Div([
            html.Label(f'Select {dim} range', className="mb-2"),
            dcc.RangeSlider(
                id={'type': 'dimension-slider', 'index': dim},
                min=min_idx,
                max=max_idx,
                value=[range_25, range_75],
                marks=marks,
                step=1,
                tooltip={"placement": "bottom", "always_visible": True},
                className="mb-2"
            ),
            html.Div([
                html.Small(f"Range: {self._format_dim_value(sorted_dim_values[range_25])} to {self._format_dim_value(sorted_dim_values[range_75])}",
                           className="text-muted")
            ], id={'type': 'slider-output', 'index': dim})
        ])

    def create_dropdown(self, ds, dim, selected_var):
        dim_values = ds[selected_var][dim].values

        # For large dimensions, show a subset of values to avoid overwhelming the dropdown
        if len(dim_values) > 100:
            # Sample every nth value for display
            step = len(dim_values) // 100
            display_values = dim_values[::step]
            display_indices = list(range(0, len(dim_values), step))
        else:
            display_values = dim_values
            display_indices = list(range(len(dim_values)))

        # Create better labels for the dropdown
        options = []
        for idx, val in zip(display_indices, display_values):
            if isinstance(val, (np.datetime64, np.timedelta64)):
                label = str(val)
            else:
                label = f"{val:.4g}" if isinstance(
                    val, (int, float)) else str(val)
            options.append({'label': label, 'value': idx})

        return html.Div([
            html.Label(f'Select {dim}', className="mb-2"),
            dcc.Dropdown(
                id={'type': 'dimension-dropdown', 'index': dim},
                options=options,
                placeholder=f"Select {dim}",
                className="mb-2"
            ),
            html.Div([
                html.Small(f"Available: {len(dim_values)} values",
                           className="text-muted")
            ])
        ])

    def _create_slider_marks(self, ds, selected_var, dim, sorted_dim_values, step):
        """Create marks for the slider with proper formatting"""
        marks = {}

        # Try to get coordinate reference system for spatial dimensions
        crs = self._get_crs(ds, selected_var)

        dim_lower = dim.lower()

        # Create marks with appropriate formatting
        for i in range(0, len(sorted_dim_values), step):
            val = sorted_dim_values[i]

            if dim_lower in ['x', 'lon'] and crs is not None:
                try:
                    # Convert to degrees for display
                    transformer = pyproj.Transformer.from_crs(
                        crs, 4326, always_xy=True)
                    lon, _ = transformer.transform(val, 0)
                    marks[i] = f"{lon:.2f}°"
                except Exception:
                    marks[i] = self._format_dim_value(val)
            elif dim_lower in ['y', 'lat'] and crs is not None:
                try:
                    # Convert to degrees for display
                    transformer = pyproj.Transformer.from_crs(
                        crs, 4326, always_xy=True)
                    _, lat = transformer.transform(0, val)
                    marks[i] = f"{lat:.2f}°"
                except Exception:
                    marks[i] = self._format_dim_value(val)
            elif isinstance(val, (np.datetime64, np.timedelta64)):
                marks[i] = str(val)[:10]  # Truncate long datetime strings
            else:
                marks[i] = self._format_dim_value(val)

        return marks

    def _get_crs(self, ds, selected_var):
        """Get coordinate reference system from dataset"""
        try:
            grid_mapping = ds[selected_var].attrs.get('grid_mapping')
            if grid_mapping and grid_mapping in ds:
                gm_var = ds[grid_mapping]
                return pyproj.CRS.from_cf(gm_var.attrs)
        except Exception:
            pass
        return None

    def _format_dim_value(self, val):
        """Format dimension values for display with proper type handling"""
        if isinstance(val, (np.datetime64, np.timedelta64)):
            return str(val)[:10]  # Truncate long datetime strings
        elif isinstance(val, (int, float)):
            return f"{val:.4g}"
        else:
            return str(val)
