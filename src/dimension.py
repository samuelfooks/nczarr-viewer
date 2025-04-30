# dimension_selection.py
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State, ALL, MATCH
import pyproj
import json

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
            prevent_initial_call=True
        )
        def store_selected_dimensions(slider_values, dropdown_values, checked_dims, selected_var):
            ds = self.ds_getter()
            if not checked_dims or ds is None:
                return {}
            selected_dims = {}
            slider_idx = 0
            dropdown_idx = 0
            for dim in checked_dims:
                dim_lower = dim.lower()
                if any(key in dim_lower for key in ['lat', 'lon', 'x', 'y']):
                    if slider_idx < len(slider_values) and slider_values[slider_idx] is not None:
                        val = slider_values[slider_idx]
                        selected_dims[dim] = val
                    slider_idx += 1
                elif any(key in dim_lower for key in ['depth', 'time']):
                    if dropdown_idx < len(dropdown_values) and dropdown_values[dropdown_idx] is not None:
                        val = dropdown_values[dropdown_idx]
                        selected_dims[dim] = val
                    dropdown_idx += 1
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
            html.Label("Select dimensions to filter:"),
            dcc.Checklist(
                id='dimension-checklist',
                options=[{'label': dim, 'value': dim} for dim in dimensions],
                value=dimensions
            )
        ])

    def generate_dimension_controls(self, ds, selected_dims, selected_var):
        if selected_var is None or selected_dims is None:
            return []
        dimension_controls = []
        for dim in selected_dims:
            dim_lower = dim.lower()
            controls = [
                html.Label(f"{dim} control type:"),
                dcc.RadioItems(
                    id={'type': 'dim-control-type', 'index': dim},
                    options=[
                        {'label': 'Slider', 'value': 'slider'},
                        {'label': 'Dropdown', 'value': 'dropdown'}
                    ],
                    value='slider' if ds[selected_var][dim].size > 10 else 'dropdown',
                    inline=True,
                    style={'marginBottom': '8px'}
                ),
                html.Div(id={'type': 'dim-control-widget', 'index': dim})
            ]
            dimension_controls.append(html.Div(controls, style={'marginBottom': '16px'}))
        return dimension_controls

    def create_range_slider(self, ds, dim, selected_var):
        dim_values = ds[selected_var][dim].values
        sorted_dim_values = sorted(dim_values)
        min_val = 0
        max_val = len(sorted_dim_values) - 1
        range_25 = int(0.25 * max_val)
        range_75 = int(0.75 * max_val)
        step = max(1, len(sorted_dim_values) // 10)
        def format_mark(val):
            try:
                return f"{float(val):.4f}"
            except (ValueError, TypeError):
                return str(val)
        marks = None
        grid_mapping = ds[selected_var].attrs.get('grid_mapping')
        crs = None
        if grid_mapping and grid_mapping in ds:
            gm_var = ds[grid_mapping]
            try:
                # Try to construct a pyproj CRS from grid mapping attributes
                crs = pyproj.CRS.from_cf(gm_var.attrs)
            except Exception as e:
                print(f"Could not parse grid mapping CRS: {e}")
        if dim.lower() == 'x' and crs is not None:
            try:
                # Assume y=0 for x axis
                transformer = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
                marks = {i: f"{transformer.transform(sorted_dim_values[i], 0)[0]:.2f}°" for i in range(0, len(sorted_dim_values), step)}
            except Exception as e:
                print(f"CRS conversion error for x: {e}")
                marks = {i: format_mark(sorted_dim_values[i]) for i in range(0, len(sorted_dim_values), step)}
        elif dim.lower() == 'y' and crs is not None:
            try:
                # Assume x=0 for y axis
                transformer = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
                marks = {i: f"{transformer.transform(0, sorted_dim_values[i])[1]:.2f}°" for i in range(0, len(sorted_dim_values), step)}
            except Exception as e:
                print(f"CRS conversion error for y: {e}")
                marks = {i: format_mark(sorted_dim_values[i]) for i in range(0, len(sorted_dim_values), step)}
        else:
            marks = {i: format_mark(sorted_dim_values[i]) for i in range(0, len(sorted_dim_values), step)}
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

    def create_dropdown(self, ds, dim, selected_var):
        dim_values = ds[selected_var][dim].values
        return html.Div([
            html.Label(f'Select {dim}'),
            dcc.Dropdown(
                id={'type': 'dimension-dropdown', 'index': dim},
                options=[{'label': str(val), 'value': idx} for idx, val in enumerate(dim_values)],
                placeholder=f"Select {dim}"
            )
        ])