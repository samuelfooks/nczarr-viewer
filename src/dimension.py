# dimension_selection.py
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State, ALL, MATCH
import json

class DimensionSelection:
    def __init__(self, app, ds):
        self.app = app
        self.ds = ds
        self.dim_selections = {}

    def setup_callbacks(self):
        # Checklist for choosing which dimensions to filter
        @self.app.callback(
            Output('dimension-checklist-container', 'children'),
            Input('variable-dropdown', 'value')
        )
        def update_dimension_checklist(selected_var):
            if selected_var:
                return self.generate_dimension_checklist(selected_var)
            return ""

        # Populate and show/hide lat/lon dropdowns dynamically
        @self.app.callback(
            Output('lat-dim-dropdown', 'options'),
            Output('lat-dim-dropdown', 'style'),
            Output('lon-dim-dropdown', 'options'),
            Output('lon-dim-dropdown', 'style'),
            Input('variable-dropdown', 'value')
        )
        def update_lat_lon_dropdowns(selected_var):
            if selected_var is None:
                return [], {'display': 'none'}, [], {'display': 'none'}

            dims = self.ds[selected_var].dims
            options = [{'label': d, 'value': d} for d in dims]
            return