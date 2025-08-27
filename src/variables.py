from dash import Output, Input

class VariableSelection:
    def __init__(self, app, ds_getter):
        self.app = app
        self.ds_getter = ds_getter  # Function that returns xarray.Dataset

    def setup_callbacks(self):
        @self.app.callback(
            Output('variable-dropdown', 'options'),
            Output('variable-dropdown', 'value'),
            Input('load-status', 'children')
        )
        def update_variable_options(_):
            ds = self.ds_getter()
            if ds is None or not hasattr(ds, 'data_vars'):
                return [], None

            options = []
            for var in ds.data_vars:
                # Only exclude variables with no dimensions (e.g., scalar metadata)
                if len(ds[var].dims) == 0:
                    continue
                # Build a user-friendly label
                attrs = ds[var].attrs
                label_parts = [var]
                if 'long_name' in attrs:
                    label_parts.append(f"[{attrs['long_name']}]")
                if 'units' in attrs:
                    label_parts.append(f"({attrs['units']})")
                label = " ".join(label_parts)
                options.append({'label': label.strip(), 'value': var})

            default = options[0]['value'] if options else None
            return options, default