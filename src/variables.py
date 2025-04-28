from dash import Output, Input

class VariableSelection:
    def __init__(self, app, ds_getter):
        self.app = app
        self.ds_getter = ds_getter  # Pass a function to get the current dataset

    def setup_callbacks(self):
        @self.app.callback(
            Output('variable-dropdown', 'options'),
            Output('variable-dropdown', 'value'),
            Input('load-status', 'children')
        )
        def update_variable_options(_):
            ds = self.ds_getter()
            if ds is None:
                return [], None
            print('Available data_vars:', list(ds.data_vars))
            options = [{'label': var, 'value': var} for var in ds.data_vars]
            value = options[0]['value'] if options else None
            return options, value