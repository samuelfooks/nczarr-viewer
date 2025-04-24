from dash import Output, Input
from dash import dcc, html

class VariableSelection:
    def __init__(self, app, ds):
        self.app = app
        self.ds = ds

    def setup_callbacks(self):
        @self.app.callback(
            Output('variable-dropdown', 'options'),
            Input('variable-dropdown', 'value')
        )
        def update_variable_options(selected_var):
            
            return [{'label': var, 'value': var} for var in self.ds.data_vars]
