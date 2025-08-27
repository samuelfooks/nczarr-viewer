API Reference
=============

Core Modules
------------

.. automodule:: main
   :members:
   :undoc-members:
   :show-inheritance:

Data Loading and Processing
---------------------------

.. automodule:: data
   :members:
   :undoc-members:
   :show-inheritance:

Variable and Dimension Handling
------------------------------

.. automodule:: variables
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: dimension
   :members:
   :undoc-members:
   :show-inheritance:

Main Application Class
---------------------

The main application class ``ViewerApp`` provides the following key methods:

- **setup_layout()**: Creates the main application layout
- **setup_callbacks()**: Sets up all interactive callbacks
- **load_dataset()**: Handles dataset loading with error handling
- **run()**: Starts the Dash application server

For detailed implementation, see the source code in ``src/main.py``.