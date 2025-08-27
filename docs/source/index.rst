.. dash_nczarr_viewer documentation master file, created by
   sphinx-quickstart on Thu Oct 14 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Dash NCZarr Viewer's documentation!
==============================================

The Dash NCZarr Viewer is a powerful web application for exploring and visualizing geospatial NetCDF and Zarr datasets. Built with Dash and optimized for both local development and cloud deployment.

Key Features
------------

- **Multi-Format Support**: NetCDF, Zarr, HDF5, and GRIB files
- **Cloud Storage Access**: Direct S3, Minio, and cloud storage access
- **Marine Data Integration**: Copernicus Marine Service (CMEMS) support
- **Interactive Visualization**: Dynamic plotting with Plotly and Matplotlib
- **Smart Data Exploration**: Variable browsing, dimension filtering, and statistics
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Docker Ready**: Containerized for easy deployment

Quick Start
-----------

.. code-block:: bash

   # Clone and setup
   git clone https://github.com/samuelfooks/dash_nczarr_viewer.git
   cd dash_nczarr_viewer
   
   # Install dependencies with uv
   uv sync
   
   # Run the viewer
   python run.py

The viewer will be available at http://localhost:8050

Documentation Contents
======================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   dockerfile
   dashapp
   api

Additional Resources
===================

- **GitHub Repository**: https://github.com/samuelfooks/dash_nczarr_viewer
- **Docker Hub**: samfooks/zarr-netcdf-viewer
- **Presentation**: See `docs/dash_nczarr_viewer_presentation.md` for usage examples
- **Examples**: Explore `docs/explore_data/` for sample datasets and workflows

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`