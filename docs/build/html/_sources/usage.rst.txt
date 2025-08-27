Usage
=====

The NCZarr Viewer is a powerful web application for exploring and visualizing geospatial NetCDF and Zarr datasets. This guide covers various ways to use the application.

Quick Start
-----------

The fastest way to get started is using the local development setup:

.. code-block:: bash

   # Clone and setup
   git clone git@github.com:EDITO-Infra/nczarr-viewer.git
   cd nczarr-viewer
   
   # Install dependencies with uv
   uv sync
   
   # Run the viewer
   python run.py

The viewer will be available at http://localhost:8050

Local Development
-----------------

For development and testing, use the local Python setup:

.. code-block:: bash

   # Install in development mode
   pip install -e .
   
   # Run the viewer
   python run.py

The application will automatically reload when you make changes to the source code.

Docker Deployment
-----------------

For production deployment or consistent environments, use Docker:

Quick Docker Run
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Pull and run the latest image
   docker run -p 8050:8050 samfooks/nczarr-viewer:latest

With Local Files
^^^^^^^^^^^^^^^^

1. Place your NetCDF file (e.g., `Water_body_chlorophyll-a.nc`) in a folder named `myfiles`.
2. Run the following Docker command:

   .. code-block:: bash

      docker run -v $(pwd)/myfiles:/app/myfiles -p 8050:8050 samfooks/nczarr-viewer:latest

3. Once the container is running, open your browser to `http://localhost:8050`
4. Enter the path to your dataset: `/app/myfiles/Water_body_chlorophyll-a.nc`

With Online Datasets
^^^^^^^^^^^^^^^^^^^^

Access Zarr datasets directly from cloud storage:

.. code-block:: bash

   docker run -p 8050:8050 samfooks/nczarr-viewer:latest

Once the container is running:
1. Open your browser to `http://localhost:8050`
2. Enter the Zarr dataset URL: `https://s3.waw3-1.cloudferro.com/mdl-arco-geo-041/arco/NWSHELF_ANALYSISFORECAST_BGC_004_002/cmems_mod_nws_bgc_anfc_0.027deg-3D_P1D-m_202311/geoChunked.zarr`

S3/Cloud Storage Access
-----------------------

For NetCDF files stored on S3-compatible storage (like Minio), add `#mode=bytes` to the end of your URL:

.. code-block:: bash

   # Example for Minio/EDITO storage
   docker run -p 8050:8050 samfooks/nczarr-viewer:latest

Once the container is running:
1. Open your browser to `http://localhost:8050`
2. Enter the S3 URL with #mode=bytes: `https://minio.lab.dive.edito.eu/oidc-YOURUSERNAME/folder/mynetcdf.nc#mode=bytes`

The `#mode=bytes` parameter ensures proper binary data access for NetCDF files in object storage.

CMEMS Integration
-----------------

To access Copernicus Marine Service datasets:

1. Set your CMEMS credentials as environment variables:

   .. code-block:: bash

      export CMEMS_USERNAME=your_username
      export CMEMS_PASSWORD=your_password

2. Run the Docker container with credentials:

   .. code-block:: bash

      docker run -e CMEMS_USERNAME=$CMEMS_USERNAME -e CMEMS_PASSWORD=$CMEMS_PASSWORD -p 8050:8050 samfooks/nczarr-viewer:latest

Configuration
-------------

The application supports various configuration options through the web interface:

- **Backend Selection**: Choose between xarray, Copernicus Marine, or auto-detect
- **Engine Configuration**: Specify engines like 'zarr', 'netcdf4', 'h5netcdf'
- **Additional Parameters**: Pass xarray-specific parameters like chunks, decode_timedelta, etc.

Example JSON configuration:

.. code-block:: json

   {
     "backend": "xarray",
     "engine": "zarr",
     "chunks": {"time": 1},
     "decode_timedelta": true
   }

Troubleshooting
--------------

Common Issues and Solutions:

1. **NetCDF Loading Fails**: 
   - For S3 storage: Add `#mode=bytes` to the URL
   - Try using 'zarr' engine instead of 'netcdf4'
   - Check file integrity and S3 credentials

2. **Memory Issues**:
   - Use chunking parameters to limit memory usage
   - Consider using smaller datasets for testing

3. **Authentication Errors**:
   - Verify CMEMS credentials are correct
   - Check S3 access permissions for cloud storage

For more help, check the GitHub issues or documentation in the `docs/` folder.