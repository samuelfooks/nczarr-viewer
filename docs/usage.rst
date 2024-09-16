Usage
=====

This application allows you to view local NetCDF files or online Zarr files.

Running with a Local NetCDF File
--------------------------------

1. Place your NetCDF file (e.g., `Water_body_chlorophyll-a.nc`) in a folder named `myfiles`.
2. Run the following Docker command:
   
   .. code-block:: bash

      docker run -v $(pwd)/myfiles:/app/myfiles -p 8050:8050 samfooks/zarrdashapp:latest /app/myfiles/Water_body_chlorophyll-a.nc

Running with an Online Zarr File
--------------------------------

Run the following Docker command:

.. code-block:: bash

   docker run -it -p 8050:8050 samfooks/zarr-netcdf-viewer:latest https://s3.waw3-1.cloudferro.com/mdl-arco-geo-041/arco/NWSHELF_ANALYSISFORECAST_BGC_004_002/cmems_mod_nws_bgc_anfc_0.027deg-3D_P1D-m_202311/geoChunked.zarr

Running with a NetCDF stored on s3
----------------------------------

If your NetCDF is stored on an s3 bucket, you can add `#mode=bytes` to the end of the link and it can be used.
For example, a NetCDF stored on your personal s3 storage that is publicly viewable:

.. code-block:: bash

   docker run -it -p 8050:8050 samfooks/zarr-netcdf-viewer:latest https://minio.lab.dive.edito.eu/oidc-YOURUSERNAME/folder/mynetcdf.nc#mode=bytes