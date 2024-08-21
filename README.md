# dash-nczarr-viewer


Simple Geo NetCDF and Zarr dataset viewing app

Can view a local NetCDF or an online Zarr file

Will first show all available variables, and you select one

Then the dimensions that apply to that variable, latitude and longitude must be available!

You can then select the values you need from the dimensions, to select the array you need to view

Click either Max/Min/Mean/Med/STDEV for quick statistics from your selected array

Or Click plot data to plot the data using matplot lib and cartopy.

Enjoy!


To run on local NetCDF

put your netcdf in a folder 'myfiles'

docker run -v $(pwd)/myfiles:/app/myfiles -p 8050:8050 samfooks/zarrdashapp:latest /app/myfiles/Water_body_chlorophyll-a.nc


To run with online zarr file

docker run -it -p 8050:8050 samfooks/zarrdashapp:latest https://s3.waw3-1.cloudferro.com/mdl-arco-geo-041/arco/NWSHELF_ANALYSISFORECAST_BGC_004_002/cmems_mod_nws_bgc_anfc_0.027deg-3D_P1D-m_202311/geoChunked.zarr

