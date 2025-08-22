#!/usr/bin/env python3
"""
Direct test of CMEMS functionality using copernicus toolbox
"""


def test_cmems_direct():
    """Test CMEMS dataset opening directly"""

    # Test CMEMS URL
    cmems_url = "https://s3.waw3-1.cloudferro.com/mdl-arco-geo-041/arco/NWSHELF_ANALYSISFORECAST_BGC_004_002/cmems_mod_nws_bgc_anfc_0.027deg-3D_P1D-m_202311/geoChunked.zarr"

    print(f"Testing CMEMS URL: {cmems_url}")
    print("-" * 80)

    # Test 1: Direct import
    try:
        import copernicusmarine
        print("✓ Copernicus toolbox import successful")
        print(
            f"Available functions: {[f for f in dir(copernicusmarine) if not f.startswith('_')]}")
    except ImportError as e:
        print(f"✗ Copernicus toolbox import failed: {e}")
        return

    # Test 2: Try different copernicusmarine approaches
    print("\nTesting different copernicusmarine approaches...")

    # Approach 1: Try with xarray directly
    try:
        import xarray as xr
        print("\nApproach 1: Using xarray.open_zarr directly...")
        ds = xr.open_zarr(cmems_url)
        print("✓ Successfully opened with xarray.open_zarr directly!")
        print(f"Dataset info: {ds}")

        # Try to get basic info
        if hasattr(ds, 'data_vars'):
            print(f"✓ Data variables: {list(ds.data_vars.keys())}")
        if hasattr(ds, 'dims'):
            print(f"✓ Dimensions: {dict(ds.dims)}")
        return

    except Exception as e:
        print(f"✗ Approach 1 failed: {e}")

    # Approach 2: Try copernicusmarine.open_dataset
    try:
        print("\nApproach 2: Using copernicusmarine.open_dataset...")
        ds = copernicusmarine.open_dataset(cmems_url)
        print("✓ Successfully opened with copernicusmarine.open_dataset!")
        print(f"Dataset info: {ds}")

        # Try to get basic info
        if hasattr(ds, 'data_vars'):
            print(f"✓ Data variables: {list(ds.data_vars.keys())}")
        if hasattr(ds, 'dims'):
            print(f"✓ Dimensions: {dict(ds.dims)}")
        return

    except Exception as e:
        print(f"✗ Approach 2 failed: {e}")

    # Approach 3: Try with authentication
    try:
        print("\nApproach 3: Using copernicusmarine.open_dataset with username...")
        ds = copernicusmarine.open_dataset(
            cmems_url,
            username='sfooks'
        )
        print("✓ Successfully opened with copernicusmarine.open_dataset (with username)!")
        print(f"Dataset info: {ds}")

        # Try to get basic info
        if hasattr(ds, 'data_vars'):
            print(f"✓ Data variables: {list(ds.data_vars.keys())}")
        if hasattr(ds, 'dims'):
            print(f"✓ Dimensions: {dict(ds.dims)}")
        return

    except Exception as e:
        print(f"✗ Approach 3 failed: {e}")

    # Approach 4: Try custom_open_zarr
    try:
        print("\nApproach 4: Using custom_open_zarr...")
        from copernicusmarine.core_functions import custom_open_zarr
        store = custom_open_zarr.open_zarr(cmems_url)
        print(f"✓ Got store object: {type(store)}")

        # Try to use the store with xarray
        ds = xr.open_zarr(store)
        print("✓ Successfully opened with xarray using store!")
        print(f"Dataset info: {ds}")

        # Try to get basic info
        if hasattr(ds, 'data_vars'):
            print(f"✓ Data variables: {list(ds.data_vars.keys())}")
        if hasattr(ds, 'dims'):
            print(f"✓ Dimensions: {dict(ds.dims)}")
        return

    except Exception as e:
        print(f"✗ Approach 4 failed: {e}")

    print("\nAll approaches failed. The CMEMS dataset may require authentication or be inaccessible.")


if __name__ == "__main__":
    test_cmems_direct()
