import numpy as np
import xarray as xr
import regionmask

PATH = "/mnt/PROVIDE/SCALES/AR6_LAND_OCEAN_REGIONS_ABBREVIATION.nc"

def generate_ar6_region_masks(lat, lon):
    """
    Generate an xarray.Dataset with separate masks for each AR6 region.
    """
    
    ar6 = regionmask.defined_regions.ar6.all
    full_mask = ar6.mask(lon, lat)

    ds = xr.Dataset(coords={'lat': lat, 'lon': lon})

    for region in ar6:
        region_mask = (full_mask == region.number).astype(int)
        ds[region.abbrev.replace(" ", "-").replace("/", "-")] = xr.DataArray(
            region_mask, coords={'lat': lat, 'lon': lon}, dims=["lat", "lon"]
        )

    return ds


lats = np.arange(-88.75, 90.5, 2.5)
lons = np.arange(1.25, 358.8, 2.5)
mask = generate_ar6_region_masks(lats, lons)
mask.to_netcdf(PATH)
