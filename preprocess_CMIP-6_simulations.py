import xarray as xr
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import gc
import time



def create_regional_averages_cmip6(
    simulation_directory,
    simulation_file,
    TARGET_DIRECTORY,
    masks,
    indicator,
):

    simulation_path = f"{simulation_directory}/{simulation_file}"

    MODEL = simulation_file.split("_")[2]
    scenario = simulation_file.split("_")[3]
    ensemble = simulation_file.split("_")[4]
    SCENARIO = f"{scenario}-{ensemble}"
    INDICATOR = simulation_file.split("_")[0]

    simulation = xr.open_dataset(simulation_path).load()
    #simulation["lon"] = xr.where(
    #    simulation["lon"] > 180, simulation["lon"] - 360, simulation["lon"]
    #)
    #simulation = simulation.sortby("lon")

    masks = masks.broadcast_like(simulation.isel(time=0))

    masks = xr.where(masks == 0, np.nan, masks)

    averages = masks * simulation[indicator]
    averages = averages.sum(dim=["lat", "lon"]) / masks.sum(dim=["lat", "lon"])
    
    out_df =         {
                "time": averages["time"].values}
    
    for region in list(averages.keys()):

        out_df[region.replace("m_", "")] = averages[region].values
    
    out_df = pd.DataFrame(out_df)
  
    
    directory = f'{TARGET_DIRECTORY}/{MODEL}'
    filename = f'{MODEL}_{SCENARIO}_{INDICATOR}_IPCC-REGIONS_latweight.csv'.lower()
    print(filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    out_df.to_csv(f"{directory}/{filename}", index=False, sep=",")

    simulation.close()
    del simulation
    del averages
    gc.collect()
    time.sleep(4)



if __name__ == "__main__":

    simulation_directories = {
        "tas": "/mnt/CMIP6_storage/cmip6-ng/tas/mon/g025",
        "hurs": "/mnt/CMIP6_storage/cmip6-ng/hurs/hurs/mon/g025",
        "tasmax": "/mnt/CMIP6_storage/cmip6-ng/tasmax/mon/g025",
        "tasmin": "/mnt/CMIP6_storage/cmip6-ng/tasmin/mon/g025",
        "pr": "/mnt/CMIP6_storage/cmip6-ng/pr/mon/g025",
        "mrso": "/mnt/CMIP6_storage/cmip6-ng/mrso/mrso/mon/g025",
        "rsds": "/mnt/CMIP6_storage/cmip6-ng/rsds/rsds/mon/g025",
    }

    for indicator in ["tas", "pr", "tasmax", "tasmin", "rsds", "mrso", "hurs"]:

        TARGET_DIRECTORY = "/mnt/PROVIDE/SCALES/cmip6-ng-inc-oceans"
        MESMER_MASKS = "/mnt/PROVIDE/SCALES/AR6_LAND_OCEAN_REGIONS_ABBREVIATION.nc"

        masks = xr.open_dataset(MESMER_MASKS)
        
        simulation_directory = simulation_directories[indicator]

        lat_da = xr.DataArray(masks.lat.values, dims="lat")
        cos_lat = np.cos(np.deg2rad(lat_da))
        cos_lat = xr.DataArray(cos_lat, coords=[lat_da], dims=["lat"]).broadcast_like(
            masks
        )
        
        masks = masks * cos_lat
        masks["GLOBAL"] = cos_lat

       
        all_files = os.listdir(simulation_directory)

        for file in tqdm(all_files):

            create_regional_averages_cmip6(
                simulation_directory, file, TARGET_DIRECTORY, masks, indicator
            )

           