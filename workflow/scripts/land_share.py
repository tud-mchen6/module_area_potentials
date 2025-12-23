"""
This script combines all technologies (or the defined technologies) and prepare the data needed for making the 
cost curve (cumulated cost against yearly aggregated electricity production) and the land curve (land used 
against yearly aggregated electricity production).
"""

import xarray as xr
import numpy as np
from internal.helper_functions import *


def land_share_calculation(
        inputs,
        resampled_input,
        land_share_type,
        output_path,
        single_tech,
):
    """_summary_

    Args:
        inputs:
        resampled_input:
        land_share_type:
        output_path:
        single_tech:
    """

    
    # wind offshore will not have land issues, therefore will only appear in
    # cost curves
    
    # read data from all technology Datasets
    datasets = []
    for item in inputs:
        # rooftop PV is not part of the game, will not appear anywhere
        if 'rooftop' in item:
            continue
        ds = xr.open_dataset(item)
        datasets.append(ds)
    combined = xr.concat(datasets, dim="tech")
    # get the area per pixel to prepare for the land overlap calculation
    resampled = xr.open_dataset(resampled_input)
    pixel_area = resampled.pixel_area
    ds_combined = combined.assign(pixel_area=pixel_area)
    
    # assigned land share type between technologies. If valued -1, then one
    # pixel can only be used for exclusively one technology.
    if not land_share_type:
        land_share_type = -1

    ds_land_processed = allocate_with_sharing(ds_combined, land_share_type)
    
    if len(single_tech) > 0:
        # if only require the curves of one specific technology (TODO: can be modified into several specified technologies)
        # keep the tech dimension in the Dataset
        ds_land_processed = ds_land_processed.where(ds_land_processed.tech==single_tech, drop=True)

    ds_land_processed.to_netcdf(output_path)






if __name__ == "__main__":
    land_share_calculation(
        inputs=snakemake.input.inputs,
        resampled_input=snakemake.input.resampled_input,
        land_share_type=snakemake.params.land_share_type,
        output_path=snakemake.output.curve_data,
        single_tech=snakemake.params.single_tech,
    )