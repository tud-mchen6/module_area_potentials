"""
This script combines all technologies (or the defined technologies) and prepare the data needed for making the 
cost curve (cumulated cost against yearly aggregated electricity production) and the land curve (land used 
against yearly aggregated electricity production).
"""

import xarray as xr
import numpy as np
from internal.helper_functions import *


def land_cost_curves(
        inputs,
        resampled_input,
        land_share_type,
        output_path,
        single_tech : str = None,
):
    """_summary_

    Args:
        inputs:
        resampled_input:
        output_path:
        single_tech:
    """

    
    # wind offshore will not have land issues, therefore will only appear in
    # cost curves
    
    datasets = []
    for item in inputs:
        # rooftop PV is not part of the game, will not appear anywhere
        if 'rooftop' in item:
            continue
        ds = xr.open_dataset(item)
        datasets.append(ds)
    combined = xr.concat(datasets, dim="tech")
    resampled = xr.open_dataset(resampled_input)
    pixel_area = resampled.pixel_area
    ds = combined.assign(pixel_area=pixel_area)
    breakpoint()

    if not land_share_type:
        land_share_type = -1

    ds_land_processed = allocate_with_sharing(ds, land_share_type)
    # TODO: check if this function actually does what I want it to do


    




if __name__ == "__main__":
    land_cost_curves(
        inputs=snakemake.input.inputs,
        resampled_input=snakemake.input.resampled_input,
        land_share_type=snakemake.params.land_share_type,
        output_path=snakemake.output.curve_data,
        single_tech=snakemake.params.single_tech,
    )