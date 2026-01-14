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
):
    """Produce the dataset that allows the cost curve and land curve 
    based on the ranking of vRES production LCOE of pixels.

    Args:
        inputs: list of str - paths to datasets of yearly production of all technologies
        resampled_input: str - path to the resampled input dataset
        land_share_type: int - the integer of land sharing mode between onshore wind and 
            open field PV.
        output_path: str - path to the output NetCDF file
    """

    
    # wind offshore will not have land issues, therefore will only appear in
    # cost curves


    # # assigned land share type between technologies. If valued -1, then one
    # # pixel can only be used for exclusively one technology.
    # if not land_share_type:
    #     land_share_type = -1
    # # read data from all technology Datasets
    # datasets = []
    # for item in inputs:
    #     # rooftop PV is not part of the game, will not appear anywhere
    #     if 'rooftop' in item:
    #         continue
    #     ds = xr.open_dataset(item)
    #     datasets.append(ds)
    # combined = xr.concat(datasets, dim="tech")
    # # get the area per pixel to prepare for the land overlap calculation
    # resampled = xr.open_dataset(resampled_input)
    # pixel_area = resampled.pixel_area
    # ds_combined = combined.assign(pixel_area=pixel_area)
    # ds_land_processed = allocate_with_sharing_old(ds_combined, land_share_type)
    # ds_land_processed.to_netcdf(output_path)







    # assigned land share type between technologies. If valued -1, then one
    # pixel can only be used for exclusively one technology.
    if not land_share_type:
        land_share_type = -1

    # Filter out rooftop PV as it doesn't have land use issues
    # and is not used as power plants
    tech_files = [p for p in inputs if "rooftop" not in p]

    # Open all files lazily, concatenate along 'tech'
    # Used to increase speed and reduce memory usage
    combined = xr.open_mfdataset(
        tech_files,
        chunks={"y": 512, "x": 512}, # heuristic chunk size
        combine="nested",
        concat_dim="tech",
        parallel=True
    )

    # Add pixel_area to prepare for land area sharing calculation
    resampled = xr.open_dataset(resampled_input, chunks={"y": 512, "x": 512})
    ds_combined = combined.assign(pixel_area=resampled["pixel_area"].chunk({"y": 512, "x": 512}))

    ds_land_processed = allocate_with_sharing(ds_combined, land_share_type)

    # Choose output chunking
    ds_to_write = ds_land_processed.chunk({'tech': -1, 'x': 512, 'y': 512})

    # TEMPORARY TO TEST SPEED
    ds_to_write = ds_to_write.astype({v: 'float32' for v in ds_to_write.data_vars})

    # encoding = {
    #     'lcoe': {'zlib': True, 'complevel': 1},
    #     'prod': {'zlib': True, 'complevel': 1},
    #     'area': {'zlib': True, 'complevel': 1},
    # }

    print("is dask-backed:", any(hasattr(ds_to_write[v].data, "chunks") for v in ds_to_write.data_vars))
    print("approx GB:", ds_to_write.nbytes / 1e9)
    print("dask chunks:", ds_to_write.chunks)  # if dask-backed
    # breakpoint()
    ds_materialised = ds_to_write.compute()

    # ds_materialised.to_netcdf(output_path, encoding=encoding, engine='netcdf4')

    # Flatten the dataset and save to output
    prep = prepare_global_order(ds_materialised)

    # x_points, y_points, x_label = build_land_use_curve_points(
    #     ds_materialised,
    #     prep)
    np.savez_compressed(output_path, **prep,
                        # lcoe=prep['lcoe'],
                        # prod=prep['prod'],
                        # tech=prep['tech'],
                        # area=prep['area'],
                        # overlap=prep['overlap'],
                        )




if __name__ == "__main__":
    land_share_calculation(
        inputs=snakemake.input.inputs,
        resampled_input=snakemake.input.resampled_input,
        land_share_type=snakemake.params.land_share_type,
        output_path=snakemake.output.prep,
    )