"""
This script combines all technologies (or the defined technologies) and prepare the data needed for making the 
cost curve (cumulated cost against yearly aggregated electricity production) and the land curve (land used 
against yearly aggregated electricity production).
"""

import xarray as xr


def land_cost_curves(
        inputs,
        output_path,
        single_tech : str = None,
):
    """_summary_

    Args:
        inputs:
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
    breakpoint()
    

    




if __name__ == "__main__":
    land_cost_curves(
        inputs=snakemake.input.inputs,
        output_path=snakemake.output.curve_data,
        single_tech=snakemake.params.single_tech,
    )