"""
Blabla
"""

import xarray as xr
import rioxarray as rxr
import numpy as np
from internal.helper_functions import *


def plot_supply_and_land_curves(
        synth_ds,
        single_tech,
        supply_curve_path,
        land_curve_path,
):
    """_summary_

    Args:
    synth_ds:
    single_tech:
    supply_curve_path:
    land_curve_path:
        
    """
    
    ds = xr.open_dataset(synth_ds[0])

    if len(single_tech) > 0:
        # if only require the curves of one specific technology (TODO: can be modified into several specified technologies)
        # keep the tech dimension in the Dataset
        ds = ds.where(ds.tech==single_tech, drop=True)

    # TODO: Take offshore wind out since it has no land use 
    prep = prepare_global_order(ds)
    

    # 1) Supply curve
    plot_supply_curve_bars(
        prep,
        output_path=supply_curve_path,
        bin_count=1000,
        rasterized=False,
    )
    
    # 2) Land-use curve (line) with overlap handling

    x_pts, y_pts, x_lab = build_land_use_curve_points(ds, prep, area_var=AREA_VAR, overlap_var=OVERLAP_VAR)
    plot_land_use_curve_line(x_pts, y_pts, x_lab, area_label="Cumulative land (km²)", output_path=land_curve_path)
    
    
    return






if __name__ == "__main__":
    plot_supply_and_land_curves(
        synth_ds=snakemake.input.synth_ds,
        single_tech=snakemake.params.single_tech,
        supply_curve_path=snakemake.output.supply_curve_path,
        land_curve_path=snakemake.output.land_curve_path,
    )