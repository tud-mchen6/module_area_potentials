"""
Blabla
"""

import xarray as xr
import numpy as np
from internal.helper_functions import *


def plot_supply_and_land_curves(
        synth_ds,
        supply_curve_path,
        land_curve_path,
):
    """_summary_

    Args:
    synth_ds:
    supply_curve_path:
    land_curve_path:
        
    """

    
    breakpoint()
    synth_ds = xr.open_dataset("your_dataset.nc")
    prep = prepare_global_order(synth_ds)
    
    # 1) Supply curve as line
    plot_supply_curve_line(prep)
    
    # 3) Land-use curve (line) with overlap handling
    x_pts, y_pts, x_lab = build_land_use_curve_points(synth_ds, prep, area_var=AREA_VAR, overlap_var=OVERLAP_VAR, output_path=supply_curve_path)
    plot_land_use_curve_line(x_pts, y_pts, x_lab, area_label="Cumulative land (e.g., km²)", output_path=land_curve_path)
    
    
    return






if __name__ == "__main__":
    plot_supply_and_land_curves(
        synth_ds=snakemake.input.synth_ds,
        supply_curve_path=snakemake.output.supply_curve_path,
        land_curve_path=snakemake.output.land_curve_path,
    )