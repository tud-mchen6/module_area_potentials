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

    
    






if __name__ == "__main__":
    plot_supply_and_land_curves(
        synth_ds=snakemake.input.synth_ds,
        supply_curve_path=snakemake.output.supply_curve_path,
        land_curve_path=snakemake.output.land_curve_path,
    )