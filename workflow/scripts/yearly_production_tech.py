"""
This script collects all technology areas of the given shape, use assumed technology
cost data, renewables density and discount rate to calculate the cost of each pixel,
then produce renewables cost curves and land curves to account for the accumulated
cost and land needed for a given amount of yearly vRES production.
"""

import os
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import numpy as np

def yearly_production_tech(
        density : float,
        lifetime : int,
        costs : dict,
        area_potentials_path : str,
        resampled_path : str,
        output_path : str,
):
    """_summary_

    Args:
        density (float): MW/m2
        lifetime (int): year
        costs (dict): EUR/MW
        area_potentials_path (str): _description_
        resampled_path (str): _description_
        output_path (str): _description_
    """
    
    # Get the tech name
    tech = snakemake.wildcards.tech
    # Get the area potentials
    area_potentials = rxr.open_rasterio(area_potentials_path)
    # Load the capacity factor within the resampled input
    cf_map = {
        'pv_rooftop': 'pv_cf',
        'pv_open_field': 'pv_cf',
        'wind_onshore': 'wind_cf',
        'wind_offshore': 'wind_cf'
    }
    resampled = xr.open_dataset(resampled_path)

    # Calculate the yearly aggregated production
    # Assuming same production level for each year

    # Reindex to keep the original coordinates, otherwise there will be
    # missing pixels in the result
    cf = resampled[cf_map[tech]].reindex(
        y=area_potentials.y,
        x=area_potentials.x,
        method='nearest',
        tolerance=1e-6
    )
    # area convert to km2; production unit is MWh
    yearly_prod = area_potentials * cf * density * 8760 * 1e-6
    # Since area_potentials have -1 values, get rid of them
    yearly_prod = yearly_prod.where(yearly_prod > 0, np.nan)

    # Calculate the rastered LCOE
    lcoe = (costs['CAPEX'] / (1 - (1+costs['WACC'])**(-lifetime)) * costs['WACC'] + \
            costs['OPEX']) / (cf * 8760)
    # Make the not-eligible areas also without lcoe data
    lcoe = lcoe.where(yearly_prod > 0, np.nan)

    # Save to .nc
    prod_cost = xr.Dataset({
        'area': area_potentials,
        'prod': yearly_prod,
        'lcoe': lcoe,
    })
    # add tech as a dimension, prepare for the synthesis
    prod_cost = prod_cost.rename_dims({"band": "tech"})
    prod_cost = prod_cost.assign_coords(tech=[tech])
    prod_cost = prod_cost.drop_vars('band')
    prod_cost.to_netcdf(output_path)

if __name__ == "__main__":
    yearly_production_tech(
        density=snakemake.params.density,
        lifetime=snakemake.params.lifetime,
        costs=snakemake.params.costs,
        area_potentials_path=snakemake.input.area_potentials_path,
        resampled_path=snakemake.input.resampled_path,
        output_path=snakemake.output.production_tech,
    )