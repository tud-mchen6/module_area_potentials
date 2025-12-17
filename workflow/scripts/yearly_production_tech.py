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
        density (float): _description_
        lifetime (int): 
        costs (dict): _description_
        area_potentials_path (_type_): _description_
        resampled_path (_type_): _description_
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
    yearly_prod = area_potentials * density * resampled[cf_map[tech]] * 8760

    # Calculate the rastered LCOE
    lcoe = costs['CAPEX'] / yearly_prod * (1 - (1+costs['WACC'])^(-lifetime)) * costs['WACC'] + \
            costs['OPEX'] / yearly_prod

    # Save to .nc
    prod_cost = xr.Dataset({
        'prod': yearly_prod,
        'lcoe': lcoe,
    })
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