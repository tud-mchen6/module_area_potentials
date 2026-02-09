"""This script calculates the area potential based on the provided configuration."""

import _geo
import click
import geopandas as gpd
import glom
import matplotlib.pyplot as plt
import xarray as xr
import yaml
from _script_utils import plot_with_zero_separate


@click.command()
@click.argument("shapes_path", type=str)
@click.argument("resampled_path", type=str)
@click.argument("config", type=str)
@click.argument("buffer_crs", type=str)
@click.argument("output_path", type=str)
@click.argument("plot_path", type=str)
@click.argument("tech", type=str)
@click.option("--override_config", type=str)
@click.option("--min_protected_share", type=float)
@click.option("--max_total_land_use_share", type=bool, default=False)
def get_area_potential(
    shapes_path,
    resampled_path,
    config,
    buffer_crs,
    output_path,
    plot_path,
    tech,
    override_config,
    min_protected_share,
    max_total_land_use_share
):
    """Calculate the area potential based on the provided configuration.

    Args:
        shapes_path (str): Path to the input shapes in the parquet format.
        resampled_path (str): Path to the resampled input data in the NetCDF format.
        config (str): Configuration YAML string.
        buffer_crs (str): Coordinate Reference System for buffering shapes.
        output_path (str): Path to save the resulting area potential raster.
        plot_path (str): Path to save the plot of the area potential.
        override_config (str): Configuration override YAML string.

    Returns:
        None

    """
    shapes = gpd.read_parquet(shapes_path)
    ds = xr.open_dataset(resampled_path, decode_coords="all")
    # NOTE: this is a workaround for the CRS not being set correctly, ideally this
    # should not be necessary
    ds.rio.write_crs(ds.spatial_ref.attrs["crs_wkt"], inplace=True)
    config = yaml.safe_load(config)
    if override_config:
        override_config = yaml.safe_load(override_config)
        config = glom.merge([config, override_config])
        print(f"\nConfig after override: {config}\n")

    # Start with the configured pixel area as a base
    potential_da = ds[config["initial_area"]].squeeze(drop=True)  # Drop `band`

    # Apply the continuous_layers criteria to zero out additional pixels
    continuous_layers = config.get("continuous_layers", {})
    for layer, layer_config in continuous_layers.items():
        if layer in ds:
            # Apply the min-max criteria
            potential_da = potential_da.where(
                (ds[layer] <= layer_config["max"]) & (ds[layer] >= layer_config["min"]),
                other=0,
            )
            # If a share is defined, multiply the pixel area by the share
            if "share" in layer_config:
                potential_da = potential_da * layer_config["share"]
        else:
            print(f"Warning: Layer '{layer}' not found in dataset. Skipping.")


    # Add feature of min protected area as share of total territory
    # If necessary, adjust land use factors to reach the minimum protected share
    if min_protected_share > 0:
        protected_share = ds['protected'].sum() / ds['pixel_area'].sum()
        if protected_share < min_protected_share:
            # incrementally decrease the land use factor for forest, shrub, grass, farm, bare to reach the min protected share
            print(f"Protected share {protected_share.values} is less than minimum {min_protected_share}, adjusting land use factors.")
            binary_layers = config.get("binary_layers", {})
            priority = ['FOREST', 'SHRUB', 'GRASS', 'FARM', 'BARE']
            land_use_types = [k for k in binary_layers.keys() if any(n in k for n in priority)]
            land_use_types = sorted(land_use_types, key=lambda x: priority.index(next(n for n in priority if n in x)))
            increased_protected = 0
            for land_use_type in land_use_types:
                if float(binary_layers[land_use_type]) == 0:
                    continue
                type_total_area = (ds[land_use_type] * potential_da).sum()
                # If this is the 'marginal land type', decrease the land use factor just enough to reach the minimum protected share, and break the loop
                if (increased_protected + binary_layers[land_use_type] * type_total_area + ds['protected'].sum()) / ds['pixel_area'].sum() >= min_protected_share:
                    binary_layers[land_use_type] = binary_layers[land_use_type] - (min_protected_share * ds['pixel_area'].sum() - ds['protected'].sum() - increased_protected) / type_total_area
                    break
                # If this is not a 'marginal land type', decrease the land use factor to 0, add to the increased protected area, and continue to the next land use type
                else:
                    increased_protected += binary_layers[land_use_type] * type_total_area
                    binary_layers[land_use_type] = 0
                    continue


    # Zero out pixels from binary layers with share 0 from potential_da
    binary_layers = config.get("binary_layers", {})
    zero_binary_layers = [layer for layer, value in binary_layers.items() if value == 0]
    for layer in zero_binary_layers:
        if layer in ds:
            potential_da = potential_da.where(~(ds[layer] > 0), other=0)
        else:
            print(f"Warning: Layer '{layer}' not found in dataset. Skipping.")

    # Multiply pixels by their share from the binary layers
    for layer, value in binary_layers.items():
        if layer in ds:
            if value != 0:
                potential_da = xr.where(
                    ds[layer] != 0, potential_da * ds[layer] * value, potential_da
                )
        else:
            print(f"Warning: Layer '{layer}' not found in dataset. Skipping.")

    if max_total_land_use_share:
        # Add feature of max land use area for onshore wind and open field PV
        # Numbers coming from current Germany capacity, estimated with technology density
        land_max_dict = {
            'wind_onshore': 0.014, # Policy goal
            'pv_open_field': 0.005, # Current land uptake estimation
        }
        if tech in land_max_dict:
            # reduce the eligible area with the following sequence
            priority = ['FOREST', 'SHRUB', 'GRASS', 'FARM', 'BARE']
            # If the total potential area exceeds the maximum land use share, reduce the potential area by zeroing out 
            # land use types in the order of priority, until the total potential area is below the maximum land use share
            if potential_da.sum().values / ds['pixel_area'].sum().values > land_max_dict[tech]:
                land_use_types = [k for k in binary_layers.keys() if any(n in k for n in priority)]
                land_use_types = sorted(land_use_types, key=lambda x: priority.index(next(n for n in priority if n in x)))
                for land_use_type in land_use_types:
                    tot_area_land_type = (ds[land_use_type] * potential_da).sum().values
                    # If this is the 'marginal land type', decrease the land use factor just enough to reach the maximum land use share, and break the loop
                    if (potential_da.sum().values - tot_area_land_type) / ds['pixel_area'].sum().values < land_max_dict[tech]:
                        proportion = 1 + (land_max_dict[tech] * ds['pixel_area'].sum().values - potential_da.sum().values) / tot_area_land_type
                        potential_da = xr.where(ds[land_use_type] != 0, potential_da * proportion, potential_da)
                        break
                    # If this is not a 'marginal land type', zero it out, and see if the next one is
                    else:
                        potential_da = xr.where(ds[land_use_type] != 0, 0, potential_da)

    # Apply shapes-based buffering
    if "shapes_buffer" in config:
        for shape_class in config["shapes_buffer"]:
            buffer_distance = config["shapes_buffer"][shape_class]
            shapes_subset = shapes[shapes["shape_class"] == shape_class]
            if buffer_crs.lower() == "utm":
                buffer = _geo.apply_utm_buffer(
                    shapes_subset, buffer_distance_m=buffer_distance
                ).to_crs(ds.rio.crs)["geometry"]
            else:
                buffer = shapes_subset.to_crs(buffer_crs).buffer(buffer_distance)

            # Clip the potential area with the buffered shapes
            potential_da.rio.write_crs(ds.rio.crs, inplace=True)
            buffer_geo = gpd.GeoDataFrame(geometry=buffer).to_crs(ds.rio.crs)
            potential_da = potential_da.rio.clip(
                buffer_geo.geometry, buffer_geo.crs, invert=True
            )

    potential_da.name = "area_potential"
    potential_da = potential_da.transpose("band", "y", "x")
    potential_da.rio.write_crs(ds.rio.crs, inplace=True)

    fig, ax = plt.subplots(1, 1)
    ax = plot_with_zero_separate(ax=ax, da=potential_da)
    plt.savefig(plot_path, bbox_inches="tight")

    # Fill NaN with a nodata value only after plotting
    nodata_value = -1
    potential_da = potential_da.fillna(nodata_value)
    potential_da.rio.write_nodata(nodata_value, inplace=True)
    potential_da.rio.to_raster(
        output_path, driver="GTiff", compress="LZW", write_nodata=True
    )


if __name__ == "__main__":
    get_area_potential()
