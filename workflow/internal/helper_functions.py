
import xarray as xr
import numpy as np


# Author: Copilot
def allocate_with_sharing(ds,
                          share,
                          area_var='area',
                          prod_var='prod',
                          lcoe_var='lcoe',
                          pixel_area_var='pixel_area'):
    """
    Adjusts the area/prod of the second-cheapest technology per pixel according to a 'share' factor
    and adds an (y,x) 'overlap' variable for the shared area. Fully vectorized (no loops).
    
    share semantics:
      -1 : only cheapest tech allowed (second tech area -> 0)
       0 : no sharing; curb second tech so area_low + area_second <= pixel_area
     0..1: partial/full sharing; allows s * min(area_low, area_second) to overlap
    """

    # --- Prepare LCOE ranking (ignore NaNs via +inf) ---
    lcoe = ds[lcoe_var]
    lcoe_filled = lcoe.where(np.isfinite(lcoe), np.inf)

    # Cheapest (index & value)
    idx_low = lcoe_filled.argmin('tech')         # (y,x)
    min_low = lcoe_filled.min('tech')            # (y,x)
    valid_low = np.isfinite(min_low)

    # Mask out cheapest to find second-cheapest
    tech_index = xr.DataArray(np.arange(ds.dims['tech']),
                              coords={'tech': ds['tech']}, dims=['tech'])  # (tech)
    mask_low = (tech_index == idx_low)                                   # (tech,y,x)
    lcoe_masked = lcoe_filled.where(~mask_low, np.inf)

    idx_second = lcoe_masked.argmin('tech')       # (y,x)
    min_second = lcoe_masked.min('tech')          # (y,x)
    valid_second = np.isfinite(min_second)

    # --- Gather areas/prod for cheapest & second-cheapest ---
    area_low = ds[area_var].isel(tech=idx_low)        # (y,x)
    area_second = ds[area_var].isel(tech=idx_second)  # (y,x)
    prod_second = ds[prod_var].isel(tech=idx_second)  # (y,x)
    pixel_area = ds[pixel_area_var]                   # (y,x)

    # valid pair: both lcoe valid and both areas finite
    valid_area_pair = (valid_low & valid_second &
                       np.isfinite(area_low) & np.isfinite(area_second))

    # --- Sharing logic ---
    s = np.clip(share, 0.0, 1.0)   # effective sharing (negative treated separately)
    only_cheapest = (share < 0)

    # Case A cap (a2' <= area_low)
    # For s < 1: capA = (pixel_area - area_low) / (1 - s)
    # For s = 1: feasible capA is area_low if area_low <= pixel_area, else 0
    denom = (1.0 - s)
    capA = xr.where(denom > 0,
                    (pixel_area - area_low) / denom,
                    area_low)
    # Bound to [0, area_low]
    capA = capA.clip(min=0).where(np.isfinite(capA), 0.0)
    capA = xr.where(capA <= area_low, capA, area_low)

    # Case B cap (a2' >= area_low): capB = pixel_area - (1 - s) * area_low
    capB = (pixel_area - (1.0 - s) * area_low).clip(min=0)

    # Unified cap: if capB >= area_low, we can be in case B; else, use case A
    cap = xr.where(capB >= area_low, capB, capA)

    # Apply "only cheapest" rule
    cap = xr.where(only_cheapest, 0.0, cap)

    # New area for second-cheapest: cap downwards; leave unchanged where pair invalid
    new_area_second = xr.where(
        valid_area_pair,
        xr.apply_ufunc(np.minimum, area_second, cap),
        area_second
    )


    # Overlap area (y,x): s * min(area_low, new_area_second); zero if invalid pair or negative share
    s_eff = 0.0 if only_cheapest else s
    overlap = xr.where(valid_area_pair, s_eff * xr.apply_ufunc(np.minimum, area_low, new_area_second),
                       0.0)

    # Scale production proportionally for second-cheapest
    scale = xr.where(valid_area_pair,
                     xr.where(area_second > 0, new_area_second / area_second, 0.0),
                     1.0)
    updated_prod_second = prod_second * scale

    # --- Write back to ds for second-cheapest positions ---
    mask_second = (tech_index == idx_second)  # (tech,y,x)
    area_updated = xr.where(mask_second, new_area_second, ds[area_var])
    prod_updated = xr.where(mask_second, updated_prod_second, ds[prod_var])

    # --- Build output dataset with 'overlap' (y,x) ---
    ds_out = ds.copy()
    ds_out[area_var] = area_updated
    ds_out[prod_var] = prod_updated
    ds_out['overlap'] = overlap  # dims (y,x)

    return ds_out
