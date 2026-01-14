
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


# Original author: Copilot
def allocate_with_sharing_old(ds,
                          share,
                          area_var='area',
                          prod_var='prod',
                          lcoe_var='lcoe',
                          pixel_area_var='pixel_area'):
    """
    Adjusts the area/prod of the second-cheapest technology per pixel according to a 'share' factor
    and adds an (y,x) 'overlap' variable for the shared area.
    
    share semantics:
      -1 : only cheapest tech allowed (second tech area -> 0)
       0 : no sharing; curb second tech so area_low + area_second <= pixel_area
     0..1: partial/full sharing; allows s * min(area_low, area_second) to overlap
    """

    # --- Prepare LCOE ranking (ignore NaNs via +inf) ---
    lcoe = ds[lcoe_var]                          # (tech,y,x)
    lcoe_filled = lcoe.where(np.isfinite(lcoe), np.inf)

    # Select the tech that has the lowest lcoe in each pixel
    # Get the index of the cheapest tech
    idx_low = lcoe_filled.argmin('tech')         # (y,x)
    # Get the lowest lcoe values
    min_low = lcoe_filled.min('tech')            # (y,x)
    # Get the mask of the non-nan pixels for the cheapest tech in each pixel
    # Meaning, if not valid, then this pixel does not have potential for any tech
    valid_low = np.isfinite(min_low)

    # Mask out cheapest to find second-cheapest
    # tech_index to map tech names to integers
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
    # important since not every pixel has both techs
    valid_area_pair = (valid_low & valid_second &
                       np.isfinite(area_low) & np.isfinite(area_second))

    # --- Sharing logic ---
    s = np.clip(share, 0.0, 1.0)   # effective sharing (negative treated separately)
    only_cheapest = (share < 0)

    # a2' is the new area for second-cheapest after applying sharing rules
    # cap is the largest value of a2' allowed

    # Case A: a2' <= area_low
    # For s < 1: capA = (pixel_area - area_low) / (1 - s)
    # For s = 1: feasible capA is area_low
    denom = (1.0 - s)
    capA = xr.where(denom > 0,
                    (pixel_area - area_low) / denom,
                    area_low)
    # Bound to [0, area_low]
    capA = capA.clip(min=0).where(np.isfinite(capA), 0.0)
    capA = xr.where(capA <= area_low, capA, area_low)

    # Case B: a2' >= area_low
    # capB = pixel_area - (1 - s) * area_low
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






import numpy as np
import xarray as xr

def allocate_with_sharing_vectorized(ds: xr.Dataset, share: float) -> xr.Dataset:
    """
    Vectorized allocation with sharing (Dask-lazy).
    - If fewer than two techs are present in a pixel, leave it unchanged.
    - share < 0: only cheapest tech survives (others present -> 0).
    - 0 <= share <= 1: allow overlap = share * min(area_lowest, area_second),
      and curb the second-cheapest tech accordingly.
    """
    lcoe, prod, area, pixel_area = ds['lcoe'], ds['prod'], ds['area'], ds['pixel_area']

    # Present mask and count per pixel
    present = np.isfinite(prod)
    two_present = present.sum('tech') >= 2

    # Effective LCOE for ranking (absent or invalid -> +inf)
    lcoe_eff = xr.where(present, lcoe, np.inf)

    # Cheapest index
    k1 = lcoe_eff.argmin('tech')

    # Mask out cheapest to get second-cheapest
    tech_pos = xr.DataArray(np.arange(ds.sizes['tech']), dims='tech', coords={'tech': ds['tech']})
    lcoe_eff2 = xr.where(tech_pos == k1, np.inf, lcoe_eff)
    k2 = lcoe_eff2.argmin('tech')

    # Build masks for selective updates
    mask_k1 = (tech_pos == k1) & two_present
    mask_k2 = (tech_pos == k2) & two_present

    # Gather values for cheapest and second-cheapest (vectorized indexing)
    area1 = xr.where(mask_k1, area, 0).max('tech')
    area2 = xr.where(mask_k2, area, 0).max('tech')
    prod2 = xr.where(mask_k2, prod, 0).max('tech')

    if share < 0:
        # Winner-takes-all: zero other present techs, keep NaNs for absent, leave single-tech pixels unchanged
        prod_updated = xr.where(two_present & ~mask_k1, 0, prod)
        area_updated = xr.where(two_present & ~mask_k1, 0, area)
        overlap = xr.zeros_like(pixel_area)
    else:
        # Sharing branch (0 <= share <= 1)
        s = float(np.clip(share, 0.0, 1.0))
        denom = 1.0 - s
        # capA: when a2' < area1 (only relevant if denom > 0)
        capA = ((pixel_area - area1) / denom) if denom > 0 else area1
        capA = capA.clip(0.0, area1)
        # capB: when a2' >= area1
        capB = xr.apply_ufunc(np.maximum, area1, pixel_area - (1.0 - s) * area1)
        # Choose cap (capB is always >= area1 by construction)
        cap = xr.where(capB >= area1, capB, capA)

        new_area2 = xr.apply_ufunc(np.minimum, area2, cap)
        overlap = s * xr.apply_ufunc(np.minimum, area1, new_area2)

        # Scale production of second-cheapest tech
        scale = xr.where(area2 > 0, new_area2 / area2, 0.0)
        updated_prod2 = prod2 * scale

        # Update only the second-cheapest tech where two techs are present
        prod_updated = xr.where(mask_k2, updated_prod2, prod)
        area_updated = xr.where(mask_k2, new_area2, area)

        # No overlap where fewer than two are present
        overlap = xr.where(two_present, overlap, 0)

    return ds.assign(prod=prod_updated, area=area_updated, overlap=overlap)





def policy_one_pixel(share, lcoe_1d, prod_1d, area_1d, pixel_area):
    """
    Inputs are 1-D arrays for a single pixel, each length = number of techs.
    lcoe_1d: float array (NaN allowed)
    prod_1d: float array (NaN indicates not present), total yearly production of the tech
    area_1d: float array (NaN indicates not present), estimated used area of the tech
    pixel_1d: float array (NaN indicates not present), total area of the given pixel

    Return two 1-D arrays: updated prod and area (same length).
    """

    # Situation 1: less than two techs are present in one pixel
    present = np.isfinite(prod_1d)
    if present.sum() <= 1:
        # Single or none present: leave unchanged
        overlap = 0
        return prod_1d, area_1d, overlap

    # Situation 2: two techs are present in one pixel, need to 
    # decide for the output
    # Situation 2.1: no sharing. Zero-out non-winners and keep the lowest LCOE
    if share < 0:
        # set NaN to inf to avoid issues with argmin
        lcoe_assess = np.where(np.isnan(lcoe_1d), np.inf, lcoe_1d)
        k = np.argmin(lcoe_assess)
        prod_out = np.zeros_like(prod_1d)
        area_out = np.zeros_like(area_1d)
        prod_out[k] = prod_1d[k]
        area_out[k] = area_1d[k]
        overlap = 0
        return prod_out, area_out, overlap
    # Situation 2.2: sharing allowed, 0 <= share <= 1
    else:
        idx_sorted = np.argsort(lcoe_1d)
        k1 = idx_sorted[0]  # index of lowest LCOE
        k2 = idx_sorted[1]  # index of second-lowest LCOE
        area1 = area_1d[k1]
        area2 = area_1d[k2]
        prod2 = prod_1d[k2]

        s = np.clip(share, 0.0, 1.0)

        # Compute cap for area2
        # Case A: a2' < area1
        denom = (1.0 - s)
        if denom > 0:
            capA = (pixel_area - area1) / denom
        else:
            capA = area1
        capA = max(0.0, min(capA, area1))
        # Case B: a2' >= area1
        capB = max(area1, pixel_area - (1.0 - s) * area1)

        # Determine cap
        cap = capB if capB >= area1 else capA
        # Assign new area2
        new_area2 = min(area2, cap)
        # Compute overlap area between two techs
        overlap = s * min(area1, new_area2)

        if area2 > 0:
            scale = new_area2 / area2
        else:
            scale = 0.0
        updated_prod2 = prod2 * scale

        # Build output arrays
        prod_out = prod_1d.copy()
        area_out = area_1d.copy()
        prod_out[k2] = updated_prod2
        area_out[k2] = new_area2

        return prod_out, area_out, overlap



def allocate_with_sharing(ds: xr.Dataset,
                          share: float,) -> xr.Dataset:
    """
    Adjusts the area/prod of the second-cheapest technology per pixel according to a 'share' factor
    and adds an (y,x) 'overlap' variable for the shared area.
    share semantics:
      -1 : only cheapest tech allowed (second tech area -> 0)
       0 : no sharing; curb second tech so area_low + area_second <= pixel_area
        0..1: partial/full sharing; allows s * min(area_low, area_second) to overlap
    """
    
    ds = ds.chunk({'tech': -1})

    prod_updated, area_updated, overlap = xr.apply_ufunc(
        policy_one_pixel,
        share, ds['lcoe'], ds['prod'], ds['area'], ds['pixel_area'],
        input_core_dims=[[],['tech'], ['tech'], ['tech'], []],  # we operate along tech per pixel
        output_core_dims=[['tech'], ['tech'], []],           # return arrays along tech
        vectorize=True,                                  # broadcast across (x,y)
        dask='parallelized',
        output_dtypes=[ds['prod'].dtype, ds['area'].dtype, ds['area'].dtype],
        output_sizes={'tech': ds.sizes['tech']},
    )

    ds_out = ds.assign(prod=prod_updated, area=area_updated, overlap=overlap)

    return ds_out


# Author: Copilot


LCOE_VAR = "lcoe"
PROD_VAR = "prod"
AREA_VAR = "area"
OVERLAP_VAR = "overlap"

TECH_DIM = "tech"          # name of technology dimension if present
TECH_COORD = "tech"        # name of technology coordinate (string labels)

# Unit conversion for x-axis (production)
PROD_TO_TWH = True
PROD_DIVISOR = 1e6  # MWh -> TWh

# Technology colors
TECH_COLORS = {
    "pv_open_field": "#FFD700",          # yellow
    "wind_onshore": "#7EC8E3",   # light blue
    "wind_offshore": "#1F4E79",  # dark blue
}
FALLBACK_COLORS = ["#999999", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


# Maybe to trim down
def _detect_tech(ds: xr.Dataset, tech_dim: str, tech_coord: str) -> Tuple[bool, Optional[List[str]]]:
    """Detects if dataset has a tech dimension+coord. Returns (has_tech, tech_names_or_None)."""
    has_dim = tech_dim in ds.dims
    has_coord = tech_coord in ds.coords
    if has_dim and has_coord:
        tech_names = [str(v) for v in ds.coords[tech_coord].values]
        return True, tech_names
    return False, None

# TODO: really verbose, try to trim down
def _flatten_for_sort(
    ds: xr.Dataset,
    lcoe_var: str,
    prod_var: str,
    area_var: str,
    overlap_var: str,
    tech_names: Optional[List[str]]
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Flattens lcoe and prod to 1D arrays aligned for global sorting.
    If multi-tech, tech axis is moved to last, then raveled.
    Returns: (lcoe_flat, prod_flat, tech_id_or_None)
    """
    lvar = ds[lcoe_var]
    pvar = ds[prod_var]
    avar = ds[area_var]
    ovar = ds[overlap_var]

    if tech_names is None:
        lcoe_flat = lvar.values.ravel()
        prod_flat = pvar.values.ravel()
        area_flat = avar.values.ravel()
        overlap_flat = ovar.values.ravel()
        tech_id = None
    else:
        dims = list(lvar.dims)
        # move tech to last axis for predictable flatten order (Y, X, ..., T)
        transpose_order = [d for d in dims if d != TECH_DIM] + [TECH_DIM]
        larr = lvar.transpose(*transpose_order).values
        parr = pvar.transpose(*transpose_order).values
        aarr = avar.transpose(*transpose_order).values
        T = len(tech_names)
        non_tech_size = larr.size // T

        lcoe_flat = larr.reshape(-1)
        prod_flat = parr.reshape(-1)
        area_flat = aarr.reshape(-1)
        overlap_flat = ovar.values.ravel()
        # For each (y,x,...) position, we have T consecutive tech entries
        tech_id = np.tile(np.arange(T, dtype=np.uint16), non_tech_size)

    return lcoe_flat, prod_flat, area_flat, overlap_flat, tech_id

# Maybe to trim down
def _flatten_aux_var(
    ds: xr.Dataset,
    var_name: str,
    mask: np.ndarray,
    tech_names: Optional[List[str]]
) -> np.ndarray:
    """
    Flattens an auxiliary var (e.g., area/overlap) to align with the mask built from lcoe/prod.
    - If the var includes the tech dim, we ravel like lcoe/prod.
    - If the var lacks the tech dim (typical for area/overlap), we repeat each (y,x,...) value T times
      to match the (y,x,tech) flattened shape used for sorting.
    Returns the masked flattened array (pre-sorting).
    """
    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' not found in dataset.")
    v = ds[var_name].astype(np.float64)  # keep higher precision for area math

    if tech_names is None:
        flat = v.values.ravel()
        return flat[mask]

    # multi-tech case
    dims = list(v.dims)
    if TECH_DIM in dims:
        transpose_order = [d for d in dims if d != TECH_DIM] + [TECH_DIM]
        arr = v.transpose(*transpose_order).values
        flat = arr.reshape(-1)
        return flat[mask]
    else:
        # no tech dim: repeat each element T times to align with (y,x,tech) raveling
        transpose_order = dims  # tech not present; just ravel in current order
        arr = v.transpose(*transpose_order).values
        base_flat = arr.reshape(-1)
        T = len(tech_names)
        # Repeat each (y,x,...) element consecutively T times
        expanded = np.repeat(base_flat, T)
        return expanded[mask]


def prepare_global_order(
    ds: xr.Dataset,
    lcoe_var: str = LCOE_VAR,
    prod_var: str = PROD_VAR,
    area_var: str = AREA_VAR,
    overlap_var: str = OVERLAP_VAR,
    tech_dim: str = TECH_DIM,
    tech_coord: str = TECH_COORD,
) -> Dict[str, np.ndarray]:
    """
    Computes a global LCOE order and returns a dictionary with:
      - 'lcoe_sorted'      : (N,) float64
      - 'prod_sorted'      : (N,) float64
      - 'area_sorted'      : (N,) float64
      - 'overlap_sorted'   : (N,) float64
      - 'tech_sorted_id'   : (N,) uint16 or None
      - 'tech_names'       : list[str] or None
      - 'left_positions'   : (N,) float64 cumulative prod "left edges" (MWh)
      - 'cum_prod'         : (N,) float64 cumulative prod at bar RIGHT edges (MWh)
      - 'mask'             : boolean mask used before sorting (for advanced reuse)
      - 'order'            : sort indices (aligned to masked arrays)
    """
    has_tech, tech_names = _detect_tech(ds, tech_dim, tech_coord)
    lcoe, prod, area, overlap, tech_id = _flatten_for_sort(ds, lcoe_var, prod_var, area_var, overlap_var, tech_names)

    mask = np.isfinite(lcoe) & np.isfinite(prod)
    l_m = lcoe[mask]
    p_m = prod[mask]
    a_m = area[mask]
    t_m = tech_id[mask] if tech_id is not None else None

    # Stable global sort
    order = np.argsort(l_m, kind="mergesort")
    lcoe_sorted = l_m[order]
    prod_sorted = p_m[order]
    area_sorted = a_m[order]
    # TODO: overlap does not have the tech dimension, so it is not ordered at the same time. Might have a problem.
    overlap_sorted = overlap[order]
    tech_sorted_id = t_m[order] if t_m is not None else None

    # Cumulative production for bar positions (in MWh)
    left_positions = np.cumsum(np.concatenate(([0.0], prod_sorted[:-1]))).astype(np.float64)
    cum_prod = (left_positions + prod_sorted).astype(np.float64)

    return {
        "lcoe_sorted": lcoe_sorted,
        "prod_sorted": prod_sorted,
        "area_sorted": area_sorted,
        "overlap_sorted": overlap_sorted,
        "tech_sorted_id": tech_sorted_id,
        "tech_names": tech_names,
        "left_positions": left_positions,
        "cum_prod": cum_prod,
        "mask": mask,
        "order": order,
    }



def _choose_color_map(tech_names: Optional[List[str]]) -> Dict[Optional[str], str]:
    """Maps technology names to colors with fallbacks."""
    if tech_names is None:
        return {None: "#7EC8E3"}  # single-tech default
    cmap = {}
    fb = iter(FALLBACK_COLORS)
    for name in tech_names:
        cmap[name] = TECH_COLORS.get(name, next(fb, "#7f7f7f"))
    return cmap



def build_supply_curve_line(prep: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Returns (x, y, x_label) for supply curve line plot:
      x = cumulative prod (MWh -> TWh if enabled)
      y = lcoe_sorted
    """
    x = prep["cum_prod"].copy()
    if PROD_TO_TWH and PROD_DIVISOR:
        x = x / PROD_DIVISOR
        x_label = "Cumulative production (TWh)"
    else:
        x_label = "Cumulative production (MWh)"
    y = prep["lcoe_sorted"]
    return x, y, x_label


def plot_supply_curve_line(prep: Dict[str, np.ndarray],
                           output_path: str,
                           title: str = "LCOE Supply Curve (line)",):
    """Plots LCOE vs cumulative production (line)."""
    x, y, x_label = build_supply_curve_line(prep)
    plt.figure(figsize=(9, 6), dpi=120)
    plt.plot(x, y, color="#1f77b4", lw=1.5)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel("LCOE (€/MWh)", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, format='png')

TECH_NAME_DICT = {
    "pv_open_field": "Open Field PV",
    "pv_rooftop": "Rooftop PV",
    "wind_onshore": "Onshore Wind",
    "wind_offshore": "Offshore Wind",
}

def plot_supply_curve_bars(
    prep: Dict[str, np.ndarray],
    output_path: str,
    bin_count: Optional[int] = None,
    rasterized: bool = False,
    title: str = "Supply Curve"
):
    """
    Plots the supply curve as adjacent bars:
      - bar width  = prod_sorted (MWh -> TWh if enabled)
      - bar height = lcoe_sorted
      - bar color  = technology (if present)
    Options:
      - bin_count: aggregate contiguous bars to reduce rectangles (preserves total cost approximately)
      - max_bars_to_plot: truncate after N bars (lowest LCOE region)
    """
    lcoe_sorted = prep["lcoe_sorted"]
    prod_sorted = prep["prod_sorted"]
    left = prep["left_positions"]
    tech_sorted_id = prep["tech_sorted_id"]
    tech_names = prep["tech_names"]

    # Convert units for x
    x_left = left.copy()
    x_width = prod_sorted.copy()
    if PROD_TO_TWH and PROD_DIVISOR:
        x_left = x_left / PROD_DIVISOR
        x_width = x_width / PROD_DIVISOR
        x_label = "Cumulative production (TWh)"
    else:
        x_label = "Cumulative production (MWh)"

    plt.figure(figsize=(11, 6), dpi=120)

    if bin_count is not None and bin_count > 0 and bin_count < len(x_width):
        # Bin adjacent bars (contiguous in sorted order)
        n = len(x_width)
        edges = np.linspace(0, n, bin_count + 1).astype(int)
        bin_left, bin_width, bin_height, bin_tech = [], [], [], []

        for i in range(bin_count):
            s, e = edges[i], edges[i + 1]
            w = x_width[s:e]
            h = lcoe_sorted[s:e]
            if w.size == 0:
                continue
            total_w = w.sum()
            if total_w <= 0:
                # empty bin
                continue
            # Weighted average LCOE so area ≈ sum(prod×lcoe)
            H = np.sum(h * w) / total_w
            bin_left.append(x_left[s])
            bin_width.append(total_w)
            bin_height.append(H)

            if tech_sorted_id is not None:
                ids = tech_sorted_id[s:e]
                if ids.size > 0:
                    # dominant tech by count (cheap heuristic)
                    uids, counts = np.unique(ids, return_counts=True)
                    bin_tech.append(uids[np.argmax(counts)])
                else:
                    bin_tech.append(None)
            else:
                bin_tech.append(None)

        # Draw binned bars
        if tech_sorted_id is None:
            plt.bar(bin_left, bin_height, width=bin_width, align="edge",
                    color="#7EC8E3", edgecolor="none", rasterized=rasterized)
            legend_items = [("Technology", "#7EC8E3")]
        else:
            cmap = _choose_color_map(tech_names)
            legend_items = []
            for l, h, w, tid in zip(bin_left, bin_height, bin_width, bin_tech):
                tname = tech_names[tid] if tid is not None else None
                color = cmap.get(tname, "#7f7f7f")
                plt.bar(l, h, width=w, align="edge", color=color, edgecolor="none", rasterized=rasterized)
            cmap_renamed = cmap.copy()
            for tname, color in cmap.items():
                cmap_renamed[TECH_NAME_DICT[tname]] = cmap_renamed.pop(tname)
            legend_items = list(cmap_renamed.items())
            
    else:
        # Draw per-tech vectorized bars
        if tech_sorted_id is None:
            plt.bar(x_left, lcoe_sorted, width=x_width, align="edge",
                    color="#7EC8E3", edgecolor="none", rasterized=rasterized)
            legend_items = [("Technology", "#7EC8E3")]
        else:
            cmap = _choose_color_map(tech_names)
            legend_items = []
            T = len(tech_names)
            for t in range(T):
                # NEEDAUNDERSTAND: what is this m? Is it a mask (an array of booleans)?
                m = (tech_sorted_id == t)
                if not np.any(m):
                    continue
                color = cmap.get(tech_names[t], "#7f7f7f")
                plt.bar(x_left[m], lcoe_sorted[m], width=x_width[m], align="edge",
                        color=color, edgecolor="none", rasterized=rasterized)
                legend_items.append((TECH_NAME_DICT[tech_names[t]], color))

    plt.xlabel(x_label, fontsize=16)
    plt.ylabel("LCOE (€/MWh)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=lab) for lab, c in legend_items]
    if handles:
        plt.legend(handles=handles, title="Technology", frameon=True, fontsize=14, title_fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, format='png')




def build_land_use_curve_points(
    # ds: xr.Dataset,
    prep: Dict[str, np.ndarray],
    # area_var: str = AREA_VAR,
    # overlap_var: str = OVERLAP_VAR
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Builds the land-use curve with overlap behavior using the same global LCOE order:
      - For each pixel i:
          prod_i (MWh), area_i, overlap_i (area units)
          prod_density_i = prod_i / area_i (MWh per area unit) if area_i>0, else 0
          overlap_prod_i = min(prod_i, overlap_i * prod_density_i)  --> flat x segment
          non_overlap_prod_i = prod_i - overlap_prod_i
          area_add_i = max(area_i - overlap_i, 0)
        The curve has 2 points per pixel: (flat end, rise end).
    Returns (x_points, y_points, x_label).
    """

    area_sorted = prep["area_sorted"]
    overlap_sorted = prep["overlap_sorted"]
    prod_sorted = prep["prod_sorted"]

    # Production per area unit (guard zeros)
    prod_density = np.where(area_sorted > 0, prod_sorted / area_sorted, 0.0)

    # Production deliverable "for free" (no new land) from overlapped area
    overlap_prod = np.minimum(prod_sorted, overlap_sorted * prod_density)

    # Remaining production that requires new land
    non_overlap_prod = prod_sorted - overlap_prod

    # New land to be added by the pixel (up to area - overlap, floored at 0)
    area_add = np.maximum(area_sorted - overlap_sorted, 0.0)

    # Build segments: two points per pixel
    N = len(prod_sorted)
    x_points = np.empty(2 * N, dtype=np.float64)
    y_points = np.empty(2 * N, dtype=np.float64)

    cum_x = 0.0
    cum_y = 0.0
    k = 0
    for i in range(N):
        # flat segment over overlap_prod
        x1 = cum_x + overlap_prod[i]
        y1 = cum_y
        x_points[k] = x1
        y_points[k] = y1
        k += 1

        # rising segment over non_overlap_prod adding exactly area_add
        x2 = x1 + non_overlap_prod[i]
        y2 = y1 + area_add[i]
        x_points[k] = x2
        y_points[k] = y2
        k += 1

        cum_x = x2
        cum_y = y2

    # Convert x units if requested
    if PROD_TO_TWH and PROD_DIVISOR:
        x_points = x_points / PROD_DIVISOR
        x_label = "Cumulative production (TWh)"
    else:
        x_label = "Cumulative production (MWh)"
    # Convert y to standard area units (km²)
    y_points = y_points / 1e6  # m² -> km²

    return x_points, y_points, x_label


def plot_land_use_curve_line(
    x_points: np.ndarray,
    y_points: np.ndarray,
    x_label: str,
    output_path: str,
    area_label: str = "Cumulative land use (km²)",
    title: str = "Land-Use Curve",
    
):
    """Plots the land-use curve as a line (piecewise linear)."""
    plt.figure(figsize=(11, 6), dpi=120)
    plt.plot(x_points, y_points, color="#2ca02c", lw=1.5)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(area_label, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, format='png')
