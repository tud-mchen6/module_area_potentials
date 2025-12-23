
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


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

# Filtering
REQUIRE_POSITIVE_PROD = True

# Numerical stability
EPS = 1e-12


# Technology colors
TECH_COLORS = {
    "pv_open_field": "#FFD700",          # yellow
    "wind_onshore": "#7EC8E3",   # light blue
    "wind_offshore": "#1F4E79",  # dark blue
}
FALLBACK_COLORS = ["#999999", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]



def _detect_tech(ds: xr.Dataset, tech_dim: str, tech_coord: str) -> Tuple[bool, Optional[List[str]]]:
    """Detects if dataset has a tech dimension+coord. Returns (has_tech, tech_names_or_None)."""
    has_dim = tech_dim in ds.dims
    has_coord = tech_coord in ds.coords
    if has_dim and has_coord:
        tech_names = [str(v) for v in ds.coords[tech_coord].values]
        return True, tech_names
    return False, None


def _flatten_for_sort(
    ds: xr.Dataset,
    lcoe_var: str,
    prod_var: str,
    tech_names: Optional[List[str]]
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Flattens lcoe and prod to 1D arrays aligned for global sorting.
    If multi-tech, tech axis is moved to last, then raveled.
    Returns: (lcoe_flat, prod_flat, tech_id_or_None)
    """
    lvar = ds[lcoe_var].astype(np.float32)
    pvar = ds[prod_var].astype(np.float32)

    if set(lvar.dims) != set(pvar.dims):
        raise ValueError(f"'{lcoe_var}' and '{prod_var}' must share the same dims. Got {lvar.dims} vs {pvar.dims}")

    if tech_names is None:
        lcoe_flat = lvar.values.ravel()
        prod_flat = pvar.values.ravel()
        tech_id = None
    else:
        dims = list(lvar.dims)
        # move tech to last axis for predictable flatten order (Y, X, ..., T)
        transpose_order = [d for d in dims if d != TECH_DIM] + [TECH_DIM]
        larr = lvar.transpose(*transpose_order).values
        parr = pvar.transpose(*transpose_order).values
        T = len(tech_names)
        non_tech_size = larr.size // T

        lcoe_flat = larr.reshape(-1)
        prod_flat = parr.reshape(-1)
        # For each (y,x,...) position, we have T consecutive tech entries
        tech_id = np.tile(np.arange(T, dtype=np.uint16), non_tech_size)

    return lcoe_flat, prod_flat, tech_id


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
    tech_dim: str = TECH_DIM,
    tech_coord: str = TECH_COORD,
    require_positive_prod: bool = REQUIRE_POSITIVE_PROD
) -> Dict[str, np.ndarray]:
    """
    Computes a global LCOE order and returns a dictionary with:
      - 'lcoe_sorted'      : (N,) float32
      - 'prod_sorted'      : (N,) float32
      - 'tech_sorted_id'   : (N,) uint16 or None
      - 'tech_names'       : list[str] or None
      - 'left_positions'   : (N,) float64 cumulative prod "left edges" (MWh)
      - 'cum_prod'         : (N,) float64 cumulative prod at bar RIGHT edges (MWh)
      - 'mask'             : boolean mask used before sorting (for advanced reuse)
      - 'order'            : sort indices (aligned to masked arrays)
    """
    has_tech, tech_names = _detect_tech(ds, tech_dim, tech_coord)
    lcoe, prod, tech_id = _flatten_for_sort(ds, lcoe_var, prod_var, tech_names)

    mask = np.isfinite(lcoe) & np.isfinite(prod)
    if require_positive_prod:
        mask &= (prod > 0)

    l_m = lcoe[mask]
    p_m = prod[mask]
    t_m = tech_id[mask] if tech_id is not None else None

    # Stable global sort
    order = np.argsort(l_m, kind="mergesort")
    lcoe_sorted = l_m[order]
    prod_sorted = p_m[order]
    tech_sorted_id = t_m[order] if t_m is not None else None

    # Cumulative production for bar positions (in MWh)
    left_positions = np.cumsum(np.concatenate(([0.0], prod_sorted[:-1]))).astype(np.float64)
    cum_prod = (left_positions + prod_sorted).astype(np.float64)

    return {
        "lcoe_sorted": lcoe_sorted,
        "prod_sorted": prod_sorted,
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
    plt.xlabel(x_label)
    plt.ylabel("LCOE (€/MWh)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, format='png')



def plot_supply_curve_bars(
    prep: Dict[str, np.ndarray],
    output_path: str,
    bin_count: Optional[int] = None,
    max_bars_to_plot: Optional[int] = None,
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

    # Truncate (if requested)
    if max_bars_to_plot is not None and len(x_width) > max_bars_to_plot:
        lcoe_sorted = lcoe_sorted[:max_bars_to_plot]
        x_width = x_width[:max_bars_to_plot]
        x_left = x_left[:max_bars_to_plot]
        if tech_sorted_id is not None:
            tech_sorted_id = tech_sorted_id[:max_bars_to_plot]

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
            legend_items = list(cmap.items())
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
                m = (tech_sorted_id == t)
                if not np.any(m):
                    continue
                color = cmap.get(tech_names[t], "#7f7f7f")
                plt.bar(x_left[m], lcoe_sorted[m], width=x_width[m], align="edge",
                        color=color, edgecolor="none", rasterized=rasterized)
                legend_items.append((tech_names[t], color))

    plt.xlabel(x_label)
    plt.ylabel("LCOE (€/MWh)")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=lab) for lab, c in legend_items]
    if handles:
        plt.legend(handles=handles, title="Technology", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, format='png')




def build_land_use_curve_points(
    ds: xr.Dataset,
    prep: Dict[str, np.ndarray],
    area_var: str = AREA_VAR,
    overlap_var: str = OVERLAP_VAR
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
    mask = prep["mask"]
    order = prep["order"]
    # Flatten aux vars aligned to mask
    area = _flatten_aux_var(ds, area_var, mask, prep["tech_names"])
    overlap = _flatten_aux_var(ds, overlap_var, mask, prep["tech_names"])

    # We need area/overlap sorted to match prod_sorted order
    area_sorted = area[order]
    overlap_sorted = overlap[order]

    # Ensure non-negative
    area_sorted = np.clip(area_sorted, 0, None)
    overlap_sorted = np.clip(overlap_sorted, 0, None)

    # prod_sorted already aligned with order
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
    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(x_points, y_points, color="#2ca02c", lw=1.5)
    plt.xlabel(x_label)
    plt.ylabel(area_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, format='png')
