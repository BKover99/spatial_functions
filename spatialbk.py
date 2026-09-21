"""Shared utilities for 10X Xenium analysis (human or mouse).

This module is import-safe: nothing is plotted, cropped, or written until you
call a function. Plotting functions require ``base_dir`` so each project can
choose its own figure root.

Typical workflow::

    from spatialbk import crop_sdatas, plot_gene_on_cells, plot_morphology_panels

    entries = crop_sdatas(sdatas, crop_dict)  # list of {sample_id, name, sdata}
    plot_gene_on_cells(entries[0]["sdata"], entries[0]["sample_id"], FIGDIR, gene="SOX2")

    plot_morphology_panels(
        entries,
        FIGDIR,
        dapi_channel="DAPI",
        dapi_color="blue",
        channels=["ATP1A1/CD45/E-Cadherin", "18S", "AlphaSMA/Vimentin"],
        colors=["magenta", "magenta", "magenta"],
        overlay_channels=["ATP1A1/CD45/E-Cadherin", "AlphaSMA/Vimentin"],
        overlay_colors=["cyan", "magenta"],
        overlay_dapi_color="white",
    )

Entry dicts
-----------
``crop_sdatas`` returns dicts with keys ``sample_id``, ``name``, and ``sdata``.
``name`` is ``"all"`` for the uncropped object, or the crop key (e.g. ``"CXM_1"``).

Dependencies
------------
spatialdata, spatialdata-plot, anndata, matplotlib, numpy, scipy, pandas.
scanpy and seaborn are imported only by the QC helpers.
"""

from __future__ import annotations

import gc
import os
import traceback
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

import anndata as ad
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.sparse as sp
import spatialdata as sd
import spatialdata_plot  # noqa: F401  # registers SpatialData.pl
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from scipy.stats import beta, median_abs_deviation as mad
from scipy.stats import rankdata

__all__ = [
    # QC
    "mad_outlier",
    "violinplot",
    "qc1",
    "qc2",
    # rank aggregation
    "robust_rank_aggregation",
    # subsetting / genes
    "crop_sdata",
    "crop_sdatas",
    "check_genes",
    # colormaps / morphology inspection
    "cmap_from_black",
    "cmap_from_white",
    "get_morphology_image",
    "morphology_channel_names",
    "percentile_norm",
    # plotting
    "plot_cell_types",
    "plot_cell_types_on_grid",
    "plot_obs_continuous",
    "plot_gene_on_cells",
    "plot_gene_transcripts",
    "plot_single_tx",
    "plot_gene_pair_boundaries",
    "plot_gene_pair_transcripts",
    "plot_gene_cell_nucleus",
    "plot_morphology",
    "plot_morphology_panels",
    # batch
    "plot_genes_for_entries",
    "plot_gene_pairs_for_entries",
]


class SDataEntry(TypedDict):
    """One SpatialData object tagged with a sample id and crop name."""

    sample_id: str
    name: str
    sdata: Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_str_path(base_dir: str | os.PathLike) -> str:
    return os.fspath(base_dir)


def _safe_filename(value: str) -> str:
    return str(value).replace("/", "-").replace(os.sep, "-")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _cleanup() -> None:
    plt.close("all")
    gc.collect()


def _skip_existing(path: str, skip_existing: bool) -> bool:
    if skip_existing and os.path.exists(path):
        print(f"Skipping {path}, already exists.")
        return True
    return False


def _table(sdata: Any, table_name: str = "table") -> ad.AnnData:
    tables = getattr(sdata, "tables", None)
    if tables is not None and table_name in tables:
        return tables[table_name]
    return sdata[table_name]


def _unpack_entry(entry: Mapping[str, Any]) -> tuple[str, str, Any]:
    try:
        return str(entry["sample_id"]), str(entry["name"]), entry["sdata"]
    except KeyError as exc:
        raise KeyError(
            "Each entry must be a dict with keys 'sample_id', 'name', and "
            "'sdata'. Build these with crop_sdatas()."
        ) from exc


def _as_pair(values: Sequence[Any], label: str = "genes") -> tuple[Any, Any]:
    values = list(values)
    if len(values) != 2:
        raise ValueError(f"{label} must have length 2, got {values!r}")
    return values[0], values[1]


def _as_pairs(gene_pairs: Sequence[Sequence[str]]) -> list[tuple[str, str]]:
    pairs = list(gene_pairs)
    if pairs and isinstance(pairs[0], str):
        raise TypeError(
            "gene_pairs must be a sequence of pairs, e.g. [('SOX2', 'SOX9')], "
            "not a single pair of strings."
        )
    out: list[tuple[str, str]] = []
    for pair in pairs:
        a, b = _as_pair(pair, "gene pair")
        out.append((str(a), str(b)))
    return out


def _is_cmap_name(spec: str) -> bool:
    registry = getattr(plt, "colormaps", None)
    if registry is not None:
        try:
            return spec in registry
        except TypeError:
            pass
    try:
        plt.get_cmap(spec)
        return spec in getattr(plt.cm, "cmap_d", {})
    except (ValueError, KeyError):
        return False


def _as_cmap(spec: Any, *, background: str) -> Any:
    """Return a matplotlib colormap.

    * Existing ``Colormap`` objects are passed through.
    * Matplotlib colormap names (``"gray"``, ``"Reds"``) are passed through.
    * Anything else is treated as a colour (name, hex, or RGB tuple) and
      turned into a gradient from ``background`` to that colour.
    """
    if spec is None:
        raise ValueError("color/cmap is required")
    if isinstance(spec, Colormap):
        return spec
    if isinstance(spec, str) and _is_cmap_name(spec):
        return spec
    if background == "black":
        return cmap_from_black(spec)
    return cmap_from_white(spec)


def _resolve_one_gene(var_names: Sequence[str], gene: str) -> str | None:
    if gene in var_names:
        return gene
    lower = gene.lower()
    hits = [name for name in var_names if name.lower() == lower]
    if len(hits) == 1:
        warnings.warn(
            f"Gene {gene!r} not found; using case-insensitive match {hits[0]!r}.",
            UserWarning,
            stacklevel=3,
        )
        return hits[0]
    if len(hits) > 1:
        warnings.warn(
            f"Gene {gene!r} is ambiguous under case-insensitive match: {hits}. "
            "Skipping.",
            UserWarning,
            stacklevel=3,
        )
        return None
    return None


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------

def cmap_from_black(color: Any, name: str | None = None) -> LinearSegmentedColormap:
    """Build a colormap that goes from black to ``color`` (additive overlays).

    Parameters
    ----------
    color : color-like
        Any matplotlib colour (name, hex, RGB tuple).
    name : str, optional
        Colormap name. Inferred from ``color`` when omitted.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    label = name or f"black_{color}".replace(" ", "")
    return LinearSegmentedColormap.from_list(str(label), ["black", color])


def cmap_from_white(color: Any, name: str | None = None) -> LinearSegmentedColormap:
    """Build a colormap that goes from white to ``color``.

    Parameters
    ----------
    color : color-like
        Any matplotlib colour (name, hex, RGB tuple).
    name : str, optional
        Colormap name. Inferred from ``color`` when omitted.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    label = name or f"white_{color}".replace(" ", "")
    return LinearSegmentedColormap.from_list(str(label), ["white", color])


def _binary_cmap(color: Any) -> LinearSegmentedColormap:
    target = np.array(mcolors.to_rgba(color))
    target[3] = 1.0
    white = np.array([1.0, 1.0, 1.0, 1.0])
    return LinearSegmentedColormap.from_list(
        f"binary_{color}",
        [(0.0, white), (1e-6, target), (1.0, target)],
        N=256,
    )


def _gradient_cmap(color: Any) -> LinearSegmentedColormap:
    target = np.array(mcolors.to_rgba(color))
    target[3] = 1.0
    white = np.array([1.0, 1.0, 1.0, 1.0])
    return LinearSegmentedColormap.from_list(
        f"gradient_{color}",
        [white, target],
        N=256,
    )


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------

def mad_outlier(
    adata: ad.AnnData,
    metric: str,
    nmads: float,
    upper_only: bool = False,
    value: bool = False,
):
    """Flag cells whose ``obs[metric]`` is more than ``nmads`` MADs from the median.

    Parameters
    ----------
    adata : anndata.AnnData
        Object whose ``obs`` holds ``metric``.
    metric : str
        Column in ``adata.obs``.
    nmads : float
        Number of median absolute deviations from the median.
    upper_only : bool, default False
        If True, only the upper tail is considered.
    value : bool, default False
        If True, return the threshold(s) instead of a boolean mask.
        Upper-only thresholds are returned as ``[upper, 0]`` so they can be
        drawn as horizontal lines on a violin plot.

    Returns
    -------
    pandas.Series or list of float
        Boolean mask aligned to ``adata.obs``, or ``[upper, lower_or_0]``.
    """
    values = adata.obs[metric]
    median = np.median(values)
    spread = mad(values)
    upper_val = median + nmads * spread
    lower_val = median - nmads * spread
    if value:
        return [upper_val, 0.0 if upper_only else lower_val]
    if upper_only:
        return values > upper_val
    return (values < lower_val) | (values > upper_val)


def violinplot(ax, adata: ad.AnnData, col: str, vals: Sequence[float]) -> None:
    """Draw a violin of ``adata.obs[col]`` with red horizontal lines at ``vals``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    adata : anndata.AnnData
        Object holding the metric in ``obs``.
    col : str
        Column in ``adata.obs``.
    vals : sequence of float
        Horizontal reference lines (typically MAD thresholds).
    """
    import seaborn as sns

    sns.violinplot(y=adata.obs[col].values, ax=ax)
    for val in vals:
        ax.axhline(val, color="red")
    ax.set_title(col)
    ax.set_ylabel("")


def qc1(
    adata: ad.AnnData,
    *,
    min_genes: int = 50,
    min_counts: int = 100,
    percent_top: tuple[int, ...] = (10, 20, 50, 150),
) -> ad.AnnData:
    """Hard filters plus QC metrics (Xenium cell table).

    Drops cells with fewer than ``min_genes`` detected genes or ``min_counts``
    counts, then computes Scanpy QC metrics including ``pct_counts_in_top_*``.

    Parameters
    ----------
    adata : anndata.AnnData
        Cell-level Xenium table. Modified in place by Scanpy filters.
    min_genes : int, default 50
        Passed to ``scanpy.pp.filter_cells``.
    min_counts : int, default 100
        Passed to ``scanpy.pp.filter_cells``.
    percent_top : tuple of int, default (10, 20, 50, 150)
        Rank cutoffs for ``pct_counts_in_top_*`` metrics. ``qc2`` expects
        ``50`` (and uses ``20`` for the violin display).

    Returns
    -------
    anndata.AnnData
        The same object, filtered and with QC columns in ``obs``.
    """
    import scanpy as sc

    print("QC1")
    print(adata.shape)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.calculate_qc_metrics(adata, percent_top=percent_top, inplace=True)
    print(adata.shape)
    return adata


def qc2(
    adata: ad.AnnData,
    *,
    n_noisy_genes: int = 50,
    min_counts_after_noisy: int = 150,
    mad_n_genes: float = 4,
    mad_total_counts: float = 4,
    mad_top_genes: float = 5,
    top_genes_filter_key: str = "pct_counts_in_top_50_genes",
    top_genes_plot_key: str = "pct_counts_in_top_20_genes",
) -> tuple[ad.AnnData, plt.Figure]:
    """Drop cells dominated by noisy genes, then MAD-filter remaining outliers.

    1. Compute counts not in the top ``n_noisy_genes`` and drop cells below
       ``min_counts_after_noisy``.
    2. Draw violins with MAD thresholds.
    3. Drop cells that are MAD outliers on log1p gene count, log1p total
       count, or ``top_genes_filter_key``.

    Parameters
    ----------
    adata : anndata.AnnData
        Output of :func:`qc1` (must already have QC metric columns).
    n_noisy_genes : int, default 50
        Used with ``pct_counts_in_top_{n_noisy_genes}_genes``.
    min_counts_after_noisy : int, default 150
        Minimum leftover counts after subtracting the top-gene fraction.
    mad_n_genes, mad_total_counts, mad_top_genes : float
        MAD cutoffs for the three outlier metrics.
    top_genes_filter_key : str, default ``pct_counts_in_top_50_genes``
        Column used for the upper-only MAD filter.
    top_genes_plot_key : str, default ``pct_counts_in_top_20_genes``
        Column shown on the third violin (display only).

    Returns
    -------
    adata : anndata.AnnData
        Filtered copy. ``adata.uns['cells_removed']`` is the MAD-filter count.
    fig : matplotlib.figure.Figure
        Four-panel violin diagnostic. The caller can save it if desired.

    Notes
    -----
    The leftover-count violin is annotated with ``min_counts_after_noisy``,
    which is the threshold actually applied in step 1.
    """
    pct_key = f"pct_counts_in_top_{n_noisy_genes}_genes"
    leftover = adata.obs["transcript_counts"] - (
        adata.obs[pct_key] * adata.obs["transcript_counts"] / 100.0
    )
    adata = adata.copy()
    adata.obs["counts_left_after_noisy_genes"] = np.asarray(leftover)

    print("Removing where its mostly noisy genes")
    adata = adata[
        adata.obs["counts_left_after_noisy_genes"] > min_counts_after_noisy
    ].copy()
    print(adata.shape)

    print("QC2")
    print(adata.shape)
    log1p_n_genes_val = mad_outlier(
        adata, "log1p_n_genes_by_counts", mad_n_genes, upper_only=False, value=True
    )
    log1p_total_val = mad_outlier(
        adata, "log1p_total_counts", mad_total_counts, upper_only=False, value=True
    )
    top_genes_val = mad_outlier(
        adata, top_genes_plot_key, mad_top_genes, upper_only=True, value=True
    )

    fig, axs = plt.subplots(1, 4, figsize=(15, 10))
    violinplot(axs[0], adata, "log1p_n_genes_by_counts", log1p_n_genes_val)
    violinplot(axs[1], adata, "log1p_total_counts", log1p_total_val)
    violinplot(axs[2], adata, top_genes_plot_key, top_genes_val)
    violinplot(
        axs[3], adata, "counts_left_after_noisy_genes", [min_counts_after_noisy]
    )
    fig.tight_layout()

    outlier = (
        mad_outlier(adata, "log1p_total_counts", mad_total_counts)
        | mad_outlier(adata, "log1p_n_genes_by_counts", mad_n_genes)
        | mad_outlier(adata, top_genes_filter_key, mad_top_genes, upper_only=True)
    )
    n_removed = int(np.asarray(outlier).sum())
    adata = adata[~np.asarray(outlier)].copy()
    adata.uns["cells_removed"] = n_removed
    print(adata.shape)
    return adata, fig


# ---------------------------------------------------------------------------
# Rank aggregation
# ---------------------------------------------------------------------------

def robust_rank_aggregation(lists: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Robust Rank Aggregation (Kolde et al., 2012).

    Parameters
    ----------
    lists : sequence of sequences
        Each inner sequence is gene names in rank order (best first). Genes
        missing from a list are treated as rank 1 (worst).

    Returns
    -------
    pandas.DataFrame
        Columns ``gene``, ``rho_score``, ``corrected_score``, ``rank``, sorted
        by ``corrected_score`` (best first).
    """
    if not lists:
        return pd.DataFrame(
            columns=["gene", "rho_score", "corrected_score", "rank"]
        )

    all_genes = list({gene for sublist in lists for gene in sublist})
    n_genes = len(all_genes)
    gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
    rank_matrix = np.ones((n_genes, len(lists)))

    for list_idx, gene_list in enumerate(lists):
        list_length = len(gene_list)
        if list_length == 0:
            continue
        ranks = rankdata(range(list_length), method="average") / list_length
        for rank_idx, gene in enumerate(gene_list):
            rank_matrix[gene_to_idx[gene], list_idx] = ranks[rank_idx]

    rho_scores = np.zeros(n_genes)
    k_values = np.arange(1, len(lists) + 1)
    dist_a = k_values
    dist_b = len(lists) - k_values + 1
    for i in range(n_genes):
        gene_ranks = np.sort(rank_matrix[i])
        p_values = np.array(
            [
                beta.cdf(rank, a, b)
                for rank, a, b in zip(gene_ranks, dist_a, dist_b)
            ]
        )
        rho_scores[i] = np.min(p_values)

    corrected_scores = np.minimum(rho_scores * len(lists), 1.0)
    result = pd.DataFrame(
        {
            "gene": all_genes,
            "rho_score": rho_scores,
            "corrected_score": corrected_scores,
        }
    )
    result["rank"] = result["corrected_score"].rank(method="dense")
    return result.sort_values("corrected_score")


# ---------------------------------------------------------------------------
# Subsetting and gene helpers
# ---------------------------------------------------------------------------

def crop_sdata(
    sdata: Any,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    *,
    target_coordinate_system: str = "global",
):
    """Crop a SpatialData object to a bounding box in ``target_coordinate_system``.

    Parameters
    ----------
    sdata : SpatialData
        Object to crop.
    xmin, ymin, xmax, ymax : float
        Bounding box in the target coordinate system (Xenium global µm).
    target_coordinate_system : str, default ``global``
        Coordinate system passed to ``spatialdata.bounding_box_query``.

    Returns
    -------
    SpatialData
        Cropped object (shapes, points, images, and tables).
    """
    return sd.bounding_box_query(
        sdata,
        min_coordinate=[xmin, ymin],
        max_coordinate=[xmax, ymax],
        axes=("x", "y"),
        target_coordinate_system=target_coordinate_system,
    )


def crop_sdatas(
    sdatas: Sequence[Any],
    crop_dict: Mapping[str, Sequence[float]],
    *,
    sample_id_key: str = "ID",
    table_name: str = "table",
) -> list[SDataEntry]:
    """Attach uncropped objects plus any matching crops from ``crop_dict``.

    Each SpatialData object is stored once with ``name="all"``. Every crop key
    that starts with ``{sample_id}_`` is then applied and appended.

    Parameters
    ----------
    sdatas : sequence of SpatialData
        One object per sample.
    crop_dict : mapping
        Keys like ``"CXM_1"``; values ``(xmin, ymin, xmax, ymax)``.
    sample_id_key : str, default ``ID``
        ``obs`` column used to read the sample id from ``table_name``.
    table_name : str, default ``table``
        Table that holds ``sample_id_key``.

    Returns
    -------
    list of dict
        Each dict has ``sample_id``, ``name``, and ``sdata``.

    Notes
    -----
    Crop matching is ``crop_name.startswith(sample_id + "_")``. Sample ids
    should not be prefixes of one another (e.g. avoid both ``"L"`` and
    ``"LY"``).
    """
    entries: list[SDataEntry] = []
    for sdata in sdatas:
        ids = _table(sdata, table_name).obs[sample_id_key].unique()
        if len(ids) != 1:
            warnings.warn(
                f"Expected one unique {sample_id_key!r}, got {list(ids)!r}. "
                "Using the first.",
                UserWarning,
                stacklevel=2,
            )
        sample_id = str(ids[0])
        entries.append({"sample_id": sample_id, "name": "all", "sdata": sdata})
        prefix = f"{sample_id}_"
        for crop_name, bbox in crop_dict.items():
            if not str(crop_name).startswith(prefix):
                continue
            try:
                xmin, ymin, xmax, ymax = bbox
                cropped = crop_sdata(sdata, xmin, ymin, xmax, ymax)
                entries.append(
                    {
                        "sample_id": sample_id,
                        "name": str(crop_name),
                        "sdata": cropped,
                    }
                )
            except Exception as exc:
                print(f"Failed to crop {crop_name}: {exc}")
    print(f"Built {len(entries)} entries (full + crops).")
    return entries


def check_genes(
    sdata: Any,
    genes: Sequence[str],
    *,
    table_name: str = "table",
    verbose: bool = True,
) -> list[str]:
    """Keep genes present in ``sdata``, with unique case-insensitive fallback.

    Exact ``var_names`` matches are preferred. If a name is missing, a unique
    case-insensitive match is used and a warning is emitted (human ``SOX2`` vs
    mouse ``Sox2``). Ambiguous or missing names are dropped.

    Parameters
    ----------
    sdata : SpatialData
        Object whose table holds ``var_names``.
    genes : sequence of str
        Requested gene symbols, in the order you want them plotted.
    table_name : str, default ``table``
        Table to inspect.
    verbose : bool, default True
        Print how many genes were kept or dropped.

    Returns
    -------
    list of str
        Symbols as they appear in ``var_names`` (possibly case-corrected).
    """
    var_names = list(_table(sdata, table_name).var_names)
    kept: list[str] = []
    missing: list[str] = []
    for gene in genes:
        resolved = _resolve_one_gene(var_names, gene)
        if resolved is None:
            missing.append(gene)
        else:
            kept.append(resolved)
    if verbose:
        print(f"Kept {len(kept)} genes, dropped {len(missing)}")
        if missing:
            print(f"Missing: {missing}")
    return kept


def _require_obs_column(sdata: Any, column: str, table_name: str = "table") -> None:
    table = _table(sdata, table_name)
    if column in table.obs.columns:
        return
    available = list(table.obs.columns)
    raise KeyError(
        f"{column!r} is not in {table_name!r}.obs. "
        f"Available columns: {available}"
    )


def _as_palette_color(color: Any) -> str:
    if isinstance(color, str):
        return color
    return mcolors.to_hex(color)


def _prepare_categorical_plot(
    sdata: Any,
    column: str,
    groups: Sequence[str] | None,
    colors: Sequence | None,
    table_name: str = "table",
) -> tuple[list[str] | None, list[str] | None]:
    """Ensure ``column`` is categorical and drop groups missing from this object."""
    _require_obs_column(sdata, column, table_name=table_name)
    table = _table(sdata, table_name)
    series = table.obs[column]
    if not isinstance(series.dtype, pd.CategoricalDtype):
        print(
            f"Converting {column!r} from {series.dtype} to category for plotting."
        )
        table.obs[column] = series.astype("category")
        series = table.obs[column]

    present = {str(value) for value in series.dropna().unique()}
    present.update(str(value) for value in series.cat.categories)

    palette: list[str] | None = None
    if colors is not None:
        palette = [_as_palette_color(color) for color in colors]

    if groups is None:
        return None, palette

    groups_list = list(groups)
    if palette is not None and len(palette) != len(groups_list):
        if len(palette) > len(groups_list):
            warnings.warn(
                f"colors has {len(palette)} entries but groups has "
                f"{len(groups_list)}; extra colours are ignored.",
                UserWarning,
                stacklevel=3,
            )
            palette = palette[: len(groups_list)]
        else:
            raise ValueError(
                f"groups and colors must have the same length "
                f"(got {len(groups_list)} groups and {len(palette)} colors)."
            )

    kept_groups: list[str] = []
    kept_colors: list[str] = []
    missing: list[str] = []
    for i, group in enumerate(groups_list):
        if str(group) in present:
            kept_groups.append(group)
            if palette is not None:
                kept_colors.append(palette[i])
        else:
            missing.append(group)
    if missing:
        print(
            f"Skipping {len(missing)} groups not present in this object: {missing}"
        )
    if not kept_groups:
        raise ValueError(
            f"None of the requested groups are in {column!r}. "
            f"Requested: {groups_list}. Present: {sorted(present)}"
        )
    return kept_groups, (kept_colors if palette is not None else None)


def _log_plot_error(func_name: str, exc: BaseException, **ctx: Any) -> None:
    bits = " ".join(f"{key}={value!r}" for key, value in ctx.items())
    extra = f" ({bits})" if bits else ""
    print(f"{func_name} failed{extra}: {type(exc).__name__}: {exc}")
    traceback.print_exc()


def _require_gene(sdata: Any, gene: str, table_name: str = "table") -> str:
    resolved = check_genes(sdata, [gene], table_name=table_name, verbose=False)
    if not resolved:
        raise KeyError(
            f"Gene {gene!r} not found in table {table_name!r} "
            f"(var_names have {len(_table(sdata, table_name).var_names)} genes)."
        )
    return resolved[0]


# ---------------------------------------------------------------------------
# Morphology image helpers
# ---------------------------------------------------------------------------

def get_morphology_image(sdata: Any, key: str = "morphology_focus"):
    """Return the highest-resolution morphology image array.

    Parameters
    ----------
    sdata : SpatialData
        Xenium SpatialData object.
    key : str, default ``morphology_focus``
        Image key. Multi-scale images use ``scale0``.

    Returns
    -------
    xarray-like
        Image with a channel coordinate ``c``.
    """
    image = sdata[key]
    if "scale0" in image:
        return image["scale0"]["image"]
    return image


def morphology_channel_names(
    sdata: Any, key: str = "morphology_focus"
) -> list[str]:
    """List morphology channel names (useful when panel names differ by species).

    Parameters
    ----------
    sdata : SpatialData
        Xenium SpatialData object.
    key : str, default ``morphology_focus``
        Image key.

    Returns
    -------
    list of str
    """
    image = get_morphology_image(sdata, key=key)
    if hasattr(image, "coords") and "c" in image.coords:
        return [str(name) for name in np.asarray(image.coords["c"].values)]
    if hasattr(image, "c"):
        return [str(name) for name in np.asarray(image.c.values)]
    raise KeyError(f"Could not read channel names from {key!r}.")


def percentile_norm(
    image: Any, channel: str, lo: float, hi: float
) -> Normalize:
    """Percentile-based ``Normalize`` for one morphology channel.

    Parameters
    ----------
    image : xarray-like
        Morphology image from :func:`get_morphology_image`.
    channel : str
        Channel name (coordinate ``c``).
    lo, hi : float
        Percentiles used as ``vmin`` / ``vmax``.

    Returns
    -------
    matplotlib.colors.Normalize
    """
    array = np.asarray(image.sel(c=channel).values)
    vmin, vmax = np.percentile(array, [lo, hi])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return Normalize(vmin=vmin, vmax=vmax, clip=True)


# ---------------------------------------------------------------------------
# Plotting: obs columns
# ---------------------------------------------------------------------------

def plot_cell_types(
    sdata: Any,
    sample_id: str,
    base_dir: str | os.PathLike,
    *,
    name: str = "plot",
    groups: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    column: str = "cell_type",
    shape_key: str = "cell_boundaries",
    dpi: int = 600,
    skip_existing: bool = True,
    method: str = "matplotlib",
    table_name: str = "table",
) -> str:
    """Plot a categorical ``obs`` column on cell (or other) boundaries.

    Groups that are not present in this object are skipped (crops often lack
    some cell types). The column is converted to a pandas categorical if needed.

    Parameters
    ----------
    sdata : SpatialData
        Object to plot.
    sample_id : str
        Sample identifier used in the filename.
    base_dir : path-like
        Directory where the PNG is written (created if needed).
    name : str, default ``plot``
        Crop / panel name (e.g. ``all`` or ``CXM_1``).
    groups : sequence of str, optional
        Subset of categories to show. ``None`` shows all.
    colors : sequence of str, optional
        Palette aligned with ``groups`` (or with the category order).
    column : str, default ``cell_type``
        Categorical column to colour by.
    shape_key : str, default ``cell_boundaries``
        Shapes element to render.
    dpi : int, default 600
        Output resolution.
    skip_existing : bool, default True
        Skip rendering when the target PNG already exists.
    method : str, default ``matplotlib``
        Passed to ``render_shapes``.
    table_name : str, default ``table``
        Table that holds ``column``.

    Returns
    -------
    str
        Output path (also when the file already existed and was skipped).
    """
    groups, colors = _prepare_categorical_plot(
        sdata, column, groups, colors, table_name=table_name
    )
    base_dir = _ensure_dir(_as_str_path(base_dir))
    save_path = os.path.join(
        base_dir, f"celltypes_{sample_id}_{dpi}_dpi_{name}.png"
    )
    if _skip_existing(save_path, skip_existing):
        return save_path
    kwargs: dict[str, Any] = {"method": method, "table_name": table_name}
    if groups is not None:
        kwargs["groups"] = list(groups)
    if colors is not None:
        kwargs["palette"] = list(colors)
    try:
        try:
            rendered = sdata.pl.render_shapes(shape_key, color=column, **kwargs)
        except TypeError:
            kwargs.pop("table_name", None)
            rendered = sdata.pl.render_shapes(shape_key, color=column, **kwargs)
        rendered.pl.show(dpi=dpi, save=save_path)
    except Exception as exc:
        _log_plot_error(
            "plot_cell_types",
            exc,
            sample_id=sample_id,
            name=name,
            column=column,
            save_path=save_path,
        )
        raise
    _cleanup()
    return save_path


def plot_cell_types_on_grid(
    sdata: Any,
    sample_id: str,
    base_dir: str | os.PathLike,
    *,
    name: str = "plot",
    groups: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    column: str = "cell_type",
    shape_key: str = "cell_boundaries",
    dpi: int = 400,
    skip_existing: bool = True,
    method: str = "matplotlib",
    major_frac: float = 0.01,
    minor_frac: float = 0.002,
) -> str:
    """Like :func:`plot_cell_types` but with percentage ticks for picking crops.

    Major ticks are ``major_frac`` of the axis range (labelled); minor ticks
    are ``minor_frac``. Coordinates are in the plotted coordinate system
    (Xenium global µm).

    Parameters
    ----------
    sdata, sample_id, base_dir, name, groups, colors, column, shape_key, dpi,
    skip_existing, method
        See :func:`plot_cell_types`. Default ``dpi`` is 400.
    major_frac, minor_frac : float
        Tick spacing as a fraction of the axis range.

    Returns
    -------
    str
        Output path (also when the file already existed and was skipped).
    """
    base_dir = _ensure_dir(_as_str_path(base_dir))
    save_path = os.path.join(
        base_dir, f"celltypes_{sample_id}_{dpi}_dpi_{name}.png"
    )
    if _skip_existing(save_path, skip_existing):
        return save_path
    kwargs: dict[str, Any] = {"method": method}
    if groups is not None:
        kwargs["groups"] = list(groups)
    if colors is not None:
        kwargs["palette"] = list(colors)
    ret = sdata.pl.render_shapes(shape_key, color=column, **kwargs).pl.show(
        dpi=dpi, return_ax=True
    )
    if isinstance(ret, (list, tuple)):
        ax = ret[0]
    else:
        ax = ret
    if ax is None:
        ax = plt.gca()
    fig = ax.figure

    xmin, xmax = sorted(ax.get_xlim())
    ymin, ymax = sorted(ax.get_ylim())
    x_range = xmax - xmin
    y_range = ymax - ymin
    if x_range > 0 and y_range > 0:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_range * major_frac))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_range * minor_frac))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_range * major_frac))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_range * minor_frac))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax.minorticks_on()
        ax.tick_params(
            axis="x",
            which="major",
            labelsize=3,
            length=4,
            width=0.5,
            direction="out",
            rotation=90,
            pad=1,
        )
        ax.tick_params(
            axis="y",
            which="major",
            labelsize=3,
            length=4,
            width=0.5,
            direction="out",
            pad=1,
        )
        ax.tick_params(
            axis="both", which="minor", length=1.5, width=0.25, direction="out"
        )

    fig.canvas.draw()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    _cleanup()
    return save_path


def plot_obs_continuous(
    sdata: Any,
    sample_id: str,
    base_dir: str | os.PathLike,
    color: str,
    *,
    name: str = "plot",
    cmap: Any = "Spectral_r",
    shape_key: str = "cell_boundaries",
    dpi: int = 600,
    skip_existing: bool = True,
) -> str:
    """Plot a numeric ``obs`` column (AUCell scores, QC metrics, …) on shapes.

    Parameters
    ----------
    sdata : SpatialData
        Object to plot.
    sample_id : str
        Sample identifier used in the filename.
    base_dir : path-like
        Directory where the PNG is written.
    color : str
        Numeric ``obs`` column (or gene) to colour by.
    name : str, default ``plot``
        Crop / panel name.
    cmap : colormap or str, default ``Spectral_r``
        Matplotlib colormap.
    shape_key : str, default ``cell_boundaries``
        Shapes element to render.
    dpi : int, default 600
        Output resolution.
    skip_existing : bool, default True
        Skip rendering when the target PNG already exists.

    Returns
    -------
    str
        Output path (also when the file already existed and was skipped).
    """
    base_dir = _ensure_dir(_as_str_path(base_dir))
    save_path = os.path.join(
        base_dir, f"{_safe_filename(color)}_{sample_id}_{dpi}_dpi_{name}.png"
    )
    if _skip_existing(save_path, skip_existing):
        return save_path
    sdata.pl.render_shapes(shape_key, color=color, cmap=cmap).pl.show(
        dpi=dpi, save=save_path
    )
    _cleanup()
    return save_path


# ---------------------------------------------------------------------------
# Plotting: genes
# ---------------------------------------------------------------------------

def plot_gene_on_cells(
    sdata: Any,
    sample_id: str,
    base_dir: str | os.PathLike,
    gene: str,
    *,
    name: str = "plot",
    cmap: Any = "Reds",
    shape_key: str = "cell_boundaries",
    table_name: str = "table",
    dpi: int = 600,
    skip_existing: bool = True,
) -> str:
    """Plot one gene on cell boundaries.

    Parameters
    ----------
    sdata : SpatialData
        Object to plot.
    sample_id : str
        Sample identifier used in the filename.
    base_dir : path-like
        Root figure directory. A subdirectory named after the gene is created.
    gene : str
        Gene symbol. Case-insensitive unique matches are accepted.
    name : str, default ``plot``
        Crop / panel name.
    cmap : colormap or str, default ``Reds``
        Matplotlib colormap.
    shape_key : str, default ``cell_boundaries``
        Shapes element to render.
    table_name : str, default ``table``
        Table used for gene-name lookup.
    dpi : int, default 600
        Output resolution.
    skip_existing : bool, default True
        Skip rendering when the target PNG already exists.

    Returns
    -------
    str
        Output path (also when the file already existed and was skipped).
    """
    gene = _require_gene(sdata, gene, table_name=table_name)
    gene_dir = _ensure_dir(os.path.join(_as_str_path(base_dir), _safe_filename(gene)))
    save_path = os.path.join(
        gene_dir, f"{_safe_filename(gene)}_{sample_id}_{dpi}_dpi_{name}.png"
    )
    if _skip_existing(save_path, skip_existing):
        return save_path
    sdata.pl.render_shapes(shape_key, color=gene, cmap=cmap).pl.show(
        dpi=dpi, save=save_path
    )
    _cleanup()
    return save_path


def plot_gene_transcripts(
    sdata: Any,
    sample_id: str,
    base_dir: str | os.PathLike,
    gene: str,
    *,
    name: str = "plot",
    color: str = "magenta",
    shape_key: str = "cell_boundaries",
    transcript_key: str = "transcripts",
    size: float = 0.2,
    alpha: float = 1.0,
    method: str = "matplotlib",
    dpi: int = 600,
    skip_existing: bool = True,
    skip_names: Sequence[str] = (),
) -> str | None:
    """Plot one gene as transcript dots on top of cell boundaries.

    Full-slide transcript plots are memory-heavy. Pass
    ``skip_names=('all',)`` to skip uncropped objects (the original notebook
    always skipped ``name=='all'``). Available as ``plot_single_tx`` as well.

    Parameters
    ----------
    sdata : SpatialData
        Object to plot.
    sample_id : str
        Sample identifier used in the filename.
    base_dir : path-like
        Root figure directory. A subdirectory named after the gene is created.
    gene : str
        Feature name in the transcript table.
    name : str, default ``plot``
        Crop / panel name.
    color : str, default ``magenta``
        Palette colour for the gene's transcripts.
    shape_key : str, default ``cell_boundaries``
        Outlines drawn under the transcripts.
    transcript_key : str, default ``transcripts``
        Points element.
    size, alpha : float
        Passed to ``render_points``.
    method : str, default ``matplotlib``
        Passed to ``render_points``.
    dpi : int, default 600
        Output resolution.
    skip_existing : bool, default True
        Skip rendering when the target PNG already exists.
    skip_names : sequence of str, default ()
        If ``name`` is in this sequence, return without plotting.

    Returns
    -------
    str or None
        Output path, or ``None`` when ``name`` is in ``skip_names``.
    """
    if name in set(skip_names):
        print(f"Skipping transcript plot for name={name!r}.")
        return None
    gene_dir = _ensure_dir(os.path.join(_as_str_path(base_dir), _safe_filename(gene)))
    save_path = os.path.join(
        gene_dir,
        f"{name}_{_safe_filename(gene)}_{color}_{sample_id}_{dpi}_dpi_tx.png",
    )
    if _skip_existing(save_path, skip_existing):
        return save_path
    (
        sdata.pl.render_shapes(shape_key)
        .pl.render_points(
            transcript_key,
            color="feature_name",
            groups=gene,
            size=size,
            alpha=alpha,
            method=method,
            palette=color,
        )
        .pl.show(dpi=dpi, save=save_path)
    )
    _cleanup()
    return save_path


plot_single_tx = plot_gene_transcripts


def plot_gene_pair_boundaries(
    sdata: Any,
    sample_id: str,
    base_dir: str | os.PathLike,
    genes: Sequence[str],
    colors: Sequence,
    *,
    name: str = "plot",
    cell_shape_key: str = "cell_boundaries",
    nucleus_shape_key: str = "nucleus_boundaries",
    table_name: str = "table",
    dpi: int = 600,
    skip_existing: bool = True,
) -> str:
    """Colour cell boundaries by gene 0 and nucleus boundaries by gene 1.

    Parameters
    ----------
    sdata : SpatialData
        Object to plot.
    sample_id : str
        Sample identifier used in the filename.
    base_dir : path-like
        Root figure directory. A subdirectory ``{gene0}_{gene1}`` is created.
    genes : sequence of str, length 2
        Gene on cells, then gene on nuclei.
    colors : sequence, length 2
        Matplotlib colormap names, or colour names (converted to white-to-colour
        gradients).
    name : str, default ``plot``
        Crop / panel name.
    cell_shape_key, nucleus_shape_key : str
        Shape elements to render.
    table_name : str, default ``table``
        Table used for gene-name lookup.
    dpi : int, default 600
        Output resolution.
    skip_existing : bool, default True
        Skip rendering when the target PNG already exists.

    Returns
    -------
    str
        Output path (also when the file already existed and was skipped).
    """
    gene0, gene1 = _as_pair(genes, "genes")
    color0, color1 = _as_pair(colors, "colors")
    gene0 = _require_gene(sdata, gene0, table_name=table_name)
    gene1 = _require_gene(sdata, gene1, table_name=table_name)
    folder = _ensure_dir(
        os.path.join(
            _as_str_path(base_dir),
            f"{_safe_filename(gene0)}_{_safe_filename(gene1)}",
        )
    )
    save_path = os.path.join(
        folder,
        f"{_safe_filename(gene0)}_{_safe_filename(gene1)}_{color0}_{color1}_"
        f"{sample_id}_{dpi}_dpi.png",
    )
    if _skip_existing(save_path, skip_existing):
        return save_path
    (
        sdata.pl.render_shapes(
            cell_shape_key, color=gene0, cmap=_as_cmap(color0, background="white")
        )
        .pl.render_shapes(
            nucleus_shape_key,
            color=gene1,
            cmap=_as_cmap(color1, background="white"),
        )
        .pl.show(dpi=dpi, save=save_path)
    )
    _cleanup()
    return save_path


def plot_gene_pair_transcripts(
    sdata: Any,
    sample_id: str,
    base_dir: str | os.PathLike,
    genes: Sequence[str],
    colors: Sequence[str],
    *,
    name: str = "plot",
    shape_key: str = "cell_boundaries",
    transcript_key: str = "transcripts",
    size: float = 0.2,
    alpha: float = 1.0,
    method: str = "matplotlib",
    dpi: int = 600,
    skip_existing: bool = True,
    skip_names: Sequence[str] = (),
) -> str | None:
    """Plot two genes as transcript dots on cell-boundary outlines.

    Parameters
    ----------
    sdata : SpatialData
        Object to plot.
    sample_id : str
        Sample identifier used in the filename.
    base_dir : path-like
        Root figure directory. A subdirectory ``{gene0}_{gene1}`` is created.
    genes : sequence of str, length 2
        The two feature names.
    colors : sequence of str, length 2
        Palette colours for the two genes.
    name : str, default ``plot``
        Crop / panel name.
    shape_key, transcript_key : str
        Shape outlines and points element.
    size, alpha : float
        Passed to ``render_points``.
    method : str, default ``matplotlib``
        Passed to ``render_points``.
    dpi : int, default 600
        Output resolution.
    skip_existing : bool, default True
        Skip rendering when the target PNG already exists.
    skip_names : sequence of str, default ()
        If ``name`` is in this sequence, return without plotting.

    Returns
    -------
    str or None
        Output path, or ``None`` when ``name`` is in ``skip_names``.
    """
    if name in set(skip_names):
        print(f"Skipping transcript pair plot for name={name!r}.")
        return None
    gene0, gene1 = _as_pair(genes, "genes")
    color0, color1 = _as_pair(colors, "colors")
    folder = _ensure_dir(
        os.path.join(
            _as_str_path(base_dir),
            f"{_safe_filename(gene0)}_{_safe_filename(gene1)}",
        )
    )
    save_path = os.path.join(
        folder,
        f"{name}_{_safe_filename(gene0)}_{_safe_filename(gene1)}_{color0}_{color1}_"
        f"{sample_id}_{dpi}_dpi_tx.png",
    )
    if _skip_existing(save_path, skip_existing):
        return save_path
    (
        sdata.pl.render_shapes(shape_key)
        .pl.render_points(
            transcript_key,
            color="feature_name",
            groups=gene0,
            size=size,
            alpha=alpha,
            method=method,
            palette=color0,
        )
        .pl.render_points(
            transcript_key,
            color="feature_name",
            groups=gene1,
            size=size,
            alpha=alpha,
            method=method,
            palette=color1,
        )
        .pl.show(dpi=dpi, save=save_path)
    )
    _cleanup()
    return save_path


def _norm_preserve_zero(vals: np.ndarray) -> np.ndarray:
    vmax = vals.max()
    if vmax == 0:
        return vals.copy()
    out = vals / vmax
    out[vals == 0] = 0.0
    return out


def _make_binary_table(table: ad.AnnData, gene: str) -> ad.AnnData:
    """Return a new AnnData whose ``gene`` column is 0 / scaled-positive."""
    idx = table.var_names.get_loc(gene)
    vals = table.X[:, idx]
    if hasattr(vals, "toarray"):
        vals = vals.toarray().flatten()
    vals = _norm_preserve_zero(np.asarray(vals, dtype=float).reshape(-1))
    if sp.issparse(table.X):
        x = table.X.tocsc().copy()
    else:
        x = sp.csc_matrix(table.X.copy())
    x[:, idx] = sp.csc_matrix(vals[:, None])
    x = x.tocsr()
    return ad.AnnData(X=x, obs=table.obs, var=table.var, uns=table.uns, obsm=table.obsm)


def plot_gene_cell_nucleus(
    sdata: Any,
    sample_id: str,
    base_dir: str | os.PathLike,
    genes: Sequence[str],
    colors: Sequence[str],
    *,
    name: str = "plot",
    mode: Literal["gradient", "binary", "both"] = "both",
    dpi: int = 600,
    cell_alpha: float = 1.0,
    nucleus_alpha: float = 1.0,
    outline_alpha: float = 0.0,
    cell_table: str = "table",
    nucleus_table: str = "nucleus_table",
    cell_shape_key: str = "cell_boundaries",
    nucleus_shape_key: str = "nucleus_boundaries",
    skip_existing: bool = True,
) -> dict[str, str | None]:
    """Overlay gene 0 on cells and gene 1 on nuclei (gradient and/or binary).

    *gradient* maps expression through a white-to-colour colormap.
    *binary* rescales the gene column so zeros stay white and any positive
    count is the solid colour (tables are swapped in place and always restored).

    Parameters
    ----------
    sdata : SpatialData
        Object to plot. Tables named ``cell_table`` / ``nucleus_table`` are
        temporarily replaced in binary mode and restored afterwards.
    sample_id : str
        Sample identifier used in the filename.
    base_dir : path-like
        Root figure directory. A subdirectory ``{gene0}_cell_{gene1}_nuc`` is
        created.
    genes : sequence of str, length 2
        Gene on cells, then gene on nuclei.
    colors : sequence of str, length 2
        Matplotlib colour names for the two genes.
    name : str, default ``plot``
        Crop / panel name.
    mode : {'gradient', 'binary', 'both'}, default ``both``
        Which PNG(s) to write.
    dpi : int, default 600
        Output resolution.
    cell_alpha, nucleus_alpha, outline_alpha : float
        Fill / outline alphas passed to ``render_shapes``.
    cell_table, nucleus_table : str
        Table keys holding cell- and nucleus-level expression.
    cell_shape_key, nucleus_shape_key : str
        Shape elements to render.
    skip_existing : bool, default True
        Skip each PNG that already exists.

    Returns
    -------
    dict
        Keys ``gradient`` and/or ``binary`` mapping to output paths (or
        ``None`` if that mode was not requested).
    """
    if mode not in {"gradient", "binary", "both"}:
        raise ValueError("mode must be 'gradient', 'binary', or 'both'")
    gene0, gene1 = _as_pair(genes, "genes")
    color0, color1 = _as_pair(colors, "colors")
    gene0 = _require_gene(sdata, gene0, table_name=cell_table)
    gene1 = _require_gene(sdata, gene1, table_name=nucleus_table)

    folder = _ensure_dir(
        os.path.join(
            _as_str_path(base_dir),
            f"{_safe_filename(gene0)}_cell_{_safe_filename(gene1)}_nuc",
        )
    )
    stem = (
        f"{name}_{_safe_filename(gene0)}cell_{_safe_filename(gene1)}nuc_"
        f"{color0}_{color1}_{sample_id}_{dpi}_dpi"
    )
    paths = {
        "gradient": os.path.join(folder, f"{stem}_gradient.png"),
        "binary": os.path.join(folder, f"{stem}_binary.png"),
    }
    out: dict[str, str | None] = {"gradient": None, "binary": None}

    if mode in {"gradient", "both"}:
        save_gradient = paths["gradient"]
        if _skip_existing(save_gradient, skip_existing):
            out["gradient"] = save_gradient
        else:
            print(f"  [run] rendering gradient → {save_gradient}")
            (
                sdata.pl.render_shapes(
                    cell_shape_key,
                    color=gene0,
                    cmap=_gradient_cmap(color0),
                    table_name=cell_table,
                    fill_alpha=cell_alpha,
                    outline_alpha=outline_alpha,
                )
                .pl.render_shapes(
                    nucleus_shape_key,
                    color=gene1,
                    cmap=_gradient_cmap(color1),
                    table_name=nucleus_table,
                    fill_alpha=nucleus_alpha,
                    outline_alpha=outline_alpha,
                )
                .pl.show(
                    dpi=dpi,
                    title=f"{gene0} cell · {gene1} nucleus — gradient",
                    colorbar=False,
                    save=save_gradient,
                )
            )
            out["gradient"] = save_gradient
            _cleanup()

    if mode in {"binary", "both"}:
        save_binary = paths["binary"]
        if _skip_existing(save_binary, skip_existing):
            out["binary"] = save_binary
        else:
            print(f"  [run] rendering binary → {save_binary}")
            orig_cell_table = sdata[cell_table]
            orig_nuc_table = sdata[nucleus_table]
            try:
                sdata[cell_table] = _make_binary_table(orig_cell_table, gene0)
                sdata[nucleus_table] = _make_binary_table(orig_nuc_table, gene1)
                (
                    sdata.pl.render_shapes(
                        cell_shape_key,
                        color=gene0,
                        cmap=_binary_cmap(color0),
                        table_name=cell_table,
                        fill_alpha=cell_alpha,
                        outline_alpha=outline_alpha,
                    )
                    .pl.render_shapes(
                        nucleus_shape_key,
                        color=gene1,
                        cmap=_binary_cmap(color1),
                        table_name=nucleus_table,
                        fill_alpha=nucleus_alpha,
                        outline_alpha=outline_alpha,
                    )
                    .pl.show(
                        dpi=dpi,
                        title=f"{gene0} cell · {gene1} nucleus — binary",
                        colorbar=False,
                        save=save_binary,
                    )
                )
                out["binary"] = save_binary
            finally:
                sdata[cell_table] = orig_cell_table
                sdata[nucleus_table] = orig_nuc_table
                _cleanup()

    return out


# ---------------------------------------------------------------------------
# Plotting: morphology
# ---------------------------------------------------------------------------

def plot_morphology(
    sdata: Any,
    sample_id: str,
    base_dir: str | os.PathLike,
    *,
    dapi_channel: str,
    dapi_color: Any,
    channels: Sequence[str],
    colors: Sequence,
    name: str = "plot",
    pair_dapi_color: Any = "gray",
    overlay_channels: Sequence[str] | None = None,
    overlay_colors: Sequence | None = None,
    overlay_dapi_color: Any | None = None,
    overlay_dirname: str = "overlay",
    dapi_percentiles: tuple[float, float] = (1.0, 99.0),
    channel_percentiles: tuple[float, float] = (50.0, 99.5),
    morphology_key: str = "morphology_focus",
    dpi: int = 600,
    skip_existing: bool = True,
) -> list[str]:
    """Save DAPI, DAPI+each channel, and an optional multi-channel overlay.

    Colour *names* are converted to black-to-colour colormaps so channels add
    on a dark background. Matplotlib colormap names (e.g. ``"gray"``) are
    passed through.

    Parameters
    ----------
    sdata : SpatialData
        Object to plot.
    sample_id : str
        Sample identifier used in the filename.
    base_dir : path-like
        Root figure directory. One subdirectory is created per channel, plus
        ``overlay_dirname`` when an overlay is requested.
    dapi_channel : str
        Nuclear stain channel name in the morphology image.
    dapi_color : color or colormap
        Colour for the DAPI-only panel.
    channels : sequence of str
        Additional morphology channels. Each is saved as DAPI + that channel.
    colors : sequence
        One colour/colormap per entry in ``channels`` (same length).
    name : str, default ``plot``
        Crop / panel name.
    pair_dapi_color : color or colormap, default ``gray``
        DAPI colour in the pairwise DAPI+channel panels.
    overlay_channels : sequence of str, optional
        Extra channels to overlay on DAPI in one RGB-style panel.
    overlay_colors : sequence, optional
        Colours for ``overlay_channels``. Required when ``overlay_channels``
        is set.
    overlay_dapi_color : color or colormap, optional
        DAPI colour in the overlay panel. Required when ``overlay_channels``
        is set.
    overlay_dirname : str, default ``overlay``
        Subfolder name for the overlay PNG.
    dapi_percentiles, channel_percentiles : tuple of float
        Contrast percentiles for DAPI vs the other channels.
    morphology_key : str, default ``morphology_focus``
        Image key.
    dpi : int, default 600
        Output resolution.
    skip_existing : bool, default True
        Skip each PNG that already exists.

    Returns
    -------
    list of str
        Paths that were written or already present.
    """
    channels = list(channels)
    colors = list(colors)
    if len(channels) != len(colors):
        raise ValueError(
            f"channels and colors must have the same length "
            f"(got {len(channels)} and {len(colors)})."
        )
    if overlay_channels is not None:
        if overlay_colors is None or overlay_dapi_color is None:
            raise ValueError(
                "overlay_channels requires overlay_colors and overlay_dapi_color."
            )
        overlay_channels = list(overlay_channels)
        overlay_colors = list(overlay_colors)
        if len(overlay_channels) != len(overlay_colors):
            raise ValueError(
                "overlay_channels and overlay_colors must have the same length."
            )

    base_dir = _ensure_dir(_as_str_path(base_dir))
    saved: list[str] = []
    image = get_morphology_image(sdata, key=morphology_key)
    dapi_norm = percentile_norm(image, dapi_channel, *dapi_percentiles)
    channel_norms = {
        ch: percentile_norm(image, ch, *channel_percentiles) for ch in channels
    }
    if overlay_channels:
        for ch in overlay_channels:
            if ch not in channel_norms:
                channel_norms[ch] = percentile_norm(
                    image, ch, *channel_percentiles
                )

    dapi_cmap = _as_cmap(dapi_color, background="black")
    pair_dapi_cmap = _as_cmap(pair_dapi_color, background="black")
    channel_cmaps = [_as_cmap(c, background="black") for c in colors]

    dapi_dir = _ensure_dir(os.path.join(base_dir, _safe_filename(dapi_channel)))
    dapi_path = os.path.join(
        dapi_dir,
        f"{_safe_filename(dapi_channel)}_{sample_id}_{dpi}_dpi_{name}.png",
    )
    if not _skip_existing(dapi_path, skip_existing):
        sdata.pl.render_images(
            morphology_key,
            channel=dapi_channel,
            cmap=dapi_cmap,
            norm=dapi_norm,
        ).pl.show(
            dpi=dpi, title=dapi_channel, colorbar=False, save=dapi_path
        )
        _cleanup()
    saved.append(dapi_path)

    for channel, cmap in zip(channels, channel_cmaps):
        ch_dir = _ensure_dir(os.path.join(base_dir, _safe_filename(channel)))
        ch_path = os.path.join(
            ch_dir, f"{_safe_filename(channel)}_{sample_id}_{dpi}_dpi_{name}.png"
        )
        if not _skip_existing(ch_path, skip_existing):
            sdata.pl.render_images(
                morphology_key,
                channel=[dapi_channel, channel],
                cmap=[pair_dapi_cmap, cmap],
                norm=[dapi_norm, channel_norms[channel]],
            ).pl.show(
                dpi=dpi,
                title=f"{dapi_channel} + {channel}",
                colorbar=False,
                save=ch_path,
            )
            _cleanup()
        saved.append(ch_path)

    if overlay_channels:
        overlay_dir = _ensure_dir(os.path.join(base_dir, overlay_dirname))
        overlay_path = os.path.join(
            overlay_dir, f"overlay_{sample_id}_{dpi}_dpi_{name}.png"
        )
        if not _skip_existing(overlay_path, skip_existing):
            overlay_cmaps = [
                _as_cmap(overlay_dapi_color, background="black")
            ] + [_as_cmap(c, background="black") for c in overlay_colors]
            overlay_norms = [dapi_norm] + [
                channel_norms[ch] for ch in overlay_channels
            ]
            sdata.pl.render_images(
                morphology_key,
                channel=[dapi_channel, *overlay_channels],
                cmap=overlay_cmaps,
                norm=overlay_norms,
            ).pl.show(
                dpi=dpi,
                title=(
                    f"{dapi_channel} + " + " + ".join(overlay_channels)
                ),
                colorbar=False,
                save=overlay_path,
            )
            _cleanup()
        saved.append(overlay_path)

    return saved


def plot_morphology_panels(
    entries: Sequence[Mapping[str, Any]],
    base_dir: str | os.PathLike,
    *,
    dapi_channel: str,
    dapi_color: Any,
    channels: Sequence[str],
    colors: Sequence,
    pair_dapi_color: Any = "gray",
    overlay_channels: Sequence[str] | None = None,
    overlay_colors: Sequence | None = None,
    overlay_dapi_color: Any | None = None,
    overlay_dirname: str = "overlay",
    dapi_percentiles: tuple[float, float] = (1.0, 99.0),
    channel_percentiles: tuple[float, float] = (50.0, 99.5),
    morphology_key: str = "morphology_focus",
    dpi: int = 600,
    skip_existing: bool = True,
) -> None:
    """Run :func:`plot_morphology` for every entry; failures are printed, not raised.

    Parameters
    ----------
    entries : sequence of dict
        Dicts with ``sample_id``, ``name``, and ``sdata`` (from
        :func:`crop_sdatas`).
    base_dir : path-like
        Root figure directory.
    dapi_channel, dapi_color, channels, colors
        Required morphology specification; see :func:`plot_morphology`.
    pair_dapi_color, overlay_channels, overlay_colors, overlay_dapi_color,
    overlay_dirname, dapi_percentiles, channel_percentiles, morphology_key,
    dpi, skip_existing
        Passed through to :func:`plot_morphology`.
    """
    for entry in entries:
        sample_id, name, sdata = _unpack_entry(entry)
        print(f"morphology {sample_id} {name}")
        try:
            plot_morphology(
                sdata,
                sample_id,
                base_dir,
                name=name,
                dapi_channel=dapi_channel,
                dapi_color=dapi_color,
                channels=channels,
                colors=colors,
                pair_dapi_color=pair_dapi_color,
                overlay_channels=overlay_channels,
                overlay_colors=overlay_colors,
                overlay_dapi_color=overlay_dapi_color,
                overlay_dirname=overlay_dirname,
                dapi_percentiles=dapi_percentiles,
                channel_percentiles=channel_percentiles,
                morphology_key=morphology_key,
                dpi=dpi,
                skip_existing=skip_existing,
            )
        except Exception as exc:
            print(f"Morphology failed for {sample_id} {name}: {exc}")
        finally:
            _cleanup()


# ---------------------------------------------------------------------------
# Batch wrappers
# ---------------------------------------------------------------------------

def plot_genes_for_entries(
    entries: Sequence[Mapping[str, Any]],
    genes: Sequence[str],
    base_dir: str | os.PathLike,
    *,
    layer: Literal["cells", "transcripts", "both"] = "cells",
    cmap: Any = "Reds",
    transcript_color: str = "magenta",
    dpi: int = 600,
    skip_existing: bool = True,
    skip_transcript_names: Sequence[str] = ("all",),
) -> None:
    """Plot each gene for each entry as cell fills and/or transcript dots.

    Missing genes are skipped per object (see :func:`check_genes`). Errors on
    a single gene do not stop the rest of the batch.

    Parameters
    ----------
    entries : sequence of dict
        Dicts with ``sample_id``, ``name``, and ``sdata``.
    genes : sequence of str
        Gene symbols to plot.
    base_dir : path-like
        Root figure directory.
    layer : {'cells', 'transcripts', 'both'}, default ``cells``
        Which representation(s) to draw.
    cmap : colormap or str, default ``Reds``
        Colormap for cell-boundary plots.
    transcript_color : str, default ``magenta``
        Palette colour for transcript plots.
    dpi : int, default 600
        Output resolution.
    skip_existing : bool, default True
        Skip PNGs that already exist.
    skip_transcript_names : sequence of str, default ('all',)
        Crop names skipped for transcript plots. Full slides are omitted by
        default because they are memory-heavy.
    """
    if layer not in {"cells", "transcripts", "both"}:
        raise ValueError("layer must be 'cells', 'transcripts', or 'both'")
    genes = list(genes)
    for entry in entries:
        sample_id, name, sdata = _unpack_entry(entry)
        present = check_genes(sdata, genes, verbose=True)
        for gene in present:
            if layer in {"cells", "both"}:
                try:
                    plot_gene_on_cells(
                        sdata,
                        sample_id,
                        base_dir,
                        gene,
                        name=name,
                        cmap=cmap,
                        dpi=dpi,
                        skip_existing=skip_existing,
                    )
                except Exception as exc:
                    print(
                        f"plot_gene_on_cells failed for {sample_id} {name} "
                        f"{gene}: {exc}"
                    )
                finally:
                    _cleanup()
            if layer in {"transcripts", "both"}:
                try:
                    plot_gene_transcripts(
                        sdata,
                        sample_id,
                        base_dir,
                        gene,
                        name=name,
                        color=transcript_color,
                        dpi=dpi,
                        skip_existing=skip_existing,
                        skip_names=skip_transcript_names,
                    )
                except Exception as exc:
                    print(
                        f"plot_gene_transcripts failed for {sample_id} {name} "
                        f"{gene}: {exc}"
                    )
                finally:
                    _cleanup()


def plot_gene_pairs_for_entries(
    entries: Sequence[Mapping[str, Any]],
    gene_pairs: Sequence[Sequence[str]],
    base_dir: str | os.PathLike,
    colors: Sequence[str],
    *,
    style: Literal["boundaries", "transcripts", "cell_nucleus"] = "cell_nucleus",
    mode: Literal["gradient", "binary", "both"] = "both",
    dpi: int = 600,
    skip_existing: bool = True,
    skip_transcript_names: Sequence[str] = ("all",),
) -> None:
    """Plot each gene pair for each entry.

    Parameters
    ----------
    entries : sequence of dict
        Dicts with ``sample_id``, ``name``, and ``sdata``.
    gene_pairs : sequence of pairs
        e.g. ``[('SOX2', 'SOX9'), ('PDCD1', 'CD274')]``.
    base_dir : path-like
        Root figure directory.
    colors : sequence of str, length 2
        Colours for gene 0 and gene 1.
    style : {'boundaries', 'transcripts', 'cell_nucleus'}, default ``cell_nucleus``
        Which pair plotter to call.
    mode : {'gradient', 'binary', 'both'}, default ``both``
        Used only when ``style='cell_nucleus'``.
    dpi : int, default 600
        Output resolution.
    skip_existing : bool, default True
        Skip PNGs that already exist.
    skip_transcript_names : sequence of str, default ('all',)
        Crop names skipped when ``style='transcripts'``.
    """
    if style not in {"boundaries", "transcripts", "cell_nucleus"}:
        raise ValueError(
            "style must be 'boundaries', 'transcripts', or 'cell_nucleus'"
        )
    pairs = _as_pairs(gene_pairs)
    color0, color1 = _as_pair(colors, "colors")
    pair_colors = (str(color0), str(color1))
    for entry in entries:
        sample_id, name, sdata = _unpack_entry(entry)
        for pair in pairs:
            try:
                if style == "boundaries":
                    plot_gene_pair_boundaries(
                        sdata,
                        sample_id,
                        base_dir,
                        pair,
                        pair_colors,
                        name=name,
                        dpi=dpi,
                        skip_existing=skip_existing,
                    )
                elif style == "transcripts":
                    plot_gene_pair_transcripts(
                        sdata,
                        sample_id,
                        base_dir,
                        pair,
                        pair_colors,
                        name=name,
                        dpi=dpi,
                        skip_existing=skip_existing,
                        skip_names=skip_transcript_names,
                    )
                else:
                    plot_gene_cell_nucleus(
                        sdata,
                        sample_id,
                        base_dir,
                        pair,
                        pair_colors,
                        name=name,
                        mode=mode,
                        dpi=dpi,
                        skip_existing=skip_existing,
                    )
            except Exception as exc:
                print(
                    f"plot_gene_pairs_for_entries ({style}) failed for "
                    f"{sample_id} {name} {pair}: {exc}"
                )
            finally:
                _cleanup()
