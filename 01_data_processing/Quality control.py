#!/usr/bin/env python3
"""
filter_patches.py
=================
Quality-control filter for histopathology image patches.

Removes (or moves) background-dominated, blurry, or otherwise
uninformative patches produced by a whole-slide-image tiling pipeline.

Decision criteria
-----------------
1. **White / black fraction** – rejects near-blank or near-black tiles.
2. **Tissue fraction** – based on optical-density + saturation masking.
3. **Edge content** – Sobel-like gradient check guards against out-of-focus
   or featureless regions.

Usage
-----
    python filter_patches.py ./Patch_Output --ext .png --workers 8
    python filter_patches.py ./Patch_Output --min-tissue-frac 0.55 --dry-run

Run ``python filter_patches.py --help`` for the full option list.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter parameters (dataclass for clarity & serialisability)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FilterParams:
    """Thresholds that govern the keep / discard decision for a single patch."""

    min_tissue_frac: float = 0.65
    """Minimum fraction of tissue pixels required to keep the patch."""

    od_thresh: float = 0.12
    """Optical-density sum threshold used to identify tissue pixels."""

    sat_thresh: float = 0.065
    """HSV saturation threshold for the tissue mask."""

    edge_thresh: float = 0.045
    """Gradient magnitude threshold for the edge mask."""

    min_edge_frac: float = 0.012
    """Minimum fraction of edge pixels (sharpness guard)."""

    white_frac_limit: float = 0.85
    """Drop patch if the fraction of near-white pixels exceeds this value."""

    black_frac_limit: float = 0.85
    """Drop patch if the fraction of near-black pixels exceeds this value."""


# ---------------------------------------------------------------------------
# Core per-patch quality check
# ---------------------------------------------------------------------------
def is_informative_patch(img: Image.Image, params: FilterParams) -> bool:
    """Return *True* if the patch contains enough tissue to be useful.

    The check is intentionally fast (pure NumPy, no OpenCV dependency)
    so it can run in a thread-pool on thousands of tiles.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    if rgb.size == 0:
        return False

    # ---- 1. Global intensity gates (white / black background) -------------
    mean_intensity = rgb.mean(axis=-1)
    if float((mean_intensity > 0.93).mean()) >= params.white_frac_limit:
        return False
    if float((mean_intensity < 0.08).mean()) >= params.black_frac_limit:
        return False

    # ---- 2. Tissue-fraction via optical density + saturation --------------
    cmax = rgb.max(axis=-1)
    cmin = rgb.min(axis=-1)
    saturation = (cmax - cmin) / (cmax + 1e-6)
    sat_mean = float(saturation.mean())

    od_sum = (-np.log(rgb + 1e-6)).sum(axis=-1)
    tissue_mask = (od_sum > params.od_thresh) & (
        (saturation > params.sat_thresh) | (od_sum > params.od_thresh * 2.0)
    )
    tissue_frac = float(tissue_mask.mean())

    if tissue_frac < params.min_tissue_frac:
        return False

    # ---- 3. Edge / sharpness guard ----------------------------------------
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    grad = np.zeros_like(gray)
    grad[:, 1:] += gx
    grad[1:, :] += gy
    edge_frac = float((grad > params.edge_thresh).mean())

    if edge_frac < params.min_edge_frac and sat_mean < params.sat_thresh * 1.5:
        return False

    return True


# ---------------------------------------------------------------------------
# Single-file worker (called inside the thread-pool)
# ---------------------------------------------------------------------------
@dataclass
class PatchResult:
    """Outcome for one processed file."""

    path: str
    kept: Optional[bool]  # True = kept, False = removed, None = error
    detail: str = ""


def _process_single_patch(
    filepath: Path,
    bad_dir: Path,
    move_bad: bool,
    params: FilterParams,
) -> PatchResult:
    """Open *filepath*, run the quality filter, and move/delete if rejected."""
    try:
        with Image.open(filepath) as im:
            keep = is_informative_patch(im, params)

        if keep:
            return PatchResult(path=str(filepath), kept=True)

        # Ensure destination exists (thread-safe via exist_ok)
        bad_dir.mkdir(parents=True, exist_ok=True)
        dst = bad_dir / filepath.name

        if move_bad:
            shutil.move(str(filepath), str(dst))
        else:
            filepath.unlink(missing_ok=True)

        return PatchResult(path=str(filepath), kept=False, detail=str(dst))

    except Exception as exc:
        return PatchResult(path=str(filepath), kept=None, detail=str(exc))


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
@dataclass
class FilterStats:
    """Aggregate statistics returned by the batch filter."""

    kept: int = 0
    removed: int = 0
    errors: int = 0

    @property
    def total(self) -> int:
        return self.kept + self.removed + self.errors


def filter_patch_directory(
    out_root: str | Path,
    *,
    ext: str = ".png",
    move_bad: bool = True,
    bad_dir_name: str = "_filtered_bad",
    num_workers: int = 8,
    params: Optional[FilterParams] = None,
) -> FilterStats:
    """Walk *out_root* recursively, filtering every ``*{ext}`` patch.

    Parameters
    ----------
    out_root : path-like
        Root directory that contains slide sub-folders with patch images.
    ext : str
        File extension to glob for (e.g. ``".png"``).
    move_bad : bool
        If *True*, rejected patches are moved into *bad_dir_name*;
        otherwise they are deleted.
    bad_dir_name : str
        Name of the sub-directory (under *out_root*) that collects
        rejected patches.
    num_workers : int
        Number of threads for parallel I/O + filtering.
    params : FilterParams | None
        Override default filter thresholds.  ``None`` → use defaults.

    Returns
    -------
    FilterStats
        Summary counts (kept / removed / errors).
    """
    out_root = Path(out_root)
    if params is None:
        params = FilterParams()

    # Collect candidate files, skipping the bad-patch directory itself
    files = [
        f
        for f in out_root.rglob(f"*{ext}")
        if bad_dir_name not in f.parts
    ]

    if not files:
        logger.info("No *%s files found under %s — nothing to do.", ext, out_root)
        return FilterStats()

    bad_root = out_root / bad_dir_name
    bad_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Found %d candidate patches.  bad_root=%s  move_bad=%s",
        len(files),
        bad_root,
        move_bad,
    )

    stats = FilterStats()
    futures = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for filepath in files:
            # Derive the slide-level sub-folder name for organising rejects
            rel = filepath.relative_to(out_root)
            slide_name = rel.parts[0] if len(rel.parts) > 1 else filepath.parent.name
            bad_dir = bad_root / slide_name

            fut = executor.submit(
                _process_single_patch, filepath, bad_dir, move_bad, params
            )
            futures[fut] = filepath

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Filtering patches",
            unit="patch",
        ):
            result: PatchResult = fut.result()
            if result.kept is True:
                stats.kept += 1
            elif result.kept is False:
                stats.removed += 1
            else:
                stats.errors += 1
                logger.warning("Error processing %s: %s", result.path, result.detail)

    logger.info(
        "Done — kept=%d  removed=%d  errors=%d  total=%d",
        stats.kept,
        stats.removed,
        stats.errors,
        stats.total,
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Filter uninformative histopathology patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("out_root", type=str, help="Root directory containing patch images.")
    p.add_argument("--ext", default=".png", help="Image file extension to glob for.")
    p.add_argument(
        "--workers", type=int, default=8, help="Number of parallel worker threads."
    )
    p.add_argument(
        "--bad-dir-name",
        default="_filtered_bad",
        help="Sub-directory name for rejected patches.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Delete bad patches instead of moving them.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG-level logging."
    )

    # Filter thresholds (mirror FilterParams fields)
    g = p.add_argument_group("filter thresholds")
    defaults = FilterParams()
    g.add_argument("--min-tissue-frac", type=float, default=defaults.min_tissue_frac)
    g.add_argument("--od-thresh", type=float, default=defaults.od_thresh)
    g.add_argument("--sat-thresh", type=float, default=defaults.sat_thresh)
    g.add_argument("--edge-thresh", type=float, default=defaults.edge_thresh)
    g.add_argument("--min-edge-frac", type=float, default=defaults.min_edge_frac)
    g.add_argument("--white-frac-limit", type=float, default=defaults.white_frac_limit)
    g.add_argument("--black-frac-limit", type=float, default=defaults.black_frac_limit)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    params = FilterParams(
        min_tissue_frac=args.min_tissue_frac,
        od_thresh=args.od_thresh,
        sat_thresh=args.sat_thresh,
        edge_thresh=args.edge_thresh,
        min_edge_frac=args.min_edge_frac,
        white_frac_limit=args.white_frac_limit,
        black_frac_limit=args.black_frac_limit,
    )

    stats = filter_patch_directory(
        out_root=args.out_root,
        ext=args.ext,
        move_bad=not args.dry_run,
        bad_dir_name=args.bad_dir_name,
        num_workers=args.workers,
        params=params,
    )

    print(
        f"\n{'='*50}\n"
        f"  Kept:    {stats.kept}\n"
        f"  Removed: {stats.removed}\n"
        f"  Errors:  {stats.errors}\n"
        f"  Total:   {stats.total}\n"
        f"{'='*50}"
    )
    return 0 if stats.errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

