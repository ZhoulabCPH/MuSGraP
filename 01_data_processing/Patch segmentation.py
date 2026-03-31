import os
import math
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import openslide


def _get_level0_objective_power(slide: openslide.OpenSlide):
    """读取WSI的分辨倍率的大小"""
    for k in ("openslide.objective-power", "aperio.AppMag"):
        v = slide.properties.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            pass
    return None


def _get_mpp_x(slide: openslide.OpenSlide):
    """Microns per pixel in X at level0, if available."""
    for k in ("openslide.mpp-x", "aperio.MPP"):
        v = slide.properties.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            pass
    return None


def _estimate_desired_downsample_to_20x(slide: openslide.OpenSlide, target_objective=20.0):
    """
    Return desired downsample relative to level0 to reach target 20x.
    Prefer objective-power; fallback to mpp heuristic.
    """
    obj = _get_level0_objective_power(slide)
    if obj is not None and obj > 0:
        return obj / target_objective  # e.g., 40->20 => 2.0 ; 20->20 => 1.0

    # Fallback: use mpp heuristic (typical: 40x~0.25, 20x~0.5)
    mpp = _get_mpp_x(slide)
    if mpp is not None and mpp > 0:
        # assume target 20x mpp is ~2x of 40x; if slide already ~0.5 then ds≈1
        # So desired downsample to 20x = target_mpp / mpp0. Use target_mpp=0.5 as a practical default.
        target_mpp = 0.5
        return target_mpp / mpp

    # Last resort: assume level0 is 40x
    return 2.0


def _pick_best_level(slide: openslide.OpenSlide, desired_downsample: float):
    """Pick slide level whose native downsample is closest to desired_downsample."""
    downsamples = np.array(slide.level_downsamples, dtype=np.float64)
    idx = int(np.argmin(np.abs(downsamples - desired_downsample)))
    return idx, float(downsamples[idx])


def _is_tissue(pil_img: Image.Image, white_thresh=220, tissue_frac_thresh=0.40):
    """
    40%有组织的Patch就会被保留
    Simple background filter:
    - Convert to RGB
    - Tissue if fraction of non-white pixels >= tissue_frac_thresh
    """
    arr = np.array(pil_img.convert("RGB"))
    # non-white mask
    non_white = np.any(arr < white_thresh, axis=-1)
    frac = non_white.mean()
    return frac >= tissue_frac_thresh


def tile_svs_to_patches_20x(
    svs_path: str,
    out_root: str,
    patch_size: int = 224,
    target_objective: float = 20.0,
    save_ext: str = ".png",
    skip_background: bool = True,
    tissue_frac_thresh: float = 0.05,
):
    """
    For one SVS:
    - Convert to target 20x equivalent by choosing best OpenSlide level and resizing if needed
    - Tile into patch_size x patch_size patches
    - Save patches under out_root/<slide_stem>/
    """
    svs_path = Path(svs_path)
    out_root = Path(out_root)
    out_dir = out_root / svs_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    slide = openslide.OpenSlide(str(svs_path))

    # Desired downsample to reach 20x vs level0 (ds=1 means already at 20x if level0 is 20x)
    desired_ds = _estimate_desired_downsample_to_20x(slide, target_objective=target_objective)

    # Pick closest native level to reduce heavy resizing
    level, level_ds = _pick_best_level(slide, desired_ds)

    w0, h0 = slide.dimensions  # level0
    # Step in level0 pixels corresponding to 224 px at target 20x
    step0 = int(round(patch_size * desired_ds))
    if step0 <= 0:
        raise ValueError("Invalid step size computed. Check slide metadata.")

    # When reading at chosen level, read size in that level's pixels:
    # region at level0 of size (patch_size*desired_ds) corresponds to (patch_size*desired_ds/level_ds) at this level
    read_size_L = int(round(patch_size * desired_ds / level_ds))
    read_size_L = max(1, read_size_L)

    n_x = max(0, (w0 - step0) // step0 + 1)
    n_y = max(0, (h0 - step0) // step0 + 1)

    total = n_x * n_y
    if total == 0:
        slide.close()
        return

    pbar = tqdm(total=total, desc=f"{svs_path.name} (level={level}, ds~{level_ds:.2f})", leave=False)

    idx = 0
    for y0 in range(0, h0 - step0 + 1, step0):
        for x0 in range(0, w0 - step0 + 1, step0):
            # read_region: location in level0 coords; size in level coords
            patch_rgba = slide.read_region((x0, y0), level, (read_size_L, read_size_L))
            patch_rgb = patch_rgba.convert("RGB")

            # Resize to exact 224x224 if needed (to match target 20x scale)
            if read_size_L != patch_size:
                patch_rgb = patch_rgb.resize((patch_size, patch_size), resample=Image.BILINEAR)

            if skip_background and (not _is_tissue(patch_rgb, tissue_frac_thresh=tissue_frac_thresh)):
                pbar.update(1)
                continue

            # save with coordinates
            out_name = f"{svs_path.stem}_x{x0}_y{y0}{save_ext}"
            patch_rgb.save(out_dir / out_name)

            idx += 1
            pbar.update(1)

    pbar.close()
    slide.close()


def batch_process_folder(
    svs_folder: str,
    out_root: str,
    patch_size: int = 224,
    target_objective: float = 20.0,
):
    svs_folder = Path(svs_folder)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    svs_files = sorted([p for p in svs_folder.glob("*.svs")])
    if not svs_files:
        print(f"No .svs files found in: {svs_folder}")
        return

    for svs in svs_files:
        try:
            tile_svs_to_patches_20x(
                svs_path=str(svs),
                out_root=str(out_root),
                patch_size=patch_size,
                target_objective=target_objective,
                save_ext=".png",
                skip_background=True,
                tissue_frac_thresh=0.40,
            )
            print(f"[OK] {svs.name}")
        except Exception as e:
            print(f"[FAIL] {svs.name}: {e}")


if __name__ == "__main__":
    # 修改为你的路径
    SVS_FOLDER = "./SVS_Folder"
    OUT_ROOT = "./Patch_Output"

    batch_process_folder(
        svs_folder=SVS_FOLDER,
        out_root=OUT_ROOT,
        patch_size=224,
        target_objective=10.0,
    )
