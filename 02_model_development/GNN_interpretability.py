
"""
GNN interpretability script.
"""

import argparse
import os
import random
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

OPTIONAL_IMPORT_ERRORS: Dict[str, Exception] = {}

try:
    import cv2
except Exception as exc:  # pragma: no cover - optional dependency
    cv2 = None
    OPTIONAL_IMPORT_ERRORS["cv2"] = exc

try:
    import tables
except Exception as exc:  # pragma: no cover - optional dependency
    tables = None
    OPTIONAL_IMPORT_ERRORS["tables"] = exc

try:
    import Train_Data
except Exception as exc:  # pragma: no cover - project dependency
    Train_Data = None
    OPTIONAL_IMPORT_ERRORS["Train_Data"] = exc

try:
    from dataset import null_collate
except Exception as exc:  # pragma: no cover - project dependency
    null_collate = None
    OPTIONAL_IMPORT_ERRORS["dataset.null_collate"] = exc

try:
    from models import Model_Foundation
except Exception as exc:  # pragma: no cover - project dependency
    Model_Foundation = None
    OPTIONAL_IMPORT_ERRORS["models.Model_Foundation"] = exc


# Illustrator-friendly vector export settings.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["axes.unicode_minus"] = False


def _require_dependency(obj, name: str):
    """Raise a clear error message when an optional dependency is unavailable."""
    if obj is None:
        original_error = OPTIONAL_IMPORT_ERRORS.get(name)
        raise ImportError(
            f"Required dependency '{name}' is unavailable. "
            f"Original error: {original_error}"
        )
    return obj


def _get_device_from_model(model: torch.nn.Module) -> torch.device:
    """Return the device of the model parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_int_from_patient_name(patient_name_value) -> int:
    """Extract a stable integer patient id from tensors, arrays, strings, or scalars."""
    if isinstance(patient_name_value, torch.Tensor):
        patient_name_value = patient_name_value.detach().cpu().numpy()

    if isinstance(patient_name_value, np.ndarray):
        if patient_name_value.size == 0:
            raise ValueError("patient_name is empty.")
        patient_name_value = patient_name_value.reshape(-1)[0]

    if isinstance(patient_name_value, (list, tuple)):
        if len(patient_name_value) == 0:
            raise ValueError("patient_name is empty.")
        patient_name_value = patient_name_value[0]

    patient_name_str = str(patient_name_value).strip()
    patient_name_digits = "".join(ch for ch in patient_name_str if ch.isdigit())
    if patient_name_digits:
        return int(patient_name_digits)
    return int(float(patient_name_str))


def _safe_to_numpy(data) -> np.ndarray:
    """Convert tensors or array-like objects to numpy arrays."""
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _normalize_patch_name_column(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Make sure the patch name column is named consistently."""
    df = patient_df.copy()
    if "Patch_Name" not in df.columns and "PatchName" in df.columns:
        df = df.rename(columns={"PatchName": "Patch_Name"})
    return df


def _ensure_attention_column(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Make sure the attention column is named consistently."""
    df = patient_df.copy()
    if "Attention" not in df.columns and "AttentionScore" in df.columns:
        df["Attention"] = df["AttentionScore"]
    return df


def _parse_coordinates(coord_text: str) -> Tuple[int, int]:
    """Parse coordinates from '(x,y)' style text."""
    coord_text = str(coord_text).strip()
    coord_text = coord_text.strip("()")
    x_text, y_text = coord_text.split(",")
    return int(x_text), int(y_text)


def _load_patch_image_from_array(patch_array: np.ndarray, out_size: int) -> np.ndarray:
    """Convert a patch array to a square RGB thumbnail."""
    patch_array = np.asarray(patch_array)

    if patch_array.ndim == 2:
        patch_array = np.stack([patch_array] * 3, axis=-1)
    elif patch_array.ndim == 3 and patch_array.shape[2] == 1:
        patch_array = np.repeat(patch_array, 3, axis=2)

    patch_array = patch_array.astype(np.uint8, copy=False)

    if cv2 is not None:
        # Preserve the original project behavior when OpenCV is available.
        if patch_array.ndim == 3 and patch_array.shape[2] == 3:
            rgb_image = cv2.cvtColor(patch_array, cv2.COLOR_BGR2RGB)
            thumb = cv2.resize(rgb_image, (out_size, out_size), interpolation=cv2.INTER_AREA)
            return thumb

    return np.array(Image.fromarray(patch_array).resize((out_size, out_size), Image.Resampling.LANCZOS))


def yaml_config_hook(config_file):
    """Load a YAML configuration file."""
    with open(config_file, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def get_learning_rate(optimizer):
    """Return the current learning rate from the first optimizer param group."""
    return optimizer.param_groups[0]["lr"]


def normalization(data):
    """Apply min-max normalization with zero-division protection."""
    data = np.asarray(data, dtype=np.float32)
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    denom = data_max - data_min
    if abs(float(denom)) < 1e-12:
        return np.zeros_like(data, dtype=np.float32)
    return (data - data_min) / denom


def BoxCox_Change(data):
    """
    Apply a stable Box-Cox transform and then normalize the output.

    The function name and signature are preserved for compatibility.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.size == 0:
        return data

    min_value = np.nanmin(data)
    shift = 0.0
    if min_value <= 0:
        shift = abs(min_value) + 1e-6

    transformed, _ = stats.boxcox(data + shift + 1e-18)
    return normalization(transformed)


def initialize(batch):
    """
    Move known tensor fields in a batch to the best available device.

    The original function always used CUDA directly, which could fail on CPU-only
    environments. This version keeps the same interface but makes device handling safe.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_keys = ["OS", "OSState", "DFS", "DFSState", "WSI_feature"]
    for key in tensor_keys:
        value = batch.get(key)
        if torch.is_tensor(value):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def best_threshold_youden(y_true, score, pos_label=1, auto_flip=True):
    """
    Compute the best threshold using Youden's J statistic.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    score = np.asarray(score).astype(float).ravel()

    valid_mask = np.isfinite(score) & np.isfinite(y_true)
    y_true = y_true[valid_mask]
    score = score[valid_mask]

    if y_true.size == 0:
        raise ValueError("No valid samples are available after filtering invalid values.")
    if np.unique(y_true).size < 2:
        raise ValueError("y_true must contain at least two classes to compute ROC metrics.")

    if auto_flip:
        auc = roc_auc_score(y_true, score)
        if auc < 0.5:
            score = -score
            auc = roc_auc_score(y_true, score)
    else:
        auc = roc_auc_score(y_true, score)

    fpr, tpr, thresholds = roc_curve(y_true, score, pos_label=pos_label)
    youden_j = tpr - fpr
    best_index = int(np.argmax(youden_j))

    best_threshold = thresholds[best_index]
    best_tpr = tpr[best_index]
    best_fpr = fpr[best_index]
    best_spec = 1.0 - best_fpr
    y_pred = (score >= best_threshold).astype(int)

    return {
        "best_threshold": float(best_threshold),
        "youden_J": float(youden_j[best_index]),
        "sensitivity_TPR": float(best_tpr),
        "specificity_TNR": float(best_spec),
        "auc": float(auc),
        "y_pred": y_pred,
        "used_score": score,
    }


def validate_epoch(model, datas, args, Type=None):
    """
    Run inference on a dataset and collect attention weights and adjacency matrices.
    """
    _require_dependency(null_collate, "dataset.null_collate")

    model.eval()
    device = _get_device_from_model(model)

    train_Patient_Attention = {}
    train_Patient_Adj = {}
    Patch_Attention_list = {}

    loader = DataLoader(
        datas,
        shuffle=False,
        batch_size=1,
        drop_last=False,
        num_workers=int(getattr(args, "workers", 0)),
        pin_memory=False,
        collate_fn=null_collate,
    )

    with torch.no_grad():
        for _, batch in enumerate(loader):
            batch = initialize(batch)
            x_wsi = batch["WSI_feature"].to(device)

            patient_name = _safe_int_from_patient_name(batch["patient_name"])

            h = model.fc_start(x_wsi)
            adj = model.create_knn_graph(h, k=7)
            _, attn_weights, _ = model(x_wsi)

            train_Patient_Adj[patient_name] = _safe_to_numpy(adj).squeeze()

            attn = _safe_to_numpy(attn_weights).squeeze()
            patch_name_value = batch.get("patch_name", [])
            patch_name_list = patch_name_value[0] if isinstance(patch_name_value, (list, tuple)) else patch_name_value

            try:
                attention_df = pd.DataFrame(
                    {"PatchName": patch_name_list, "AttentionScore": np.asarray(attn).reshape(-1)}
                )
            except ValueError:
                flattened_attn = np.asarray(attn[0]).reshape(-1)
                attention_df = pd.DataFrame(
                    {"PatchName": patch_name_list, "AttentionScore": flattened_attn}
                )

            Patch_Attention_list[str(patient_name)] = attention_df

    return Patch_Attention_list, train_Patient_Adj


def build_sparse_coordinate_system(coord_list):
    """Build a compact coordinate index map."""
    unique_coords = sorted(set(coord_list))
    return {coord: idx for idx, coord in enumerate(unique_coords)}


def get_canvas_size(wsi_original):
    """Return the canvas size as (width, height)."""
    if isinstance(wsi_original, Image.Image):
        return wsi_original.size
    array = np.asarray(wsi_original)
    return int(array.shape[1]), int(array.shape[0])


def rect_from_center(center, size):
    """Create a rectangle [x1, y1, x2, y2] from its center and size."""
    center_x, center_y = center
    half = size / 2.0
    return [center_x - half, center_y - half, center_x + half, center_y + half]


def rect_overlap(a, b, pad=6):
    """Check whether two rectangles overlap with extra padding."""
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def intersection_area(a, b):
    """Compute the intersection area of two rectangles."""
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    return max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)


def generate_candidate_offsets(radii, n_angles=16):
    """Generate candidate offsets on concentric rings."""
    offsets = [(0.0, 0.0)]
    for radius in radii:
        for angle_idx in range(n_angles):
            theta = 2.0 * np.pi * angle_idx / n_angles
            offsets.append((radius * np.cos(theta), radius * np.sin(theta)))
    return offsets


def choose_non_overlapping_position(
    anchor,
    display_size,
    canvas_size,
    occupied_rects,
    radii,
    pad=8,
    distance_penalty_scale=0.01,
):
    """
    Search around an anchor position for a low-overlap display location.
    """
    canvas_width, canvas_height = canvas_size
    candidates = generate_candidate_offsets(radii, n_angles=16)

    best_pos = anchor
    best_rect = rect_from_center(anchor, display_size)
    best_score = float("inf")

    for dx, dy in candidates:
        candidate = (anchor[0] + dx, anchor[1] + dy)
        rect = rect_from_center(candidate, display_size)

        x1, y1, x2, y2 = rect
        if x1 < 5 or y1 < 5 or x2 > canvas_width - 5 or y2 > canvas_height - 5:
            continue

        overlap_count = 0
        overlap_area_penalty = 0.0
        for occupied_rect in occupied_rects:
            if rect_overlap(rect, occupied_rect, pad=pad):
                overlap_count += 1
                overlap_area_penalty += intersection_area(rect, occupied_rect)

        distance_penalty = distance_penalty_scale * (dx * dx + dy * dy)
        score = overlap_count * 1e7 + overlap_area_penalty + distance_penalty

        if score < best_score:
            best_score = score
            best_pos = candidate
            best_rect = rect

    return best_pos, best_rect


def compute_seed_display_positions(
    seeds,
    pos_spatial,
    all_nodes,
    neighbor_display_size,
    seed_display_size,
    canvas_size,
):
    """
    Find display positions for seed patches.
    """
    occupied_rects = [
        rect_from_center(pos_spatial[int(node)], max(neighbor_display_size * 0.95, 18))
        for node in all_nodes
    ]

    seed_display_pos = {}
    seed_display_rects = {}

    seed_radii = [
        seed_display_size * 0.80,
        seed_display_size * 1.05,
        seed_display_size * 1.35,
        seed_display_size * 1.65,
    ]

    for seed in seeds:
        seed = int(seed)
        anchor = pos_spatial[seed]
        pos, rect = choose_non_overlapping_position(
            anchor=anchor,
            display_size=seed_display_size,
            canvas_size=canvas_size,
            occupied_rects=occupied_rects,
            radii=seed_radii,
            pad=12,
            distance_penalty_scale=0.01,
        )

        seed_display_pos[seed] = pos
        seed_display_rects[seed] = rect
        occupied_rects.append(rect)

    return seed_display_pos, seed_display_rects, occupied_rects


def compute_neighbor_display_positions(
    neighbor_nodes,
    pos_spatial,
    canvas_size,
    occupied_rects,
    neighbor_display_size,
    neighbor_owner_map,
    neighbor_min_dist,
):
    """
    Find display positions for neighbor patches.
    """
    neighbor_display_pos = {}
    neighbor_display_rects = {}

    neighbor_order = sorted(
        list(neighbor_nodes),
        key=lambda node: (
            -len(neighbor_owner_map.get(int(node), [])),
            neighbor_min_dist.get(int(node), 999999.0),
            int(node),
        ),
    )

    neighbor_radii = [
        neighbor_display_size * 0.70,
        neighbor_display_size * 1.00,
        neighbor_display_size * 1.35,
        neighbor_display_size * 1.70,
        neighbor_display_size * 2.05,
        neighbor_display_size * 2.40,
    ]

    for neighbor in neighbor_order:
        neighbor = int(neighbor)
        anchor = pos_spatial[neighbor]
        pos, rect = choose_non_overlapping_position(
            anchor=anchor,
            display_size=neighbor_display_size,
            canvas_size=canvas_size,
            occupied_rects=occupied_rects,
            radii=neighbor_radii,
            pad=10,
            distance_penalty_scale=0.012,
        )

        neighbor_display_pos[neighbor] = pos
        neighbor_display_rects[neighbor] = rect
        occupied_rects.append(rect)

    return neighbor_display_pos, neighbor_display_rects


def generate_wsi_heatmap_circle_optimized(
    patient_df,
    tile_size=224,
    thumbnail_ratio=0.1,
    cmap_name="jet",
    circle_scale=1.0,
    bg_color=(0, 0, 0),
):
    """
    Generate a thumbnail heatmap where each patch is shown as a colored circle.
    """
    patient_df = _normalize_patch_name_column(_ensure_attention_column(patient_df))

    patch_names = patient_df["Patch_Name"].values
    weights = patient_df["Attention"].values.astype(np.float32)

    try:
        x_coords = np.array([int(str(name).split("(")[1].split(",")[0]) for name in patch_names])
        y_coords = np.array([int(str(name).split(",")[1].split(")")[0]) for name in patch_names])
    except (IndexError, ValueError):
        return None

    grid_x = x_coords // tile_size
    grid_y = y_coords // tile_size
    norm_grid_x = grid_x - grid_x.min()
    norm_grid_y = grid_y - grid_y.min()

    thumb_size = max(1, int(tile_size * thumbnail_ratio))
    canvas_width = int((norm_grid_x.max() + 1) * thumb_size)
    canvas_height = int((norm_grid_y.max() + 1) * thumb_size)

    norm = mcolors.Normalize(vmin=weights.min(), vmax=weights.max())
    try:
        colormap = cm.colormaps.get_cmap(cmap_name)
    except AttributeError:  # pragma: no cover - matplotlib compatibility
        colormap = cm.get_cmap(cmap_name)

    colors_u8 = (colormap(norm(weights))[:, :3] * 255).astype(np.uint8)

    canvas = Image.new("RGB", (canvas_width, canvas_height), bg_color)
    draw = ImageDraw.Draw(canvas)
    radius = (thumb_size / 2) * circle_scale

    for grid_x_value, grid_y_value, color in zip(norm_grid_x, norm_grid_y, colors_u8):
        center_x = (grid_x_value * thumb_size) + (thumb_size / 2)
        center_y = (grid_y_value * thumb_size) + (thumb_size / 2)
        bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
        draw.ellipse(bbox, fill=tuple(color))

    return canvas


def generate_wsi_image(Patch_Index, patch_coords, patch_dir, tile_size=224, thumbnail_ratio=0.1):
    """
    Generate a compact WSI thumbnail from patch coordinates and patch image storage.
    """
    thumb_size = max(1, int(tile_size * thumbnail_ratio))
    x_list = [coord[0] // tile_size for coord in patch_coords]
    y_list = [coord[1] // tile_size for coord in patch_coords]

    x_map = build_sparse_coordinate_system(x_list)
    y_map = build_sparse_coordinate_system(y_list)

    canvas_width = max(1, len(x_map) * thumb_size)
    canvas_height = max(1, len(y_map) * thumb_size)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (242, 242, 242))

    if patch_dir is None:
        return canvas

    for index, (orig_x, orig_y) in enumerate(patch_coords):
        grid_x = x_map[orig_x // tile_size]
        grid_y = y_map[orig_y // tile_size]

        try:
            patch = patch_dir[int(Patch_Index[index])]
            thumb = _load_patch_image_from_array(patch, thumb_size)
            patch_image = Image.fromarray(thumb)
        except Exception:
            patch_image = Image.new("RGB", (thumb_size, thumb_size), (180, 180, 180))

        pos_x = grid_x * thumb_size
        pos_y = grid_y * thumb_size
        canvas.paste(patch_image, (pos_x, pos_y))

    return canvas


def select_exact_topk(patient_df, k=10):
    """Return the indices of the top-k attention values."""
    patient_df = _ensure_attention_column(patient_df)
    attention = patient_df["Attention"].to_numpy().astype(float)
    return np.argsort(-attention)[: min(k, len(attention))]


def get_seed_topk_feature_neighbors(seeds, Patch_Features, topn=5):
    """
    Return the nearest neighbors for each seed patch in feature space.
    """
    if torch.is_tensor(Patch_Features):
        features = Patch_Features.detach().float()
    else:
        features = torch.tensor(np.asarray(Patch_Features), dtype=torch.float32)

    if features.is_cuda:
        features = features.cpu()

    seed_to_all_dist = torch.cdist(features[seeds], features).numpy()

    seed_neighbors = {}
    for row_idx, seed in enumerate(seeds):
        order = np.argsort(seed_to_all_dist[row_idx])
        order = [int(node) for node in order if node != int(seed)][:topn]
        seed_neighbors[int(seed)] = [(node, float(seed_to_all_dist[row_idx, node])) for node in order]

    return seed_neighbors


def safe_get_patch_image(patch_dir, h5_idx, out_size):
    """Read a patch image safely and return a resized RGB numpy array."""
    try:
        patch_array = patch_dir[int(h5_idx)]
        return _load_patch_image_from_array(patch_array, out_size)
    except Exception:
        return np.array(Image.new("RGB", (out_size, out_size), "gray"))


def map_distance_to_edge_style(
    dist,
    dmin,
    dmax,
    min_width=1.2,
    max_width=7.5,
    min_alpha=0.25,
    max_alpha=0.92,
):
    """
    Map a feature distance to line width and alpha.

    Smaller distance means a stronger visual edge.
    """
    if abs(dmax - dmin) < 1e-12:
        width = (min_width + max_width) / 2.0
        alpha = (min_alpha + max_alpha) / 2.0
        return width, alpha

    ratio = (dist - dmin) / (dmax - dmin)
    width = max_width - ratio * (max_width - min_width)
    alpha = max_alpha - ratio * (max_alpha - min_alpha)
    return width, alpha


def build_neighbor_owner_map(seeds, seed_neighbors):
    """
    Build neighbor ownership metadata for visualization labels.
    """
    neighbor_owner_map = {}
    neighbor_min_dist = {}

    for rank, seed in enumerate(seeds, start=1):
        for neighbor, distance in seed_neighbors[int(seed)]:
            neighbor = int(neighbor)
            neighbor_owner_map.setdefault(neighbor, []).append(rank)
            if neighbor not in neighbor_min_dist:
                neighbor_min_dist[neighbor] = float(distance)
            else:
                neighbor_min_dist[neighbor] = min(neighbor_min_dist[neighbor], float(distance))

    for neighbor in neighbor_owner_map:
        neighbor_owner_map[neighbor] = sorted(list(set(neighbor_owner_map[neighbor])))

    return neighbor_owner_map, neighbor_min_dist


def format_owner_text(owner_list):
    """
    Convert owner ranks to a display label such as 'Top1/Top3'.
    """
    if len(owner_list) == 0:
        return ""
    return "/".join([f"Top{owner}" for owner in owner_list])


def add_top5_neighbor_gallery(
    fig,
    seeds,
    seed_neighbors,
    patch_dir,
    Patch_Index,
    patient_df,
    panel_bottom=0.08,
    panel_height=0.18,
):
    """
    Draw a gallery panel that shows top seeds and their nearest neighbors.
    """
    n_rows = len(seeds)
    n_cols = 6

    title_ax = fig.add_axes([0.05, panel_bottom + panel_height + 0.005, 0.90, 0.02])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.5,
        "Top5 Seed Patches and Their 5 Nearest Patches (Feature Distance)",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    label_left = 0.02
    label_width = 0.08
    content_left = 0.11
    content_right = 0.96
    content_width = content_right - content_left

    col_gap = 0.008
    row_gap = 0.010

    cell_w = (content_width - col_gap * (n_cols - 1)) / n_cols
    cell_h = (panel_height - row_gap * (n_rows - 1)) / max(n_rows, 1)

    for row_index, seed in enumerate(seeds):
        y_pos = panel_bottom + (n_rows - 1 - row_index) * (cell_h + row_gap)

        label_ax = fig.add_axes([label_left, y_pos, label_width, cell_h])
        label_ax.axis("off")
        attention_value = float(patient_df.iloc[int(seed)]["Attention"])
        label_ax.text(
            0.5,
            0.55,
            f"Top{row_index + 1}\nP{int(seed)}\nA={attention_value:.3f}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        items = [(int(seed), None)] + seed_neighbors[int(seed)]

        for col_index, item in enumerate(items):
            x_pos = content_left + col_index * (cell_w + col_gap)
            ax_cell = fig.add_axes([x_pos, y_pos, cell_w, cell_h])

            if col_index == 0:
                patch_idx = int(item[0])
                edge_color = "#E53935"
                title_text = f"P{patch_idx}\nSeed"
            else:
                patch_idx, dist_value = int(item[0]), float(item[1])
                edge_color = "#111111"
                title_text = f"P{patch_idx}\nd={dist_value:.3f}"

            h5_idx = int(Patch_Index[patch_idx])
            image_array = safe_get_patch_image(patch_dir, h5_idx, out_size=138)

            ax_cell.imshow(image_array)
            ax_cell.set_xticks([])
            ax_cell.set_yticks([])
            ax_cell.set_title(title_text, fontsize=8, pad=2)

            for spine in ax_cell.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.6 if col_index == 0 else 2.2)
                spine.set_edgecolor(edge_color)


def generate_wsi_spatial_network_vector_v16_Top10(
    WSIname,
    patient_df,
    adj_matrix,
    Patch_Index,
    patch_dir,
    wsi_original,
    save_path,
    Patch_Features,
    tile_size=224,
    thumbnail_ratio=0.1,
    Top_patch=5,
    Top_Size=10,
    Patch_Neighbor=10,
    Neighbor=5,
):
    """
    Generate the final vector visualization on top of the WSI overview image.
    """
    patient_df = _normalize_patch_name_column(_ensure_attention_column(patient_df))

    attention = patient_df["Attention"].values
    num_nodes_total = len(attention)
    sorted_idx = np.argsort(attention)[::-1]

    seeds = select_exact_topk(patient_df, k=Top_patch)
    seed_set = set(map(int, seeds))

    seed_neighbors = get_seed_topk_feature_neighbors(seeds, Patch_Features, topn=Patch_Neighbor)
    neighbor_owner_map, neighbor_min_dist = build_neighbor_owner_map(seeds, seed_neighbors)

    neighbor_set = set()
    feature_edges = []
    for seed in seeds:
        for neighbor, distance in seed_neighbors[int(seed)]:
            neighbor_set.add(int(neighbor))
            feature_edges.append((int(seed), int(neighbor), float(distance)))

    all_nodes = list(seed_set | neighbor_set)

    coords_list = []
    for _, row in patient_df.iterrows():
        x_px, y_px = _parse_coordinates(row["Coordinates"])
        coords_list.append((x_px, y_px))

    x_grid_list = [x // tile_size for x, _ in coords_list]
    y_grid_list = [y // tile_size for _, y in coords_list]
    x_map = {coord: idx for idx, coord in enumerate(sorted(set(x_grid_list)))}
    y_map = {coord: idx for idx, coord in enumerate(sorted(set(y_grid_list)))}

    thumb_size = max(1, int(tile_size * thumbnail_ratio))

    pos_spatial = {}
    for node in all_nodes:
        grid_x = coords_list[node][0] // tile_size
        grid_y = coords_list[node][1] // tile_size
        center_x = x_map[grid_x] * thumb_size + (thumb_size / 2)
        center_y = y_map[grid_y] * thumb_size + (thumb_size / 2)
        pos_spatial[node] = (center_x, center_y)

    graph = nx.Graph()
    rank_pos = {idx: rank for rank, idx in enumerate(sorted_idx)}
    denominator = max(num_nodes_total - 1, 1)
    rank_norm = {node: 1.0 - (rank_pos[node] / denominator) for node in all_nodes}

    for node in all_nodes:
        role = "seed" if node in seed_set else "neighbor"
        graph.add_node(node, h5=int(Patch_Index[node]), role=role, score=rank_norm[node])

    feature_edge_set = set()
    for seed, neighbor, distance in feature_edges:
        graph.add_edge(seed, neighbor, feature_dist=distance, edge_type="feature")
        feature_edge_set.add(tuple(sorted((seed, neighbor))))

    context_edges = []
    for first_idx in range(len(all_nodes)):
        for second_idx in range(first_idx + 1, len(all_nodes)):
            u = all_nodes[first_idx]
            v = all_nodes[second_idx]
            pair_key = tuple(sorted((u, v)))
            if pair_key in feature_edge_set:
                continue
            weight = float(adj_matrix[u, v])
            if weight > 0.001:
                context_edges.append((u, v, weight))

    fig, ax = plt.subplots(figsize=(20, 28))
    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.30)

    ax.imshow(wsi_original, zorder=0)
    ax.set_title(
        f"GNN Semantic Communication Network on WSI\n{WSIname}",
        fontsize=28,
        fontweight="bold",
        pad=20,
    )
    ax.axis("off")

    for u, v, _ in context_edges:
        nx.draw_networkx_edges(
            graph,
            pos_spatial,
            edgelist=[(u, v)],
            ax=ax,
            width=0.8,
            alpha=0.08,
            edge_color="#BDBDBD",
            connectionstyle="arc3,rad=0.10",
        )

    feature_distances = [distance for _, _, distance in feature_edges]
    dmin = min(feature_distances) if feature_distances else 0.0
    dmax = max(feature_distances) if feature_distances else 1.0

    for seed, neighbor, distance in feature_edges:
        width, alpha = map_distance_to_edge_style(
            distance,
            dmin,
            dmax,
            min_width=1.2,
            max_width=7.5,
            min_alpha=0.25,
            max_alpha=0.92,
        )
        radius = 0.12 if seed < neighbor else -0.12
        nx.draw_networkx_edges(
            graph,
            pos_spatial,
            edgelist=[(seed, neighbor)],
            ax=ax,
            width=width,
            alpha=alpha,
            edge_color="#FFD700",
            connectionstyle=f"arc3,rad={radius}",
        )

    neighbor_display_size = int(max(thumb_size * Neighbor, 48))
    seed_display_size = int(max(thumb_size * Top_Size, 82))
    canvas_size = get_canvas_size(wsi_original)

    seed_display_pos, _, occupied_rects = compute_seed_display_positions(
        seeds=seeds,
        pos_spatial=pos_spatial,
        all_nodes=all_nodes,
        neighbor_display_size=neighbor_display_size,
        seed_display_size=seed_display_size,
        canvas_size=canvas_size,
    )

    neighbor_display_pos, _ = compute_neighbor_display_positions(
        neighbor_nodes=neighbor_set,
        pos_spatial=pos_spatial,
        canvas_size=canvas_size,
        occupied_rects=occupied_rects,
        neighbor_display_size=neighbor_display_size,
        neighbor_owner_map=neighbor_owner_map,
        neighbor_min_dist=neighbor_min_dist,
    )

    for neighbor in sorted(list(neighbor_set)):
        anchor_x, anchor_y = pos_spatial[int(neighbor)]
        disp_x, disp_y = neighbor_display_pos[int(neighbor)]
        anchor_half = max(neighbor_display_size * 0.12, 4.5)

        anchor_rect = patches.Rectangle(
            (anchor_x - anchor_half, anchor_y - anchor_half),
            2 * anchor_half,
            2 * anchor_half,
            linewidth=1.6,
            edgecolor="#111111",
            facecolor=(0.0, 0.0, 0.0, 0.05),
            zorder=3,
        )
        ax.add_patch(anchor_rect)

        if abs(disp_x - anchor_x) > 1 or abs(disp_y - anchor_y) > 1:
            ax.plot(
                [anchor_x, disp_x],
                [anchor_y, disp_y],
                linestyle="--",
                linewidth=1.0,
                color="#111111",
                alpha=0.45,
                zorder=3,
            )

    for rank, seed in enumerate(seeds):
        anchor_x, anchor_y = pos_spatial[int(seed)]
        disp_x, disp_y = seed_display_pos[int(seed)]
        anchor_half = max(neighbor_display_size * 0.22, 6)

        anchor_rect = patches.Rectangle(
            (anchor_x - anchor_half, anchor_y - anchor_half),
            2 * anchor_half,
            2 * anchor_half,
            linewidth=2.2,
            edgecolor="#E53935",
            facecolor=(1.0, 0.0, 0.0, 0.08),
            zorder=4,
        )
        ax.add_patch(anchor_rect)

        if abs(disp_x - anchor_x) > 1 or abs(disp_y - anchor_y) > 1:
            ax.plot(
                [anchor_x, disp_x],
                [anchor_y, disp_y],
                linestyle="--",
                linewidth=1.5,
                color="#E53935",
                alpha=0.75,
                zorder=4,
            )

    neighbor_order = sorted(
        list(neighbor_set),
        key=lambda node: (
            -len(neighbor_owner_map.get(int(node), [])),
            neighbor_min_dist.get(int(node), 999999.0),
            int(node),
        ),
    )

    for neighbor in neighbor_order:
        neighbor = int(neighbor)
        h5_idx = graph.nodes[neighbor]["h5"]
        patch_image = safe_get_patch_image(patch_dir, h5_idx, neighbor_display_size)

        imagebox = OffsetImage(np.array(patch_image), zoom=1.0)
        imagebox.set_alpha(0.98)

        disp_x, disp_y = neighbor_display_pos[neighbor]
        annotation = AnnotationBbox(
            imagebox,
            (disp_x, disp_y),
            frameon=True,
            annotation_clip=False,
            bboxprops=dict(edgecolor="#111111", linewidth=2.5, boxstyle="square,pad=0.0"),
        )
        annotation.zorder = 5
        ax.add_artist(annotation)

        owner_text = format_owner_text(neighbor_owner_map.get(neighbor, []))
        if owner_text:
            ax.text(
                disp_x,
                disp_y - neighbor_display_size / 2 - 8,
                owner_text,
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#111111",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", facecolor=(1, 1, 1, 0.82), edgecolor="none"),
            )

    for rank, seed in enumerate(seeds):
        seed = int(seed)
        h5_idx = graph.nodes[seed]["h5"]
        patch_image = safe_get_patch_image(patch_dir, h5_idx, seed_display_size)

        imagebox = OffsetImage(np.array(patch_image), zoom=1.0)
        imagebox.set_alpha(0.90)

        disp_x, disp_y = seed_display_pos[seed]
        annotation = AnnotationBbox(
            imagebox,
            (disp_x, disp_y),
            frameon=True,
            annotation_clip=False,
            bboxprops=dict(edgecolor="#E53935", linewidth=5.8, boxstyle="square,pad=0.0"),
        )
        annotation.zorder = 7
        ax.add_artist(annotation)

        ax.text(
            disp_x,
            disp_y - seed_display_size / 2 - 10,
            f"Top{rank + 1}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#E53935",
            zorder=8,
            bbox=dict(boxstyle="round,pad=0.16", facecolor=(1, 1, 1, 0.80), edgecolor="none"),
        )

    add_top5_neighbor_gallery(
        fig=fig,
        seeds=seeds,
        seed_neighbors=seed_neighbors,
        patch_dir=patch_dir,
        Patch_Index=Patch_Index,
        patient_df=patient_df,
        panel_bottom=0.08,
        panel_height=0.18,
    )

    cbar_ax = fig.add_axes([0.30, 0.035, 0.40, 0.015])
    scalar_mappable = plt.cm.ScalarMappable(
        cmap=plt.cm.coolwarm,
        norm=mcolors.Normalize(vmin=0, vmax=1),
    )
    scalar_mappable.set_array([])
    cbar = plt.colorbar(scalar_mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(
        "Stratified Attention Risk Score (From Low to Extreme)",
        fontsize=16,
        fontweight="bold",
    )
    cbar.set_ticks([0, 0.25, 0.75, 1])
    cbar.set_ticklabels(["Low Risk", "Mid-Low", "Mid-High", "Extreme Risk"])

    save_path = str(save_path)
    if not save_path.lower().endswith(".pdf"):
        save_path = str(Path(save_path).with_suffix(".pdf"))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300, transparent=True)
    print(f"Vector PDF saved to: {save_path}")
    plt.close(fig)


def generate_top_patches_grid(
    patient_df,
    Patch_Index,
    patch_dir,
    tile_size=224,
    rows=3,
    cols=10,
    padding=20,
    draw_border=True,
):
    """
    Create a grid image using the highest-attention patches.
    """
    patient_df = _ensure_attention_column(patient_df)

    top_k = rows * cols
    attention = patient_df["Attention"].values.astype(np.float32)
    actual_k = min(top_k, len(attention))
    top_indices = np.argsort(attention)[-actual_k:][::-1]

    grid_width = cols * tile_size + (cols + 1) * padding
    grid_height = rows * tile_size + (rows + 1) * padding

    grid_img = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid_img)

    for index, patch_idx in enumerate(top_indices):
        row_idx = index // cols
        col_idx = index % cols
        real_h5_idx = int(Patch_Index[patch_idx])

        try:
            patch_array = patch_dir[real_h5_idx]
            patch_img = Image.fromarray(np.asarray(patch_array).astype(np.uint8))
        except Exception:
            patch_img = Image.new("RGB", (tile_size, tile_size), color=(220, 220, 220))

        if patch_img.size != (tile_size, tile_size):
            patch_img = patch_img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

        paste_x = padding + col_idx * (tile_size + padding)
        paste_y = padding + row_idx * (tile_size + padding)
        grid_img.paste(patch_img, (paste_x, paste_y))

        if draw_border:
            border_rect = [paste_x - 2, paste_y - 2, paste_x + tile_size + 1, paste_y + tile_size + 1]
            draw.rectangle(border_rect, outline=(220, 50, 50), width=3)

    return grid_img


def robust_clip_normalize(data, p_min=5, p_max=95):
    """
    Normalize data after percentile clipping.
    """
    data = np.asarray(data)
    if data.size == 0:
        return data

    v_min, v_max = np.percentile(data, [p_min, p_max])
    clipped_data = np.clip(data, v_min, v_max)
    return (clipped_data - v_min) / (v_max - v_min + 1e-8)


def _resolve_runtime_path(args, arg_name: str, env_name: str, default_value: str) -> str:
    """Resolve a runtime path from args, environment variables, or a default."""
    arg_value = getattr(args, arg_name, None)
    if arg_value not in [None, "", "None"]:
        return str(arg_value)
    return os.getenv(env_name, default_value)


def _load_slide_feature_mapping(slide_feature_path: str):
    """Load slide feature data and create a patch-name index mapping."""
    slide_feature = torch.load(slide_feature_path, map_location="cpu")

    if "Patch_name" not in slide_feature or "feature" not in slide_feature:
        raise KeyError("Slide feature file must contain 'Patch_name' and 'feature'.")

    name_to_idx = {name: idx for idx, name in enumerate(slide_feature["Patch_name"])}
    return slide_feature, name_to_idx


def _prepare_patient_information(patch_csv: pd.DataFrame, patient_attention_df: pd.DataFrame, wsi_name: str):
    """Merge metadata and model attention into a single patient dataframe."""
    patient_rows = patch_csv[patch_csv["Patient_Name"].astype(str) == str(wsi_name)].reset_index(drop=True)
    if patient_rows.empty:
        raise ValueError(f"No patient rows were found for WSI '{wsi_name}'.")

    patch_weight = patient_attention_df.copy()
    patch_weight["Attention"] = patch_weight["AttentionScore"]
    patch_weight = patch_weight[["PatchName", "Attention"]].rename(columns={"PatchName": "Patch_Name"})
    patch_weight["Attention"] = robust_clip_normalize(patch_weight["Attention"].values, p_min=1, p_max=99)

    patient_information = pd.merge(patient_rows, patch_weight, how="inner", on="Patch_Name")
    if patient_information.empty:
        raise ValueError(f"No matched patches were found for WSI '{wsi_name}' after merging.")

    patient_information["Coordinates"] = [
        str(name).split("_")[1] for name in patient_information["Patch_Name"].values
    ]
    return patient_information


def _generate_patient_visualization(
    wsi_name: str,
    patch_csv: pd.DataFrame,
    patient_attention_df: pd.DataFrame,
    patient_adj: np.ndarray,
    patch_dir,
    slide_feature,
    slide_feature_name_to_idx,
    save_root: str,
):
    """Generate a visualization PDF for one WSI."""
    patient_information = _prepare_patient_information(patch_csv, patient_attention_df, str(wsi_name))

    patch_coords = [_parse_coordinates(coord) for coord in patient_information["Coordinates"]]
    patch_index = patient_information["Unnamed: 0"].values

    wsi_original = generate_wsi_image(
        Patch_Index=patch_index,
        patch_coords=patch_coords,
        patch_dir=patch_dir,
        tile_size=224,
        thumbnail_ratio=0.1,
    )

    target_list = patient_information["Patch_Name"].values
    valid_indices = [slide_feature_name_to_idx[name] for name in target_list if name in slide_feature_name_to_idx]
    if not valid_indices:
        raise ValueError(f"No slide features were found for WSI '{wsi_name}'.")

    patch_features = slide_feature["feature"][valid_indices]
    save_file = os.path.join(save_root, f"{wsi_name}_interpretability_GNN.pdf")

    generate_wsi_spatial_network_vector_v16_Top10(
        wsi_name,
        patient_df=patient_information,
        adj_matrix=patient_adj,
        Patch_Index=patch_index,
        patch_dir=patch_dir,
        wsi_original=wsi_original,
        save_path=save_file,
        Patch_Features=patch_features,
        tile_size=224,
        thumbnail_ratio=0.1,
        Top_patch=6,
        Top_Size=7,
        Patch_Neighbor=5,
        Neighbor=4,
    )


def main():
    """Main entry point."""
    _require_dependency(Train_Data, "Train_Data")
    _require_dependency(Model_Foundation, "models.Model_Foundation")

    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    args = parser.parse_args()

    initial_checkpoint = getattr(args, "initial_checkpoint", "None")

    _, cohorts, dataset_val = Train_Data.Fundation_Cohort(args)
    args.dim = cohorts["dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model_Foundation.HGSurv(input_dim=1024).to(device)

    if initial_checkpoint not in ["None", None, ""]:
        checkpoint = torch.load(initial_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    train_data = dataset_val["train_dataset"]
    train_patient_attention, train_patient_adj = validate_epoch(model, train_data, args, Type="Train")

    wsi_h5_path = _resolve_runtime_path(args, "wsi_h5_path", "WSI_H5_PATH", "./data/CHCAMS.h5")
    patch_csv_path = _resolve_runtime_path(args, "patch_csv_path", "PATCH_CSV_PATH", "./data/CHCAMS.csv")
    slide_feature_path = _resolve_runtime_path(
        args,
        "slide_feature_path",
        "SLIDE_FEATURE_PATH",
        "./data/CHCAMS_10X_UNI.pkl",
    )
    save_root = _resolve_runtime_path(args, "save_root", "SAVE_ROOT", "./outputs/model_visualization")

    Path(save_root).mkdir(parents=True, exist_ok=True)

    if not Path(patch_csv_path).exists():
        raise FileNotFoundError(f"PATCH_CSV_PATH does not exist: {patch_csv_path}")
    if not Path(slide_feature_path).exists():
        raise FileNotFoundError(f"SLIDE_FEATURE_PATH does not exist: {slide_feature_path}")

    patch_csv = pd.read_csv(patch_csv_path)
    if "Patch_Name" not in patch_csv.columns:
        raise KeyError("The patch CSV must contain a 'Patch_Name' column.")

    patch_csv["Patient_Name"] = [str(name).split("_")[0] for name in patch_csv["Patch_Name"].values]
    slide_feature, slide_feature_name_to_idx = _load_slide_feature_mapping(slide_feature_path)

    store = None
    patch_dir = None
    if Path(wsi_h5_path).exists():
        _require_dependency(tables, "tables")
        store = tables.open_file(wsi_h5_path, mode="r")
        patch_dir = store.root.patches
    else:
        warnings.warn(
            f"WSI_H5_PATH does not exist: {wsi_h5_path}. "
            "A blank WSI overview will be used instead of real patches."
        )

    try:
        for wsi_name, patient_attention_df in train_patient_attention.items():
            patient_key = _safe_int_from_patient_name(wsi_name)
            if patient_key not in train_patient_adj:
                warnings.warn(f"Adjacency matrix is missing for WSI '{wsi_name}'. Skipped.")
                continue

            print(f"Processing WSI: {wsi_name}")
            try:
                _generate_patient_visualization(
                    wsi_name=str(wsi_name),
                    patch_csv=patch_csv,
                    patient_attention_df=patient_attention_df,
                    patient_adj=train_patient_adj[patient_key],
                    patch_dir=patch_dir,
                    slide_feature=slide_feature,
                    slide_feature_name_to_idx=slide_feature_name_to_idx,
                    save_root=save_root,
                )
            except Exception as patient_exc:
                warnings.warn(f"Failed to process WSI '{wsi_name}': {patient_exc}")
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    main()
