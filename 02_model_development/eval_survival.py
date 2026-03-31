"""
Survival Analysis — Multi-Cohort Inference & Evaluation
=======================================================
This script loads a pre-trained WSI-based survival prediction model and
evaluates it across multiple cohorts (train, validation, and two external
test sets). Per-patient hazard predictions are exported as CSV files, and
performance metrics (C-Index, p-value, time-dependent AUC) are reported.

Interfaces (unchanged):
    - Survival.CIndex, Survival.cox_log_rank, Survival.coxph_log_rank
    - Train_Data.Fundation_Cohort
    - Model_Foundation.HGSurv
    - dataset.null_collate, dataset.DataLoader
    - utils.yaml_config_hook, utils.save_model
"""

# ============================================================
# Imports
# ============================================================
import os
import random
import warnings
import argparse

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, roc_curve
from sksurv.metrics import cumulative_dynamic_auc

import Survival
import Train_Data
from dataset import null_collate
from models import Model_Foundation

warnings.filterwarnings("ignore")


# ============================================================
# Configuration & Reproducibility
# ============================================================

def yaml_config_hook(config_file):
    """Load a YAML configuration file and return a dict."""
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Batch Initialization (move tensors to GPU)
# ============================================================

def initialize(batch: dict) -> dict:
    """Transfer all relevant tensors in a batch to CUDA device."""
    for key in ("OS", "OSState", "DFS", "DFSState", "WSI_feature"):
        batch[key] = batch[key].cuda()
    return batch


# ============================================================
# Validation Utilities
# ============================================================

def Model_Val(net, dataset_, args):
    """
    Run full-dataset inference and compute C-Index / p-value.

    Args:
        net:       The survival prediction model (already on GPU).
        dataset_:  A PyTorch Dataset object.
        args:      Parsed argument namespace (must contain `workers`).

    Returns:
        MyResult:  pd.DataFrame with per-patient predictions.
        CIndex:    Concordance index.
        Pvalues:   Log-rank p-value.
    """
    loader = DataLoader(
        dataset_,
        shuffle=False,
        batch_size=1,
        drop_last=False,
        num_workers=int(args.workers),
        pin_memory=False,
        collate_fn=null_collate,
    )
    net.eval()
    net.output_type = ["inference"]

    results = []
    for _, batch in enumerate(loader):
        batch = initialize(batch)
        with torch.no_grad():
            output = net(batch)
            result = {
                "Patient": batch["patient_name"],
                "OS": batch["OS"].detach().cpu().numpy().squeeze(),
                "OSState": batch["OSState"].detach().cpu().numpy().squeeze(),
                "DFS": batch["DFS"].detach().cpu().numpy().squeeze(),
                "DFSState": batch["DFSState"].detach().cpu().numpy().squeeze(),
                "hazards": output["hazards"].detach().cpu().numpy().squeeze(),
            }
            results.append(pd.DataFrame(result))

    MyResult = pd.concat(results, ignore_index=True)
    CIndex = Survival.CIndex(
        hazards=MyResult["hazards"],
        labels=MyResult["OSState"],
        survtime_all=MyResult["OS"],
    )
    Pvalues = Survival.cox_log_rank(
        hazardsdata=MyResult["hazards"],
        labels=MyResult["OSState"],
        survtime_all=MyResult["OS"],
    )
    return MyResult, CIndex, Pvalues


def best_threshold_youden(y_true, score, pos_label=1, auto_flip=True):
    """
    Find the optimal binary classification threshold using the Youden J statistic.

    Args:
        y_true:     Ground truth binary labels.
        score:      Continuous risk / prediction scores.
        pos_label:  The label considered as positive (default 1).
        auto_flip:  If True, automatically flip score direction when AUC < 0.5.

    Returns:
        A dict containing: best_threshold, youden_J, sensitivity_TPR,
        specificity_TNR, auc, y_pred, used_score.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    score = np.asarray(score).astype(float).ravel()

    # 1) Remove NaN / Inf entries to keep arrays aligned
    valid_mask = np.isfinite(score) & np.isfinite(y_true)
    y_true = y_true[valid_mask]
    score = score[valid_mask]

    # 2) Optionally auto-flip score direction (higher hazard may map to negative class)
    if auto_flip:
        auc = roc_auc_score(y_true, score)
        if auc < 0.5:
            score = -score
            auc = roc_auc_score(y_true, score)
    else:
        auc = roc_auc_score(y_true, score)

    # 3) Compute ROC curve and Youden J statistic
    fpr, tpr, thresholds = roc_curve(y_true, score, pos_label=pos_label)
    J = tpr - fpr
    best_idx = np.argmax(J)

    best_thr = thresholds[best_idx]
    best_tpr = tpr[best_idx]
    best_spec = 1.0 - fpr[best_idx]

    # 4) Generate binary predictions at the optimal threshold
    y_pred = (score >= best_thr).astype(int)

    return {
        "best_threshold": float(best_thr),
        "youden_J": float(J[best_idx]),
        "sensitivity_TPR": float(best_tpr),
        "specificity_TNR": float(best_spec),
        "auc": float(auc),
        "y_pred": y_pred,
        "used_score": score,  # flipped scores if auto_flip was triggered
    }


def validate_epoch(model, datas, args, Type=None):
    """
    Validate one epoch: compute C-Index, p-value, time-dependent AUC, and
    per-patient hazard predictions.

    Args:
        model:  The survival model (already on GPU).
        datas:  A PyTorch Dataset for the target cohort.
        args:   Parsed argument namespace (must contain `workers`).
        Type:   Optional string label for logging purposes.

    Returns:
        Cindex:         Concordance index.
        p_value:        Cox log-rank p-value.
        auc_dict:       Dict with keys like 'AUC_1yr', 'AUC_3yr', 'AUC_5yr'.
        Patient_Result: pd.DataFrame with columns ['PatientID', 'PreHazard'].
    """
    model.eval()

    patient_namelist = []
    all_survival_time = []
    all_event_status = []
    all_hazards = []

    loader = DataLoader(
        datas,
        shuffle=False,
        batch_size=1,
        drop_last=False,
        num_workers=int(args.workers),
        pin_memory=False,
        collate_fn=null_collate,
    )

    with torch.no_grad():
        for _, batch in enumerate(loader):
            batch = initialize(batch)
            x_wsi = batch["WSI_feature"]
            patient_name = batch["patient_name"].numpy()

            # Forward pass — use the `model` parameter (not a global variable)
            hazards, attn_weights, slide_emb = model(x=x_wsi)

            patient_namelist.extend(patient_name)
            all_hazards.extend(hazards.cpu().numpy())
            all_survival_time.extend(batch["OS"].cpu().numpy())
            all_event_status.extend(batch["OSState"].cpu().numpy())

    # Flatten collected arrays
    all_hazards = np.concatenate(all_hazards)

    # Build per-patient result DataFrame
    patient_ids = np.array(patient_namelist)
    Patient_Result = pd.DataFrame({
        "PatientID": patient_ids,
        "PreHazard": all_hazards,
    })

    # Remove samples with NaN values before computing metrics
    all_survival_time = np.array(all_survival_time)
    all_event_status = np.array(all_event_status)
    nan_mask = np.isnan(all_survival_time) | np.isnan(all_event_status)
    event_clean = all_event_status[~nan_mask]
    hazards_clean = all_hazards[~nan_mask]
    time_clean = all_survival_time[~nan_mask]

    # --- 1. Compute C-Index and Cox log-rank test ---
    Cindex, p_value, hr, hr_lower, hr_upper = Survival.coxph_log_rank(
        survtime_all=time_clean,
        labels=event_clean,
        covariates=hazards_clean,
    )

    # --- 2. Compute time-dependent AUC at 1 / 3 / 5 years ---
    # Build structured array required by scikit-survival
    dtype = [("event", bool), ("time", float)]
    structured_y = np.array(
        [(bool(e), float(t)) for e, t in zip(event_clean, time_clean)],
        dtype=dtype,
    )

    # Define evaluation time points (unit: months)
    time_points = np.array([12, 36, 60])

    # Safety check: drop time points beyond the maximum follow-up
    max_follow_up = np.max(time_clean)
    valid_time_points = time_points[time_points < max_follow_up]

    auc_dict = {}
    if len(valid_time_points) > 0:
        # Note: ideally, survival_train should come from the training set.
        # Here we use the validation set itself as an approximation.
        auc_values, mean_auc = cumulative_dynamic_auc(
            structured_y,
            structured_y,
            hazards_clean,
            valid_time_points,
        )
        for i, tp in enumerate(valid_time_points):
            year_label = int(tp / 12)
            auc_dict[f"AUC_{year_label}yr"] = auc_values[i]

    return Cindex, p_value, auc_dict, Patient_Result


# ============================================================
# Main — Inference & Export
# ============================================================

if __name__ == "__main__":

    # --- Reproducibility ---
    set_seed(42)

    # --- Parse hyperparameters from YAML config ---
    parser = argparse.ArgumentParser(description="WSI Survival Prediction Inference")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    initial_checkpoint = args.initial_checkpoint

    # --- Prepare datasets ---
    train_loader, Cohorts, dataset_val = Train_Data.Fundation_Cohort(args)
    args.dim = Cohorts["dim"]

    # --- Build model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Model_Foundation.HGSurv(input_dim=1024).to(device)

    # --- Load checkpoint (if provided) ---
    if initial_checkpoint != "None":
        ckpt = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        net.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # --- Define cohort configurations ---
    # Maps: dataset dict key -> (display name, output CSV filename)
    # Dataset dict keys are kept as-is to preserve upstream interface.
    output_dir = "../../Analysis/Log"
    os.makedirs(output_dir, exist_ok=True)

    cohort_configs = [
        ("train_dataset",  "Train",      "train_result.csv"),
        ("val_dataset",    "Val",        "val_result.csv"),
        ("TMUGH_dataset",  "External_1", "external_1_result.csv"),
        ("HMUCH_dataset",  "External_2", "external_2_result.csv"),
    ]

    # --- Run inference on each cohort ---
    print("=" * 60)
    print(f"{'Cohort':<14} | {'C-Index':<8} | {'P-Value':<10} | {'AUC (1y/3y/5y)'}")
    print("=" * 60)

    for ds_key, display_name, csv_filename in cohort_configs:
        print(f"[{display_name}] Evaluating ...")

        cohort_data = dataset_val[ds_key]
        c_idx, p_val, auc_dict, patient_result = validate_epoch(
            net, cohort_data, args, Type=display_name
        )

        # Export per-patient predictions
        csv_path = os.path.join(output_dir, csv_filename)
        patient_result.to_csv(csv_path, index=None)

        # Format AUC string
        auc_parts = []
        for key in ("AUC_1yr", "AUC_3yr", "AUC_5yr"):
            auc_parts.append(f"{auc_dict[key]:.3f}" if key in auc_dict else "N/A")
        auc_str = "/".join(auc_parts)

        print(f"{'':>2}{display_name:<12} | {c_idx:<8.3f} | {p_val:<10.4f} | {auc_str}")
        print(f"{'':>2}Results saved to: {csv_path}")
        print("-" * 60)

    print("All cohorts evaluated successfully.")
