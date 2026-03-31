import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# Configuration
# =========================
CUTOFF = 0.5172764

FILE_NE_SUBTYPE = "xx.xlsx"
FILE_CLINICAL = "xs.xlsx"
FILE_TRAIN_PRED = "xx.csv"
FILE_VAL_PRED = "xx.csv"

OUTPUT_DIR = Path("processed_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "CHCAMS_NE_anonymized.csv"

ID_COLUMN = "PatientID"
HAZARD_COLUMN = "PreHazard"
LABEL_COLUMN = "Pre_Label"


# =========================
# Utility functions
# =========================
def anonymize_patient_id(patient_id: object, salt: str = "ne_project_salt") -> str:
    """
    Convert the original patient ID into a non-reversible anonymized ID.

    Parameters
    ----------
    patient_id : object
        Original patient identifier.
    salt : str
        Salt string used to strengthen the hash.

    Returns
    -------
    str
        An anonymized patient ID.
    """
    raw_value = f"{salt}_{str(patient_id)}"
    hashed_value = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:12]
    return f"PID_{hashed_value}"


def validate_columns(df: pd.DataFrame, required_columns: list, file_name: str) -> None:
    """
    Validate whether all required columns exist in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    required_columns : list
        Required column names.
    file_name : str
        File name for error reporting.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {file_name}: {missing_columns}")


def load_ne_clinical_data(subtype_file: str, clinical_file: str) -> pd.DataFrame:
    """
    Load and merge NE subtype data with clinical data.

    Parameters
    ----------
    subtype_file : str
        Excel file containing ID and NE subtype.
    clinical_file : str
        Excel file containing pathology number and ID.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with standardized columns:
        ['PatientID', 'NE_Type']
    """
    ne_subtype_df = pd.read_excel(subtype_file, sheet_name=0)[["ID", "NE subtype"]]
    clinical_df = pd.read_excel(clinical_file, sheet_name=0)

    validate_columns(ne_subtype_df, ["ID", "NE subtype"], subtype_file)
    validate_columns(clinical_df, ["ID", "病理号"], clinical_file)

    merged_df = pd.merge(clinical_df, ne_subtype_df, how="inner", on="ID")
    merged_df = merged_df[["病理号", "NE subtype"]].copy()
    merged_df.columns = ["PatientID", "NE_Type"]

    return merged_df


def load_prediction_data(train_file: str, val_file: str) -> pd.DataFrame:
    """
    Load and concatenate prediction datasets.

    Parameters
    ----------
    train_file : str
        Training prediction CSV file.
    val_file : str
        Validation prediction CSV file.

    Returns
    -------
    pd.DataFrame
        Concatenated prediction DataFrame.
    """
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    prediction_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    return prediction_df


def add_risk_label(df: pd.DataFrame, hazard_col: str, cutoff: float, label_col: str) -> pd.DataFrame:
    """
    Add a risk label column based on the hazard score cutoff.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    hazard_col : str
        Name of the hazard score column.
    cutoff : float
        Threshold used for classification.
    label_col : str
        Name of the output label column.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional risk label column.
    """
    df = df.copy()
    df[label_col] = np.where(df[hazard_col] > cutoff, "High_Risk", "Low_Risk")
    return df


def deidentify_dataframe(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    De-identify the DataFrame by anonymizing the patient ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    id_col : str
        Name of the patient ID column.

    Returns
    -------
    pd.DataFrame
        De-identified DataFrame.
    """
    df = df.copy()
    if id_col in df.columns:
        df[id_col] = df[id_col].apply(anonymize_patient_id)
    return df


# =========================
# Main workflow
# =========================
def main():
    # Load NE clinical information
    ne_clinical_df = load_ne_clinical_data(FILE_NE_SUBTYPE, FILE_CLINICAL)

    # Load prediction results
    pred_df = load_prediction_data(FILE_TRAIN_PRED, FILE_VAL_PRED)

    validate_columns(pred_df, [ID_COLUMN, HAZARD_COLUMN], "Prediction files")
    validate_columns(ne_clinical_df, [ID_COLUMN, "NE_Type"], "Merged NE clinical data")

    # Merge prediction data with NE subtype information
    chcams_df = pd.merge(pred_df, ne_clinical_df, how="inner", on=ID_COLUMN)

    # Add risk label
    chcams_df = add_risk_label(
        chcams_df,
        hazard_col=HAZARD_COLUMN,
        cutoff=CUTOFF,
        label_col=LABEL_COLUMN
    )

    # De-identify patient IDs
    chcams_df = deidentify_dataframe(chcams_df, id_col=ID_COLUMN)

    # Save result
    chcams_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Data processing completed successfully.")
    print(f"Output file saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
