import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# Configuration
# =========================
CUTOFF = 0.5172764
RISK_COLUMN = "PreHazard"
LABEL_COLUMN = "Pre_Label"
ID_COLUMN = "PatientID"

INPUT_FILES = {
    "CHCAMS_Train": "CHCAMS_Train.csv",
    "CHCAMS_Val": "CHCAMS_Val.csv",
    "HMUCH_Val": "HMUCH_Val.csv",
    "TMUGH_Val": "TMUGH_Val.csv",
}

OUTPUT_DIR = Path("processed_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FINAL_OUTPUT_FILE = OUTPUT_DIR / "Stage_anonymized.csv"

FINAL_COLUMNS = [
    "PatientID",
    "Gender",
    "Age",
    "SmokingHistory",
    "DFSState",
    "DFS",
    "OSState",
    "OS",
    "AJCCStage",
    "PreHazard",
    "Pre_Label",
]


# =========================
# Utility functions
# =========================
def anonymize_patient_id(patient_id: object, salt: str = "clinical_study_salt") -> str:
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
        raise ValueError(
            f"Missing columns in {file_name}: {missing_columns}"
        )


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


def process_file(file_path: str, output_dir: Path) -> pd.DataFrame:
    """
    Read, validate, label, de-identify, and save a single CSV file.

    Parameters
    ----------
    file_path : str
        Path to the input CSV file.
    output_dir : Path
        Directory for saving processed files.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame.
    """
    df = pd.read_csv(file_path)

    validate_columns(
        df,
        required_columns=[ID_COLUMN, RISK_COLUMN],
        file_name=file_path
    )

    df = add_risk_label(df, hazard_col=RISK_COLUMN, cutoff=CUTOFF, label_col=LABEL_COLUMN)
    df = deidentify_dataframe(df, id_col=ID_COLUMN)

    output_file = output_dir / f"{Path(file_path).stem}_anonymized.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    return df


# =========================
# Main workflow
# =========================
def main():
    processed_dfs = []

    for dataset_name, file_path in INPUT_FILES.items():
        print(f"Processing: {dataset_name} -> {file_path}")
        processed_df = process_file(file_path, OUTPUT_DIR)
        processed_dfs.append(processed_df)

    all_data = pd.concat(processed_dfs, axis=0, ignore_index=True)

    validate_columns(
        all_data,
        required_columns=FINAL_COLUMNS,
        file_name="Concatenated dataset"
    )

    all_data = all_data[FINAL_COLUMNS]
    all_data.to_csv(FINAL_OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"All files processed successfully.")
    print(f"Combined anonymized file saved to: {FINAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
