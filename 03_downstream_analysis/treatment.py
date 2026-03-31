import hashlib
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# =========================
# Configuration
# =========================
CUTOFF = 0.5172764
ID_COLUMN = "PatientID"
HAZARD_COLUMN = "PreHazard"
LABEL_COLUMN = "Pre_Label"
TREATMENT_COLUMN = "TreatmentModes"

INPUT_TREATMENT_FILE = Path("data/treatment/Allcohort_TreatmentModes.csv")
INPUT_SURVIVAL_DIR = Path("data/survival")
OUTPUT_DIR = Path("outputs/treatment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS: Dict[str, str] = {
    "CHCAMS_Train": "CHCAMS_Train.csv",
    "CHCAMS_Val": "CHCAMS_Val.csv",
    "TMUGH_Val": "TMUGH_Val.csv",
    "HMUCH_Val": "HMUCH_Val.csv",
}

OUTPUT_FILES: Dict[str, str] = {
    "CHCAMS_Train": "CHCAMS_Train_T_anonymized.csv",
    "CHCAMS_Val": "CHCAMS_Val_T_anonymized.csv",
    "TMUGH_Val": "TMUGH_Val_T_anonymized.csv",
    "HMUCH_Val": "HMUCH_Val_T_anonymized.csv",
}

COMBINED_OUTPUT_FILE = OUTPUT_DIR / "Treatment_T_anonymized.csv"


# =========================
# Utility functions
# =========================
def get_hash_salt() -> str:
    """
    Read the anonymization salt from an environment variable.

    Returns
    -------
    str
        Salt string used for hashing.

    Raises
    ------
    ValueError
        If the environment variable is missing.
    """
    salt = os.getenv("PATIENT_ID_SALT")
    if not salt:
        raise ValueError(
            "Environment variable 'PATIENT_ID_SALT' is not set. "
            "Please define it before running the script."
        )
    return salt


def anonymize_patient_id(patient_id: object, salt: str) -> str:
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
    hashed_value = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:16]
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


def load_treatment_information(file_path: Path, salt: str) -> pd.DataFrame:
    """
    Load treatment information and anonymize patient IDs.

    Parameters
    ----------
    file_path : Path
        Path to the treatment CSV file.
    salt : str
        Salt string used for ID anonymization.

    Returns
    -------
    pd.DataFrame
        Cleaned and anonymized treatment DataFrame.
    """
    df = pd.read_csv(file_path)
    validate_columns(df, [ID_COLUMN, TREATMENT_COLUMN], str(file_path))

    df = df[[ID_COLUMN, TREATMENT_COLUMN]].copy()
    df[ID_COLUMN] = df[ID_COLUMN].apply(lambda x: anonymize_patient_id(x, salt))

    return df


def add_risk_label(df: pd.DataFrame, hazard_col: str, cutoff: float, label_col: str) -> pd.DataFrame:
    """
    Add a risk label column based on the hazard score cutoff.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    hazard_col : str
        Hazard score column name.
    cutoff : float
        Classification threshold.
    label_col : str
        Output label column name.

    Returns
    -------
    pd.DataFrame
        DataFrame with risk label column added.
    """
    df = df.copy()
    df[label_col] = np.where(df[hazard_col] > cutoff, "High_Risk", "Low_Risk")
    return df


def process_survival_dataset(
    survival_file: Path,
    treatment_df: pd.DataFrame,
    output_file: Path,
    salt: str
) -> pd.DataFrame:
    """
    Process a survival dataset by adding risk labels, anonymizing IDs,
    merging treatment information, and saving the result.

    Parameters
    ----------
    survival_file : Path
        Path to the survival CSV file.
    treatment_df : pd.DataFrame
        Treatment information DataFrame with anonymized patient IDs.
    output_file : Path
        Output file path.
    salt : str
        Salt string used for ID anonymization.

    Returns
    -------
    pd.DataFrame
        Processed and merged DataFrame.
    """
    df = pd.read_csv(survival_file)
    validate_columns(df, [ID_COLUMN, HAZARD_COLUMN], str(survival_file))

    df = add_risk_label(df, hazard_col=HAZARD_COLUMN, cutoff=CUTOFF, label_col=LABEL_COLUMN)
    df[ID_COLUMN] = df[ID_COLUMN].apply(lambda x: anonymize_patient_id(x, salt))

    merged_df = pd.merge(df, treatment_df, how="inner", on=ID_COLUMN)
    merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    return merged_df


# =========================
# Main workflow
# =========================
def main() -> None:
    salt = get_hash_salt()

    treatment_df = load_treatment_information(INPUT_TREATMENT_FILE, salt)

    processed_datasets = []

    for dataset_name, file_name in DATASETS.items():
        input_file = INPUT_SURVIVAL_DIR / file_name
        output_file = OUTPUT_DIR / OUTPUT_FILES[dataset_name]

        print(f"Processing {dataset_name} ...")
        processed_df = process_survival_dataset(
            survival_file=input_file,
            treatment_df=treatment_df,
            output_file=output_file,
            salt=salt
        )
        processed_datasets.append(processed_df)

    combined_df = pd.concat(processed_datasets, axis=0, ignore_index=True)
    combined_df.to_csv(COMBINED_OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("All datasets processed successfully.")
    print(f"Combined output saved to: {COMBINED_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
