import hashlib
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# =========================
# Configuration
# =========================
ID_COLUMN = "PatientID"
HAZARD_COLUMN = "PreHazard"

CLINICAL_FILES: Dict[str, Path] = {
    "CHCAMS_Train": Path("../Datasets/Clincial/CHCAMS_Train.csv"),
    "CHCAMS_Val": Path("../Datasets/Clincial/CHCAMS_Test.csv"),
    "TMUGH_Val": Path("../Datasets/Clincial/TMUGH_External_Clincial.csv"),
    "HMUCH_Val": Path("../Datasets/Clincial/HMUCH_External_Clincial.csv"),
}

RESULT_FILES: Dict[str, Path] = {
    "CHCAMS_Train": Path("Log/train_result.csv"),
    "CHCAMS_Val": Path("Log/val_result.csv"),
    "TMUGH_Val": Path("Log/TMUGH_result.csv"),
    "HMUCH_Val": Path("Log/HMUCH_result.csv"),
}

OUTPUT_DIR = Path("outputs/survival")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILES: Dict[str, Path] = {
    "CHCAMS_Train": OUTPUT_DIR / "CHCAMS_Train_anonymized.csv",
    "CHCAMS_Val": OUTPUT_DIR / "CHCAMS_Val_anonymized.csv",
    "TMUGH_Val": OUTPUT_DIR / "TMUGH_Val_anonymized.csv",
    "HMUCH_Val": OUTPUT_DIR / "HMUCH_Val_anonymized.csv",
}


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


def load_clinical_data(file_path: Path, drop_first_col: bool = False, drop_last_col: bool = False) -> pd.DataFrame:
    """
    Load clinical data and optionally drop the first and/or last column.

    Parameters
    ----------
    file_path : Path
        Path to the clinical CSV file.
    drop_first_col : bool
        Whether to drop the first column.
    drop_last_col : bool
        Whether to drop the last column.

    Returns
    -------
    pd.DataFrame
        Processed clinical DataFrame.
    """
    df = pd.read_csv(file_path)

    start_idx = 1 if drop_first_col else 0
    end_idx = -1 if drop_last_col else None
    df = df.iloc[:, start_idx:end_idx].copy()

    validate_columns(df, [ID_COLUMN], str(file_path))
    return df


def load_result_data(file_path: Path) -> pd.DataFrame:
    """
    Load prediction result data.

    Parameters
    ----------
    file_path : Path
        Path to the result CSV file.

    Returns
    -------
    pd.DataFrame
        Prediction result DataFrame.
    """
    df = pd.read_csv(file_path)
    validate_columns(df, [ID_COLUMN, HAZARD_COLUMN], str(file_path))
    return df


def fit_scaler(train_result_df: pd.DataFrame) -> MinMaxScaler:
    """
    Fit a MinMaxScaler using the training hazard scores.

    Parameters
    ----------
    train_result_df : pd.DataFrame
        Training result DataFrame.

    Returns
    -------
    MinMaxScaler
        Fitted scaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_result_df[[HAZARD_COLUMN]])
    return scaler


def transform_hazard(df: pd.DataFrame, scaler: MinMaxScaler, clip: bool = False) -> pd.DataFrame:
    """
    Transform hazard scores using the fitted scaler.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    scaler : MinMaxScaler
        Fitted scaler.
    clip : bool
        Whether to clip transformed values into [0, 1].

    Returns
    -------
    pd.DataFrame
        DataFrame with transformed hazard scores.
    """
    df = df.copy()
    transformed = scaler.transform(df[[HAZARD_COLUMN]])

    if clip:
        transformed = np.clip(transformed, 0, 1)

    df[HAZARD_COLUMN] = transformed
    return df


def anonymize_dataframe(df: pd.DataFrame, id_col: str, salt: str) -> pd.DataFrame:
    """
    De-identify the DataFrame by anonymizing the patient ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    id_col : str
        Name of the patient ID column.
    salt : str
        Salt string used for anonymization.

    Returns
    -------
    pd.DataFrame
        De-identified DataFrame.
    """
    df = df.copy()
    df[id_col] = df[id_col].apply(lambda x: anonymize_patient_id(x, salt))
    return df


def process_dataset(
    dataset_name: str,
    clinical_file: Path,
    result_file: Path,
    output_file: Path,
    scaler: MinMaxScaler,
    salt: str,
    drop_first_col: bool = False,
    drop_last_col: bool = False,
    clip: bool = True
) -> pd.DataFrame:
    """
    Process one dataset by loading clinical data, transforming hazard scores,
    merging prediction results, anonymizing patient IDs, and saving the output.

    Parameters
    ----------
    dataset_name : str
        Dataset name for logging.
    clinical_file : Path
        Path to the clinical CSV file.
    result_file : Path
        Path to the prediction result CSV file.
    output_file : Path
        Path to the output CSV file.
    scaler : MinMaxScaler
        Fitted scaler based on the training cohort.
    salt : str
        Salt string used for anonymization.
    drop_first_col : bool
        Whether to drop the first column from clinical data.
    drop_last_col : bool
        Whether to drop the last column from clinical data.
    clip : bool
        Whether to clip transformed hazard scores into [0, 1].

    Returns
    -------
    pd.DataFrame
        Processed merged DataFrame.
    """
    print(f"Processing {dataset_name} ...")

    clinical_df = load_clinical_data(
        clinical_file,
        drop_first_col=drop_first_col,
        drop_last_col=drop_last_col
    )
    result_df = load_result_data(result_file)
    result_df = transform_hazard(result_df, scaler=scaler, clip=clip)

    merged_df = pd.merge(clinical_df, result_df, how="inner", on=ID_COLUMN)
    merged_df = anonymize_dataframe(merged_df, id_col=ID_COLUMN, salt=salt)

    merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    return merged_df


# =========================
# Main workflow
# =========================
def main() -> None:
    salt = get_hash_salt()

    train_result_df = load_result_data(RESULT_FILES["CHCAMS_Train"])
    scaler = fit_scaler(train_result_df)

    # Process training dataset
    chcams_train_df = process_dataset(
        dataset_name="CHCAMS_Train",
        clinical_file=CLINICAL_FILES["CHCAMS_Train"],
        result_file=RESULT_FILES["CHCAMS_Train"],
        output_file=OUTPUT_FILES["CHCAMS_Train"],
        scaler=scaler,
        salt=salt,
        drop_first_col=True,
        drop_last_col=True,
        clip=False
    )

    # Process internal validation dataset
    chcams_val_df = process_dataset(
        dataset_name="CHCAMS_Val",
        clinical_file=CLINICAL_FILES["CHCAMS_Val"],
        result_file=RESULT_FILES["CHCAMS_Val"],
        output_file=OUTPUT_FILES["CHCAMS_Val"],
        scaler=scaler,
        salt=salt,
        drop_first_col=True,
        drop_last_col=True,
        clip=True
    )

    # Process external validation datasets
    tmugh_val_df = process_dataset(
        dataset_name="TMUGH_Val",
        clinical_file=CLINICAL_FILES["TMUGH_Val"],
        result_file=RESULT_FILES["TMUGH_Val"],
        output_file=OUTPUT_FILES["TMUGH_Val"],
        scaler=scaler,
        salt=salt,
        drop_first_col=False,
        drop_last_col=False,
        clip=True
    )

    hmuch_val_df = process_dataset(
        dataset_name="HMUCH_Val",
        clinical_file=CLINICAL_FILES["HMUCH_Val"],
        result_file=RESULT_FILES["HMUCH_Val"],
        output_file=OUTPUT_FILES["HMUCH_Val"],
        scaler=scaler,
        salt=salt,
        drop_first_col=False,
        drop_last_col=False,
        clip=True
    )

    print("All datasets processed successfully.")


if __name__ == "__main__":
    main()
