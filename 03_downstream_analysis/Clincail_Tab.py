from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# =========================
# Configuration
# =========================
SURVIVAL_TRAIN_FILE = "xx.csv"
SURVIVAL_VAL_FILE = "xx.csv"

TREATMENT_TRAIN_FILE = "xx.csv"
TREATMENT_VAL_FILE = "xx.csv"

OUTPUT_DIR = Path("processed_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SURVIVAL_SUMMARY_FILE = OUTPUT_DIR / "survival_clinical_summary.csv"
TREATMENT_SUMMARY_FILE = OUTPUT_DIR / "treatment_summary.csv"


# =========================
# Utility functions
# =========================
def load_and_concat_csv(
    train_file: str,
    val_file: str,
    drop_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load two CSV files and concatenate them into one DataFrame.

    Parameters
    ----------
    train_file : str
        Path to the training CSV file.
    val_file : str
        Path to the validation CSV file.
    drop_columns : list, optional
        Columns to drop if they exist.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame.
    """
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    combined_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    if drop_columns:
        existing_drop_columns = [col for col in drop_columns if col in combined_df.columns]
        combined_df = combined_df.drop(columns=existing_drop_columns)

    return combined_df


def summarize_binary_column(
    df: pd.DataFrame,
    column: str,
    mapping: Dict[int, str]
) -> pd.DataFrame:
    """
    Summarize a binary or categorical numeric column by count and percentage.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to summarize.
    mapping : dict
        Mapping from raw value to human-readable label.

    Returns
    -------
    pd.DataFrame
        Summary table with count and percentage.
    """
    records = []
    total_valid = df[column].isin(mapping.keys()).sum()

    for raw_value, label in mapping.items():
        count = (df[column] == raw_value).sum()
        percentage = (count / total_valid * 100) if total_valid > 0 else np.nan
        records.append({
            "Variable": column,
            "Category": label,
            "Count": count,
            "Percentage": round(percentage, 2)
        })

    return pd.DataFrame(records)


def summarize_categorical_column(
    df: pd.DataFrame,
    column: str,
    category_order: List,
    category_labels: Optional[Dict] = None,
    include_missing: bool = True
) -> pd.DataFrame:
    """
    Summarize a categorical column by count and percentage.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to summarize.
    category_order : list
        Ordered list of categories to report.
    category_labels : dict, optional
        Mapping from raw category value to display label.
    include_missing : bool
        Whether to include missing values as a separate category.

    Returns
    -------
    pd.DataFrame
        Summary table with count and percentage.
    """
    records = []
    total_count = len(df)

    for value in category_order:
        count = (df[column] == value).sum()
        label = category_labels[value] if category_labels and value in category_labels else str(value)
        percentage = (count / total_count * 100) if total_count > 0 else np.nan
        records.append({
            "Variable": column,
            "Category": label,
            "Count": count,
            "Percentage": round(percentage, 2)
        })

    if include_missing:
        missing_count = df[column].isna().sum()
        missing_percentage = (missing_count / total_count * 100) if total_count > 0 else np.nan
        records.append({
            "Variable": column,
            "Category": "Missing",
            "Count": missing_count,
            "Percentage": round(missing_percentage, 2)
        })

    return pd.DataFrame(records)


def summarize_continuous_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Summarize a continuous variable with median, minimum, and maximum.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to summarize.

    Returns
    -------
    pd.DataFrame
        Summary table for the continuous variable.
    """
    series = df[column].dropna()

    return pd.DataFrame([{
        "Variable": column,
        "Median": round(series.median(), 2) if not series.empty else np.nan,
        "Min": round(series.min(), 2) if not series.empty else np.nan,
        "Max": round(series.max(), 2) if not series.empty else np.nan,
        "Missing": df[column].isna().sum()
    }])


def summarize_clinical_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Generate summary tables for the clinical dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical dataset.

    Returns
    -------
    dict
        Dictionary containing multiple summary DataFrames.
    """
    gender_summary = summarize_binary_column(
        df,
        column="Gender",
        mapping={1: "Male", 2: "Female"}
    )

    smoking_summary = summarize_binary_column(
        df,
        column="SmokingHistory",
        mapping={1: "Ever", 0: "Never"}
    )

    stage_summary = summarize_categorical_column(
        df,
        column="AJCCStage",
        category_order=[1, 2, 3, 4],
        category_labels={
            1: "Stage I",
            2: "Stage II",
            3: "Stage III",
            4: "Stage IV"
        },
        include_missing=True
    )

    age_summary = summarize_continuous_column(df, column="Age")

    return {
        "Gender": gender_summary,
        "SmokingHistory": smoking_summary,
        "AJCCStage": stage_summary,
        "Age": age_summary
    }


def summarize_treatment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for treatment mode.

    Parameters
    ----------
    df : pd.DataFrame
        Treatment dataset.

    Returns
    -------
    pd.DataFrame
        Summary table for treatment mode.
    """
    treatment_summary = summarize_categorical_column(
        df,
        column="TreatmentModes",
        category_order=[0, 1, 2, 3],
        category_labels={
            0: "Mode 0",
            1: "Mode 1",
            2: "Mode 2",
            3: "Mode 3"
        },
        include_missing=True
    )

    return treatment_summary


# =========================
# Main workflow
# =========================
def main():
    # Load and combine clinical survival data
    clinical_df = load_and_concat_csv(
        SURVIVAL_TRAIN_FILE,
        SURVIVAL_VAL_FILE
    )

    # Generate clinical summaries
    clinical_summaries = summarize_clinical_data(clinical_df)

    # Combine all clinical summaries into one output table
    clinical_output = pd.concat(
        [
            clinical_summaries["Gender"],
            clinical_summaries["SmokingHistory"],
            clinical_summaries["AJCCStage"]
        ],
        axis=0,
        ignore_index=True
    )

    # Save categorical clinical summary
    clinical_output.to_csv(SURVIVAL_SUMMARY_FILE, index=False, encoding="utf-8-sig")

    # Save age summary separately
    clinical_summaries["Age"].to_csv(
        OUTPUT_DIR / "age_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Load and combine treatment data
    treatment_df = load_and_concat_csv(
        TREATMENT_TRAIN_FILE,
        TREATMENT_VAL_FILE
    )

    # Generate treatment summary
    treatment_summary = summarize_treatment_data(treatment_df)
    treatment_summary.to_csv(TREATMENT_SUMMARY_FILE, index=False, encoding="utf-8-sig")

    # Print summaries
    print("Clinical categorical summary:")
    print(clinical_output)
    print("\nAge summary:")
    print(clinical_summaries["Age"])
    print("\nTreatment summary:")
    print(treatment_summary)


if __name__ == "__main__":
    main()
