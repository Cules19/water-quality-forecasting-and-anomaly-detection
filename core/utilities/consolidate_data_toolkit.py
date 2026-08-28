#!/usr/bin/env python3
"""
Site Data Format Inspector and Consolidator

This script consolidates data files for the same site ID by:
1. Finding all data files matching the site ID pattern
2. Reading and parsing each file, carefully handling data types
3. Converting timestamp columns to datetime64 format for proper handling
4. Making the first column the timestamp in datetime64 format
5. Consolidating all data into a single dataframe with proper column alignment
6. Removing duplicate entries and sorting by timestamp
7. Performing data quality checks and logging missing data statistics
8. Saving the consolidated data to data/clean/ directory

The script ensures proper data handling by:
- Converting timestamp columns to pandas datetime64 format
- Placing timestamp column first in the output
- Maintaining data integrity during consolidation
- Providing detailed logging of the consolidation process

Output format: <SiteID>_<param_codes>_<startdate>_<enddate>.csv

Check mode: Validates consolidated files for data quality issues including:
- Mixed data types in numeric columns
- Unclean/unclear headers
- String values in numeric columns
- Data type inconsistencies
"""

import argparse
import glob
import json
import logging
import os
import re
from datetime import datetime

import pandas as pd


def check_consolidated_file(file_path):
    """
    Check a consolidated CSV file for data quality issues

    Returns dict with issues found:
    - mixed_types: columns with mixed data types
    - string_in_numeric: numeric columns containing string values
    - unclear_headers: headers that don't follow standard patterns
    - type_errors: potential type conversion errors
    """
    issues = {
        "mixed_types": [],
        "string_in_numeric": [],
        "unclear_headers": [],
        "type_errors": [],
        "duplicate_headers": [],
        "empty_headers": [],
    }

    try:
        # Read the file
        df = pd.read_csv(file_path, parse_dates=["TIMESTAMP"])
        print(f"🔍 Checking file: {os.path.basename(file_path)}")
        logging.info(f"Checking consolidated file: {file_path}")

        # Check timestamp column format
        if "TIMESTAMP" in df.columns:
            timestamp_dtype = df["TIMESTAMP"].dtype
            print(f"📅 TIMESTAMP column type: {timestamp_dtype}")
            if "datetime64" in str(timestamp_dtype):
                print("✅ TIMESTAMP is properly formatted as datetime64")
            else:
                print(f"⚠️  TIMESTAMP is not datetime64 format: {timestamp_dtype}")
                issues["type_errors"].append(
                    {"column": "TIMESTAMP", "error": f"Expected datetime64, got {timestamp_dtype}"}
                )

        # Check for duplicate headers
        if df.columns.duplicated().any():
            duplicates = df.columns[df.columns.duplicated()].tolist()
            issues["duplicate_headers"] = duplicates
            print(f"❌ Duplicate headers found: {duplicates}")

        # Check for empty/unnamed headers
        empty_headers = [
            col for col in df.columns if str(col).strip() == "" or "Unnamed:" in str(col)
        ]
        if empty_headers:
            issues["empty_headers"] = empty_headers
            print(f"❌ Empty/unnamed headers found: {empty_headers}")

        # Check header clarity (should be parameter codes or standard names)
        expected_patterns = [
            "TIMESTAMP",
            "Site_ID",
            "site_no",  # Standard columns
            r"^\d{5}$",  # 5-digit parameter codes
            r"^TEMP \[deg C\]$",
            r"^Turbidity \[NTU\]\*$",
            r"^fDOM \[RFU\]\*$",  # Standard units
            r"^Conductance \[uS/cm\]\*$",  # More standard units
        ]

        for col in df.columns:
            col_str = str(col)
            is_clear = any(
                re.match(pattern, col_str) or pattern == col_str for pattern in expected_patterns
            )
            if not is_clear and col_str not in ["TIMESTAMP", "Site_ID"]:
                # Check if it's a reasonable parameter name
                if (
                    len(col_str) > 50
                    or any(char in col_str for char in ["/", "\\", "?", "*"])
                    or col_str.count(" ") > 5
                ):
                    issues["unclear_headers"].append(col_str)

        # Check each column for data type issues
        for col in df.columns:
            if col in ["TIMESTAMP", "Site_ID"]:
                continue

            # Get non-null values
            non_null_values = df[col].dropna()
            if len(non_null_values) == 0:
                continue

            # Check for mixed types in what should be numeric columns
            numeric_count = 0
            string_count = 0
            string_examples = []

            for value in non_null_values.head(100):  # Sample first 100 non-null values
                try:
                    # Try to convert to float
                    float(value)
                    numeric_count += 1
                except (ValueError, TypeError):
                    string_count += 1
                    if len(string_examples) < 5:  # Keep first 5 examples
                        string_examples.append(str(value))

            # If we have both numeric and string values, it's mixed
            if numeric_count > 0 and string_count > 0:
                issues["mixed_types"].append(
                    {
                        "column": col,
                        "numeric_count": numeric_count,
                        "string_count": string_count,
                        "string_examples": string_examples,
                    }
                )
                print(
                    f"❌ Mixed types in column '{col}': {numeric_count} numeric, {string_count} string values"
                )
                print(f"   String examples: {string_examples}")

            # If this looks like a numeric column but has strings, flag it
            elif string_count > 0 and (
                col.isdigit()
                or any(
                    unit in col.lower()
                    for unit in ["temp", "ph", "turbidity", "conductance", "discharge", "gage"]
                )
            ):
                issues["string_in_numeric"].append(
                    {
                        "column": col,
                        "string_count": string_count,
                        "string_examples": string_examples,
                    }
                )
                print(
                    f"⚠️  Numeric column '{col}' contains {string_count} string values: {string_examples}"
                )

        # Test for specific type errors that cause pandas operations to fail
        for col in df.columns:
            if col in ["TIMESTAMP", "Site_ID"]:
                continue

            try:
                # Try basic operations that would fail with mixed types
                series = df[col].dropna()
                if len(series) > 0:
                    # Try arithmetic operations
                    _ = series / 1  # This would fail with strings
                    _ = series + 1  # This would also fail
            except TypeError as e:
                issues["type_errors"].append({"column": col, "error": str(e)})
                print(f"❌ Type error in column '{col}': {e}")
            except Exception as e:
                # Other potential issues
                if "unsupported operand type" in str(e).lower():
                    issues["type_errors"].append({"column": col, "error": str(e)})
                    print(f"❌ Operation error in column '{col}': {e}")

        # Summary
        total_issues = sum(len(v) for v in issues.values())
        if total_issues == 0:
            print("✅ No data quality issues found!")
        else:
            print(f"❌ Found {total_issues} data quality issues")

        return issues

    except Exception as e:
        print(f"❌ Error checking file: {e}")
        logging.error(f"Error checking file {file_path}: {e}")
        return issues


def setup_logging():
    """Set up logging configuration"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"data_consolidator_{timestamp}.log")

    # Configure logging - only file logging, minimal console output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file)],
    )

    return log_file


def find_site_files(site_id, data_dir="data/raw"):
    """Find all data files for a given site ID"""
    pattern = f"{data_dir}/*{site_id}*.csv"
    files = glob.glob(pattern)
    return files


def extract_parameters_from_filename(filename):
    """Extract parameter codes from filename pattern: siteID_param1_param2_..._startdate_enddate.csv"""
    basename = os.path.basename(filename).replace("usgs_", "")  # Remove usgs_ prefix if present
    name_without_ext = basename.replace(".csv", "")
    parts = name_without_ext.split("_")

    # Find the date parts (8-digit numbers that look like dates)
    date_indices = []
    for i, part in enumerate(parts):
        if len(part) == 8 and part.isdigit():
            # Check if it looks like a date (year should be reasonable)
            if part.startswith("20"):
                date_indices.append(i)

    if len(date_indices) >= 2:
        # Parameters are between site_id (index 0) and first date
        # Skip the site_id (first part) and get parameters only
        param_codes = parts[1 : date_indices[0]]
        return param_codes
    else:
        # Fallback: assume last 2 parts are dates, skip site_id
        param_codes = parts[1:-2]
        return param_codes


_TIMESTAMP_KEYWORDS = ("timestamp", "datetime", "date", "time")
_BAD_COLUMNS = ("10s", "14n")
_SITE_ID_LIKE = ("Site_ID", "site_no")

# Maps canonical USGS parameter code -> known alternative column names found in raw files.
_PARAM_NAME_TO_CODE = {
    "00010": ["TEMP [deg C]", "Temperature, water, degrees Celsius"],
    "00045": ["00045", "Precipitation, total, inches"],
    "00060": ["Discharge, cubic feet per second", "Streamflow, ft³/s"],
    "00065": ["Gage height, feet", "Gage height, ft"],
    "00095": [
        "Conductance [uS/cm]*",
        "Specific conductance, water, unfiltered, microsiemens per centimeter",
    ],
    "00300": ["00300", "Dissolved oxygen, water, unfiltered, milligrams per liter"],
    "00301": ["00301", "Dissolved oxygen, water, unfiltered, percent of saturation"],
    "00400": ["00400", "pH, water, unfiltered, field, standard units"],
    "32295": ["fDOM [RFU]*", "32295"],
    "32315": ["32315", "Chlorophyll a, phytoplankton, fluorometric method"],
    "32321": ["32321", "Phycocyanin, phytoplankton, fluorometric method"],
    "62619": ["62619", "Blue-green algae, phycocyanin, fluorometer"],
    "62620": ["62620", "Blue-green algae, phycocyanin, fluorometer"],
    "63160": ["63160", "Stream velocity, mean, ft/s"],
    "63680": ["63680", "Turbidity, water, unfiltered, monochrome near infra-red LED light"],
    "70232": ["70232", "Total phosphorus, water, unfiltered, milligrams per liter"],
    "72254": ["72254", "Blue-green algae, phycocyanin, in-situ fluorometer"],
    "72255": ["72255", "Blue-green algae, phycocyanin, in-situ fluorometer"],
    "99133": [
        "Turbidity [NTU]*",
        "99133",
        "Turbidity, water, unfiltered, nephelometric turbidity units",
    ],
    "99404": ["99404", "Turbidity, water, unfiltered, nephelometric turbidity units"],
}


def _find_timestamp_column(df):
    for col in df.columns:
        if any(keyword in col.lower() for keyword in _TIMESTAMP_KEYWORDS):
            return col
    return None


def _build_column_mapping(df_columns, timestamp_col, site_id):
    column_mapping = {
        "Site_ID": "Site_ID",
        "site_no": "Site_ID",
        timestamp_col: "TIMESTAMP",
    }
    for param_code, possible_names in _PARAM_NAME_TO_CODE.items():
        for col_name in df_columns:
            if col_name in possible_names or col_name == param_code:
                column_mapping[col_name] = param_code
                break
    for col_name in df_columns:
        if col_name == site_id:
            logging.warning(f"Found site ID {site_id} as column name, removing it")
        elif col_name in _BAD_COLUMNS:
            logging.warning(f"Found unknown column {col_name}, removing it")
    return column_mapping


def _consolidate_site_id_columns(df):
    site_id_like_cols = [col for col in df.columns if col in _SITE_ID_LIKE]
    if len(site_id_like_cols) > 1:
        combined_site_id = df[site_id_like_cols].bfill(axis=1).iloc[:, 0]
        df = df.drop(columns=site_id_like_cols)
        df["Site_ID"] = combined_site_id
    elif len(site_id_like_cols) == 1 and site_id_like_cols[0] != "Site_ID":
        df = df.rename(columns={site_id_like_cols[0]: "Site_ID"})
    return df


def _drop_problem_columns(df, site_id):
    cols_to_drop = [col for col in df.columns if col == site_id or col in _BAD_COLUMNS]
    quality_flag_cols = [
        col for col in df.columns if col.startswith("10s.") or col.startswith("14n.")
    ]
    cols_to_drop.extend(quality_flag_cols)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logging.info(f"Dropped problematic columns: {cols_to_drop}")
    return df


def _coerce_numeric_columns(df):
    for col in df.columns:
        if col in ("TIMESTAMP", "Site_ID"):
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception as e:
            logging.warning(f"Could not clean column {col}: {e}")
    return df


def _process_single_file(file_path, site_id):
    """Read and normalise one raw CSV. Returns the cleaned DataFrame or None on failure."""
    try:
        df = pd.read_csv(file_path)
        timestamp_col = _find_timestamp_column(df)
        if timestamp_col is None:
            logging.error(f"No timestamp column found in {os.path.basename(file_path)}")
            return None
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        except Exception as e:
            logging.error(f"Failed to convert timestamp in {os.path.basename(file_path)}: {e}")
            return None

        column_mapping = _build_column_mapping(df.columns, timestamp_col, site_id)
        df = df.rename(columns=column_mapping)
        df = _consolidate_site_id_columns(df)
        df = _drop_problem_columns(df, site_id)
        df = _coerce_numeric_columns(df)
        return df
    except Exception as e:
        logging.error(f"Error processing {os.path.basename(file_path)}: {e}")
        return None


def _compute_final_column_order(dataframes):
    all_columns = set()
    for df in dataframes:
        all_columns.update(df.columns)
    param_columns = sorted([col for col in all_columns if col not in ("TIMESTAMP", "Site_ID")])
    return ["TIMESTAMP"] + param_columns + ["Site_ID"]


def _align_dataframes(dataframes, final_columns):
    processed_dfs = []
    for i, df in enumerate(dataframes):
        if df.empty:
            continue
        if df.columns.duplicated().any():
            logging.warning(f"Duplicate columns found in dataframe {i}, removing duplicates")
            df = df.loc[:, ~df.columns.duplicated()]
        for col in final_columns:
            if col not in df.columns:
                df[col] = None
        df = df[final_columns]
        processed_dfs.append(df)
    return processed_dfs


def _normalise_timestamp_and_sort(consolidated):
    if "TIMESTAMP" in consolidated.columns:
        consolidated["TIMESTAMP"] = pd.to_datetime(
            consolidated["TIMESTAMP"], utc=True
        ).dt.tz_convert(None)
        consolidated["TIMESTAMP"] = consolidated["TIMESTAMP"].astype("datetime64[ns]")
        consolidated = consolidated.sort_values("TIMESTAMP")
    return consolidated


def _log_missing_data(consolidated):
    for col in consolidated.columns:
        if col in ("datetime", "site_id"):
            continue
        try:
            missing_count = consolidated[col].isna().sum()
            missing_pct = (missing_count / len(consolidated)) * 100
            if isinstance(missing_pct, (int, float)):
                logging.info(f"Missing data - {col}: {missing_count} ({missing_pct:.1f}%)")
            else:
                logging.info(f"Missing data - {col}: {missing_count}")
        except Exception as e:
            logging.warning(f"Could not analyze missing data for column {col}: {e}")


def _build_output_filename(site_id, expected_params, consolidated):
    has_ts = "TIMESTAMP" in consolidated.columns and not consolidated["TIMESTAMP"].isnull().all()
    if not has_ts:
        return f"{site_id}_consolidated.csv"
    min_date = consolidated["TIMESTAMP"].min().strftime("%Y%m%d")
    max_date = consolidated["TIMESTAMP"].max().strftime("%Y%m%d")
    print(f"   • Date range: {min_date} to {max_date}")
    logging.info(f"Date range: {min_date} to {max_date}")
    return f"{site_id}_{'_'.join(expected_params)}_{min_date}_{max_date}.csv"


def _save_consolidated(consolidated, output_filename):
    output_dir = "data/clean"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_filename)
    consolidated.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"💾 Saved to: {output_filename}")
    logging.info(f"Output saved to: {output_path}")
    logging.info(f"TIMESTAMP column dtype: {consolidated['TIMESTAMP'].dtype}")
    return output_path


def consolidate_site_data(site_id):
    """Consolidate all data files for a given site ID"""

    pattern = f"data/raw/*{site_id}*.csv"
    files = glob.glob(pattern)

    if not files:
        logging.error(f"No files found for site {site_id}")
        print(f"❌ No files found for site {site_id}")
        return None

    print(f"📁 Processing {len(files)} files for site {site_id}")
    logging.info(f"Found {len(files)} files for site {site_id}")

    expected_params = extract_parameters_from_filename(files[0])
    logging.info(f"Expected parameter codes: {expected_params}")

    dataframes = []
    processed_files = 0
    failed_files = 0
    for file_path in files:
        df = _process_single_file(file_path, site_id)
        if df is None:
            failed_files += 1
        else:
            dataframes.append(df)
            processed_files += 1

    if not dataframes:
        logging.error("No valid dataframes to consolidate")
        print(f"❌ No valid files processed for site {site_id}")
        return None

    print(f"✅ Successfully processed {processed_files} files ({failed_files} failed)")
    logging.info(f"Processing summary: {processed_files} successful, {failed_files} failed")

    final_columns = _compute_final_column_order(dataframes)
    aligned = _align_dataframes(dataframes, final_columns)

    print("🔄 Consolidating data...")
    consolidated = pd.concat(aligned, ignore_index=True)
    consolidated = _normalise_timestamp_and_sort(consolidated)

    initial_rows = len(consolidated)
    consolidated = consolidated.drop_duplicates()
    final_rows = len(consolidated)
    duplicates_removed = initial_rows - final_rows

    print("📊 Data Summary:")
    print(f"   • Total rows: {final_rows:,}")
    print(f"   • Duplicates removed: {duplicates_removed:,}")
    print(f"   • Columns: {len(consolidated.columns)}")

    logging.info(
        f"Consolidation complete - Final rows: {final_rows}, Duplicates removed: {duplicates_removed}"
    )
    _log_missing_data(consolidated)
    output_filename = _build_output_filename(site_id, expected_params, consolidated)
    return _save_consolidated(consolidated, output_filename)


def load_site_registry():
    """Load site registry from JSON file"""
    try:
        with open("data/reference/sites_register.json", "r") as f:
            data = json.load(f)
            # Convert dictionary to list of site IDs
            if isinstance(data, dict):
                return list(data.keys())
            return data
    except FileNotFoundError:
        logging.error("sites_register.json not found in data/reference/")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing sites_register.json: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Consolidate site data files")
    parser.add_argument("--site_id", help="Specific site ID to consolidate (optional)")
    parser.add_argument("--check", help="Check a consolidated file for data quality issues")
    args = parser.parse_args()

    log_file = setup_logging()

    # Check mode
    if args.check:
        print("🔍 Data Quality Check Mode")
        print(f"📝 Log: {log_file}")
        print()
        check_consolidated_file(args.check)
        return

    print("🚀 Data Consolidator Started")
    print(f"📝 Detailed log: {log_file}")
    print()

    if args.site_id:
        # Process single site
        print(f"🎯 Processing site: {args.site_id}")
        logging.info(f"Starting consolidation for site: {args.site_id}")
        result = consolidate_site_data(args.site_id)
        if result:
            print("✅ Consolidation completed successfully!")
        else:
            print("❌ Consolidation failed")
    else:
        # Process all sites from registry
        sites = load_site_registry()
        if not sites:
            print("❌ No sites found in registry")
            return

        print(f"🔍 Found {len(sites)} sites in registry")
        successful = 0
        failed = 0

        for site_id in sites:
            print(f"\n🎯 Processing site: {site_id}")
            logging.info(f"Starting consolidation for site: {site_id}")
            result = consolidate_site_data(site_id)
            if result:
                successful += 1
            else:
                failed += 1

        print("\n📈 Final Summary:")
        print(f"   • Sites processed successfully: {successful}")
        print(f"   • Sites failed: {failed}")
        print(f"   • Total sites: {successful + failed}")


if __name__ == "__main__":
    # Quick check mode - if we want to run the check directly
    file_to_check = "/Users/tslr/Documents/Smartwater_buoy/Pollution_ID/data/clean/01389005_00010_00065_00095_00300_00301_00400_32315_32321_63680_99133_99134_20111222_20250825.csv"

    if os.path.exists(file_to_check):
        print("🔍 Running data quality check on newly consolidated file...")
        setup_logging()
        check_consolidated_file(file_to_check)
    else:
        print(f"❌ File not found: {file_to_check}")
        main()
