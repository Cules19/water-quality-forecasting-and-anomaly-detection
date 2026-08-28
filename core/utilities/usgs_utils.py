"""
Various utilities for working with USGS water quality data.
Includes parameter code mappings, data cleaning functions, and API interaction helpers.
Author: TSLR
Date: 2024-10-23
"""

import requests
import pandas as pd
from datetime import datetime
import logging
from typing import List, Dict

import os
import re
import json

# Set up logging for USGS utilities
logger = logging.getLogger(__name__)

# USGS Parameter Code Mappings by Type
# NOTE: This is now derived from the centralized parameter_registry for consistency
# Import is done below to avoid circular dependencies
PARAMETER_MAPPINGS = None


def _get_parameter_mappings():
    """Lazy load parameter mappings from registry"""
    global PARAMETER_MAPPINGS
    if PARAMETER_MAPPINGS is None:
        try:
            from .parameter_registry import registry

            # Build type-to-codes mapping from registry
            PARAMETER_MAPPINGS = {}

            # Map by display name (lowercase, underscored)
            for param in registry.list_all():
                key = param.display_name.lower().replace(" ", "_").replace("(", "").replace(")", "")

                if key not in PARAMETER_MAPPINGS:
                    PARAMETER_MAPPINGS[key] = []

                # Add primary code
                if param.usgs_code not in PARAMETER_MAPPINGS[key]:
                    PARAMETER_MAPPINGS[key].append(param.usgs_code)

                # Add alternative codes
                for alt_code in param.alternative_codes:
                    if alt_code not in PARAMETER_MAPPINGS[key]:
                        PARAMETER_MAPPINGS[key].append(alt_code)

            # Also create simplified keys for common lookups
            simplified_mapping = {
                "discharge": [],
                "water_level": [],
                "temperature": [],
                "rainfall": [],
                "conductivity": [],
                "turbidity": [],
                "ph": [],
                "dissolved_oxygen": [],
                "fdom": [],
            }

            for param in registry.list_all():
                display_lower = param.display_name.lower()

                if "discharge" in display_lower and "daily mean" in display_lower:
                    simplified_mapping["discharge"].extend(
                        [param.usgs_code] + param.alternative_codes
                    )
                elif "gage height" in display_lower or "water level" in display_lower:
                    simplified_mapping["water_level"].extend(
                        [param.usgs_code] + param.alternative_codes
                    )
                elif "temperature" in display_lower:
                    simplified_mapping["temperature"].extend(
                        [param.usgs_code] + param.alternative_codes
                    )
                elif "rainfall" in display_lower or "precipitation" in display_lower:
                    simplified_mapping["rainfall"].extend(
                        [param.usgs_code] + param.alternative_codes
                    )
                elif "conductivity" in display_lower or "conductance" in display_lower:
                    simplified_mapping["conductivity"].extend(
                        [param.usgs_code] + param.alternative_codes
                    )
                elif "turbidity" in display_lower:
                    simplified_mapping["turbidity"].extend(
                        [param.usgs_code] + param.alternative_codes
                    )
                elif "ph" in display_lower:
                    simplified_mapping["ph"].extend([param.usgs_code] + param.alternative_codes)
                elif "dissolved oxygen" in display_lower:
                    simplified_mapping["dissolved_oxygen"].extend(
                        [param.usgs_code] + param.alternative_codes
                    )
                elif "fdom" in display_lower or "fluorescent" in display_lower:
                    simplified_mapping["fdom"].extend([param.usgs_code] + param.alternative_codes)

            # Deduplicate and merge
            PARAMETER_MAPPINGS.update(simplified_mapping)

            # Remove empty lists
            PARAMETER_MAPPINGS = {k: list(set(v)) for k, v in PARAMETER_MAPPINGS.items() if v}

        except ImportError:
            # Fallback to hardcoded values if registry not available
            PARAMETER_MAPPINGS = {
                "discharge": ["00060", "00061"],
                "water_level": ["00065"],
                "temperature": ["00010", "00011"],
                "rainfall": ["00045", "00046"],
                "conductivity": [
                    "00094",
                    "00095",
                    "90094",
                    "90095",
                    "90096",
                    "99982",
                    "00096",
                    "00480",
                ],
                "turbidity": [
                    "00070",
                    "00075",
                    "00076",
                    "61028",
                    "63675",
                    "63676",
                    "63677",
                    "63678",
                    "63679",
                    "63680",
                    "63681",
                    "63682",
                    "63683",
                    "63684",
                    "72213",
                    "72208",
                    "72209",
                    "72337",
                    "82079",
                    "99872",
                ],
                "pH": ["00400"],
                "dissolved_oxygen": ["00300"],
                "fdom": ["32295"],
            }

    return PARAMETER_MAPPINGS


# Initialize on import
PARAMETER_MAPPINGS = _get_parameter_mappings()

# USGS API Configuration
USGS_API_CONFIG = {
    "base_url": "https://waterservices.usgs.gov/nwis/iv",
    "site_info_url": "https://waterservices.usgs.gov/nwis/site",
    "timeout": 30,
    "format": "json",
}


# Helper: Get the average of the last 3 points for each row (rolling window)
def last3_avg(series):
    return series.rolling(window=3, min_periods=1).mean()


def clip_spikes(df, variable_col, avg_col, jump, flag_col, end_flag_col):
    """
    Detects spikes in a variable column compared to its average column, using a jump boolean Series.
    Flags start and end of spikes in the DataFrame.

    Args:
        df: pandas DataFrame with a datetime index.
        variable_col: str, name of the variable column (e.g., 'Cond [uS/cm]').
        avg_col: str, name of the average column (e.g., 'Cond_AVG [uS/cm]').
        jump: pd.Series of bool, True where a spike/jump is detected.
        flag_col: str, name of the column to flag spike starts.
        end_flag_col: str, name of the column to flag spike ends.
    Returns:
        The DataFrame with flag columns added/updated.
    """

    VAL = df[variable_col]
    in_spike = False
    last_spike_val = 0

    # Calculate dy/dx (rate of change) using simple difference over 4 hours
    dy_dx = VAL.diff(periods=4) / (df.index.to_series().diff(periods=4).dt.total_seconds() / 3600.0)

    df[flag_col] = 0
    df[end_flag_col] = 0

    for i in range(2, len(df)):
        if not in_spike:
            if jump.iloc[i] and dy_dx.iloc[i] > 0:
                df.at[df.index[i], flag_col] = 1
                in_spike = True
                spike_start = VAL.iloc[i]
                last_spike_val = VAL.iloc[i]
                spike_start_time = df.index[i]
        elif in_spike:
            # Update last_spike_val to the maximum value seen during the spike
            if VAL.iloc[i] > last_spike_val:
                last_spike_val = VAL.iloc[i]
            # Require a drop of at least 5% from last_spike_val before allowing a new spike
            df.at[df.index[i], flag_col] = 0
            spike_time = (df.index[i] - spike_start_time).total_seconds() / 3600.0

            if spike_time > 12:  # 2 hours
                # If 2-hourly dy/dx becomes negative and there is a 5% drop from the peak, mark end of spike
                if dy_dx.iloc[i] < 0.1 or (
                    (VAL.iloc[i] - spike_start) < (0.90 * last_spike_val - spike_start)
                ):
                    df.at[df.index[i], end_flag_col] = 1
                    in_spike = False
            else:
                if dy_dx.iloc[i] < 0.1 and (
                    (VAL.iloc[i] - spike_start) < (0.90 * last_spike_val - spike_start)
                ):
                    df.at[df.index[i], end_flag_col] = 1
                    in_spike = False

    return df


def clean_usgs_tsv_to_csv(tsv_path, csv_path):

    # Extract all parameter codes from the file name (assuming codes are always 5 digits)
    file_codes = re.findall(r"_(\d{5})(?=_)", tsv_path.split("/")[-1])
    # print("Extracted parameter codes from file name:", file_codes)
    # Read the TSV file, skipping the first two lines
    df = pd.read_csv(tsv_path, sep="\t", skiprows=2)
    # print("Number of parameter codes found:", len(file_codes))

    # Dynamically calculate columns to drop: always drop 0, and for each code in file_codes, drop the next odd column after its data column
    cols_to_drop = [0, 3, 4, 5] + [6 + 2 * idx + 1 for idx in range(len(file_codes))]
    # print("Columns to drop:", cols_to_drop)
    df = df.drop(df.columns[cols_to_drop], axis=1)

    # Define the header row
    header = ["Site_ID"]

    # Read the first line (column names) from the file
    with open(tsv_path, "r") as f:
        lines = f.readlines()
        original_header = lines[1].strip().split("\t")[1:]  # skip the first column

    # Combine the new first column with the rest
    new_header = header + original_header

    # Set the first three column headers
    new_header[0] = "Site_ID"
    new_header[1] = "TIMESTAMP"

    # Assign headers based on codes present in the file name
    for i, code in enumerate(file_codes):
        if 3 + i < len(new_header):
            new_header[2 + i] = usgs_code_map.get(code, code)

    # Assign only as many headers as you have columns
    df.columns = new_header[: len(df.columns)]

    # Convert 'TIMESTAMP' to datetime
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")

    # Apply conversions to each parameter column based on file_codes
    for i, code in enumerate(file_codes):
        col_idx = 2 + i  # 0: Site_ID, 1: TIMESTAMP, 2+: parameter columns
        if col_idx >= len(df.columns):
            continue
        col = df.columns[col_idx]
        if code in SPECIFIC_CONDUCTANCE_CODES:
            df[col] = df[col].apply(lambda x: convert_to_uS_cm(code, x))
        elif code in TURBIDITY_CODES:
            df[col] = df[col].apply(lambda x: convert_to_NTU(code, x))
        elif code in FDOM_CODES:
            df[col] = df[col].apply(lambda x: convert_to_fDOM_standard(code, x))
        elif code in TEMPERATURE_CODES:
            df[col] = df[col].apply(lambda x: convert_to_Celsius(code, x))
        else:
            # No conversion, keep as is
            pass

        if code in SPECIFIC_CONDUCTANCE_CODES:
            df = df.rename(columns={col: "Conductance [uS/cm]*"})
        elif code in TURBIDITY_CODES:
            df = df.rename(columns={col: "Turbidity [NTU]*"})
        elif code in FDOM_CODES:
            df = df.rename(columns={col: "fDOM [RFU]*"})
        elif code in TEMPERATURE_CODES:
            df = df.rename(columns={col: "TEMP [deg C]"})
        else:
            # No conversion, keep as is
            pass

    # Save the cleaned data to a new CSV file
    df.to_csv(csv_path, index=False)
    # print(f"Cleaned data saved to {csv_path}")

    return df


def get_active_usgs_parameters(site_id):
    """
    Returns a dict of {parameter_code: description} for parameters active in the last 7 days.
    """
    url = "https://waterservices.usgs.gov/nwis/iv/" f"?sites={site_id}" "&format=rdb" "&period=P7D"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching active parameters: {e}")
        return {}

    parameters = {}
    parsing = False

    for line in resp.text.splitlines():
        if line.strip().startswith("# Data provided for site"):
            parsing = True
            continue
        if parsing and line.strip().startswith("#"):
            parts = line.strip("#").strip().split(None, 2)
            if len(parts) == 3:
                _, param_code, description = parts
                parameters[param_code] = description.strip()

    return parameters


def get_time_ranges_for_active_parameters(site_id, active_params):
    """
    Returns time ranges only for the given active parameter codes.
    Format: {parameter_code: (description, begin_date, end_date, duration_days)}
    """
    url = (
        "https://waterservices.usgs.gov/nwis/site/"
        f"?site={site_id}"
        "&seriesCatalogOutput=true"
        "&format=rdb"
    )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching time ranges: {e}")
        return {}

    lines = resp.text.splitlines()
    header = []
    results = {}

    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith("agency_cd"):
            header = line.strip().split("\t")
            continue
        if header:
            values = line.strip().split("\t")
            entry = dict(zip(header, values))
            param_code = entry.get("parm_cd")
            if param_code in active_params and entry.get("begin_date") and entry.get("end_date"):
                try:
                    start = datetime.strptime(entry["begin_date"], "%Y-%m-%d")
                    end = datetime.strptime(entry["end_date"], "%Y-%m-%d")
                    duration = (end - start).days
                except ValueError:
                    start = end = duration = None
                results[param_code] = (
                    active_params[param_code],  # use active param's description
                    entry["begin_date"],
                    entry["end_date"],
                    duration,
                )

    return results


usgs_code_map = {
    # Temperature
    "00010": "Temp, water, deg C",
    "00011": "Temp, water, deg F",
    "85557": "Temp, low saturation, deg C",
    "85558": "Temp high saturation, deg C",
    "90010": "Temp, area-weighted mean, deg C",
    "99984": "Temp, water, mid-depth, deg C",
    # Discharge and Gage Height
    "00060": "Discharge, cubic feet per second",
    "00065": "Gage height, feet",
    # Specific Conductance & Salinity
    "00094": "Specific conductance, water, unfiltered, field, uS/cm @25C",
    "00095": "Specific conductance, water, unfiltered, uS/cm @25C",
    "90094": "Specific conductance, area weighted average, uS/cm @25C",
    "90095": "Specific conductance, laboratory, uS/cm @25C",
    "90096": "Specific conductance, filtered, laboratory, uS/cm @25C",
    "99982": "Specific conductance, mid-depth, uS/cm @25C",
    "00096": "Salinity, mg/mL @25C",
    "00480": "Salinity, (ppt)",
    # Turbidity
    "00070": "Turbidity, Jackson Turbidity Units (JTU)",
    "00075": "Turbidity, Hellige turbidimeter, mg/L as SiO2",
    "00076": "Turbidity, Nephelometric Turbidity Units (NTU)",
    "61028": "Turbidity, field, NTU",
    "63675": "Turbidity, broad band, 90 deg, NTU",
    "63676": "Turbidity, NTRU (Ratiometric)",
    "63677": "Turbidity, Backscatter units (BU)",
    "63678": "Turbidity, Attenuation units (ATU)",
    "63679": "Turbidity, Multiple beam, NTMU",
    "63680": "Turbidity, Formazin Nephelometric Units (FNU)",
    "63681": "Turbidity, FNRU (Ratiometric)",
    "63682": "Turbidity, FBU (Backscatter)",
    "63683": "Turbidity, FAU (Attenuation)",
    "63684": "Turbidity, FNMU (Multiple beam)",
    "72213": "Turbidity, FBRU (Ratiometric)",
    "72208": "Turbidity, mid-depth, NTU",
    "72209": "Turbidity, mid-depth, FNU",
    "72337": "Turbidity, SBU (SDVB backscatter units)",
    "82079": "Turbidity, laboratory, NTU",
    "99872": "Turbidity, Hach 2100AN, NTU",
    # fDOM and DOM
    "32295": "fDOM, ug/L QSE",
    "32322": "fDOM, RFU",
    "32330": "fDOM, mg/L C",
    "32357": "fDOM, PTSA (ug/L)",
    "83565": "fDOM, ug/s QSE (load)",
    # Add more as needed
}

# Conversion factors to convert various units to microsiemens per centimeter (uS/cm)
# For specific conductance, all codes are assumed to be in uS/cm @25C or convertible as 1:1.
# For salinity (mg/mL, ppt, etc.), we use an approximate conversion: ppt * 1500 = uS/cm.
# Lookup list for specific conductance and salinity parameter codes
SPECIFIC_CONDUCTANCE_CODES = [
    "00094",
    "00095",
    "90094",
    "90095",
    "90096",
    "99982",
    "00096",
    "00480",
]

# 00094	Physical	Specific conductance, water, unfiltered, field, microsiemens per centimeter at 25 degrees Celsius	Agree					Total	25 deg C		Specific conductance	uS/cm @25C
# 00095	Physical	Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius	 Agree					Total	25 deg C		Specific conductance	uS/cm @25C
# 00096 Physical	Salinity, water, unfiltered, milligrams per milliliter at 25 degrees Celsius	 Agree					Total	25 deg C		Salinity	mg/mL @25C
# 00480	Physical	Salinity, water, unfiltered, parts per thousand	 Not checked					Total
# 90094	Physical	Specific conductance, water, unfiltered, area weighted average, microsiemens per centimeter at 25 degrees Celsius	USGS use and no use by EPA					Total	25 deg C		Specific conductance	uS/cm @25C
# 90095	Physical	Specific conductance, water, unfiltered, laboratory, microsiemens per centimeter at 25 degrees Celsius	USGS use and no use by EPA					Total	25 deg C		Specific conductance	uS/cm @25C
# 90096	Physical	Specific conductance, water, filtered, laboratory, microsiemens per centimeter at 25 degrees Celsius	Not checked					Dissolved			Specific conductance	uS/cm @25C
# 99982	Physical	Specific conductance, water, mid-depth of water column, microsiemens per centimeter at 25 degrees Celsius	USGS use and no use by EPA						25 deg C		Specific conductance	uS/cm @25C

USGS_CONDUCTANCE_CONVERSIONS = {
    # code: (unit, conversion_factor_to_uS_per_cm, notes)
    "00094": ("uS/cm @25C", 1.0, "Specific conductance, field"),
    "00095": ("uS/cm @25C", 1.0, "Specific conductance"),
    "90094": ("uS/cm @25C", 1.0, "Area weighted average"),
    "90095": ("uS/cm @25C", 1.0, "Laboratory"),
    "00480": ("ppt", 1500.0, "Salinity, ppt (ppt*1500)"),
    "90096": ("uS/cm @25C", 1.0, "Filtered, laboratory"),
    "99982": ("uS/cm @25C", 1.0, "Mid-depth water column"),
}


def convert_to_uS_cm(param_code, value):
    """
    Convert a value for a given USGS parameter code to uS/cm if possible.
    Returns the converted value or None if conversion is not possible.
    """
    conv = USGS_CONDUCTANCE_CONVERSIONS.get(param_code)
    if conv is None:
        return None
    factor = conv[1]
    if factor is None:
        return None
    try:
        return float(value) * factor
    except Exception:
        return None


# Turbidity parameter codes and their units/conversions to NTU (Nephelometric Turbidity Units)
# For most USGS turbidity codes, the standard unit is NTU or a closely related unit.
# Some codes use alternative units (e.g., JTU, FNU, NTRU, etc.), which are not always directly convertible to NTU.
# Where possible, we provide a 1:1 conversion for NTU-like units; otherwise, conversion is not attempted.

TURBIDITY_CODES = [
    "00070",
    "00075",
    "00076",
    "61028",
    "63675",
    "63676",
    "63677",
    "63678",
    "63679",
    "63680",
    "63681",
    "63682",
    "63683",
    "63684",
    "72213",
    "72208",
    "72209",
    "72337",
    "82079",
    "99872",
]

USGS_TURBIDITY_CONVERSIONS = {
    # code: (unit, conversion_factor_to_NTU, notes)
    "00070": ("JTU", None, "Jackson Turbidity Units (not directly convertible to NTU)"),
    "00075": (
        "mg/L SiO2",
        None,
        "Hellige turbidimeter, mg/L as SiO2 (not directly convertible to NTU)",
    ),
    "00076": ("NTU", 1.0, "Nephelometric Turbidity Units"),
    "61028": ("NTU", 1.0, "Field, NTU"),
    "63675": ("NTU", 1.0, "Broad band, 90 deg, NTU"),
    "63676": ("NTRU", None, "Ratiometric, NTRU (not directly convertible to NTU)"),
    "63677": ("BU", None, "Backscatter units (not directly convertible to NTU)"),
    "63678": ("ATU", None, "Attenuation units (not directly convertible to NTU)"),
    "63679": ("NTMU", None, "Multiple beam, NTMU (not directly convertible to NTU)"),
    "63680": (
        "FNU",
        1.0,
        "Formazin Nephelometric Units (ISO 7027, often treated as equivalent to NTU)",
    ),
    "63681": ("FNRU", None, "Ratiometric, FNRU (not directly convertible to NTU)"),
    "63682": ("FBU", None, "Backscatter, FBU (not directly convertible to NTU)"),
    "63683": ("FAU", None, "Attenuation, FAU (not directly convertible to NTU)"),
    "63684": ("FNMU", None, "Multiple beam, FNMU (not directly convertible to NTU)"),
    "72213": ("FBRU", None, "Ratiometric, FBRU (not directly convertible to NTU)"),
    "72208": ("NTU", 1.0, "Mid-depth, NTU"),
    "72209": ("FNU", 1.0, "Mid-depth, FNU (treated as NTU)"),
    "72337": ("SBU", None, "SDVB backscatter units (not directly convertible to NTU)"),
    "82079": ("NTU", 1.0, "Laboratory, NTU"),
    "99872": ("NTU", 1.0, "Hach 2100AN, NTU"),
}


def convert_to_NTU(param_code, value):
    """
    Convert a value for a given USGS turbidity parameter code to NTU if possible.
    Returns the converted value or None if conversion is not possible.
    """
    conv = USGS_TURBIDITY_CONVERSIONS.get(param_code)
    if conv is None:
        return None
    factor = conv[1]
    if factor is None:
        return None
    try:
        return float(value) * factor
    except Exception:
        return None


# fDOM (fluorescent dissolved organic matter) parameter codes and their units/conversions.
# There is no universal conversion between fDOM units (QSE, RFU, PTSA, mg/L C, etc.), so conversions are not attempted.
# Codes are listed for reference and possible identification.
# 32295	Physical	Dissolved organic matter fluorescence (fDOM), water, in situ, concentration estimated from reference material, micrograms per liter as quinine sulfate equivalents (QSE)	Not checked					Dissolved			Colored dissolved organic matter (CDOM)	ug/l QSE

FDOM_CODES = [
    "32295",  # ug/L QSE
    "32322",  # RFU
    "32330",  # mg/L C
    "32357",  # ug/L PTSA
    "83565",  # ug/s QSE (load)
]


def convert_to_fDOM_standard(param_code, value):
    """
    No standard conversion for fDOM units. Returns value as float if possible, else None.
    """
    if param_code not in FDOM_CODES:
        return None
    try:
        return float(value)
    except Exception:
        return None


# Temperature parameter codes and their units/conversions to degrees Celsius (°C)
# Most USGS temperature codes are in °C or °F. We provide conversion for °F to °C.

TEMPERATURE_CODES = ["00010", "00011", "85557", "85558", "90010", "99984"]

USGS_TEMPERATURE_CONVERSIONS = {
    # code: (unit, conversion_function, notes)
    "00010": ("deg C", lambda v: float(v), "Temperature, water, degrees Celsius"),
    "00011": (
        "deg F",
        lambda v: (float(v) - 32) * 5.0 / 9.0,
        "Temperature, water, degrees Fahrenheit",
    ),
    "85557": ("deg C", lambda v: float(v), "Temperature, low saturation, degrees Celsius"),
    "85558": ("deg C", lambda v: float(v), "Temperature, high saturation, degrees Celsius"),
    "90010": ("deg C", lambda v: float(v), "Temperature, area-weighted mean, degrees Celsius"),
    "99984": (
        "deg C",
        lambda v: float(v),
        "Temperature, water, mid-depth of water column, degrees Celsius",
    ),
}


def convert_to_Celsius(param_code, value):
    """
    Convert a value for a given USGS temperature parameter code to degrees Celsius if possible.
    Returns the converted value or None if conversion is not possible.
    """
    conv = USGS_TEMPERATURE_CONVERSIONS.get(param_code)
    if conv is None:
        return None
    try:
        return conv[1](value)
    except Exception:
        return None


def get_usgs_parameter_time_ranges(site_id):
    """
    Returns a list of (parameter code, description, begin_date, end_date, duration_days)
    """
    url = (
        "https://waterservices.usgs.gov/nwis/site/"
        f"?site={site_id}"
        "&seriesCatalogOutput=true"
        "&format=rdb"
    )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching site metadata: {e}")
        return []

    lines = resp.text.splitlines()
    header = []
    param_data = []

    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith("agency_cd"):
            header = line.strip().split("\t")
            continue
        if header:
            values = line.strip().split("\t")
            entry = dict(zip(header, values))
            if entry.get("parm_cd") and entry.get("begin_date") and entry.get("end_date"):
                try:
                    start = datetime.strptime(entry["begin_date"], "%Y-%m-%d")
                    end = datetime.strptime(entry["end_date"], "%Y-%m-%d")
                    duration = (end - start).days
                except ValueError:
                    start = end = duration = None
                param_data.append(
                    (
                        entry["parm_cd"],
                        entry.get("parm_nm", "No description"),
                        entry["begin_date"],
                        entry["end_date"],
                        duration,
                    )
                )
    return param_data


def get_site_coordinates(site_id, project_root):
    """
    Get latitude and longitude coordinates for a USGS site from the sites registry.

    Args:
        site_id (str): USGS site identifier
        project_root (str): Path to the project root directory

    Returns:
        tuple: (latitude, longitude) if found, (None, None) if not found
    """
    try:
        sites_registry_path = os.path.join(project_root, "data", "sites_register.json")

        if os.path.exists(sites_registry_path):
            with open(sites_registry_path, "r") as f:
                sites_data = json.load(f)

            # Look for the site in the registry
            if site_id in sites_data:
                site_info = sites_data[site_id]
                latitude = site_info.get("latitude")
                longitude = site_info.get("longitude")
                print(f"Found coordinates for {site_id}: lat={latitude}, lon={longitude}")
                return latitude, longitude
            else:
                print(f"Site {site_id} not found in sites registry")
                return None, None
        else:
            print(f"Sites registry file not found at {sites_registry_path}")
            return None, None

    except Exception as e:
        print(f"Error reading sites registry: {e}")
        return None, None


# ============================================================================
# USGS API Utility Functions
# ============================================================================


def get_parameters_by_type(param_type: str) -> List[str]:
    """
    Get parameter codes for a specific data type.

    Args:
        param_type: Type of parameter (e.g., 'discharge', 'temperature')

    Returns:
        List of parameter codes
    """
    return PARAMETER_MAPPINGS.get(param_type.lower(), [])


def build_data_url(site_id: str, start_date: str, end_date: str, parameters: List[str]) -> str:
    """
    Build USGS data API URL.

    Args:
        site_id: USGS site identifier
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        parameters: List of parameter codes

    Returns:
        Complete API URL
    """
    base_url = USGS_API_CONFIG["base_url"]
    param_string = ",".join(parameters)

    url = (
        f"{base_url}/?format={USGS_API_CONFIG['format']}&sites={site_id}"
        f"&startDT={start_date}&endDT={end_date}"
        f"&parameterCd={param_string}"
    )

    return url


def build_site_info_url(site_id: str) -> str:
    """
    Build USGS site information API URL.

    Args:
        site_id: USGS site identifier

    Returns:
        Site info API URL
    """
    base_url = USGS_API_CONFIG["site_info_url"]
    return f"{base_url}/?format={USGS_API_CONFIG['format']}&sites={site_id}"


def validate_date_format(date_string: str) -> bool:
    """
    Validate date string format (YYYY-MM-DD).

    Args:
        date_string: Date string to validate

    Returns:
        True if valid format, False otherwise
    """
    try:
        datetime.strptime(date_string, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def validate_site_id(site_id: str) -> bool:
    """
    Validate USGS site ID format.

    Args:
        site_id: Site identifier to validate

    Returns:
        True if valid format, False otherwise
    """
    # USGS site IDs are typically 8-15 digit strings
    return site_id.isdigit() and 8 <= len(site_id) <= 15


def extract_parameters_from_filename(filename: str) -> List[str]:
    """
    Extract parameter codes from a USGS data filename.

    Args:
        filename: Data filename in USGS format

    Returns:
        List of parameter codes found in filename
    """
    # Remove file extension
    if "." in filename:
        filename = filename.rsplit(".", 1)[0]

    # Split by underscores and get middle parts (excluding site ID and dates)
    parts = filename.split("_")

    if len(parts) < 3:
        return []

    # Skip site ID (first part) and date parts (last two parts)
    param_parts = parts[1:-2]

    # Filter to keep only 5-digit parameter codes
    parameters = [part for part in param_parts if part.isdigit() and len(part) == 5]

    return parameters


def get_data_type_flags(parameters: List[str]) -> Dict[str, bool]:
    """
    Generate data type flags based on parameter codes.

    Args:
        parameters: List of parameter codes

    Returns:
        Dictionary with boolean flags for each data type
    """
    flags = {}

    for data_type, param_codes in PARAMETER_MAPPINGS.items():
        flags[f"has_{data_type}"] = any(param in param_codes for param in parameters)

    # Special combined flag for water quality
    water_quality_params = (
        PARAMETER_MAPPINGS["conductivity"]
        + PARAMETER_MAPPINGS["turbidity"]
        + PARAMETER_MAPPINGS["ph"]
        + PARAMETER_MAPPINGS["dissolved_oxygen"]
        + PARAMETER_MAPPINGS["fdom"]
    )

    flags["has_water_quality"] = any(param in water_quality_params for param in parameters)

    return flags


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate approximate distance between two geographic points.

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Distance in degrees (approximate)
    """
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5


def format_usgs_response_data(response_data: Dict) -> List[Dict]:
    """
    Format USGS API response data into standardized structure.

    Args:
        response_data: Raw USGS API response

    Returns:
        List of formatted data points
    """
    try:
        value_section = response_data.get("value", {})
        time_series = value_section.get("timeSeries", [])

        # Collect all data by timestamp
        timestamp_data = {}
        site_no = None

        for ts in time_series:
            # Get site number from first time series
            if site_no is None:
                source_info = ts.get("sourceInfo", {})
                site_no = source_info.get("siteCode", [{}])[0].get("value", "unknown")

            variable = ts.get("variable", {})
            var_code_info = variable.get("variableCode", [])

            if var_code_info:
                param_code = var_code_info[0].get("value", "unknown")
            else:
                param_code = "unknown"

            # Process each data point for this parameter
            values_section = ts.get("values", [])
            for value_set in values_section:
                data_point_list = value_set.get("value", [])
                for data_point in data_point_list:
                    timestamp = data_point.get("dateTime", "")
                    if not timestamp:
                        continue

                    # Initialize timestamp entry if not exists
                    if timestamp not in timestamp_data:
                        timestamp_data[timestamp] = {"site_no": site_no, "datetime": timestamp}

                    # Add parameter value
                    timestamp_data[timestamp][param_code] = data_point.get("value", "")

        # Convert to list format
        data_points = list(timestamp_data.values())
        return data_points

    except Exception as e:
        logger.error(f"Error formatting USGS response data: {e}")
        return []


def get_usgs_parameter_name(param_code, format_type="short"):
    """
    Returns the parameter name for a given USGS 5-digit parameter code.

    Args:
        param_code (str): 5-digit USGS parameter code
        format_type (str): "short" for brief name, "long" for full name with units

    Returns:
        str: Parameter name based on format_type, or "Unknown parameter" if not found
    """

    # Most common parameters - checked first for speed
    common_params = {
        "00060": {"short": "Discharge", "long": "Discharge, cubic feet per second"},
        "00065": {"short": "Gage height", "long": "Gage height, feet"},
        "00010": {"short": "Temperature", "long": "Temperature, water, degrees Celsius"},
        "00095": {
            "short": "Conductance",
            "long": "Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius",
        },
        "00400": {"short": "pH", "long": "pH, water, unfiltered, field, standard units"},
        "00300": {
            "short": "Dissolved oxygen",
            "long": "Dissolved oxygen, water, unfiltered, milligrams per liter",
        },
        "00076": {
            "short": "Turbidity",
            "long": "Turbidity, water, unfiltered, nephelometric turbidity units",
        },
        "61028": {
            "short": "Turbidity",
            "long": "Turbidity, water, unfiltered, field, nephelometric turbidity units",
        },
        "00631": {
            "short": "Nitrate+Nitrite",
            "long": "Nitrate plus nitrite, water, filtered, milligrams per liter as nitrogen",
        },
        "00665": {
            "short": "Phosphorus",
            "long": "Phosphorus, water, unfiltered, milligrams per liter as phosphorus",
        },
        # Temperature parameters
        "00011": {"short": "Temperature", "long": "Temperature, water, degrees Fahrenheit"},
        "85557": {"short": "Temperature", "long": "Temperature, low saturation, degrees Celsius"},
        "85558": {"short": "Temperature", "long": "Temperature, high saturation, degrees Celsius"},
        "90010": {
            "short": "Temperature",
            "long": "Temperature, area-weighted mean, degrees Celsius",
        },
        "99984": {"short": "Temperature", "long": "Temperature, water, mid-depth, degrees Celsius"},
        # Specific Conductance & Salinity
        "00094": {
            "short": "Conductance",
            "long": "Specific conductance, water, unfiltered, field, microsiemens per centimeter at 25 degrees Celsius",
        },
        "90094": {
            "short": "Conductance",
            "long": "Specific conductance, area weighted average, microsiemens per centimeter at 25 degrees Celsius",
        },
        "90095": {
            "short": "Conductance",
            "long": "Specific conductance, laboratory, microsiemens per centimeter at 25 degrees Celsius",
        },
        "90096": {
            "short": "Conductance",
            "long": "Specific conductance, filtered, laboratory, microsiemens per centimeter at 25 degrees Celsius",
        },
        "99982": {
            "short": "Conductance",
            "long": "Specific conductance, mid-depth, microsiemens per centimeter at 25 degrees Celsius",
        },
        "00096": {
            "short": "Salinity",
            "long": "Salinity, milligrams per milliliter at 25 degrees Celsius",
        },
        "00480": {"short": "Salinity", "long": "Salinity, parts per thousand"},
        # Turbidity parameters
        "00070": {"short": "Turbidity", "long": "Turbidity, Jackson Turbidity Units (JTU)"},
        "00075": {
            "short": "Turbidity",
            "long": "Turbidity, Hellige turbidimeter, milligrams per liter as SiO2",
        },
        "63675": {"short": "Turbidity", "long": "Turbidity, broad band, 90 deg, NTU"},
        "63676": {"short": "Turbidity", "long": "Turbidity, NTRU (Ratiometric)"},
        "63677": {"short": "Turbidity", "long": "Turbidity, Backscatter units (BU)"},
        "63678": {"short": "Turbidity", "long": "Turbidity, Attenuation units (ATU)"},
        "63679": {"short": "Turbidity", "long": "Turbidity, Multiple beam, NTMU"},
        "63680": {"short": "Turbidity", "long": "Turbidity, Formazin Nephelometric Units (FNU)"},
        "63681": {"short": "Turbidity", "long": "Turbidity, FNRU (Ratiometric)"},
        "63682": {"short": "Turbidity", "long": "Turbidity, FBU (Backscatter)"},
        "63683": {"short": "Turbidity", "long": "Turbidity, FAU (Attenuation)"},
        "63684": {"short": "Turbidity", "long": "Turbidity, FNMU (Multiple beam)"},
        "72213": {"short": "Turbidity", "long": "Turbidity, FBRU (Ratiometric)"},
        "72208": {"short": "Turbidity", "long": "Turbidity, mid-depth, NTU"},
        "72209": {"short": "Turbidity", "long": "Turbidity, mid-depth, FNU"},
        "72337": {"short": "Turbidity", "long": "Turbidity, SBU (SDVB backscatter units)"},
        "82079": {"short": "Turbidity", "long": "Turbidity, laboratory, NTU"},
        "99872": {"short": "Turbidity", "long": "Turbidity, Hach 2100AN, NTU"},
        # fDOM and DOM parameters
        "32295": {"short": "fDOM", "long": "fDOM, micrograms per liter QSE"},
        "32322": {"short": "fDOM", "long": "fDOM, RFU"},
        "32330": {"short": "fDOM", "long": "fDOM, milligrams per liter C"},
        "32357": {"short": "fDOM", "long": "fDOM, PTSA (micrograms per liter)"},
        "83565": {"short": "fDOM", "long": "fDOM, micrograms per second QSE (load)"},
        # Additional common parameters
        "00061": {"short": "Discharge", "long": "Discharge, instantaneous, cubic feet per second"},
        "30208": {"short": "Discharge", "long": "Discharge, cubic meters per second"},
        "30209": {
            "short": "Discharge",
            "long": "Discharge, instantaneous, cubic meters per second",
        },
        "00403": {"short": "pH", "long": "pH, water, unfiltered, laboratory, standard units"},
        "00530": {
            "short": "Suspended solids",
            "long": "Suspended solids, water, unfiltered, milligrams per liter",
        },
        "00625": {
            "short": "Kjeldahl nitrogen",
            "long": "Ammonia plus organic nitrogen, water, unfiltered, milligrams per liter as nitrogen",
        },
        "00608": {
            "short": "Ammonia",
            "long": "Ammonia (NH3 + NH4+), water, filtered, milligrams per liter as nitrogen",
        },
        "00610": {
            "short": "Ammonia",
            "long": "Ammonia (NH3 + NH4+), water, unfiltered, milligrams per liter as nitrogen",
        },
    }

    # Check common parameters first
    if param_code in common_params:
        return common_params[param_code][format_type]

    # Try loading from JSON file first
    try:
        json_path = "data/reference/usgs_parameter_mapping.json"
        with open(json_path, "r") as f:
            json_params = json.load(f)
        if param_code in json_params:
            return json_params[param_code].get(format_type, f"Unknown parameter ({param_code})")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    # If not found in either list
    return f"Unknown parameter ({param_code})"
