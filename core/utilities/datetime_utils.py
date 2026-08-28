"""
Datetime Column Utilities

Helper functions for handling datetime column naming consistently across the system.
Supports the project standard:
- Data files use 'TIMESTAMP' (uppercase)
- Code parameters use 'timestamp' (lowercase)
- Code supports both variants

Author: TSLR
Created: 2026-02-09
Updated: 2026-02-09
"""

import pandas as pd
from typing import Optional, List

# Standard datetime column variants in order of preference
DATETIME_VARIANTS = [
    "TIMESTAMP",  # Standard for data files
    "timestamp",  # Standard for code parameters
    "datetime",  # Common alternative
    "DATETIME",
    "date_time",
    "dateTime",
    "date",
    "Date",
    "time",
    "Time",
]

# Standard name for data files
STANDARD_DATA_COLUMN = "TIMESTAMP"

# Standard name for code parameters
STANDARD_PARAM_NAME = "timestamp"


def detect_datetime_column(
    df: pd.DataFrame, candidates: Optional[List[str]] = None
) -> Optional[str]:
    """
    Detect the datetime column in a DataFrame.

    Checks for common datetime column names and validates the data type.
    Returns the first matching column found.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to search
    candidates : list of str, optional
        Custom list of candidate column names to check.
        If None, uses DATETIME_VARIANTS

    Returns
    -------
    str or None
        Name of the datetime column, or None if not found

    Examples
    --------
    >>> df = pd.read_csv('data.csv')
    >>> dt_col = detect_datetime_column(df)
    >>> if dt_col:
    ...     df[dt_col] = pd.to_datetime(df[dt_col])
    """
    if candidates is None:
        candidates = DATETIME_VARIANTS

    # Check exact matches first
    for variant in candidates:
        if variant in df.columns:
            return variant

    # Check case-insensitive matches
    columns_lower = {col.lower(): col for col in df.columns}
    for variant in candidates:
        if variant.lower() in columns_lower:
            return columns_lower[variant.lower()]

    # Check if any column contains datetime-like data
    for col in df.columns:
        if any(keyword in str(col).lower() for keyword in ["time", "date"]):
            # Verify it's actually datetime-like
            try:
                pd.to_datetime(df[col].head(10), errors="raise")
                return col
            except (ValueError, TypeError):
                continue

    return None


def ensure_datetime_column(
    df: pd.DataFrame, timestamp_col: Optional[str] = None, inplace: bool = False
) -> pd.DataFrame:
    """
    Ensure DataFrame has a datetime column with standard name.

    If timestamp_col is provided, uses that column.
    Otherwise, auto-detects the datetime column.
    Converts to datetime type if needed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to process
    timestamp_col : str, optional
        Name of timestamp column. If None, auto-detects.
    inplace : bool, default False
        If True, modify df in place. Otherwise return a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime column ensured

    Raises
    ------
    ValueError
        If no datetime column found and none specified

    Examples
    --------
    >>> df = pd.read_csv('data.csv')
    >>> df = ensure_datetime_column(df)  # Auto-detect and convert
    >>> # Or specify column name
    >>> df = ensure_datetime_column(df, timestamp_col='datetime')
    """
    if not inplace:
        df = df.copy()

    # Auto-detect if not specified
    if timestamp_col is None:
        timestamp_col = detect_datetime_column(df)
        if timestamp_col is None:
            raise ValueError("No datetime column found in DataFrame")

    # Verify column exists
    if timestamp_col not in df.columns:
        # Try to detect
        detected = detect_datetime_column(df)
        if detected:
            timestamp_col = detected
        else:
            raise ValueError(f"Column '{timestamp_col}' not found in DataFrame")

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    return df


def standardize_datetime_column(
    df: pd.DataFrame, timestamp_col: Optional[str] = None, inplace: bool = False
) -> pd.DataFrame:
    """
    Standardize datetime column name to STANDARD_DATA_COLUMN ('TIMESTAMP').

    Useful before saving data to ensure consistent naming.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to standardize
    timestamp_col : str, optional
        Current name of timestamp column. If None, auto-detects.
    inplace : bool, default False
        If True, modify df in place. Otherwise return a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized datetime column name

    Examples
    --------
    >>> df = pd.read_csv('data.csv')  # Has 'datetime' column
    >>> df = standardize_datetime_column(df)
    >>> df.to_csv('output.csv')  # Saves with 'TIMESTAMP' column
    """
    if not inplace:
        df = df.copy()

    # Auto-detect if not specified
    if timestamp_col is None:
        timestamp_col = detect_datetime_column(df)
        if timestamp_col is None:
            raise ValueError("No datetime column found in DataFrame")

    # Rename to standard if not already
    if timestamp_col != STANDARD_DATA_COLUMN:
        df.rename(columns={timestamp_col: STANDARD_DATA_COLUMN}, inplace=True)

    return df


def get_datetime_column_or_index(
    df: pd.DataFrame, timestamp_col: Optional[str] = None
) -> pd.Series:
    """
    Get datetime data as a Series, from either column or index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to extract datetime from
    timestamp_col : str, optional
        Name of timestamp column. If None, auto-detects or uses index.

    Returns
    -------
    pd.Series
        Datetime data as a Series

    Raises
    ------
    ValueError
        If no datetime column or index found

    Examples
    --------
    >>> df = pd.read_csv('data.csv', index_col=0, parse_dates=True)
    >>> dt = get_datetime_column_or_index(df)  # Gets from index
    >>> print(dt.min(), dt.max())
    """
    # Try specified column first
    if timestamp_col and timestamp_col in df.columns:
        return pd.to_datetime(df[timestamp_col])

    # Try to detect column
    detected = detect_datetime_column(df)
    if detected:
        return pd.to_datetime(df[detected])

    # Try index
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)

    # Try to convert index
    try:
        return pd.to_datetime(df.index)
    except (ValueError, TypeError):
        pass

    raise ValueError("No datetime column or index found in DataFrame")


def has_datetime_column(df: pd.DataFrame) -> bool:
    """
    Check if DataFrame has a datetime column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check

    Returns
    -------
    bool
        True if datetime column found, False otherwise

    Examples
    --------
    >>> if has_datetime_column(df):
    ...     print("Ready for time series analysis")
    """
    return detect_datetime_column(df) is not None


# Convenience function for the most common use case
def prepare_timeseries(
    df: pd.DataFrame,
    timestamp_col: Optional[str] = None,
    set_index: bool = True,
    sort: bool = True,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Prepare DataFrame for time series analysis.

    Combines common operations:
    1. Detect/validate datetime column
    2. Convert to datetime type
    3. Optionally set as index
    4. Optionally sort by datetime
    5. Optionally remove duplicate timestamps

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to prepare
    timestamp_col : str, optional
        Name of timestamp column. If None, auto-detects.
    set_index : bool, default True
        Set datetime column as index
    sort : bool, default True
        Sort by datetime
    drop_duplicates : bool, default True
        Remove duplicate timestamps

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame

    Examples
    --------
    >>> df = pd.read_csv('data.csv')
    >>> df = prepare_timeseries(df)  # One-liner to prepare data
    >>> # Now ready for analysis
    """
    df = df.copy()

    # Ensure datetime column exists and is correct type
    df = ensure_datetime_column(df, timestamp_col=timestamp_col, inplace=True)

    # Get the actual column name (in case it was auto-detected)
    if timestamp_col is None:
        timestamp_col = detect_datetime_column(df)

    # Sort by datetime if requested
    if sort and timestamp_col in df.columns:
        df = df.sort_values(timestamp_col)

    # Remove duplicates if requested
    if drop_duplicates:
        if set_index:
            # Will drop duplicate indices after setting index
            pass
        elif timestamp_col in df.columns:
            df = df.drop_duplicates(subset=[timestamp_col], keep="first")

    # Set as index if requested
    if set_index and timestamp_col in df.columns:
        df = df.set_index(timestamp_col)
        if drop_duplicates:
            df = df[~df.index.duplicated(keep="first")]

    return df


# Example usage and tests
if __name__ == "__main__":
    import numpy as np

    print("Testing datetime column utilities...")

    # Test 1: Detection with standard name
    df1 = pd.DataFrame(
        {
            "TIMESTAMP": pd.date_range("2024-01-01", periods=10, freq="H"),
            "00010": np.random.randn(10),
        }
    )

    detected = detect_datetime_column(df1)
    print(f"✓ Test 1 - Detected: {detected}")
    assert detected == "TIMESTAMP"

    # Test 2: Detection with alternative name
    df2 = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=10, freq="H"),
            "00060": np.random.randn(10),
        }
    )

    detected = detect_datetime_column(df2)
    print(f"✓ Test 2 - Detected: {detected}")
    assert detected == "datetime"

    # Test 3: Standardization
    df3 = standardize_datetime_column(df2)
    print(f"✓ Test 3 - Standardized columns: {list(df3.columns)}")
    assert "TIMESTAMP" in df3.columns
    assert "datetime" not in df3.columns

    # Test 4: Prepare for time series
    df4 = prepare_timeseries(df2)
    print(f"✓ Test 4 - Prepared: index type = {type(df4.index).__name__}")
    assert isinstance(df4.index, pd.DatetimeIndex)

    print("\n✓ All tests passed!")
    print(f"\nStandard column name for data files: {STANDARD_DATA_COLUMN}")
    print(f"Standard parameter name for code: {STANDARD_PARAM_NAME}")
