#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for Water Quality Data

Provides standardized preprocessing including:
- Min-Max scaling (0-1) for parameters with known bounds
- Standardization (mean=0, std=1) for parameters without fixed bounds
- Intelligent imputation for small gaps in time series
- Conversion to efficient Parquet format for storage
- Metadata tracking for reproducibility

Usage:
    from core.data.preprocessing_pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline()
    df_processed = pipeline.fit_transform(df)
    pipeline.save_to_parquet(df_processed, 'output.parquet')

Author: TSLR
Date: 2026-02-09
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Comprehensive preprocessing pipeline for water quality data.

    Features:
    - Flexible scaling strategies (MinMax or Standardization)
    - Smart imputation for missing values
    - Parameter-specific preprocessing rules
    - Metadata tracking for reproducibility
    - Efficient Parquet export
    """

    # Define parameters with known physical bounds (use MinMax scaling)
    BOUNDED_PARAMETERS = {
        "00400": (0, 14, "pH"),  # pH: 0-14
        "00300": (0, 20, "Dissolved_Oxygen_mg_L"),  # DO: 0-20 mg/L (saturation ~14)
        "32359": (0, 100, "fDOM_QSU"),  # fDOM: 0-100 QSU typical
        "00010": (-5, 50, "Temperature_C"),  # Water temp: -5 to 50°C
        "63680": (0, 1000, "Turbidity_FNU"),  # Turbidity: 0-1000 FNU
        "00065": (0, 50, "Gage_Height_ft"),  # Gage height: 0-50 ft typical
    }

    # Define parameters without fixed bounds (use Standardization)
    UNBOUNDED_PARAMETERS = {
        "00060": "Discharge_cfs",
        "00095": "Specific_Conductance_uS_cm",
        "99407": "EColi_CFU_100mL",
        "00045": "Precipitation_in",
    }

    def __init__(
        self,
        scaling_strategy: str = "auto",
        impute_method: str = "knn",
        max_gap_hours: float = 6.0,
        knn_neighbors: int = 5,
        verbose: bool = True,
    ):
        """
        Initialize preprocessing pipeline.

        Parameters:
        -----------
        scaling_strategy : str
            'auto' (use parameter-specific), 'minmax', 'standard', or 'none'
        impute_method : str
            'knn', 'linear', 'forward_fill', or 'none'
        max_gap_hours : float
            Maximum gap size to impute (hours)
        knn_neighbors : int
            Number of neighbors for KNN imputation
        verbose : bool
            Print progress messages
        """
        self.scaling_strategy = scaling_strategy
        self.impute_method = impute_method
        self.max_gap_hours = max_gap_hours
        self.knn_neighbors = knn_neighbors
        self.verbose = verbose

        # Storage for fitted scalers and metadata
        self.scalers_ = {}
        self.imputers_ = {}
        self.metadata_ = {
            "created": datetime.now().isoformat(),
            "scaling_strategy": scaling_strategy,
            "impute_method": impute_method,
            "max_gap_hours": max_gap_hours,
            "parameters_processed": [],
            "original_shape": None,
            "final_shape": None,
            "missing_before": {},
            "missing_after": {},
        }

    def _identify_parameter_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify which columns correspond to which parameter types.

        Returns dict with 'bounded', 'unbounded', and 'other' lists.
        """
        result = {"bounded": [], "unbounded": [], "other": []}

        for col in df.columns:
            # Skip timestamp and metadata columns
            if col.lower() in ["timestamp", "datetime", "date", "time", "site_id", "station_id"]:
                continue

            # Check if it's a known bounded parameter
            matched = False
            for param_code in self.BOUNDED_PARAMETERS:
                if param_code in col or self.BOUNDED_PARAMETERS[param_code][2] in col:
                    result["bounded"].append(col)
                    matched = True
                    break

            if matched:
                continue

            # Check if it's a known unbounded parameter
            for param_code in self.UNBOUNDED_PARAMETERS:
                if param_code in col or self.UNBOUNDED_PARAMETERS[param_code] in col:
                    result["unbounded"].append(col)
                    matched = True
                    break

            if not matched:
                result["other"].append(col)

        return result

    def _detect_gaps(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Detect gaps in time series and mark them for selective imputation.

        Returns DataFrame with gap information.
        """
        if timestamp_col not in df.columns:
            logger.warning(f"Timestamp column '{timestamp_col}' not found")
            return pd.DataFrame()

        df_sorted = df.sort_values(timestamp_col).copy()
        df_sorted["time_diff_hours"] = df_sorted[timestamp_col].diff().dt.total_seconds() / 3600

        # Identify gaps larger than threshold
        large_gaps = df_sorted[df_sorted["time_diff_hours"] > self.max_gap_hours]

        if self.verbose and len(large_gaps) > 0:
            logger.info(f"Found {len(large_gaps)} gaps larger than {self.max_gap_hours} hours")

        return large_gaps

    def _impute_small_gaps(
        self, df: pd.DataFrame, columns: List[str], timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Impute only small gaps in the data based on max_gap_hours threshold.

        Large gaps are left as NaN to avoid creating artificial patterns.
        """
        df_imputed = df.copy()

        if self.impute_method == "none":
            return df_imputed

        # Detect gaps
        gaps = self._detect_gaps(df, timestamp_col)

        if self.verbose:
            print(f"\n{'='*60}", flush=True)
            print(f"Starting imputation for {len(columns)} parameters", flush=True)
            print(f"Method: {self.impute_method} (two-pass optimization)", flush=True)
            print(f"{'='*60}\n", flush=True)

        # Progress tracking
        try:
            from tqdm import tqdm

            columns_iter = tqdm(columns, desc="Imputing parameters", disable=not self.verbose)
            use_tqdm = True
        except ImportError:
            columns_iter = columns
            use_tqdm = False
            if self.verbose:
                print("💡 Install tqdm for progress bar: pip install tqdm\n", flush=True)

        for col_idx, col in enumerate(columns_iter, 1):
            if col not in df.columns:
                continue

            missing_count = df[col].isna().sum()
            if missing_count == 0:
                if self.verbose and not use_tqdm:
                    print(
                        f"[{col_idx}/{len(columns)}] {col}: No missing values - skipping",
                        flush=True,
                    )
                continue

            if self.verbose and not use_tqdm:
                print(f"\n[{col_idx}/{len(columns)}] Processing {col}:", flush=True)
                print(
                    f"  Missing: {missing_count:,} values ({missing_count/len(df)*100:.1f}%)",
                    flush=True,
                )

            # Store original NaN positions
            original_nans = df[col].isna()

            if self.impute_method == "linear":
                # Time-based linear interpolation (only small gaps)
                df_imputed[col] = df_imputed[col].interpolate(
                    method="time",
                    limit=int(self.max_gap_hours * 4),  # Assuming 15-min data
                    limit_direction="both",
                )

            elif self.impute_method == "forward_fill":
                # Forward fill (good for precipitation). pandas 3.0 removed the
                # fillna(method=...) keyword; use .ffill() directly.
                df_imputed[col] = df_imputed[col].ffill(limit=int(self.max_gap_hours * 4))

            elif self.impute_method == "knn":
                # Two-pass approach for speed and quality:
                # Pass 1: Fast linear interpolation for tiny gaps (up to 3 points)
                # Pass 2: KNN for remaining gaps up to max_gap_hours (better quality)

                # Pass 1: Fill tiny isolated gaps (much faster than KNN)
                tiny_gap_limit = 3  # Fill gaps of 1-3 consecutive missing points
                df_imputed[col] = df_imputed[col].interpolate(
                    method="linear", limit=tiny_gap_limit, limit_direction="both"
                )

                if self.verbose:
                    after_linear = df_imputed[col].isna().sum()
                    filled = missing_count - after_linear
                    if filled > 0 and not use_tqdm:
                        print(
                            f"  ↳ Pass 1 (linear): Filled {filled:,} tiny gaps (≤3 points) in <1s",
                            flush=True,
                        )

                # Pass 2: KNN for remaining gaps up to max_gap_hours
                remaining_missing = df_imputed[col].isna().sum()
                if remaining_missing > 0:
                    if self.verbose and not use_tqdm:
                        print(
                            f"  ↳ Pass 2 (KNN): Processing {remaining_missing:,} remaining values...",
                            flush=True,
                        )

                    # Use only columns without too many missing values
                    valid_cols = [
                        c
                        for c in columns
                        if c in df_imputed.columns and df_imputed[c].notna().sum() > len(df) * 0.3
                    ]

                    if len(valid_cols) < 2:
                        # Fall back to linear interpolation with max_gap_hours limit
                        max_limit = int(self.max_gap_hours * 4)  # Assuming 15-min data
                        df_imputed[col] = df_imputed[col].interpolate(
                            method="linear", limit=max_limit
                        )
                    else:
                        try:
                            import time

                            knn_start = time.time()

                            # Apply KNN only to gaps within max_gap_hours
                            if self.verbose and not use_tqdm:
                                print(
                                    f"     Running KNN (k={self.knn_neighbors}) on {len(valid_cols)} parameters...",
                                    flush=True,
                                )
                                print(
                                    "     ⏳ This may take 30s-2min depending on data size...",
                                    flush=True,
                                )

                            imputer = KNNImputer(n_neighbors=self.knn_neighbors, weights="distance")
                            df_imputed[valid_cols] = imputer.fit_transform(df_imputed[valid_cols])

                            knn_elapsed = time.time() - knn_start

                            if self.verbose:
                                after_knn = df_imputed[col].isna().sum()
                                filled_knn = remaining_missing - after_knn
                                if filled_knn > 0 and not use_tqdm:
                                    print(
                                        f"     ✓ KNN complete: Filled {filled_knn:,} gaps in {knn_elapsed:.1f}s",
                                        flush=True,
                                    )
                        except Exception as e:
                            logger.warning(f"KNN imputation failed for {col}: {e}. Using linear.")
                            max_limit = int(self.max_gap_hours * 4)
                            df_imputed[col] = df_imputed[col].interpolate(
                                method="linear", limit=max_limit
                            )

            imputed_count = original_nans.sum() - df_imputed[col].isna().sum()
            if self.verbose and imputed_count > 0 and not use_tqdm:
                print(
                    f"  ✓ Total: {imputed_count:,} values imputed ({imputed_count/original_nans.sum()*100:.1f}% of missing)\n",
                    flush=True,
                )

            # Restore NaNs for large gaps (don't impute across them)
            if timestamp_col in df.columns:
                # Mark regions near large gaps as NaN
                for idx in gaps.index:
                    # Get window around large gap
                    gap_start = idx - pd.Timedelta(hours=self.max_gap_hours)
                    gap_end = idx + pd.Timedelta(hours=self.max_gap_hours)
                    mask = (df_imputed.index >= gap_start) & (df_imputed.index <= gap_end)

                    # Restore original NaNs in gap region
                    df_imputed.loc[mask & original_nans, col] = np.nan

        if self.verbose:
            print(f"\n{'='*60}", flush=True)
            print(f"✓ Imputation complete for {len(columns)} parameters", flush=True)
            print(f"{'='*60}\n", flush=True)

        return df_imputed

    def _scale_parameters(
        self, df: pd.DataFrame, param_dict: Dict[str, List[str]], fit: bool = True
    ) -> pd.DataFrame:
        """
        Apply appropriate scaling to each parameter type.

        Parameters:
        -----------
        df : DataFrame
            Input data
        param_dict : dict
            Dict with 'bounded' and 'unbounded' parameter lists
        fit : bool
            Whether to fit scalers (True) or use existing (False)
        """
        df_scaled = df.copy()

        if self.scaling_strategy == "none":
            return df_scaled

        # Scale bounded parameters with MinMax (0-1)
        if self.scaling_strategy in ["auto", "minmax"]:
            for col in param_dict["bounded"]:
                if col not in df.columns:
                    continue

                # Find parameter bounds
                param_code = None
                for code in self.BOUNDED_PARAMETERS:
                    if code in col or self.BOUNDED_PARAMETERS[code][2] in col:
                        param_code = code
                        break

                if fit:
                    if param_code:
                        # Use known physical bounds
                        min_val, max_val, name = self.BOUNDED_PARAMETERS[param_code]
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        # Fit with theoretical bounds
                        scaler.fit([[min_val], [max_val]])
                    else:
                        # Use data-driven bounds
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        valid_data = df[col].dropna().values.reshape(-1, 1)
                        if len(valid_data) > 0:
                            scaler.fit(valid_data)
                        else:
                            continue

                    self.scalers_[col] = scaler
                else:
                    scaler = self.scalers_.get(col)
                    if scaler is None:
                        logger.warning(f"No fitted scaler for {col}, skipping")
                        continue

                # Transform
                valid_mask = df_scaled[col].notna()
                if valid_mask.sum() > 0:
                    df_scaled.loc[valid_mask, col] = scaler.transform(
                        df_scaled.loc[valid_mask, col].values.reshape(-1, 1)
                    ).flatten()

                if self.verbose and fit:
                    logger.info(f"Scaled {col} with MinMaxScaler (0-1)")

        # Scale unbounded parameters with Standardization (z-score)
        if self.scaling_strategy in ["auto", "standard"]:
            unbounded_cols = param_dict["unbounded"]
            if self.scaling_strategy == "standard":
                # If 'standard' mode, also include 'other' parameters
                unbounded_cols = unbounded_cols + param_dict["other"]

            for col in unbounded_cols:
                if col not in df.columns:
                    continue

                if fit:
                    scaler = StandardScaler()
                    valid_data = df[col].dropna().values.reshape(-1, 1)
                    if len(valid_data) > 0:
                        scaler.fit(valid_data)
                    else:
                        continue

                    self.scalers_[col] = scaler
                else:
                    scaler = self.scalers_.get(col)
                    if scaler is None:
                        logger.warning(f"No fitted scaler for {col}, skipping")
                        continue

                # Transform
                valid_mask = df_scaled[col].notna()
                if valid_mask.sum() > 0:
                    df_scaled.loc[valid_mask, col] = scaler.transform(
                        df_scaled.loc[valid_mask, col].values.reshape(-1, 1)
                    ).flatten()

                if self.verbose and fit:
                    logger.info(f"Scaled {col} with StandardScaler (z-score)")

        return df_scaled

    def fit_transform(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Fit preprocessing pipeline and transform data.

        Parameters:
        -----------
        df : DataFrame
            Input water quality data with timestamp index or column
        timestamp_col : str
            Name of timestamp column

        Returns:
        --------
        DataFrame
            Preprocessed data
        """
        if self.verbose:
            logger.info("Starting preprocessing pipeline...")
            logger.info(f"Input shape: {df.shape}")

        # Store metadata
        self.metadata_["original_shape"] = df.shape

        # Ensure timestamp is datetime
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index(timestamp_col)

        # Identify parameter types
        param_dict = self._identify_parameter_columns(df)

        if self.verbose:
            logger.info(f"Identified {len(param_dict['bounded'])} bounded parameters")
            logger.info(f"Identified {len(param_dict['unbounded'])} unbounded parameters")
            logger.info(f"Identified {len(param_dict['other'])} other parameters")

        # Track missing values before imputation
        for col in df.columns:
            self.metadata_["missing_before"][col] = int(df[col].isna().sum())

        # Step 1: Impute small gaps
        all_param_cols = param_dict["bounded"] + param_dict["unbounded"] + param_dict["other"]
        df_imputed = self._impute_small_gaps(df, all_param_cols, timestamp_col)

        # Track missing values after imputation
        for col in df_imputed.columns:
            self.metadata_["missing_after"][col] = int(df_imputed[col].isna().sum())

        # Step 2: Scale parameters
        df_scaled = self._scale_parameters(df_imputed, param_dict, fit=True)

        # Store final metadata
        self.metadata_["final_shape"] = df_scaled.shape
        self.metadata_["parameters_processed"] = list(df_scaled.columns)

        if self.verbose:
            logger.info(f"Preprocessing complete. Output shape: {df_scaled.shape}")
            total_before = sum(self.metadata_["missing_before"].values())
            total_after = sum(self.metadata_["missing_after"].values())
            if total_before > 0:
                logger.info(
                    f"Missing values: {total_before} → {total_after} ({(1-total_after/total_before)*100:.1f}% reduction)"
                )

        return df_scaled

    def transform(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Transform new data using fitted pipeline.

        Parameters:
        -----------
        df : DataFrame
            Input data
        timestamp_col : str
            Name of timestamp column

        Returns:
        --------
        DataFrame
            Transformed data
        """
        # Ensure timestamp is datetime
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index(timestamp_col)

        # Identify parameter types
        param_dict = self._identify_parameter_columns(df)

        # Apply imputation
        all_param_cols = param_dict["bounded"] + param_dict["unbounded"] + param_dict["other"]
        df_imputed = self._impute_small_gaps(df, all_param_cols, timestamp_col)

        # Apply scaling (using fitted scalers)
        df_scaled = self._scale_parameters(df_imputed, param_dict, fit=False)

        return df_scaled

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse scaling transformations to get back to original scale.

        Parameters:
        -----------
        df : DataFrame
            Scaled data

        Returns:
        --------
        DataFrame
            Data in original scale
        """
        df_original = df.copy()

        for col in df.columns:
            if col in self.scalers_:
                scaler = self.scalers_[col]
                valid_mask = df_original[col].notna()

                if valid_mask.sum() > 0:
                    df_original.loc[valid_mask, col] = scaler.inverse_transform(
                        df_original.loc[valid_mask, col].values.reshape(-1, 1)
                    ).flatten()

        return df_original

    def save_to_parquet(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        compression: str = "snappy",
        save_metadata: bool = True,
    ) -> Path:
        """
        Save preprocessed data to Parquet format.

        Parameters:
        -----------
        df : DataFrame
            Preprocessed data
        output_path : str or Path
            Output file path
        compression : str
            Compression algorithm ('snappy', 'gzip', 'brotli', 'none')
        save_metadata : bool
            Whether to save preprocessing metadata alongside data

        Returns:
        --------
        Path
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to PyArrow table for efficient storage
        table = pa.Table.from_pandas(df, preserve_index=True)

        # Write to Parquet
        pq.write_table(
            table,
            output_path,
            compression=compression,
            use_dictionary=True,  # Efficient for categorical data
            write_statistics=True,
        )

        if self.verbose:
            logger.info(f"Saved to {output_path} (compression={compression})")

        # Save metadata if requested
        if save_metadata:
            metadata_path = output_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(self.metadata_, f, indent=2)

            if self.verbose:
                logger.info(f"Saved metadata to {metadata_path}")

        return output_path

    def save_scaler_state(self, output_path: Union[str, Path]) -> Path:
        """
        Save fitted scalers for later use.

        Parameters:
        -----------
        output_path : str or Path
            Output file path

        Returns:
        --------
        Path
            Path to saved file
        """
        import joblib

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "scalers": self.scalers_,
            "metadata": self.metadata_,
            "config": {
                "scaling_strategy": self.scaling_strategy,
                "impute_method": self.impute_method,
                "max_gap_hours": self.max_gap_hours,
                "knn_neighbors": self.knn_neighbors,
            },
        }

        joblib.dump(state, output_path)

        if self.verbose:
            logger.info(f"Saved pipeline state to {output_path}")

        return output_path

    @classmethod
    def load_from_state(cls, state_path: Union[str, Path]) -> "PreprocessingPipeline":
        """
        Load pipeline from saved state.

        Parameters:
        -----------
        state_path : str or Path
            Path to saved state file

        Returns:
        --------
        PreprocessingPipeline
            Loaded pipeline
        """
        import joblib

        state = joblib.load(state_path)

        # Create new pipeline with saved config
        pipeline = cls(**state["config"])
        pipeline.scalers_ = state["scalers"]
        pipeline.metadata_ = state["metadata"]

        logger.info(f"Loaded pipeline state from {state_path}")

        return pipeline


def batch_process_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    file_pattern: str = "*_clean.csv",
    **pipeline_kwargs,
) -> Dict[str, Path]:
    """
    Batch process all CSV files in a directory.

    Parameters:
    -----------
    input_dir : str or Path
        Directory containing input CSV files
    output_dir : str or Path
        Directory for output Parquet files
    file_pattern : str
        Glob pattern for matching files
    **pipeline_kwargs
        Arguments passed to PreprocessingPipeline

    Returns:
    --------
    dict
        Mapping of input filenames to output paths
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    files = list(input_dir.glob(file_pattern))

    logger.info(f"Found {len(files)} files to process")

    for file_path in files:
        try:
            logger.info(f"Processing {file_path.name}...")

            # Read CSV. Per the project datetime standard, data files use
            # `TIMESTAMP` (uppercase); code accepts both variants and
            # normalises to lowercase for downstream pipeline steps.
            df = pd.read_csv(file_path)
            ts_col = next((c for c in df.columns if c.lower() == "timestamp"), None)
            if ts_col is None:
                raise ValueError(
                    f"No timestamp/TIMESTAMP column found; got {list(df.columns)[:8]}..."
                )
            df[ts_col] = pd.to_datetime(df[ts_col])
            if ts_col != "timestamp":
                df = df.rename(columns={ts_col: "timestamp"})

            # Create pipeline
            pipeline = PreprocessingPipeline(**pipeline_kwargs)

            # Process
            df_processed = pipeline.fit_transform(df)

            # Save to Parquet
            output_path = output_dir / file_path.with_suffix(".parquet").name
            pipeline.save_to_parquet(df_processed, output_path)

            # Save pipeline state
            state_path = output_dir / f"{file_path.stem}_pipeline.joblib"
            pipeline.save_scaler_state(state_path)

            results[file_path.name] = output_path

            logger.info(f" Saved {output_path.name}")

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            continue

    logger.info(f"Batch processing complete: {len(results)}/{len(files)} files processed")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch preprocess clean CSVs to parquet")
    parser.add_argument("input_dir", help="Directory of input CSVs")
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Directory for output parquets (default: <input_dir>_processed)",
    )
    parser.add_argument("--pattern", default="*.csv", help="Glob for input files (default: *.csv)")
    parser.add_argument(
        "--impute",
        choices=["knn", "linear", "forward_fill", "none"],
        default="knn",
        help="Imputation method. 'linear' is ~30x faster than 'knn' but only fills small gaps.",
    )
    parser.add_argument(
        "--max-gap-hours", type=float, default=6.0, help="Maximum gap (in hours) to impute through"
    )
    args = parser.parse_args()

    batch_process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir or (args.input_dir + "_processed"),
        file_pattern=args.pattern,
        scaling_strategy="auto",
        impute_method=args.impute,
        max_gap_hours=args.max_gap_hours,
    )
