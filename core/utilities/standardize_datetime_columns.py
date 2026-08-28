#!/usr/bin/env python3
"""
Datetime Column Standardization Tool

This script analyzes and standardizes datetime/timestamp column naming across the entire
water quality data system. It addresses the inconsistency between 'TIMESTAMP', 'timestamp',
'datetime', and other variants.

STANDARD DECISION:
- Data files (.csv): Use 'TIMESTAMP' (uppercase) for consistency with existing clean data
- Code defaults: Use 'timestamp' (lowercase) as parameter defaults for flexibility
- Both patterns are supported in code, but data files should be standardized to TIMESTAMP

Author: TSLR
Created: 2026-02-09
Updated: 2026-02-09
"""

import pandas as pd
import os
import glob
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import argparse
import shutil


# Setup logging
def setup_logging(log_dir: str = "logs") -> str:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"datetime_standardization_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return log_file


class DatetimeStandardizer:
    """Analyzes and standardizes datetime column naming across the system."""

    # Common datetime column variants
    DATETIME_VARIANTS = [
        "TIMESTAMP",
        "timestamp",
        "Timestamp",
        "DATETIME",
        "datetime",
        "DateTime",
        "date_time",
        "dateTime",
        "date",
        "Date",
        "DATE",
        "time",
        "Time",
        "TIME",
    ]

    # Standard column name for data files
    STANDARD_DATA_COL = "TIMESTAMP"

    # Standard parameter name for code
    STANDARD_PARAM = "timestamp"

    def __init__(self, base_dir: str = None):
        """Initialize the standardizer."""
        if base_dir is None:
            # Auto-detect base directory
            current_file = Path(__file__).resolve()
            self.base_dir = current_file.parent.parent.parent
        else:
            self.base_dir = Path(base_dir)

        self.data_dir = self.base_dir.parent / "data" / "pollution_id" / "data"
        self.results = {
            "files_analyzed": 0,
            "files_with_issues": [],
            "column_usage": {},
            "code_files": [],
            "recommendations": [],
        }

    def detect_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect which column is the datetime column."""
        for col in df.columns:
            if col in self.DATETIME_VARIANTS:
                return col
            # Check if column contains datetime-like data
            if any(keyword in str(col).lower() for keyword in ["time", "date"]):
                try:
                    pd.to_datetime(df[col].head())
                    return col
                except Exception:
                    pass
        return None

    def analyze_csv_file(self, file_path: Path) -> Dict:
        """Analyze a single CSV file for datetime column usage."""
        try:
            # Read just the header
            df = pd.read_csv(file_path, nrows=0)

            datetime_col = self.detect_datetime_column(df)

            result = {
                "file": str(file_path),
                "datetime_column": datetime_col,
                "all_columns": list(df.columns),
                "needs_fix": datetime_col not in [self.STANDARD_DATA_COL, None],
                "size_mb": file_path.stat().st_size / (1024 * 1024),
            }

            if datetime_col:
                if datetime_col not in self.results["column_usage"]:
                    self.results["column_usage"][datetime_col] = 0
                self.results["column_usage"][datetime_col] += 1

            return result

        except Exception as e:
            logging.error(f"Error analyzing {file_path}: {e}")
            return {"file": str(file_path), "error": str(e)}

    def analyze_data_directory(self, directory: Path = None) -> List[Dict]:
        """Analyze all CSV files in a directory."""
        if directory is None:
            directory = self.data_dir / "clean"

        csv_files = list(directory.glob("*.csv"))
        logging.info(f"Found {len(csv_files)} CSV files in {directory}")

        results = []
        for csv_file in csv_files:
            result = self.analyze_csv_file(csv_file)
            results.append(result)
            self.results["files_analyzed"] += 1

            if result.get("needs_fix", False):
                self.results["files_with_issues"].append(result)

        return results

    def analyze_code_files(self) -> List[Dict]:
        """Analyze Python files for datetime parameter usage."""
        code_patterns = [
            str(self.base_dir / "core" / "**" / "*.py"),
            str(self.base_dir / "tests" / "**" / "*.py"),
            str(self.base_dir / "water_quality_data_platform" / "**" / "*.py"),
        ]

        code_results = []
        for pattern in code_patterns:
            for py_file in glob.glob(pattern, recursive=True):
                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Check for datetime column references
                    datetime_refs = []
                    for variant in self.DATETIME_VARIANTS:
                        # Look for parameter definitions and usage
                        if f"'{variant}'" in content or f'"{variant}"' in content:
                            count = content.count(f"'{variant}'") + content.count(f'"{variant}"')
                            datetime_refs.append({"variant": variant, "count": count})

                    if datetime_refs:
                        code_results.append({"file": py_file, "references": datetime_refs})

                except Exception as e:
                    logging.debug(f"Error reading {py_file}: {e}")

        self.results["code_files"] = code_results
        return code_results

    def fix_csv_file(self, file_path: Path, backup: bool = True, dry_run: bool = False) -> bool:
        """Fix datetime column naming in a CSV file."""
        try:
            # Read the file
            df = pd.read_csv(file_path)

            # Detect datetime column
            datetime_col = self.detect_datetime_column(df)

            if datetime_col is None:
                logging.warning(f"No datetime column found in {file_path}")
                return False

            if datetime_col == self.STANDARD_DATA_COL:
                logging.debug(f"File already has standard column name: {file_path}")
                return False

            if dry_run:
                logging.info(
                    f"[DRY RUN] Would rename '{datetime_col}' to '{self.STANDARD_DATA_COL}' in {file_path}"
                )
                return True

            # Backup if requested
            if backup:
                backup_path = file_path.with_suffix(".csv.bak")
                shutil.copy2(file_path, backup_path)
                logging.debug(f"Created backup: {backup_path}")

            # Rename column
            df.rename(columns={datetime_col: self.STANDARD_DATA_COL}, inplace=True)

            # Save file
            df.to_csv(file_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
            logging.info(f"✓ Fixed {file_path.name}: '{datetime_col}' → '{self.STANDARD_DATA_COL}'")

            return True

        except Exception as e:
            logging.error(f"Error fixing {file_path}: {e}")
            return False

    def fix_directory(
        self,
        directory: Path = None,
        backup: bool = True,
        dry_run: bool = False,
        max_files: int = None,
    ) -> Dict:
        """Fix all CSV files in a directory."""
        if directory is None:
            directory = self.data_dir / "clean"

        # Analyze first
        logging.info(f"Analyzing files in {directory}...")
        analysis = self.analyze_data_directory(directory)

        files_to_fix = [item for item in analysis if item.get("needs_fix", False)]

        if not files_to_fix:
            logging.info("✓ No files need fixing - all are already standardized!")
            return {"fixed": 0, "failed": 0, "skipped": 0}

        logging.info(f"Found {len(files_to_fix)} files that need fixing")

        if max_files:
            files_to_fix = files_to_fix[:max_files]
            logging.info(f"Limiting to first {max_files} files")

        fixed = 0
        failed = 0

        for item in files_to_fix:
            if "error" in item:
                continue

            file_path = Path(item["file"])
            if self.fix_csv_file(file_path, backup=backup, dry_run=dry_run):
                fixed += 1
            else:
                failed += 1

        result = {"fixed": fixed, "failed": failed, "total_analyzed": len(analysis)}

        return result

    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive report of findings."""
        report = []
        report.append("=" * 80)
        report.append("DATETIME COLUMN STANDARDIZATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary
        report.append("SUMMARY")
        report.append("-" * 80)
        report.append(f"Files analyzed: {self.results['files_analyzed']}")
        report.append(f"Files with non-standard columns: {len(self.results['files_with_issues'])}")
        report.append("")

        # Column usage breakdown
        report.append("COLUMN NAME USAGE")
        report.append("-" * 80)
        for col_name, count in sorted(
            self.results["column_usage"].items(), key=lambda x: x[1], reverse=True
        ):
            status = "✓ STANDARD" if col_name == self.STANDARD_DATA_COL else "✗ NON-STANDARD"
            report.append(f"{col_name:20s} : {count:5d} files  {status}")
        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 80)
        report.append(f"1. Data files should use: '{self.STANDARD_DATA_COL}' (uppercase)")
        report.append(f"2. Code parameters should default to: '{self.STANDARD_PARAM}' (lowercase)")
        report.append("3. Code should accept both variants for backward compatibility")
        report.append("")

        # Files needing attention
        if self.results["files_with_issues"]:
            report.append("FILES NEEDING STANDARDIZATION")
            report.append("-" * 80)
            report.append(f"Total: {len(self.results['files_with_issues'])} files")
            report.append("")

            # Show first 10 as examples
            for item in self.results["files_with_issues"][:10]:
                report.append(f"  {Path(item['file']).name}")
                report.append(
                    f"    Current: '{item['datetime_column']}' → Should be: '{self.STANDARD_DATA_COL}'"
                )

            if len(self.results["files_with_issues"]) > 10:
                report.append(f"  ... and {len(self.results['files_with_issues']) - 10} more")

        report.append("")
        report.append("=" * 80)

        report_text = "\n".join(report)

        # Print to console
        print(report_text)

        # Save to file if requested
        if output_file:
            with open(output_file, "w") as f:
                f.write(report_text)
            logging.info(f"Report saved to: {output_file}")

        return report_text


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Standardize datetime column naming across water quality data files"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze files without making changes"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Fix datetime column naming in data files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without actually changing",
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Do not create backup files (.bak)"
    )
    parser.add_argument(
        "--directory", type=str, help="Specific directory to process (default: data/clean)"
    )
    parser.add_argument(
        "--max-files", type=int, help="Maximum number of files to fix (for testing)"
    )
    parser.add_argument("--report", type=str, help="Save report to specified file")

    args = parser.parse_args()

    # Set up logging
    log_file = setup_logging()
    logging.info("=" * 80)
    logging.info("Datetime Column Standardization Tool")
    logging.info("=" * 80)

    # Initialize standardizer
    standardizer = DatetimeStandardizer()

    # Determine directory to process
    if args.directory:
        target_dir = Path(args.directory)
    else:
        target_dir = standardizer.data_dir / "clean"

    logging.info(f"Target directory: {target_dir}")

    # Run analysis
    if args.analyze or args.dry_run or not args.fix:
        logging.info("\nRunning analysis...")
        standardizer.analyze_data_directory(target_dir)

        logging.info("\nAnalyzing code files...")
        standardizer.analyze_code_files()

        # Generate report
        standardizer.generate_report(output_file=args.report)

    # Run fixes if requested
    if args.fix:
        if args.dry_run:
            logging.info("\n" + "=" * 80)
            logging.info("DRY RUN MODE - No files will be modified")
            logging.info("=" * 80)

        result = standardizer.fix_directory(
            directory=target_dir,
            backup=not args.no_backup,
            dry_run=args.dry_run,
            max_files=args.max_files,
        )

        logging.info("\n" + "=" * 80)
        logging.info("RESULTS")
        logging.info("=" * 80)
        logging.info(f"Files analyzed: {result['total_analyzed']}")
        logging.info(f"Files fixed: {result['fixed']}")
        logging.info(f"Files failed: {result['failed']}")

        if not args.dry_run:
            if not args.no_backup:
                logging.info("\nBackup files created with .bak extension")
            logging.info(f"Log file: {log_file}")

    logging.info("\nDone!")


if __name__ == "__main__":
    main()
