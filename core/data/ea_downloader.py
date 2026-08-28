#!/usr/bin/env python3
"""
Enhanced EA Data Batch Downloader with Compatible Output Format

This script downloads environmental data from the UK Environment Agency API
and formats it to match standard tabular data structure with timestamp-based rows.

Features:
- Downloads data for all appropriate EA sites (1,609 sites)
- Handles proper date ranges (closed sites: date_opened to date_closed, active sites: date_opened to today)
- Converts from parameter-per-row to timestamp-per-row format (compatible with analysis tools)
- Implements timestamp rounding and filtering algorithms
- Automatic timestep detection and data alignment
- Progress tracking and error handling
- Pausable/resumable operation with download status tracking

Usage:
    python core/ea_downloader.py [--site-id SITE_ID] [--force] [--timestep MINUTES]

    --site-id: Process specific site only (for testing)
    --force: Overwrite existing files
    --timestep: Force specific timestep in minutes (default: auto-detect)
"""

import json
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import argparse
import os
import sys
from collections import defaultdict
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    try:
        from .parameters_ea_usgs_toolkit import EA_TO_USGS_PARAMETERS, create_parameter_mapping_dict
    except ImportError:
        # Support direct execution: ``python core/data/ea_downloader.py``.
        from parameters_ea_usgs_toolkit import EA_TO_USGS_PARAMETERS, create_parameter_mapping_dict
    logger.info("✅ Loaded enhanced EA-USGS parameter mappings")
    ENHANCED_MAPPINGS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Enhanced parameter mappings not available, using legacy mapping")
    EA_TO_USGS_PARAMETERS = None
    ENHANCED_MAPPINGS_AVAILABLE = False


class EADataProcessor:
    """Class to handle EA data downloading and USGS format conversion"""

    def __init__(self, base_url: str = "https://environment.data.gov.uk/hydrology"):
        self.base_url = base_url
        self.headers = {"User-Agent": "EA-Data-Fetcher/1.0", "Accept": "application/json"}

    def detect_timestep(self, timestamps: List[datetime]) -> int:
        """
        Detect the most common timestep interval in minutes

        Args:
            timestamps: List of datetime objects

        Returns:
            int: Timestep in minutes
        """
        if len(timestamps) < 2:
            return 15  # Default to 15 minutes

        # Calculate intervals between consecutive timestamps
        intervals = []
        for i in range(1, len(timestamps)):
            delta = timestamps[i] - timestamps[i - 1]
            intervals.append(delta.total_seconds() / 60)  # Convert to minutes

        # Find the most common interval (rounded to nearest minute)
        rounded_intervals = [round(interval) for interval in intervals]

        # Count occurrences
        interval_counts = defaultdict(int)
        for interval in rounded_intervals:
            interval_counts[interval] += 1

        # Get most common interval
        most_common_interval = max(interval_counts.items(), key=lambda x: x[1])[0]

        logger.info(f"Detected timestep: {most_common_interval} minutes")
        return most_common_interval

    def round_timestamp(self, timestamp: datetime, timestep_minutes: int) -> datetime:
        """
        Round timestamp to nearest timestep interval

        Args:
            timestamp: Original timestamp
            timestep_minutes: Timestep in minutes

        Returns:
            datetime: Rounded timestamp
        """
        # Calculate minutes since midnight
        total_minutes = timestamp.hour * 60 + timestamp.minute

        # Round to nearest timestep
        rounded_minutes = round(total_minutes / timestep_minutes) * timestep_minutes

        # Create new timestamp
        rounded_timestamp = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        rounded_timestamp += timedelta(minutes=rounded_minutes)

        return rounded_timestamp

    def generate_complete_timeline(
        self, start_time: datetime, end_time: datetime, timestep_minutes: int
    ) -> List[datetime]:
        """
        Generate complete timeline with regular intervals

        Args:
            start_time: Start datetime
            end_time: End datetime
            timestep_minutes: Timestep in minutes

        Returns:
            List[datetime]: Complete timeline
        """
        timeline = []
        current_time = start_time

        while current_time <= end_time:
            timeline.append(current_time)
            current_time += timedelta(minutes=timestep_minutes)

        return timeline

    def convert_to_usgs_format(
        self, raw_data: pd.DataFrame, timestep_minutes: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Convert EA raw data format to compatible tabular format

        Args:
            raw_data: DataFrame with EA raw data format
            timestep_minutes: Force specific timestep (None for auto-detect)

        Returns:
            pd.DataFrame: Compatible tabular format with timestamp-based rows
        """
        if raw_data.empty:
            return pd.DataFrame()

        # Parse datetime column
        raw_data["datetime_parsed"] = pd.to_datetime(raw_data["dateTime"])

        # Auto-detect timestep if not provided
        if timestep_minutes is None:
            timestep_minutes = self.detect_timestep(raw_data["datetime_parsed"].tolist())

        # Round timestamps
        raw_data["rounded_timestamp"] = raw_data["datetime_parsed"].apply(
            lambda x: self.round_timestamp(x, timestep_minutes)
        )

        # Get site ID and date range
        site_id = raw_data["station_id"].iloc[0]
        start_time = raw_data["rounded_timestamp"].min()
        end_time = raw_data["rounded_timestamp"].max()

        # Generate complete timeline
        complete_timeline = self.generate_complete_timeline(start_time, end_time, timestep_minutes)

        # Create parameter mapping for column names using enhanced mappings if available
        if ENHANCED_MAPPINGS_AVAILABLE and EA_TO_USGS_PARAMETERS:
            parameter_mapping = create_parameter_mapping_dict()
            logger.info(
                f"Using enhanced parameter mapping with {len(parameter_mapping)} parameters"
            )
        else:
            # Legacy parameter mapping (backwards compatibility)
            parameter_mapping = {
                "CONDUCTIVITY": "Conductance [uS/cm]",
                "TEMPERATURE": "TEMP [deg C]",
                "DISSOLVED OXYGEN": "DISSOLVED_OXYGEN [mg/L]",
                "PH": "pH [units]",
                "TURBIDITY": "Turbidity [NTU]",
                "AMMONIUM": "Ammonia [mg/L as N]",
                "NITRATE": "Nitrate [mg/L as N]",
                "CHLOROPHYLL": "Chlorophyll-a [ug/L]",
                "BLUE GREEN ALGAE": "Blue-green algae [cells/mL]",
                "SALINITY": "Salinity [PSU]",
                "FLUORESCENT DISSOLVED ORGANIC MATTER": "fDOM [QSU]",
            }
            logger.info(f"Using legacy parameter mapping with {len(parameter_mapping)} parameters")

        # Aggregate data by rounded timestamp and parameter
        aggregated_data = (
            raw_data.groupby(["rounded_timestamp", "parameter"])
            .agg(
                {
                    "value": "mean",  # Average if multiple values at same timestamp
                    "quality": "first",  # Take first quality flag
                }
            )
            .reset_index()
        )

        # Pivot to get parameters as columns
        pivoted_data = aggregated_data.pivot(
            index="rounded_timestamp", columns="parameter", values="value"
        ).reset_index()

        # Rename columns using parameter mapping
        column_rename = {}
        for col in pivoted_data.columns:
            if col in parameter_mapping:
                column_rename[col] = parameter_mapping[col]
            elif col == "rounded_timestamp":
                column_rename[col] = "TIMESTAMP"

        pivoted_data = pivoted_data.rename(columns=column_rename)

        # Create complete timeline DataFrame
        timeline_df = pd.DataFrame({"TIMESTAMP": complete_timeline})

        # Merge with data to fill missing timestamps
        usgs_format = timeline_df.merge(pivoted_data, on="TIMESTAMP", how="left")

        # Add Site_ID column at the beginning
        usgs_format.insert(0, "Site_ID", site_id)

        # Sort by timestamp
        usgs_format = usgs_format.sort_values("TIMESTAMP").reset_index(drop=True)

        logger.info(
            f"Converted {len(raw_data)} raw records to {len(usgs_format)} compatible format records"
        )
        logger.info(
            f"Parameters available: {[col for col in usgs_format.columns if col not in ['Site_ID', 'TIMESTAMP']]}"
        )

        return usgs_format

    def download_site_data(self, site_info: dict, start_date: str, end_date: str) -> pd.DataFrame:
        """Download data for a single site and all its measures"""

        site_id = site_info["station_id"]
        measures = site_info.get("measures", [])

        if not measures:
            logger.warning(f"No measures found for site {site_id}")
            return pd.DataFrame()

        all_data = []

        for measure in measures:
            measure_id = measure["measure_id"]
            parameter = measure["parameter"]

            try:
                logger.info(f"  Downloading {parameter} data for measure {measure_id}")

                # Download measure data
                measure_data = self._download_measure_data(measure_id, start_date, end_date)

                if not measure_data.empty:
                    # Add metadata
                    measure_data["measure_id"] = measure_id
                    measure_data["parameter"] = parameter
                    measure_data["station_id"] = site_id
                    all_data.append(measure_data)

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"    Error downloading {parameter} data: {str(e)}")
                continue

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"  Total records downloaded: {len(combined_data)}")
            return combined_data
        else:
            logger.warning(f"  No data downloaded for site {site_id}")
            return pd.DataFrame()

    def _download_measure_data(
        self, measure_id: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Download data for a specific measure"""

        url = f"{self.base_url}/data/readings"
        params = {
            "measure": measure_id,
            "mineq-date": start_date,
            "max-date": end_date,
            "_limit": 100000,  # Large limit to get all data
        }

        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=60)
            response.raise_for_status()

            data = response.json()

            if "items" not in data or not data["items"]:
                logger.warning(f"    No data items returned for measure {measure_id}")
                return pd.DataFrame()

            readings = []
            for item in data["items"]:
                readings.append(
                    {
                        "measure": item.get("measure", {}),
                        "date": item.get("date", ""),
                        "dateTime": item.get("dateTime", ""),
                        "value": item.get("value"),
                        "quality": item.get("quality", ""),
                    }
                )

            if readings:
                logger.debug(f"    Retrieved {len(readings)} readings for measure {measure_id}")
                return pd.DataFrame(readings)
            else:
                return pd.DataFrame()

        except requests.RequestException as e:
            logger.error(f"    Request failed: {str(e)}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"    Data parsing failed: {str(e)}")
            raise


def get_date_range_for_site(site_info: dict) -> Tuple[str, str]:
    """Get appropriate date range for a site based on its status"""

    date_opened = site_info.get("date_opened")
    date_closed = site_info.get("date_closed")

    if not date_opened:
        # Default start date if not available
        start_date = "2018-01-01"
    else:
        start_date = date_opened

    if date_closed:
        # Site is closed, use closing date
        end_date = date_closed
    else:
        # Site is active, use current date
        end_date = datetime.now().strftime("%Y-%m-%d")

    return start_date, end_date


def load_site_data(json_file: str) -> Tuple[List[dict], dict]:
    """Load site information from JSON file and return sites list and metadata"""
    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        # Handle nested structure
        if isinstance(data, dict) and "sites" in data:
            sites = data["sites"]
            meta = data.get("meta", {})
        elif isinstance(data, list):
            sites = data
            meta = {}
        else:
            logger.error(f"Unexpected JSON structure in {json_file}")
            sys.exit(1)

        return sites, meta

    except FileNotFoundError:
        logger.error(f"Site data file not found: {json_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in site data file: {str(e)}")
        sys.exit(1)


def save_site_data(json_file: str, sites: List[dict], meta: dict):
    """Save updated site data back to JSON file"""
    try:
        data = {"meta": meta, "sites": sites}

        # Create backup
        backup_file = f"{json_file}.backup"
        if os.path.exists(json_file):
            import shutil

            shutil.copy2(json_file, backup_file)
            logger.info(f"Created backup: {backup_file}")

        # Save updated data
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Updated site data saved to {json_file}")

    except Exception as e:
        logger.error(f"Error saving site data: {str(e)}")
        raise


def update_site_download_status(site: dict, success: bool, error_message: str = None):
    """Update download status for a site"""
    now = datetime.now().isoformat()

    if success:
        site["download_status"] = {
            "downloaded": True,
            "success": True,
            "last_attempt": now,
            "last_success": now,
            "error_message": None,
        }
    else:
        # Keep existing success status if this is just a retry
        existing_status = site.get("download_status", {})
        site["download_status"] = {
            "downloaded": existing_status.get("downloaded", False),
            "success": existing_status.get("success", False),
            "last_attempt": now,
            "last_success": existing_status.get("last_success"),
            "error_message": error_message,
        }


def is_site_downloaded(site: dict, force: bool = False) -> bool:
    """Check if site has already been successfully downloaded"""
    if force:
        return False

    status = site.get("download_status", {})
    return status.get("success", False)


def get_download_progress(sites: List[dict]) -> dict:
    """Calculate download progress statistics"""
    total = len(sites)
    downloaded = sum(1 for site in sites if site.get("download_status", {}).get("success", False))
    failed = sum(
        1
        for site in sites
        if site.get("download_status", {}).get("downloaded", False)
        and not site.get("download_status", {}).get("success", False)
    )
    remaining = total - downloaded - failed

    return {
        "total": total,
        "downloaded": downloaded,
        "failed": failed,
        "remaining": remaining,
        "progress_percent": (downloaded / total * 100) if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download EA data in compatible tabular format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--site-id", help="Download data for specific site only (for testing)")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing files and reset download flags"
    )
    parser.add_argument(
        "--timestep", type=int, help="Force specific timestep in minutes (default: auto-detect)"
    )
    parser.add_argument(
        "--sites-file",
        default="data/ea_appropriate_sites_enhanced.json",
        help="JSON file containing site information with USGS parameter codes",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous download session"
    )
    parser.add_argument(
        "--save-interval", type=int, default=10, help="Save progress every N sites (default: 10)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = "data/ea_compatible_format"
    os.makedirs(output_dir, exist_ok=True)

    # Load site data
    logger.info(f"Loading site data from {args.sites_file}")
    sites, meta = load_site_data(args.sites_file)
    logger.info(f"Loaded {len(sites)} sites")

    # Show current progress
    progress = get_download_progress(sites)
    logger.info(
        f"Download Progress: {progress['downloaded']}/{progress['total']} sites completed "
        f"({progress['progress_percent']:.1f}%), {progress['failed']} failed, "
        f"{progress['remaining']} remaining"
    )

    # Filter sites if specific site requested
    if args.site_id:
        sites = [site for site in sites if site.get("station_id") == args.site_id]
        if not sites:
            logger.error(f"Site {args.site_id} not found")
            sys.exit(1)
        logger.info(f"Processing single site: {args.site_id}")

    # Filter sites based on download status if resuming
    if not args.site_id:
        if args.resume and not args.force:
            original_count = len(sites)
            sites = [site for site in sites if not is_site_downloaded(site)]
            logger.info(
                f"Resume mode: processing {len(sites)} remaining sites (skipping {original_count - len(sites)} completed)"
            )
        elif args.force:
            # Reset all download flags
            for site in sites:
                if "download_status" in site:
                    del site["download_status"]
            logger.info("Force mode: reset all download flags")

    if not sites:
        logger.info("No sites to process!")
        return

    # Initialize processor
    processor = EADataProcessor()

    # Process each site
    successful_downloads = 0
    failed_downloads = 0
    sites_processed = 0

    try:
        for i, site in enumerate(sites, 1):
            site_id = site.get("station_id")
            if not site_id:
                logger.warning(f"Site {i} missing station_id, skipping")
                update_site_download_status(site, False, "Missing station_id")
                failed_downloads += 1
                continue

            logger.info(f"[{i}/{len(sites)}] Processing site: {site_id}")

            # Get date range
            start_date, end_date = get_date_range_for_site(site)
            logger.info(f"  Date range: {start_date} to {end_date}")

            # Check if output file exists
            output_file = os.path.join(output_dir, f"{site_id}_{start_date}_{end_date}.csv")
            if os.path.exists(output_file) and not args.force:
                logger.info("  File already exists, skipping (use --force to overwrite)")
                if not is_site_downloaded(site):
                    update_site_download_status(site, True)
                    successful_downloads += 1
                continue

            try:
                # Download raw data
                raw_data = processor.download_site_data(site, start_date, end_date)

                if raw_data.empty:
                    logger.warning(f"  No data downloaded for site {site_id}")
                    update_site_download_status(site, False, "No data available from API")
                    failed_downloads += 1
                else:
                    # Convert to compatible format
                    usgs_data = processor.convert_to_usgs_format(raw_data, args.timestep)

                    if usgs_data.empty:
                        logger.warning(f"  No data after conversion for site {site_id}")
                        update_site_download_status(site, False, "No data after format conversion")
                        failed_downloads += 1
                    else:
                        # Save to file
                        usgs_data.to_csv(output_file, index=False)
                        logger.info(f"  Saved {len(usgs_data)} records to {output_file}")

                        update_site_download_status(site, True)
                        successful_downloads += 1

            except Exception as e:
                error_msg = f"Error processing site {site_id}: {str(e)}"
                logger.error(f"  {error_msg}")
                update_site_download_status(site, False, error_msg)
                failed_downloads += 1

            sites_processed += 1

            # Save progress periodically
            if sites_processed % args.save_interval == 0:
                logger.info(f"  Saving progress after {sites_processed} sites...")
                save_site_data(args.sites_file, sites, meta)

            # Rate limiting between sites
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Saving progress...")
        save_site_data(args.sites_file, sites, meta)
        sys.exit(1)

    # Final save
    logger.info("Saving final progress...")
    save_site_data(args.sites_file, sites, meta)

    # Summary
    logger.info("\nDownload Summary:")
    logger.info(f"  Successful: {successful_downloads}")
    logger.info(f"  Failed: {failed_downloads}")
    logger.info(f"  Processed: {sites_processed}")

    # Overall progress
    final_progress = get_download_progress(sites)
    logger.info("\nOverall Progress:")
    logger.info(f"  Total sites: {final_progress['total']}")
    logger.info(
        f"  Downloaded: {final_progress['downloaded']} ({final_progress['progress_percent']:.1f}%)"
    )
    logger.info(f"  Failed: {final_progress['failed']}")
    logger.info(f"  Remaining: {final_progress['remaining']}")
    logger.info(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
