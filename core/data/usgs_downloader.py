#!/usr/bin/env python3
"""
USGS Smart Data Downloader

High-performance, intelligent data downloader with advanced parameter detection and multiple operation modes:

OPERATION MODES:
- download-new: Download data for a new site with specified parameters
- download-new-smart:  Intelligent download with automatic parameter detection
- update-dates: Update existing data between specific dates
- update-current: Update site data to current date
- fill-gaps: Fill data gaps from analysis files

NEW SMART DOWNLOAD FEATURES:
- Automatic parameter detection and optimal time range finding
- Downloads ALL available parameters for the optimal time window
- Intelligent overlap analysis for core water quality parameters
- Maximizes data collection efficiency per site

CORE SMART FUNCTIONS:
1. get_active_parameters(): Queries USGS API for currently active parameters
2. get_parameter_time_ranges(): Gets historical availability periods for parameters
3. find_optimal_date_range(): Calculates optimal overlap periods and selects ALL available parameters

SMART DOWNLOAD WORKFLOW:
1. Query site for active parameters (temperature, conductivity, turbidity)
2. Find time ranges for each parameter type
3. Calculate optimal overlap period with minimum duration requirement
4. Identify ALL parameters available in that optimal window
5. Download comprehensive dataset for maximum data value

EXAMPLE USAGE:
# Smart download with automatic parameter detection (recommended)
python3 usgs_downloader.py download-new-smart --site-id 01304562 --min-years 0.1

# Traditional download with manual parameter specification
python3 usgs_downloader.py download-new --site-id 01304562 --start-date 2020-01-01 --end-date 2023-01-01 --parameters 00010 00095 63680

FEATURES:
- Parallel processing with configurable workers
- Progress tracking with ETA estimates
- Resume capability for interrupted sessions
- Rate limiting and retry logic
- Comprehensive logging and monitoring
- Smart parameter type classification (temperature, conductivity, turbidity, etc.)
- Automatic optimal date range calculation
- All-parameter data collection for maximum value

Author: TSLR
Date: 2025-08-28
Version: 2.0 (Smart Detection Enhanced)
"""

import asyncio
import aiohttp
import json
import pandas as pd
import logging
import requests
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import argparse
import hashlib
import os

# Configure logging with absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
log_file = os.path.join(project_root, "logs", "enhanced_data_fetcher.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Import tools
try:
    from core.utilities.usgs_utilities import PARAMETER_MAPPINGS, get_parameters_by_type
except ImportError:
    logger.warning("Tools modules not available, some features may be limited")
    PARAMETER_MAPPINGS = {}

    def get_parameters_by_type(param_type):
        """Fallback parameter mappings."""
        fallback_params = {
            "temperature": ["00010", "00011"],
            "conductivity": ["00094", "00095"],
            "turbidity": ["63680", "00076"],
        }
        return fallback_params.get(param_type, [])


@dataclass
class FetchJob:
    """Represents a single data fetch job."""

    site_id: str
    start_date: str
    end_date: str
    url: str
    job_id: str
    status: str = "pending"
    attempts: int = 0
    last_error: Optional[str] = None
    response_data: Optional[dict] = None
    fetch_time: Optional[datetime] = None


class ProgressTracker:
    """Tracks and reports progress of data fetching operations."""

    def __init__(self, total_jobs: int):
        self.total_jobs = total_jobs
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.start_time = datetime.now()
        self.last_update = self.start_time

    def update(self, completed: int, failed: int = 0):
        """Update progress counters."""
        self.completed_jobs += completed
        self.failed_jobs += failed
        self.last_update = datetime.now()

    def get_progress_report(self) -> dict:
        """Generate current progress report."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        completed = self.completed_jobs + self.failed_jobs

        if completed > 0:
            rate = completed / elapsed
            eta_seconds = (self.total_jobs - completed) / rate if rate > 0 else 0
            eta = datetime.now() + timedelta(seconds=eta_seconds)
        else:
            rate = 0
            eta = None

        return {
            "total_jobs": self.total_jobs,
            "completed": self.completed_jobs,
            "failed": self.failed_jobs,
            "remaining": self.total_jobs - completed,
            "progress_percent": (completed / self.total_jobs) * 100,
            "elapsed_time": elapsed,
            "rate_per_second": rate,
            "eta": eta.isoformat() if eta else None,
        }


class USGSDownloader:
    """High-performance parallel data fetcher with comprehensive features."""

    def __init__(
        self,
        max_workers: int = 15,
        requests_per_second: float = 10.0,
        max_retries: int = 3,
        timeout: int = 30,
        session_file: str = "../logs/fetch_session.json",
    ):

        self.max_workers = max_workers
        self.requests_per_second = requests_per_second
        self.rate_limit_delay = 1.0 / requests_per_second
        self.max_retries = max_retries
        self.timeout = timeout
        self.session_file = Path(session_file)

        # Create logs directory if it doesn't exist
        self.session_file.parent.mkdir(exist_ok=True)

        # Load configuration
        self.config = self.load_config()

    def load_config(self) -> dict:
        """Load configuration from config file."""
        try:
            try:
                from core.utilities.usgs_utils import _get_parameter_mappings
            except ImportError:
                utilities_dir = Path(__file__).resolve().parents[1] / "utilities"
                sys.path.insert(0, str(utilities_dir))
                from usgs_utils import _get_parameter_mappings

            return {
                "base_url": "https://waterservices.usgs.gov/nwis/iv",
                "api_params": _get_parameter_mappings(),
            }
        except ImportError:
            logger.warning("Config not found, using defaults")
            return {"base_url": "https://waterservices.usgs.gov/nwis/iv", "api_params": {}}

    # =================================================================================
    # SMART PARAMETER DETECTION FUNCTIONS
    # =================================================================================

    def get_active_parameters(self, site_id: str) -> Dict[str, str]:
        """
        🔍 KEY FUNCTION 1: Get Active Parameters

        Queries USGS instantaneous values API to find parameters with data in the last 7 days.
        This identifies which parameters are currently being measured at the site.

        Args:
            site_id: USGS site identifier (e.g., '01304562')

        Returns:
            Dictionary mapping parameter codes to descriptions
            Example: {'00010': 'Temperature, water', '00095': 'Specific conductance', '63680': 'Turbidity'}

        API Endpoint: https://waterservices.usgs.gov/nwis/iv/
        Query Parameters: sites={site_id}, format=rdb, period=P7D

        Usage in Smart Download:
        - First step in parameter detection workflow
        - Filters out inactive/discontinued parameters
        - Provides foundation for time range analysis
        """
        url = (
            "https://waterservices.usgs.gov/nwis/iv/"
            f"?sites={site_id}"
            "&format=rdb"
            "&period=P7D"
        )

        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching active parameters for {site_id}: {e}")
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

        logger.info(f"Found {len(parameters)} active parameters for site {site_id}")
        return parameters

    def get_parameter_time_ranges(
        self, site_id: str, active_params: Dict[str, str]
    ) -> Dict[str, Tuple[str, str, str, int]]:
        """
        📅 KEY FUNCTION 2: Get Parameter Time Ranges

        Queries USGS site catalog to get historical data availability periods for each parameter.
        This determines when each parameter has been measured and for how long.

        Args:
            site_id: USGS site identifier
            active_params: Dictionary of active parameters from get_active_parameters()

        Returns:
            Dictionary mapping parameter codes to time range tuples
            Format: {parameter_code: (description, begin_date, end_date, duration_days)}
            Example: {'00010': ('Temperature, water', '2010-01-01', '2023-12-31', 5113)}

        API Endpoint: https://waterservices.usgs.gov/nwis/site/
        Query Parameters: site={site_id}, seriesCatalogOutput=true, format=rdb

        Usage in Smart Download:
        - Second step in parameter detection workflow
        - Provides historical context for each parameter
        - Enables overlap period calculation
        - Filters parameters by data availability
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
            logger.error(f"Error fetching time ranges for {site_id}: {e}")
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
                if (
                    param_code in active_params
                    and entry.get("begin_date")
                    and entry.get("end_date")
                ):
                    try:
                        start = datetime.strptime(entry["begin_date"], "%Y-%m-%d")
                        end = datetime.strptime(entry["end_date"], "%Y-%m-%d")
                        duration = (end - start).days
                    except ValueError:
                        start = end = duration = None
                    results[param_code] = (
                        active_params[param_code],
                        entry["begin_date"],
                        entry["end_date"],
                        duration,
                    )

        logger.info(f"Found time ranges for {len(results)} parameters for site {site_id}")
        return results

    def find_optimal_date_range(
        self, site_id: str, required_types: List[str] = None, min_years: float = 2
    ) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        🎯 KEY FUNCTION 3: Find Optimal Date Range and ALL Available Parameters

        This is the core intelligence function that:
        1. Finds overlap periods for required parameter types (temp, conductivity, turbidity)
        2. Calculates optimal date range with minimum duration requirement
        3. Identifies ALL parameters available in that optimal window
        4. Returns comprehensive parameter list for maximum data value

        Args:
            site_id: USGS site identifier
            required_types: List of required parameter types for overlap calculation
                           Default: ['temperature', 'conductivity', 'turbidity']
            min_years: Minimum years of overlap required (default: 2.0)
                      Examples: 2.0 = 2 years, 0.1 = ~36 days, 0.5 = 6 months

        Returns:
            Tuple of (start_date, end_date, ALL_available_parameter_codes)
            - start_date: Optimal start date (YYYY-MM-DD)
            - end_date: Optimal end date (YYYY-MM-DD)
            - parameter_codes: List of ALL parameters available in optimal window

        Smart Algorithm:
        1. Use get_active_parameters() to find current parameters
        2. Use get_parameter_time_ranges() to get historical availability
        3. Identify core parameters by type (temperature, conductivity, turbidity)
        4. Calculate overlap period (latest start date, earliest end date)
        5. Validate minimum duration requirement
        6. Find ALL other parameters available in that optimal window
        7. Return comprehensive parameter list for maximum data collection

        Example Output:
        ('2012-08-15', '2017-10-02', ['00010', '00095', '00300', '00400', '32316', '62619', '62620', '63680', '90860'])
        - 5+ years of overlap data
        - 9 total parameters instead of just 3 core ones
        - Maximizes data value per download
        """
        if required_types is None:
            required_types = ["temperature", "conductivity", "turbidity"]

        # Get active parameters
        active_params = self.get_active_parameters(site_id)
        if not active_params:
            logger.warning(f"No active parameters found for site {site_id}")
            return None, None, []

        # Get time ranges
        time_ranges = self.get_parameter_time_ranges(site_id, active_params)
        if not time_ranges:
            logger.warning(f"No time ranges found for site {site_id}")
            return None, None, []

        # Define parameter code sets
        temp_codes = set(get_parameters_by_type("temperature"))
        cond_codes = set(get_parameters_by_type("conductivity"))
        turb_codes = set(get_parameters_by_type("turbidity"))

        def get_earliest_code(code_set, param_type):
            """Get the earliest starting code from a parameter set."""
            codes = code_set & set(time_ranges.keys())
            if not codes:
                logger.warning(f"No {param_type} parameters found for site {site_id}")
                return None
            # Find code with earliest begin_date
            earliest_code = min(
                codes,
                key=lambda c: (
                    datetime.strptime(time_ranges[c][1], "%Y-%m-%d")
                    if time_ranges[c][1]
                    else datetime.max
                ),
            )
            logger.info(f"Selected {param_type} parameter {earliest_code} for site {site_id}")
            return earliest_code

        # Select best parameters
        selected_codes = []
        if "temperature" in required_types:
            temp_code = get_earliest_code(temp_codes, "temperature")
            if temp_code:
                selected_codes.append(temp_code)

        if "conductivity" in required_types:
            cond_code = get_earliest_code(cond_codes, "conductivity")
            if cond_code:
                selected_codes.append(cond_code)

        if "turbidity" in required_types:
            turb_code = get_earliest_code(turb_codes, "turbidity")
            if turb_code:
                selected_codes.append(turb_code)

        if not selected_codes:
            logger.warning(f"No required parameters found for site {site_id}")
            return None, None, []

        # Find optimal overlap period
        all_starts = []
        all_ends = []

        for code in selected_codes:
            if code in time_ranges:
                description, begin, end, duration = time_ranges[code]
                try:
                    all_starts.append(datetime.strptime(begin, "%Y-%m-%d"))
                    all_ends.append(datetime.strptime(end, "%Y-%m-%d"))
                except Exception:
                    continue

        if not all_starts or not all_ends:
            logger.warning(f"No valid date ranges found for site {site_id}")
            return None, None, []

        # Use latest start and earliest end for overlap
        optimal_start = max(all_starts)
        optimal_end = min(all_ends)

        # Check minimum years requirement
        duration_years = (optimal_end - optimal_start).days / 365.25
        if duration_years < min_years:
            logger.warning(
                f"Site {site_id}: Overlap period {duration_years:.1f} years is less than required {min_years} years"
            )
            return None, None, []

        # Ensure we have at least some overlap
        if optimal_start >= optimal_end:
            logger.warning(f"No overlap period found for site {site_id}")
            return None, None, []

        start_str = optimal_start.strftime("%Y-%m-%d")
        end_str = optimal_end.strftime("%Y-%m-%d")

        duration_days = (optimal_end - optimal_start).days
        logger.info(
            f"Site {site_id}: Optimal period {start_str} to {end_str} ({duration_days} days)"
        )

        # Now find ALL parameters that have data in this optimal window
        all_available_params = []
        for param_code, (description, begin_date, end_date, duration) in time_ranges.items():
            try:
                param_start = datetime.strptime(begin_date, "%Y-%m-%d")
                param_end = datetime.strptime(end_date, "%Y-%m-%d")

                # Check if parameter overlaps with our optimal window
                if param_start <= optimal_end and param_end >= optimal_start:
                    all_available_params.append(param_code)

            except Exception:
                continue

        # Sort parameters for consistent ordering
        all_available_params.sort()

        logger.info(
            f"Site {site_id}: Found {len(all_available_params)} parameters available in optimal window"
        )
        logger.info(f"Site {site_id}: Core parameters {selected_codes}")
        logger.info(f"Site {site_id}: All parameters {all_available_params}")

        return start_str, end_str, all_available_params

    def create_jobs_from_gaps(self, gaps_file: str) -> List[FetchJob]:
        """Create fetch jobs from gaps analysis file."""
        try:
            # Read CSV with site_id as string to preserve leading zeros
            df = pd.read_csv(gaps_file, dtype={"site_id": str})
            jobs = []

            for _, row in df.iterrows():
                site_id = str(row["site_id"]).zfill(8)  # Ensure 8 digits with leading zeros

                # Handle different column name formats
                start_date = row.get("gap_start") or row.get("start_date")
                end_date = row.get("gap_end") or row.get("end_date")

                if not start_date or not end_date:
                    logger.warning(f"Missing date columns for site {site_id}")
                    continue

                # Build URL
                url = f"{self.config['base_url']}?site={site_id}&startDT={start_date}&endDT={end_date}"
                for key, value in self.config["api_params"].items():
                    url += f"&{key}={value}"

                # Create unique job ID
                job_data = f"{site_id}_{start_date}_{end_date}"
                job_id = hashlib.md5(job_data.encode()).hexdigest()[:16]

                job = FetchJob(
                    site_id=site_id,
                    start_date=start_date,
                    end_date=end_date,
                    url=url,
                    job_id=job_id,
                )
                jobs.append(job)

            logger.info(f"Created {len(jobs)} fetch jobs from {gaps_file}")
            return jobs

        except Exception as e:
            logger.error(f"Error creating jobs from {gaps_file}: {e}")
            return []

    def save_session(self, jobs: List[FetchJob], progress: ProgressTracker):
        """Save current session state for resumability."""
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "progress": progress.get_progress_report(),
            "jobs": [
                {
                    "job_id": job.job_id,
                    "site_id": job.site_id,
                    "start_date": job.start_date,
                    "end_date": job.end_date,
                    "url": job.url,
                    "status": job.status,
                    "attempts": job.attempts,
                    "last_error": job.last_error,
                    "fetch_time": job.fetch_time.isoformat() if job.fetch_time else None,
                }
                for job in jobs
            ],
        }

        with open(self.session_file, "w") as f:
            json.dump(session_data, f, indent=2)

    def load_session(self) -> Tuple[List[FetchJob], Optional[ProgressTracker]]:
        """Load previous session for resuming."""
        if not self.session_file.exists():
            return [], None

        try:
            with open(self.session_file, "r") as f:
                session_data = json.load(f)

            jobs = []
            for job_data in session_data["jobs"]:
                job = FetchJob(
                    site_id=job_data["site_id"],
                    start_date=job_data["start_date"],
                    end_date=job_data["end_date"],
                    url=job_data["url"],
                    job_id=job_data["job_id"],
                    status=job_data["status"],
                    attempts=job_data["attempts"],
                    last_error=job_data.get("last_error"),
                    fetch_time=(
                        datetime.fromisoformat(job_data["fetch_time"])
                        if job_data.get("fetch_time")
                        else None
                    ),
                )
                jobs.append(job)

            # Recreate progress tracker
            progress_data = session_data.get("progress", {})
            progress = ProgressTracker(progress_data.get("total_jobs", len(jobs)))
            progress.completed_jobs = progress_data.get("completed", 0)
            progress.failed_jobs = progress_data.get("failed", 0)

            logger.info(f"Loaded session with {len(jobs)} jobs")
            return jobs, progress

        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return [], None

    async def fetch_single_job(self, session: aiohttp.ClientSession, job: FetchJob) -> FetchJob:
        """Fetch data for a single job with retry logic."""
        for attempt in range(self.max_retries):
            try:
                job.attempts = attempt + 1

                async with session.get(job.url, timeout=self.timeout) as response:
                    if response.status == 200:
                        job.response_data = await response.json()
                        job.status = "completed"
                        job.fetch_time = datetime.now()
                        logger.debug(f"Successfully fetched job {job.job_id}")
                        return job
                    else:
                        job.last_error = f"HTTP {response.status}"

            except asyncio.TimeoutError:
                job.last_error = "Timeout"
            except Exception as e:
                job.last_error = str(e)

            # Wait before retry (except on last attempt)
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff

        job.status = "failed"
        logger.warning(
            f"Job {job.job_id} failed after {self.max_retries} attempts: {job.last_error}"
        )
        return job

    async def process_jobs_batch(
        self, jobs: List[FetchJob], progress: ProgressTracker
    ) -> List[FetchJob]:
        """Process a batch of jobs in parallel."""
        connector = aiohttp.TCPConnector(limit=self.max_workers)
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.max_workers)

            async def limited_fetch(job):
                async with semaphore:
                    result = await self.fetch_single_job(session, job)
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    return result

            # Process jobs in batches to avoid memory issues
            batch_size = 100
            all_results = []

            for i in range(0, len(jobs), batch_size):
                batch = jobs[i : i + batch_size]
                pending_jobs = [job for job in batch if job.status == "pending"]

                if not pending_jobs:
                    all_results.extend(batch)
                    continue

                logger.info(f"Processing batch {i//batch_size + 1}: {len(pending_jobs)} jobs")

                # Execute batch
                batch_results = await asyncio.gather(
                    *[limited_fetch(job) for job in pending_jobs], return_exceptions=True
                )

                # Update results
                completed_count = 0
                failed_count = 0

                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        pending_jobs[j].status = "failed"
                        pending_jobs[j].last_error = str(result)
                        failed_count += 1
                    else:
                        if result.status == "completed":
                            completed_count += 1
                        else:
                            failed_count += 1

                progress.update(completed_count, failed_count)
                all_results.extend(batch)

                # Save progress periodically
                if i % (batch_size * 5) == 0:
                    self.save_session(all_results, progress)
                    report = progress.get_progress_report()
                    logger.info(
                        f"Progress: {report['progress_percent']:.1f}% - ETA: {report['eta']}"
                    )

        return all_results

    def save_results(self, jobs: List[FetchJob], output_file: str):
        """Save fetch results to JSON file."""
        results = {
            "metadata": {
                "total_jobs": len(jobs),
                "completed_jobs": len([j for j in jobs if j.status == "completed"]),
                "failed_jobs": len([j for j in jobs if j.status == "failed"]),
                "fetch_time": datetime.now().isoformat(),
            },
            "successful_fetches": {},
            "failed_fetches": {},
        }

        for job in jobs:
            if job.status == "completed" and job.response_data:
                results["successful_fetches"][job.job_id] = {
                    "site_id": job.site_id,
                    "start_date": job.start_date,
                    "end_date": job.end_date,
                    "data": job.response_data,
                    "fetch_time": job.fetch_time.isoformat(),
                }
            elif job.status == "failed":
                results["failed_fetches"][job.job_id] = {
                    "site_id": job.site_id,
                    "start_date": job.start_date,
                    "end_date": job.end_date,
                    "attempts": job.attempts,
                    "last_error": job.last_error,
                }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_file}")

    def save_tsv_csv_files(self, jobs: List[FetchJob], output_dir: str = "../data/raw"):
        """
        Save successful fetch results as individual TSV and CSV files.

        Args:
            jobs: List of completed fetch jobs
            output_dir: Directory to save TSV/CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        successful_jobs = [j for j in jobs if j.status == "completed" and j.response_data]
        logger.info(f"Converting {len(successful_jobs)} successful fetches to TSV/CSV format...")

        files_created = 0
        data_points_processed = 0
        sites_processed = set()

        # Group jobs by site_id to combine data ranges
        site_jobs = {}
        for job in successful_jobs:
            site_id = job.site_id
            if site_id not in site_jobs:
                site_jobs[site_id] = []
            site_jobs[site_id].append(job)

        for site_id, site_job_list in site_jobs.items():
            try:
                # Combine all data for this site
                all_data_points = []

                for job in site_job_list:
                    points = self._extract_data_points_from_job(job)
                    all_data_points.extend(points)

                if not all_data_points:
                    logger.debug(f"No data points found for site {site_id}")
                    continue

                # Create DataFrame
                df = pd.DataFrame(all_data_points)

                if df.empty:
                    continue

                # Sort by datetime
                try:
                    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                    df = df.sort_values("datetime")
                except Exception as e:
                    logger.warning(f"Could not parse datetime for site {site_id}: {e}")
                    continue

                # Get date range for filename
                try:
                    start_date = df["datetime"].min().strftime("%Y%m%d")
                    end_date = df["datetime"].max().strftime("%Y%m%d")
                except Exception:
                    start_date = end_date = "unknown"

                # Get unique parameter codes
                param_cols = [
                    col
                    for col in df.columns
                    if col not in ["site_no", "datetime"] and not col.endswith("_cd")
                ]
                param_codes = sorted([col for col in param_cols if col != "unknown"])

                if param_codes:
                    param_str = "_".join(param_codes)
                else:
                    param_str = "data"

                # Create filename
                base_filename = f"usgs_{site_id}_{param_str}_{start_date}_{end_date}"

                # Reorder columns: site_no, datetime, then just parameter values (no qualifier codes)
                ordered_columns = ["site_no", "datetime"] + param_codes

                # Select only the columns we want (no qualifier codes)
                df_final = df[ordered_columns]

                # Save TSV
                tsv_path = output_path / f"{base_filename}.tsv"
                df_final.to_csv(tsv_path, sep="\t", index=False)
                files_created += 1

                # Save CSV
                csv_path = output_path / f"{base_filename}.csv"
                df_final.to_csv(csv_path, index=False)
                files_created += 1

                data_points_processed += len(df_final)
                sites_processed.add(site_id)

                logger.debug(
                    f"Created files for site {site_id}: {len(df_final)} data points, {len(param_codes)} parameters"
                )

            except Exception as e:
                logger.error(f"Error processing site {site_id}: {e}")
                continue

        logger.info("TSV/CSV conversion complete!")
        logger.info(f"  Sites processed: {len(sites_processed)}")
        logger.info(f"  Files created: {files_created}")
        logger.info(f"  Data points: {data_points_processed:,}")
        logger.info(f"  Output directory: {output_path}")

    def _extract_data_points_from_job(self, job: FetchJob) -> List[Dict]:
        """
        Extract data points from a successful job's USGS response.
        Creates one row per timestamp with separate columns for each parameter.

        Args:
            job: Completed fetch job with response data

        Returns:
            List of data point dictionaries (one per timestamp)
        """
        try:
            usgs_data = job.response_data
            value_section = usgs_data.get("value", {})
            time_series = value_section.get("timeSeries", [])

            # Collect all data by timestamp
            timestamp_data = {}

            for ts in time_series:
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
                            timestamp_data[timestamp] = {
                                "site_no": job.site_id,
                                "datetime": timestamp,
                            }

                        # Add parameter value (but skip qualifier codes)
                        timestamp_data[timestamp][param_code] = data_point.get("value", "")

            # Convert to list format
            data_points = list(timestamp_data.values())
            return data_points

        except Exception as e:
            logger.error(f"Error extracting data points from job {job.job_id}: {e}")
            return []

    async def download_new_site(
        self,
        site_id: str,
        start_date: str,
        end_date: str,
        parameters: List[str],
        output_dir: str = "../data/raw",
    ) -> bool:
        """
        Download data for a completely new site with automatic chunking.

        Large date ranges are automatically split into smaller chunks (< 3 months each)
        to avoid USGS API limits and improve reliability. Results are concatenated seamlessly.

        Args:
            site_id: USGS site identifier
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            parameters: List of parameter codes to download
            output_dir: Directory to save the data

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Downloading new site {site_id} from {start_date} to {end_date}")

        # Check if date range needs chunking (more than 2.5 months)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days_diff = (end_dt - start_dt).days

        if days_diff <= 90:  # ~3 months, no chunking needed
            logger.info(f"Date range is {days_diff} days, downloading in single request")
            param_string = "_".join(parameters)
            url = self._build_url(site_id, start_date, end_date, parameters)

            job = FetchJob(
                site_id=site_id,
                start_date=start_date,
                end_date=end_date,
                url=url,
                job_id=f"{site_id}_{start_date}_{end_date}_{param_string}",
            )

            return await self._download_single_job(job, output_dir)

        else:
            # Use chunking for large date ranges
            logger.info(f"Date range is {days_diff} days, using chunked download")
            return await self._download_chunked(
                site_id, start_date, end_date, parameters, output_dir
            )

    async def _download_chunked(
        self, site_id: str, start_date: str, end_date: str, parameters: List[str], output_dir: str
    ) -> bool:
        """
        Download data in chunks and concatenate results.

        Args:
            site_id: USGS site identifier
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            parameters: List of parameter codes to download
            output_dir: Directory to save the data

        Returns:
            True if successful, False otherwise
        """
        # Create date chunks
        date_chunks = self._create_date_chunks(start_date, end_date, max_months=3)
        logger.info(f"Created {len(date_chunks)} chunks for site {site_id}")

        # Create jobs for each chunk
        chunk_jobs = []
        param_string = "_".join(parameters)

        for i, (chunk_start, chunk_end) in enumerate(date_chunks):
            url = self._build_url(site_id, chunk_start, chunk_end, parameters)

            job = FetchJob(
                site_id=site_id,
                start_date=chunk_start,
                end_date=chunk_end,
                url=url,
                job_id=f"{site_id}_chunk_{i:02d}_{chunk_start}_{chunk_end}_{param_string}",
            )
            chunk_jobs.append(job)

        # Execute all chunk downloads with parallel processing
        logger.info(f"Downloading {len(chunk_jobs)} chunks for site {site_id} in parallel")

        session_timeout = aiohttp.ClientTimeout(total=self.timeout)
        successful_chunks = []

        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            # Create semaphore to limit concurrent requests (faster than sequential)
            semaphore = asyncio.Semaphore(min(8, len(chunk_jobs)))  # Max 8 parallel chunks

            async def download_chunk(job, chunk_num):
                async with semaphore:
                    for attempt in range(self.max_retries):
                        try:
                            await asyncio.sleep(self.rate_limit_delay)

                            async with session.get(job.url) as response:
                                if response.status == 200:
                                    response_text = await response.text()
                                    job.response_data = json.loads(response_text)
                                    job.status = "completed"
                                    logger.info(
                                        f"✅ Chunk {chunk_num}/{len(chunk_jobs)} completed: {job.start_date} to {job.end_date}"
                                    )
                                    return job
                                else:
                                    logger.warning(
                                        f"HTTP {response.status} for chunk {chunk_num}, attempt {attempt + 1}"
                                    )

                        except Exception as e:
                            logger.warning(f"Chunk {chunk_num} attempt {attempt + 1} failed: {e}")
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(1)  # Shorter retry delay

                    logger.error(
                        f"❌ Failed to download chunk {chunk_num} after {self.max_retries} attempts"
                    )
                    job.status = "failed"
                    return job

            # Execute all chunks in parallel
            chunk_results = await asyncio.gather(
                *[download_chunk(job, i + 1) for i, job in enumerate(chunk_jobs)],
                return_exceptions=True,
            )

            # Collect successful chunks
            for result in chunk_results:
                if isinstance(result, FetchJob) and result.status == "completed":
                    successful_chunks.append(result)

        # Check if we have any successful chunks
        if not successful_chunks:
            logger.error(f"No chunks downloaded successfully for site {site_id}")
            return False

        if len(successful_chunks) < len(chunk_jobs):
            logger.warning(
                f"Only {len(successful_chunks)}/{len(chunk_jobs)} chunks downloaded successfully"
            )

        # Concatenate all successful chunks
        logger.info(f"Concatenating {len(successful_chunks)} chunks for site {site_id}")
        combined_data = self._concatenate_chunk_data(successful_chunks)

        if not combined_data:
            logger.error(f"Failed to concatenate chunk data for site {site_id}")
            return False

        # Create a combined job for saving
        combined_job = FetchJob(
            site_id=site_id,
            start_date=start_date,
            end_date=end_date,
            url="chunked_download",
            job_id=f"{site_id}_combined_{start_date}_{end_date}_{param_string}",
            status="completed",
            response_data=combined_data,
        )

        # Save the combined data
        self.save_tsv_csv_files([combined_job], output_dir)
        logger.info(f"Successfully downloaded and concatenated chunked data for site {site_id}")

        return True

    async def batch_download_smart(
        self,
        site_ids: List[str],
        required_types: List[str] = None,
        min_years: float = 0.1,
        output_dir: str = "../data/raw",
        continue_on_error: bool = True,
        delay: float = 1.0,
    ) -> Dict[str, bool]:
        """
        Download multiple sites in batch using smart detection.

        Args:
            site_ids: List of USGS site identifiers
            required_types: Core parameter types for overlap calculation
            min_years: Minimum overlap duration in years
            output_dir: Directory for data files
            continue_on_error: Continue with other sites if one fails
            delay: Delay between sites in seconds

        Returns:
            Dictionary mapping site_id to success status
        """
        logger.info(f"🚀 Starting batch download for {len(site_ids)} sites")
        logger.info(f"Sites: {', '.join(site_ids)}")

        results = {}
        successful = 0
        failed = 0
        start_time = datetime.now()

        for i, site_id in enumerate(site_ids, 1):
            site_start_time = datetime.now()
            logger.info(f"\n{'='*60}")
            logger.info(f"📍 Processing site {i}/{len(site_ids)}: {site_id}")
            logger.info(f"{'='*60}")

            try:
                success = await self.download_new_site_smart(
                    site_id, required_types, min_years, output_dir
                )

                site_duration = datetime.now() - site_start_time

                if success:
                    results[site_id] = True
                    successful += 1
                    logger.info(f"✅ Site {site_id} completed successfully in {site_duration}")
                else:
                    results[site_id] = False
                    failed += 1
                    logger.error(f"❌ Site {site_id} failed after {site_duration}")

                    if not continue_on_error:
                        logger.error("Stopping batch download due to failure")
                        break

            except Exception as e:
                results[site_id] = False
                failed += 1
                site_duration = datetime.now() - site_start_time
                logger.error(f"❌ Site {site_id} failed with exception after {site_duration}: {e}")

                if not continue_on_error:
                    logger.error("Stopping batch download due to exception")
                    break

            # Add delay between sites (except for last one)
            if i < len(site_ids) and delay > 0:
                logger.info(f"⏳ Waiting {delay}s before next site...")
                await asyncio.sleep(delay)

        # Final summary
        total_duration = datetime.now() - start_time
        logger.info(f"\n{'='*80}")
        logger.info("🏁 BATCH DOWNLOAD COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"⏱️  Total duration: {total_duration}")
        logger.info(f"📊 Total sites: {len(site_ids)}")
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"📈 Success rate: {(successful/len(site_ids)*100):.1f}%")

        if successful > 0:
            logger.info("\n✅ Successfully downloaded sites:")
            for site_id, success in results.items():
                if success:
                    logger.info(f"   {site_id}")

        if failed > 0:
            logger.error("\n❌ Failed sites:")
            for site_id, success in results.items():
                if not success:
                    logger.error(f"   {site_id}")

        return results

    async def download_new_site_smart(
        self,
        site_id: str,
        required_types: List[str] = None,
        min_years: float = 2,
        output_dir: str = "../data/raw",
    ) -> bool:
        """
        🚀 SMART DOWNLOAD: Intelligent Parameter Detection and Comprehensive Data Collection

        This is the flagship function that revolutionizes USGS data collection by:

        INTELLIGENT WORKFLOW:
        1. 🔍 Automatically discovers active parameters using get_active_parameters()
        2. 📅 Analyzes historical data availability using get_parameter_time_ranges()
        3. 🎯 Calculates optimal overlap periods using find_optimal_date_range()
        4. 📊 Downloads ALL available parameters for maximum data value
        5. 💾 Saves comprehensive datasets with optimal coverage

        SMART ADVANTAGES:
        ✅ No manual parameter specification required
        ✅ Maximizes data collection per site
        ✅ Finds optimal time windows automatically
        ✅ Downloads comprehensive datasets (often 5-15 parameters vs 3)
        ✅ Respects minimum duration requirements
        ✅ Handles varying site capabilities intelligently

        Args:
            site_id: USGS site identifier (e.g., '01304562')
            required_types: Core parameter types for overlap calculation
                           Default: ['temperature', 'conductivity', 'turbidity']
                           These ensure water quality measurement capability
            min_years: Minimum overlap duration in years (default: 2.0)
                      Common values: 2.0=2years, 1.0=1year, 0.1=~36days
            output_dir: Directory for data files (default: 'data/raw')

        Returns:
            bool: True if successful download, False if failed

        Example Smart Detection Results:
        Site 01304562: Found 12 active parameters
        Site 01304562: Core parameters ['00010', '00095', '63680']
        Site 01304562: ALL parameters ['00010', '00095', '00300', '00400', '32316', '62619', '62620', '63680', '90860']
        Site 01304562: Optimal period 2012-08-15 to 2017-10-02 (1874 days)
        → Downloads 9 parameters × 5+ years = comprehensive water quality dataset

        USAGE:
        # Smart download with defaults (2+ years overlap)
        await downloader.download_new_site_smart('01304562')

        # Smart download with relaxed requirements
        await downloader.download_new_site_smart('01304562', min_years=0.1)

        # CLI usage
        python3 usgs_downloader.py download-new-smart --site-id 01304562 --min-years 0.1
        """
        logger.info(f"Smart downloading for site {site_id}...")

        # Set default required types if not provided
        if required_types is None:
            required_types = ["temperature", "conductivity", "turbidity"]

        # Find optimal parameters and date range
        start_date, end_date, parameters = self.find_optimal_date_range(
            site_id, required_types, min_years
        )

        if not start_date or not end_date or not parameters:
            logger.error(f"Could not determine optimal parameters/dates for site {site_id}")
            return False

        logger.info(f"Site {site_id}: Smart download from {start_date} to {end_date}")
        logger.info(f"Site {site_id}: Parameters {parameters}")

        # Use the regular download method with the smart parameters
        return await self.download_new_site(site_id, start_date, end_date, parameters, output_dir)

    async def update_site_between_dates(
        self,
        site_id: str,
        start_date: str,
        end_date: str,
        parameters: List[str],
        output_dir: str = "../data/raw",
    ) -> bool:
        """
        Update existing site data between specific dates with automatic chunking.

        Args:
            site_id: USGS site identifier
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            parameters: List of parameter codes to update
            output_dir: Directory to save the data

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Updating site {site_id} between {start_date} and {end_date}")

        # Use the async download method with chunking
        return await self.download_new_site(site_id, start_date, end_date, parameters, output_dir)

    async def update_site_to_current(
        self, site_id: str, last_date: str, parameters: List[str], output_dir: str = "../data/raw"
    ) -> bool:
        """
        Update site data from last known date to current date.

        Args:
            site_id: USGS site identifier
            last_date: Last date of existing data in YYYY-MM-DD format
            parameters: List of parameter codes to update
            output_dir: Directory to save the data

        Returns:
            True if successful, False otherwise
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Updating site {site_id} from {last_date} to {current_date}")

        return await self.update_site_between_dates(
            site_id, last_date, current_date, parameters, output_dir
        )

    async def _download_single_job(self, job: FetchJob, output_dir: str) -> bool:
        """Execute a single download job."""
        try:
            # Simulate the fetch process
            session_timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=session_timeout) as session:
                for attempt in range(self.max_retries):
                    try:
                        await asyncio.sleep(self.rate_limit_delay)

                        async with session.get(job.url) as response:
                            if response.status == 200:
                                response_text = await response.text()
                                job.response_data = json.loads(response_text)
                                job.status = "completed"

                                # Save the data
                                self.save_tsv_csv_files([job], output_dir)
                                logger.info(f"Successfully downloaded data for site {job.site_id}")
                                return True
                            else:
                                logger.warning(
                                    f"HTTP {response.status} for {job.site_id}, attempt {attempt + 1}"
                                )

                    except Exception as e:
                        logger.warning(f"Attempt {attempt + 1} failed for {job.site_id}: {e}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(2**attempt)  # Exponential backoff

                job.status = "failed"
                logger.error(
                    f"Failed to download data for site {job.site_id} after {self.max_retries} attempts"
                )
                return False

        except Exception as e:
            logger.error(f"Error downloading site {job.site_id}: {e}")
            return False

    def _build_url(
        self, site_id: str, start_date: str, end_date: str, parameters: List[str]
    ) -> str:
        """Build USGS API URL for the given parameters."""
        base_url = "https://waterservices.usgs.gov/nwis/iv"
        param_string = ",".join(parameters)

        url = (
            f"{base_url}/?format=json&sites={site_id}"
            f"&startDT={start_date}&endDT={end_date}"
            f"&parameterCd={param_string}"
        )

        return url

    def _create_date_chunks(
        self, start_date: str, end_date: str, max_months: int = 3
    ) -> List[Tuple[str, str]]:
        """
        Break a date range into smaller chunks of maximum duration.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_months: Maximum months per chunk (default: 3 months for faster downloads)

        Returns:
            List of (start_date, end_date) tuples for each chunk
        """
        from datetime import datetime
        from dateutil.relativedelta import relativedelta

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        chunks = []
        current_start = start_dt

        while current_start < end_dt:
            # Calculate chunk end date (max_months later or end_date, whichever is earlier)
            chunk_end = current_start + relativedelta(months=max_months)
            if chunk_end > end_dt:
                chunk_end = end_dt

            chunks.append((current_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))

            # Move to next chunk (start from the day after current chunk ends)
            current_start = chunk_end + timedelta(days=1)

            # Avoid infinite loop if we're at the end date
            if current_start >= end_dt:
                break

        logger.info(
            f"Split date range {start_date} to {end_date} into {len(chunks)} chunks (max {max_months} months each)"
        )
        return chunks

    def _concatenate_chunk_data(self, chunk_results: List[FetchJob]) -> dict:
        """
        Concatenate data from multiple chunks into a single response format.

        Args:
            chunk_results: List of completed FetchJob objects with response data

        Returns:
            Combined response data in USGS JSON format
        """
        if not chunk_results:
            return {}

        # Start with the first chunk's structure
        first_chunk = chunk_results[0]
        if not first_chunk.response_data or "value" not in first_chunk.response_data:
            logger.warning("First chunk has no valid data to use as template")
            return {}

        combined_data = first_chunk.response_data.copy()

        # Collect all time series data from all chunks
        all_time_series = []

        for chunk in chunk_results:
            if (
                chunk.status == "completed"
                and chunk.response_data
                and "value" in chunk.response_data
                and "timeSeries" in chunk.response_data["value"]
            ):

                chunk_time_series = chunk.response_data["value"]["timeSeries"]
                all_time_series.extend(chunk_time_series)

        # Group time series by parameter and variable
        parameter_groups = {}
        for ts in all_time_series:
            if "variable" in ts and "variableCode" in ts["variable"]:
                var_code = ts["variable"]["variableCode"][0]["value"]
                if var_code not in parameter_groups:
                    parameter_groups[var_code] = {"metadata": ts.copy(), "values": []}
                    # Clear the values to rebuild them
                    parameter_groups[var_code]["metadata"]["values"] = []

                # Collect all values from this time series
                if "values" in ts:
                    for value_set in ts["values"]:
                        if "value" in value_set:
                            parameter_groups[var_code]["values"].extend(value_set["value"])

        # Rebuild the combined time series with concatenated values
        combined_time_series = []
        for var_code, group in parameter_groups.items():
            # Sort values by dateTime to ensure chronological order
            all_values = sorted(group["values"], key=lambda x: x.get("dateTime", ""))

            # Create the combined time series entry
            combined_ts = group["metadata"].copy()
            combined_ts["values"] = [
                {
                    "value": all_values,
                    "qualifiers": ["P"],  # Provisional data qualifier
                    "method": [{"methodDescription": "Combined from multiple API chunks"}],
                }
            ]

            combined_time_series.append(combined_ts)

        # Update the combined data structure
        combined_data["value"]["timeSeries"] = combined_time_series

        # Update query info to reflect the full date range
        if "value" in combined_data and "queryInfo" in combined_data["value"]:
            if "criteria" in combined_data["value"]["queryInfo"]:
                criteria = combined_data["value"]["queryInfo"]["criteria"]
                # Update date range to reflect full combined range
                first_date = chunk_results[0].start_date
                last_date = chunk_results[-1].end_date

                for i, criterion in enumerate(criteria):
                    if isinstance(criterion, dict):
                        if criterion.get("name") == "startDT":
                            criterion["value"] = first_date
                        elif criterion.get("name") == "endDT":
                            criterion["value"] = last_date
                    elif isinstance(criterion, str):
                        # Handle string format criteria
                        if "startDT" in criterion:
                            criteria[i] = criterion.replace(criterion.split("=")[1], first_date)
                        elif "endDT" in criterion:
                            criteria[i] = criterion.replace(criterion.split("=")[1], last_date)

        total_values = sum(
            len(ts["values"][0]["value"]) for ts in combined_time_series if ts.get("values")
        )
        logger.info(
            f"Concatenated {len(chunk_results)} chunks into {len(combined_time_series)} parameters with {total_values} total data points"
        )

        return combined_data


def _load_site_list(file_path: str) -> List[str]:
    """Load site IDs from a file (one per line)."""
    try:
        with open(file_path, "r") as f:
            sites = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        logger.info(f"Loaded {len(sites)} sites from {file_path}")
        return sites
    except Exception as e:
        logger.error(f"Error loading site list from {file_path}: {e}")
        sys.exit(1)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="USGS Data Downloader with multiple operation modes"
    )
    subparsers = parser.add_subparsers(dest="command", help="Operation mode")

    gaps_parser = subparsers.add_parser("fill-gaps", help="Fill data gaps from CSV file")
    gaps_parser.add_argument(
        "--gaps-file",
        default="../data/analysis/significant_gaps_1day_plus.csv",
        help="CSV file containing gaps to fill",
    )
    gaps_parser.add_argument("--resume", action="store_true", help="Resume previous session")
    gaps_parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Maximum number of jobs to process (for testing)",
    )

    new_parser = subparsers.add_parser("download-new", help="Download data for a new site")
    new_parser.add_argument("--site-id", required=True, help="USGS site identifier")
    new_parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    new_parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    new_parser.add_argument(
        "--parameters",
        required=True,
        nargs="+",
        help="Parameter codes to download (e.g., 00010 00060 00065)",
    )

    update_parser = subparsers.add_parser(
        "update-dates", help="Update site data between specific dates"
    )
    update_parser.add_argument("--site-id", required=True, help="USGS site identifier")
    update_parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    update_parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    update_parser.add_argument(
        "--parameters",
        required=True,
        nargs="+",
        help="Parameter codes to update (e.g., 00010 00060 00065)",
    )

    current_parser = subparsers.add_parser(
        "update-current", help="Update site data to current date"
    )
    current_parser.add_argument("--site-id", required=True, help="USGS site identifier")
    current_parser.add_argument(
        "--last-date", required=True, help="Last date of existing data (YYYY-MM-DD)"
    )
    current_parser.add_argument(
        "--parameters",
        required=True,
        nargs="+",
        help="Parameter codes to update (e.g., 00010 00060 00065)",
    )

    smart_parser = subparsers.add_parser(
        "download-new-smart",
        help="Automatically detect and download optimal data for a new site",
    )
    smart_parser.add_argument("--site-id", help="USGS site identifier")
    smart_parser.add_argument("--site-list", help="File containing list of site IDs (one per line)")
    smart_parser.add_argument("--sites", nargs="+", help="Multiple site IDs separated by spaces")
    smart_parser.add_argument(
        "--min-years",
        type=float,
        default=2,
        help="Minimum years of overlap for parameter types (default: 2)",
    )
    smart_parser.add_argument(
        "--required-types",
        nargs="+",
        default=["temperature", "conductivity", "turbidity"],
        help="Required parameter types (default: temperature conductivity turbidity)",
    )

    batch_parser = subparsers.add_parser(
        "batch-smart", help="Download multiple sites in batch with smart detection"
    )
    batch_parser.add_argument(
        "--site-list", required=True, help="File containing list of site IDs (one per line)"
    )
    batch_parser.add_argument(
        "--sites",
        nargs="+",
        help="Multiple site IDs separated by spaces (alternative to --site-list)",
    )
    batch_parser.add_argument(
        "--min-years",
        type=float,
        default=0.1,
        help="Minimum years of overlap for parameter types (default: 0.1)",
    )
    batch_parser.add_argument(
        "--required-types",
        nargs="+",
        default=["temperature", "conductivity", "turbidity"],
        help="Required parameter types (default: temperature conductivity turbidity)",
    )
    batch_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue downloading other sites if one fails",
    )
    batch_parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between sites in seconds (default: 1.0)",
    )

    for subparser in (
        gaps_parser,
        new_parser,
        update_parser,
        current_parser,
        smart_parser,
        batch_parser,
    ):
        subparser.add_argument(
            "--output",
            default="../data/fetched/gap_fill_results.json",
            help="Output file for results",
        )
        subparser.add_argument(
            "--tsv-csv-output",
            default="../data/raw",
            help="Output directory for TSV/CSV files",
        )
        subparser.add_argument(
            "--save-raw", action="store_true", help="Save TSV/CSV files in addition to JSON"
        )
        subparser.add_argument("--workers", type=int, default=15, help="Number of parallel workers")
        subparser.add_argument(
            "--rate", type=float, default=10.0, help="Requests per second rate limit"
        )
        subparser.add_argument(
            "--fast",
            action="store_true",
            help="Use aggressive settings for maximum speed (20 req/s, 20 workers)",
        )

    return parser


def _apply_fast_mode(args) -> tuple:
    workers = args.workers
    rate = args.rate
    if hasattr(args, "fast") and args.fast:
        workers = 20
        rate = 20.0
        logger.info("🚀 FAST MODE: Using aggressive settings (20 req/s, 20 workers)")
    return workers, rate


def _validate_smart_args(args) -> None:
    site_sources = sum([bool(args.site_id), bool(args.site_list), bool(args.sites)])
    if site_sources != 1:
        logger.error(
            "For download-new-smart mode, provide exactly one of: --site-id, --site-list, or --sites"
        )
        sys.exit(1)


async def _handle_download_new(downloader, args) -> None:
    success = await downloader.download_new_site(
        args.site_id, args.start_date, args.end_date, args.parameters, args.tsv_csv_output
    )
    if success:
        logger.info(f"Successfully downloaded new site {args.site_id}")
    else:
        logger.error(f"Failed to download new site {args.site_id}")


async def _handle_update_dates(downloader, args) -> None:
    success = await downloader.update_site_between_dates(
        args.site_id, args.start_date, args.end_date, args.parameters, args.tsv_csv_output
    )
    if success:
        logger.info(
            f"Successfully updated site {args.site_id} between {args.start_date} and {args.end_date}"
        )
    else:
        logger.error(f"Failed to update site {args.site_id}")


async def _handle_update_current(downloader, args) -> None:
    success = await downloader.update_site_to_current(
        args.site_id, args.last_date, args.parameters, args.tsv_csv_output
    )
    if success:
        logger.info(f"Successfully updated site {args.site_id} to current date")
    else:
        logger.error(f"Failed to update site {args.site_id}")


async def _handle_download_new_smart(downloader, args) -> None:
    if args.site_id:
        logger.info(f"Starting smart download for site {args.site_id}")
        success = await downloader.download_new_site_smart(
            args.site_id, args.required_types, args.min_years, args.tsv_csv_output
        )
        if success:
            logger.info(f"Successfully completed smart download for site {args.site_id}")
        else:
            logger.error(f"Failed smart download for site {args.site_id}")
        return

    if args.site_list:
        site_ids = _load_site_list(args.site_list)
    else:
        site_ids = args.sites

    logger.info(f"Starting batch smart download for {len(site_ids)} sites")
    results = await downloader.batch_download_smart(
        site_ids,
        args.required_types,
        args.min_years,
        args.tsv_csv_output,
        continue_on_error=True,
        delay=1.0,
    )
    successful_count = sum(results.values())
    if successful_count == len(site_ids):
        logger.info(f"Successfully completed all {len(site_ids)} downloads")
    else:
        logger.warning(f"Completed {successful_count}/{len(site_ids)} downloads successfully")


async def _handle_batch_smart(downloader, args) -> None:
    if args.site_list:
        site_ids = _load_site_list(args.site_list)
    else:
        site_ids = args.sites

    if not site_ids:
        logger.error("No sites provided for batch download")
        sys.exit(1)

    logger.info(f"Starting batch smart download for {len(site_ids)} sites")
    results = await downloader.batch_download_smart(
        site_ids,
        args.required_types,
        args.min_years,
        args.tsv_csv_output,
        continue_on_error=args.continue_on_error,
        delay=args.delay,
    )
    successful_count = sum(results.values())
    if successful_count == len(site_ids):
        logger.info(f"Successfully completed all {len(site_ids)} downloads")
    elif successful_count > 0:
        logger.warning(f"Completed {successful_count}/{len(site_ids)} downloads successfully")
    else:
        logger.error("All downloads failed")
        sys.exit(1)


async def _handle_fill_gaps(downloader, args) -> None:
    if args.resume:
        jobs, progress = downloader.load_session()
        if not jobs:
            logger.info("No previous session found, creating new jobs")
            jobs = downloader.create_jobs_from_gaps(args.gaps_file)
            progress = ProgressTracker(len(jobs))
    else:
        jobs = downloader.create_jobs_from_gaps(args.gaps_file)
        progress = ProgressTracker(len(jobs))

    if not jobs:
        logger.error("No jobs to process")
        return

    if hasattr(args, "max_jobs") and args.max_jobs:
        jobs = jobs[: args.max_jobs]
        progress = ProgressTracker(len(jobs))
        logger.info(f"Limited to {len(jobs)} jobs for testing")

    results = await downloader.process_jobs_batch(jobs, progress)
    downloader.save_session(results, progress)
    downloader.save_results(results, args.output)

    if args.save_raw:
        logger.info("Creating TSV and CSV files...")
        downloader.save_tsv_csv_files(results, args.tsv_csv_output)

    logger.info("\n=== Final Results ===")
    logger.info(f"Total jobs: {len(results)}")


_COMMAND_HANDLERS = {
    "download-new": _handle_download_new,
    "update-dates": _handle_update_dates,
    "update-current": _handle_update_current,
    "download-new-smart": _handle_download_new_smart,
    "batch-smart": _handle_batch_smart,
    "fill-gaps": _handle_fill_gaps,
}


async def main():
    """Main execution function."""
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.command == "download-new-smart":
        _validate_smart_args(args)

    Path(args.tsv_csv_output).mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting USGS Downloader in {args.command} mode...")

    workers, rate = _apply_fast_mode(args)
    downloader = USGSDownloader(max_workers=workers, requests_per_second=rate)

    handler = _COMMAND_HANDLERS.get(args.command)
    if handler is not None:
        await handler(downloader, args)


if __name__ == "__main__":
    asyncio.run(main())
