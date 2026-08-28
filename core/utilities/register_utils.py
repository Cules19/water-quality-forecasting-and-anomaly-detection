"""
Registry validation and safe modification utilities

Provides helper functions to ensure registry integrity when reading/writing.
Use these functions instead of direct JSON operations on the registry.

IMPORTANT: As of January 2026, the registry uses the standardized format
defined in data/reference/Register_README.md. Key changes:
- Primary registry file: sites_register.json (was sites_registry_enhanced.json)
- Field names: 'SITE_ID' (was 'site_id'), 'source' (was 'agency_cd')
- New boolean flags: still_active, has_discharge, has_precipitation
- Structured discharge_description.statistics with comprehensive metrics

See data/reference/Register_README.md for complete format specification.
See REGISTRY_MIGRATION_NOTES.md for migration details.

Author: TSLR
Updated: 14 January 2026
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from collections import Counter


class RegistryValidator:
    """Validate registry structure and data integrity"""

    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.errors = []
        self.warnings = []

    def validate_all(self, registry_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Run all validation checks on registry data.

        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Run all checks
        self._check_structure(registry_data)
        self._check_duplicates(registry_data)
        self._check_site_data(registry_data)
        self._check_discharge_consistency(registry_data)
        self._check_coordinates(registry_data)

        return len(self.errors) == 0, self.errors, self.warnings

    def _check_structure(self, registry_data: Dict[str, Any]):
        """Check basic registry structure"""
        if not isinstance(registry_data, dict):
            self.errors.append("Registry must be a dictionary")
            return

        if len(registry_data) == 0:
            self.warnings.append("Registry is empty")

    def _check_duplicates(self, registry_data: Dict[str, Any]):
        """Check for duplicate site IDs"""
        site_ids = list(registry_data.keys())
        duplicates = [sid for sid, count in Counter(site_ids).items() if count > 1]

        if duplicates:
            self.errors.append(f"Duplicate site IDs found: {duplicates}")

    def _check_site_data(self, registry_data: Dict[str, Any]):
        """Check each site has required fields"""
        # Updated for standardized registry format (Jan 2026)
        required_fields = ["SITE_ID", "source", "parameters", "data_types"]

        for site_id, site_data in registry_data.items():
            if not isinstance(site_data, dict):
                self.errors.append(f"Site {site_id}: data must be a dictionary")
                continue

            # Check required fields
            for field in required_fields:
                if field not in site_data:
                    self.errors.append(f"Site {site_id}: missing required field '{field}'")

            # Verify SITE_ID matches key
            if "SITE_ID" in site_data and site_data["SITE_ID"] != site_id:
                self.errors.append(
                    f"Site {site_id}: SITE_ID field '{site_data['SITE_ID']}' "
                    f"doesn't match key '{site_id}'"
                )

            # Check field types
            if "source" in site_data:
                source = site_data["source"]
                if source not in ["USGS", "EA", "NEON"]:
                    self.errors.append(f"Site {site_id}: invalid source '{source}'")

            if "parameters" in site_data:
                if not isinstance(site_data["parameters"], list):
                    self.errors.append(f"Site {site_id}: parameters must be a list")

            if "data_types" in site_data:
                if not isinstance(site_data["data_types"], dict):
                    self.errors.append(f"Site {site_id}: data_types must be a dictionary")
                else:
                    # Check boolean flags
                    for key, value in site_data["data_types"].items():
                        if not isinstance(value, bool):
                            self.errors.append(
                                f"Site {site_id}: data_types.{key} must be boolean, "
                                f"got {type(value).__name__}"
                            )

    def _check_discharge_consistency(self, registry_data: Dict[str, Any]):
        """Check discharge flag consistency"""
        for site_id, site_data in registry_data.items():
            # Use standardized has_discharge flag
            has_discharge_flag = site_data.get("has_discharge", False)

            # Check if discharge is in parameters
            has_discharge_param = any(
                param in ["00060", "00061"] for param in site_data.get("parameters", [])
            )

            # Check if linked discharge exists
            has_linked_discharge_id = bool(site_data.get("linked_discharge_site_id"))

            # Check for discharge statistics
            has_discharge_stats = bool(site_data.get("discharge_description", {}).get("statistics"))

            # Validate consistency
            should_have_discharge = (
                has_discharge_param or has_linked_discharge_id or has_discharge_stats
            )

            if has_discharge_flag != should_have_discharge:
                self.warnings.append(
                    f"Site {site_id}: has_discharge={has_discharge_flag} but "
                    f"data suggests {should_have_discharge} "
                    f"(param={has_discharge_param}, linked={has_linked_discharge_id}, "
                    f"stats={has_discharge_stats})"
                )

            # Check discharge_description structure if present
            discharge_desc = site_data.get("discharge_description", {})
            if discharge_desc:
                if not isinstance(discharge_desc, dict):
                    self.errors.append(
                        f"Site {site_id}: discharge_description must be a dictionary"
                    )
                elif "statistics" in discharge_desc:
                    stats = discharge_desc["statistics"]
                    if not isinstance(stats, dict):
                        self.errors.append(
                            f"Site {site_id}: discharge_description.statistics must be a dictionary"
                        )
                    else:
                        # Check for required stat fields
                        required_stats = ["mean_discharge", "cv_discharge", "flashiness_index"]
                        for stat in required_stats:
                            if stat not in stats:
                                self.warnings.append(
                                    f"Site {site_id}: discharge statistics missing '{stat}'"
                                )

    def _check_coordinates(self, registry_data: Dict[str, Any]):
        """Check coordinate validity"""
        for site_id, site_data in registry_data.items():
            lat = site_data.get("latitude")
            lon = site_data.get("longitude")

            if lat is not None:
                try:
                    lat_float = float(lat)
                    if not -90 <= lat_float <= 90:
                        self.errors.append(f"Site {site_id}: invalid latitude {lat_float}")
                except (ValueError, TypeError):
                    self.errors.append(f"Site {site_id}: latitude is not a number: {lat}")

            if lon is not None:
                try:
                    lon_float = float(lon)
                    if not -180 <= lon_float <= 180:
                        self.errors.append(f"Site {site_id}: invalid longitude {lon_float}")
                except (ValueError, TypeError):
                    self.errors.append(f"Site {site_id}: longitude is not a number: {lon}")


def load_registry_safe(registry_path: Path, validate: bool = True) -> Dict[str, Any]:
    """
    Safely load registry with optional validation.

    Args:
        registry_path: Path to registry JSON file
        validate: Whether to run validation checks

    Returns:
        Registry data dictionary

    Raises:
        ValueError: If registry is invalid and validation is enabled
    """
    try:
        with open(registry_path, "r") as f:
            registry_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Registry is not valid JSON: {e}")
    except FileNotFoundError:
        raise ValueError(f"Registry file not found: {registry_path}")

    if validate:
        validator = RegistryValidator(registry_path)
        is_valid, errors, warnings = validator.validate_all(registry_data)

        if warnings:
            print("️ Registry validation warnings:")
            for warning in warnings[:5]:  # Show first 5
                print(f" {warning}")
            if len(warnings) > 5:
                print(f" ... and {len(warnings) - 5} more warnings")

        if not is_valid:
            error_msg = "Registry validation failed:\n" + "\n".join(errors[:10])
            if len(errors) > 10:
                error_msg += f"\n... and {len(errors) - 10} more errors"
            raise ValueError(error_msg)

    return registry_data


def save_registry_safe(
    registry_path: Path, registry_data: Dict[str, Any], backup: bool = True, validate: bool = True
) -> Path:
    """
    Safely save registry with backup and validation.

    Args:
        registry_path: Path to registry JSON file
        registry_data: Registry data to save
        backup: Whether to create backup before saving
        validate: Whether to validate data before saving

    Returns:
        Path to backup file (if created)

    Raises:
        ValueError: If validation fails
    """
    backup_path = None

    # Validate before saving
    if validate:
        validator = RegistryValidator(registry_path)
        is_valid, errors, warnings = validator.validate_all(registry_data)

        if not is_valid:
            error_msg = "Cannot save invalid registry:\n" + "\n".join(errors[:10])
            if len(errors) > 10:
                error_msg += f"\n... and {len(errors) - 10} more errors"
            raise ValueError(error_msg)

        if warnings:
            print(f"️ Registry has {len(warnings)} warnings but saving anyway")

    # Create backup
    if backup and registry_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = registry_path.parent / f"{registry_path.stem}_backup_{timestamp}.json"
        shutil.copy2(registry_path, backup_path)
        print(f" Backup created: {backup_path.name}")

    # Save registry
    with open(registry_path, "w") as f:
        json.dump(registry_data, f, indent=2)

    print(f" Registry saved: {registry_path}")

    return backup_path


def update_site_data_safe(
    registry_path: Path,
    site_id: str,
    updates: Dict[str, Any],
    validate: bool = True,
    backup: bool = True,
) -> bool:
    """
    Safely update a single site's data in the registry.

    Args:
        registry_path: Path to registry JSON file
        site_id: Site ID to update
        updates: Dictionary of fields to update
        validate: Whether to validate after update
        backup: Whether to create backup

    Returns:
        True if update successful

    Raises:
        ValueError: If site doesn't exist or validation fails
    """
    # Load registry
    registry_data = load_registry_safe(registry_path, validate=False)

    # Check site exists
    if site_id not in registry_data:
        raise ValueError(f"Site {site_id} not found in registry")

    # Update site data
    site_data = registry_data[site_id]

    # Deep merge updates
    for key, value in updates.items():
        if isinstance(value, dict) and key in site_data and isinstance(site_data[key], dict):
            # Merge dictionaries (e.g., data_types)
            site_data[key].update(value)
        else:
            # Replace value
            site_data[key] = value

    registry_data[site_id] = site_data

    # Save with validation
    save_registry_safe(registry_path, registry_data, backup=backup, validate=validate)

    return True


def get_registry_stats(registry_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get statistics about the registry.

    Args:
        registry_data: Registry dictionary

    Returns:
        Dictionary of statistics
    """
    stats = {
        "total_sites": len(registry_data),
        "agencies": {},
        "sites_with_discharge": 0,
        "sites_with_parameters": 0,
        "avg_parameters_per_site": 0,
        "sites_with_coordinates": 0,
    }

    agency_counts = Counter()
    param_counts = []

    for site_id, site_data in registry_data.items():
        # Count by source
        source = site_data.get("source", "Unknown")
        agency_counts[source] += 1

        # Count discharge
        if site_data.get("data_types", {}).get("has_discharge", False):
            stats["sites_with_discharge"] += 1

        # Count parameters
        params = site_data.get("parameters", [])
        if params:
            stats["sites_with_parameters"] += 1
            param_counts.append(len(params))

        # Count coordinates
        if site_data.get("latitude") and site_data.get("longitude"):
            stats["sites_with_coordinates"] += 1

    stats["agencies"] = dict(agency_counts)

    if param_counts:
        stats["avg_parameters_per_site"] = sum(param_counts) / len(param_counts)

    return stats


def print_registry_report(registry_path: Path):
    """
    Print a comprehensive registry validation report.

    Args:
        registry_path: Path to registry JSON file
    """
    print("=" * 80)
    print("REGISTRY VALIDATION REPORT")
    print("=" * 80)
    print(f"File: {registry_path}")
    print()

    # Load and validate
    try:
        registry_data = load_registry_safe(registry_path, validate=True)
        print(" Registry structure: VALID")
    except ValueError as e:
        print(" Registry structure: INVALID")
        print(f" {e}")
        return

    print()

    # Statistics
    stats = get_registry_stats(registry_data)

    print("STATISTICS")
    print("-" * 80)
    print(f"Total sites: {stats['total_sites']}")
    print("Agencies:")
    for agency, count in stats["agencies"].items():
        pct = (count / stats["total_sites"]) * 100
        print(f" - {agency}: {count} ({pct:.1f}%)")
    print()

    discharge_pct = (stats["sites_with_discharge"] / stats["total_sites"]) * 100
    print(f"Sites with discharge: {stats['sites_with_discharge']} ({discharge_pct:.1f}%)")

    params_pct = (stats["sites_with_parameters"] / stats["total_sites"]) * 100
    print(f"Sites with parameters: {stats['sites_with_parameters']} ({params_pct:.1f}%)")
    print(f"Average parameters per site: {stats['avg_parameters_per_site']:.1f}")

    coords_pct = (stats["sites_with_coordinates"] / stats["total_sites"]) * 100
    print(f"Sites with coordinates: {stats['sites_with_coordinates']} ({coords_pct:.1f}%)")

    print()
    print("=" * 80)
    print(" VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    from config import config

    print_registry_report(config.REGISTRY_PATH)
