#!/usr/bin/env python3
"""
Station Class Hierarchy - Object-Oriented Station Management

This module provides a comprehensive class hierarchy for managing monitoring stations
from different data sources (EA, USGS) with proper inheritance and composition.

Class Hierarchy:
    Site (basic location info)
    └── MonitoringStation (adds temporal status)
        ├── WaterQualityStation (adds parameters and data types)
        ├── DischargeStation (gauging/streamflow)
        ├── WeatherStation (precipitation and meteorological data)
        └── GroundwaterStation (groundwater monitoring)

Author: TSLR
Date: 17 January 2026
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum
import json

# ============================================================================
# CONSTANTS (extracted from magic numbers)
# ============================================================================

LATITUDE_LIMIT = 90
LATITUDE_LIMIT_1 = 90
LONGITUDE_LIMIT = 180
LONGITUDE_LIMIT_1 = 180


class DataSource(Enum):
    """Enumeration of data sources"""

    USGS = "USGS"
    EA = "EA"
    OTHER = "OTHER"


class StationType(Enum):
    """Enumeration of station types"""

    WATER_QUALITY = "water_quality"
    DISCHARGE = "discharge"
    WEATHER = "weather"
    GROUNDWATER = "groundwater"
    HYBRID = "hybrid"  # For stations with multiple types


class WeatherDataType(Enum):
    """Type of weather data"""

    PHYSICAL = "physical"  # Directly measured (e.g., rain gauge)
    COMPUTED = "computed"  # Modeled or computed (e.g., radar estimates)


@dataclass
class Site:
    """
    Base site class with fundamental location information.

    All monitoring sites must have at minimum:
    - A unique identifier
    - Geographic coordinates
    - Data source
    """

    SITE_ID: str
    source: DataSource
    latitude: float
    longitude: float
    elevation: Optional[float] = None
    country_cd: str = "Unknown"

    def __post_init__(self):
        """Validate core fields"""
        if not self.SITE_ID:
            raise ValueError("SITE_ID cannot be empty")
        if not isinstance(self.source, DataSource):
            # Convert string to enum if needed
            self.source = DataSource(self.source) if isinstance(self.source, str) else self.source
        if not (-LATITUDE_LIMIT <= self.latitude <= LATITUDE_LIMIT_1):
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not (-LONGITUDE_LIMIT <= self.longitude <= LONGITUDE_LIMIT_1):
            raise ValueError(f"Invalid longitude: {self.longitude}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["source"] = self.source.value if isinstance(self.source, DataSource) else self.source
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Site":
        """Create from dictionary, filtering to only valid fields for this class"""
        # Convert source string to enum if needed
        if "source" in data and isinstance(data["source"], str):
            data = data.copy()
            data["source"] = DataSource(data["source"])

        # Get all field names from this class and parent classes
        import inspect

        all_fields = set()
        for klass in inspect.getmro(cls):
            if hasattr(klass, "__annotations__"):
                all_fields.update(klass.__annotations__.keys())

        # Filter data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in all_fields}

        return cls(**filtered_data)


@dataclass
class MonitoringStation(Site):
    """
    Station with temporal status information.

    Extends Site with:
    - Operational status (active/inactive)
    - Station naming and identification
    - Basin/watershed information
    """

    waterbody_name: str = "Unknown"
    station_nm: Optional[str] = None
    basin_cd: str = "Unknown"
    still_active: bool = True
    start_date: Optional[str] = (
        None  # ISO format date - renamed from date_started for compatibility
    )
    end_date: Optional[str] = None  # ISO format date - renamed from date_ended for compatibility
    status: Optional[str] = None  # Status string (Open/Closed)

    # Additional metadata fields from registry
    river_name: Optional[str] = None
    filename: Optional[str] = None
    last_checked: Optional[str] = None
    units: str = "Metric"

    def __post_init__(self):
        """Validate temporal status"""
        super().__post_init__()

        # Validate dates if provided
        if self.start_date:
            try:
                datetime.fromisoformat(self.start_date.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError(f"Invalid start_date format: {self.start_date}")

        if self.end_date:
            try:
                datetime.fromisoformat(self.end_date.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError(f"Invalid end_date format: {self.end_date}")
            # If ended date exists, station should not be active
            if self.still_active:
                self.still_active = False

    @property
    def is_active(self) -> bool:
        """Check if station is currently active"""
        return self.still_active and self.end_date is None


@dataclass
class WaterQualityStation(MonitoringStation):
    """
    Water quality monitoring station.

    Extends MonitoringStation with:
    - Parameters measured
    - Data availability flags
    - Linked stations for additional data sources
    """

    parameters: List[str] = field(default_factory=list)
    data_types: Dict[str, bool] = field(default_factory=dict)
    has_discharge: bool = False
    has_precipitation: bool = False
    has_linked_discharge: bool = False
    has_linked_precipitation: bool = False

    # Linked stations
    linked_discharge_site_id: Optional[str] = None
    linked_precipitation_site_id: Optional[str] = None

    # Discharge information
    discharge_description: Dict[str, Any] = field(default_factory=dict)
    precipitation_description: Dict[str, Any] = field(default_factory=dict)
    discharge_data_file_name: Optional[str] = None
    precipitation_data_file_name: Optional[str] = None

    # Additional metadata
    completeness_percentage: Optional[float] = None
    huc_cd: Optional[str] = None  # USGS hydrologic unit code
    parameter_count: int = 0
    total_rows: int = 0
    total_data_points: int = 0

    def __post_init__(self):
        """Validate water quality specific fields"""
        super().__post_init__()

        # Ensure parameters is a list
        if not isinstance(self.parameters, list):
            self.parameters = list(self.parameters) if self.parameters else []

        # Update parameter_count if not set
        if self.parameter_count == 0:
            self.parameter_count = len(self.parameters)

        # Auto-detect discharge/precipitation from parameters if not set
        if not self.has_discharge and self.parameters:
            discharge_codes = {"00060", "00061"}
            self.has_discharge = bool(discharge_codes & set(self.parameters))

        # Set linked flags based on linked site IDs
        if self.linked_discharge_site_id:
            self.has_linked_discharge = True
        if self.linked_precipitation_site_id:
            self.has_linked_precipitation = True

        # Validate completeness percentage
        if self.completeness_percentage is not None:
            if not (0 <= self.completeness_percentage <= 100):
                raise ValueError(f"Invalid completeness: {self.completeness_percentage}")

    def get_station_types(self) -> List[StationType]:
        """Determine all applicable station types"""
        types = [StationType.WATER_QUALITY]

        if self.has_discharge or self.discharge_description:
            types.append(StationType.DISCHARGE)

        if self.has_precipitation or self.precipitation_description:
            types.append(StationType.PRECIPITATION)

        return types


@dataclass
class DischargeStation(MonitoringStation):
    """
    Gauging station with discharge/streamflow measurements.

    Direct specialization of MonitoringStation for streamflow monitoring.
    Includes discharge-specific statistics and metadata.
    """

    discharge_description: Dict[str, Any] = field(default_factory=dict)
    parameters: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure discharge is available"""
        super().__post_init__()

        # Ensure discharge codes in parameters
        discharge_codes = {"00060", "00061"}
        if not (discharge_codes & set(self.parameters)):
            if not self.parameters:
                self.parameters = []
            self.parameters.extend(["00060"])  # Add standard discharge code

    @property
    def discharge_statistics(self) -> Optional[Dict[str, float]]:
        """Get discharge statistics if available"""
        return self.discharge_description.get("statistics")

    @property
    def mean_discharge(self) -> Optional[float]:
        """Get mean discharge value"""
        stats = self.discharge_statistics
        return stats.get("mean_discharge") if stats else None

    @property
    def cv_discharge(self) -> Optional[float]:
        """Get coefficient of variation for discharge"""
        stats = self.discharge_statistics
        return stats.get("cv_discharge") if stats else None

    @property
    def flashiness_index(self) -> Optional[float]:
        """Get flashiness index (RBI)"""
        stats = self.discharge_statistics
        return stats.get("flashiness_index") if stats else None


@dataclass
class WeatherStation(MonitoringStation):
    """
    Weather monitoring station (precipitation and meteorological data).

    Direct specialization of MonitoringStation for weather/precipitation monitoring.
    Can handle both physical measurements and computed/modeled data.
    """

    precipitation_description: Dict[str, Any] = field(default_factory=dict)
    weather_data_type: Optional[WeatherDataType] = None
    parameters: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate weather station"""
        super().__post_init__()

        # Convert string to enum if needed
        if isinstance(self.weather_data_type, str):
            self.weather_data_type = WeatherDataType(self.weather_data_type.lower())

    @property
    def is_physical(self) -> bool:
        """Check if data is from physical measurements"""
        return self.weather_data_type == WeatherDataType.PHYSICAL

    @property
    def is_computed(self) -> bool:
        """Check if data is computed/modeled"""
        return self.weather_data_type == WeatherDataType.COMPUTED

    @property
    def precipitation_statistics(self) -> Optional[Dict[str, Any]]:
        """Get precipitation statistics if available"""
        return self.precipitation_description.get("statistics")


@dataclass
class GroundwaterStation(MonitoringStation):
    """
    Groundwater monitoring station.

    Direct specialization of MonitoringStation for groundwater level and quality monitoring.
    """

    well_depth: Optional[float] = None
    aquifer_cd: Optional[str] = None
    parameters: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate groundwater specific fields"""
        super().__post_init__()

        # Validate well depth
        if self.well_depth is not None and self.well_depth < 0:
            raise ValueError(f"Invalid well_depth: {self.well_depth}")


class StationFactory:
    """
    Factory for creating appropriate station objects from registry data.

    Automatically determines the best station class based on available data.
    """

    @staticmethod
    def create_station(site_id: str, site_data: Dict[str, Any]) -> WaterQualityStation:
        """
        Create appropriate station object from registry data.

        Args:
            site_id: Unique site identifier
            site_data: Dictionary of site data from registry

        Returns:
            Station object of appropriate type

        Raises:
            ValueError: If required fields are missing
        """
        # Ensure SITE_ID is set
        if "SITE_ID" not in site_data:
            site_data["SITE_ID"] = site_id

        # Convert source to DataSource enum if string
        if "source" in site_data and isinstance(site_data["source"], str):
            site_data["source"] = DataSource(site_data["source"])

        # Determine station type based on available data
        has_discharge = site_data.get("has_discharge", False)
        has_precipitation = site_data.get("has_precipitation", False)
        discharge_desc = site_data.get("discharge_description", {})
        precip_desc = site_data.get("precipitation_description", {})

        # Check for specific station types
        parameters = site_data.get("parameters", [])
        groundwater_codes = {"72019", "72020", "72150"}  # Common groundwater level codes
        is_groundwater = bool(groundwater_codes & set(parameters))

        # Priority order: groundwater > discharge > weather > water quality
        if is_groundwater:
            target_class = GroundwaterStation
        elif has_discharge or discharge_desc:
            target_class = DischargeStation
        elif has_precipitation or precip_desc:
            target_class = WeatherStation
        else:
            # Default to WaterQualityStation if it has parameters
            target_class = WaterQualityStation

        # Get all valid field names for the target class
        import inspect

        all_fields = set()
        for klass in inspect.getmro(target_class):
            if hasattr(klass, "__annotations__"):
                all_fields.update(klass.__annotations__.keys())

        # Filter site_data to only include valid fields
        filtered_data = {k: v for k, v in site_data.items() if k in all_fields}

        # Create station object
        try:
            return target_class(**filtered_data)
        except TypeError as e:
            # If that fails, provide more helpful error
            raise ValueError(f"Could not create {target_class.__name__} for {site_id}: {e}")

    @staticmethod
    def from_registry(registry_path: str) -> Dict[str, WaterQualityStation]:
        """
        Load all stations from registry file.

        Args:
            registry_path: Path to sites_register.json

        Returns:
            Dictionary mapping site_id to station object
        """
        with open(registry_path, "r") as f:
            registry_data = json.load(f)

        stations = {}
        for site_id, site_data in registry_data.items():
            try:
                stations[site_id] = StationFactory.create_station(site_id, site_data)
            except Exception as e:
                print(f"Warning: Could not create station for {site_id}: {e}")

        return stations

    @staticmethod
    def to_registry(stations: Dict[str, WaterQualityStation], output_path: str):
        """
        Save stations to registry file.

        Args:
            stations: Dictionary of station objects
            output_path: Path to save registry
        """
        registry_data = {site_id: station.to_dict() for site_id, station in stations.items()}

        with open(output_path, "w") as f:
            json.dump(registry_data, f, indent=2, default=str)


# Example usage and testing
if __name__ == "__main__":
    # Example: Create a USGS discharge station
    usgs_station = DischargeStation(
        SITE_ID="02335000",
        source=DataSource.USGS,
        latitude=33.8923,
        longitude=-84.2341,
        elevation=305.0,
        waterbody_name="Chattahoochee River",
        station_nm="CHATTAHOOCHEE RIVER AT ATLANTA, GA",
        basin_cd="03130002",
        still_active=True,
        parameters=["00060", "00010", "00095"],
        discharge_description={
            "source": "direct",
            "statistics": {
                "mean_discharge": 2500.5,
                "cv_discharge": 0.85,
                "flashiness_index": 0.12,
            },
        },
    )

    print("USGS Discharge Station Created:")
    print(f" ID: {usgs_station.SITE_ID}")
    print(f" Active: {usgs_station.is_active}")
    print(f" Mean Discharge: {usgs_station.mean_discharge}")
    print(f" Parameters: {len(usgs_station.parameters)}")

    # Example: Create a weather station
    weather_station = WeatherStation(
        SITE_ID="WEATHER001",
        source=DataSource.EA,
        latitude=52.1234,
        longitude=-1.5678,
        elevation=120.0,
        waterbody_name="Met Office Station",
        still_active=True,
        weather_data_type=WeatherDataType.PHYSICAL,
        precipitation_description={
            "source": "rain_gauge",
            "statistics": {"total_annual_mm": 650.5, "max_daily_mm": 45.2},
        },
    )

    print("\nWeather Station Created:")
    print(f" ID: {weather_station.SITE_ID}")
    print(
        f" Data Type: Physical={weather_station.is_physical}, Computed={weather_station.is_computed}"
    )

    # Example: Create an EA water quality station
    wq_station = WaterQualityStation(
        SITE_ID="E24005",
        source=DataSource.EA,
        latitude=52.1234,
        longitude=-1.5678,
        elevation=120.0,
        waterbody_name="River Thames",
        still_active=True,
        parameters=["temperature", "ph", "dissolved_oxygen"],
        has_discharge=True,
        linked_discharge_site_id="E24001",
        completeness_percentage=85.5,
    )

    print("\nEA Water Quality Station Created:")
    print(f" ID: {wq_station.SITE_ID}")
    print(f" Types: {[t.value for t in wq_station.get_station_types()]}")
    print(f" Has Linked Discharge: {wq_station.has_linked_discharge}")
    print(f" Completeness: {wq_station.completeness_percentage}%")
