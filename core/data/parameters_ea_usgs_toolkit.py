#!/usr/bin/env python3
"""
EA to USGS Parameter Reference

Centralized parameter mapping reference for converting EA data to USGS-compatible format.
This module provides the definitive mapping between EA parameter names and USGS parameter codes.

Author: TSLR
Date: 2025-08-29
"""

# EA to USGS Parameter Mapping Dictionary
# This is the authoritative source for all EA-USGS parameter conversions
EA_TO_USGS_PARAMETERS = {
    # Core Water Quality Parameters (Original 5)
    "TEMPERATURE": {
        "usgs_code": "00010",
        "usgs_description": "Temperature, water, degrees Celsius",
        "column_name": "TEMP [deg C]",
        "units": "deg C",
        "parameter_type": "physical",
        "priority": 1,
    },
    "CONDUCTIVITY": {
        "usgs_code": "00095",
        "usgs_description": "Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius",
        "column_name": "Conductance [uS/cm]",
        "units": "uS/cm",
        "parameter_type": "water_quality",
        "priority": 1,
    },
    "TURBIDITY": {
        "usgs_code": "63680",
        "usgs_description": "Turbidity, water, unfiltered, nephelometric turbidity units",
        "column_name": "Turbidity [NTU]",
        "units": "NTU",
        "parameter_type": "water_quality",
        "priority": 1,
    },
    "DISSOLVED OXYGEN": {
        "usgs_code": "00300",
        "usgs_description": "Dissolved oxygen, water, unfiltered, milligrams per liter",
        "column_name": "DISSOLVED_OXYGEN [mg/L]",
        "units": "mg/L",
        "parameter_type": "water_quality",
        "priority": 1,
    },
    "PH": {
        "usgs_code": "00400",
        "usgs_description": "pH, water, unfiltered, field, standard units",
        "column_name": "pH [units]",
        "units": "standard units",
        "parameter_type": "water_quality",
        "priority": 1,
    },
    # Extended Parameters (Additional 6)
    "AMMONIUM": {
        "usgs_code": "00608",
        "usgs_description": "Ammonia-nitrogen, water, filtered, milligrams per liter as nitrogen",
        "column_name": "Ammonia [mg/L as N]",
        "units": "mg/L as N",
        "parameter_type": "nutrient",
        "priority": 2,
    },
    "NITRATE": {
        "usgs_code": "00618",
        "usgs_description": "Nitrate, water, filtered, milligrams per liter as nitrogen",
        "column_name": "Nitrate [mg/L as N]",
        "units": "mg/L as N",
        "parameter_type": "nutrient",
        "priority": 2,
    },
    "CHLOROPHYLL": {
        "usgs_code": "32211",
        "usgs_description": "Chlorophyll a, water, fluorometric method, micrograms per liter",
        "column_name": "Chlorophyll-a [ug/L]",
        "units": "ug/L",
        "parameter_type": "biological",
        "priority": 2,
    },
    "BLUE GREEN ALGAE": {
        "usgs_code": "32212",
        "usgs_description": "Blue-green algae, water, fluorometric method, cells per milliliter",
        "column_name": "Blue-green algae [cells/mL]",
        "units": "cells/mL",
        "parameter_type": "biological",
        "priority": 3,
    },
    "SALINITY": {
        "usgs_code": "00480",
        "usgs_description": "Salinity, water, unfiltered, practical salinity units",
        "column_name": "Salinity [PSU]",
        "units": "PSU",
        "parameter_type": "physical",
        "priority": 3,
    },
    "FLUORESCENT DISSOLVED ORGANIC MATTER": {
        "usgs_code": "32295",
        "usgs_description": "Dissolved organic matter fluorescence (fDOM), water, unfiltered, quinine sulfate units",
        "column_name": "fDOM [QSU]",
        "units": "QSU",
        "parameter_type": "water_quality",
        "priority": 3,
    },
    # Flow Parameters (UK Units)
    "DISCHARGE": {
        "usgs_code": "00060",
        "usgs_description": "Discharge, cubic meters per second",
        "column_name": "Discharge [m³/s]",
        "units": "m³/s",
        "parameter_type": "physical",
        "priority": 2,
    },
    "GAGE HEIGHT": {
        "usgs_code": "00065",
        "usgs_description": "Gage height, meters",
        "column_name": "Gage Height [m]",
        "units": "m",
        "parameter_type": "physical",
        "priority": 2,
    },
}

# USGS to EA reverse mapping (for lookups)
USGS_TO_EA_PARAMETERS = {
    info["usgs_code"]: {
        "ea_parameter": ea_param,
        "column_name": info["column_name"],
        "units": info["units"],
        "parameter_type": info["parameter_type"],
        "priority": info["priority"],
    }
    for ea_param, info in EA_TO_USGS_PARAMETERS.items()
}

# Parameter groupings by priority
CORE_PARAMETERS = {k: v for k, v in EA_TO_USGS_PARAMETERS.items() if v["priority"] == 1}
EXTENDED_PARAMETERS = {k: v for k, v in EA_TO_USGS_PARAMETERS.items() if v["priority"] == 2}
SPECIALIZED_PARAMETERS = {k: v for k, v in EA_TO_USGS_PARAMETERS.items() if v["priority"] == 3}

# Parameter groupings by type
PHYSICAL_PARAMETERS = {
    k: v for k, v in EA_TO_USGS_PARAMETERS.items() if v["parameter_type"] == "physical"
}
WATER_QUALITY_PARAMETERS = {
    k: v for k, v in EA_TO_USGS_PARAMETERS.items() if v["parameter_type"] == "water_quality"
}
NUTRIENT_PARAMETERS = {
    k: v for k, v in EA_TO_USGS_PARAMETERS.items() if v["parameter_type"] == "nutrient"
}
BIOLOGICAL_PARAMETERS = {
    k: v for k, v in EA_TO_USGS_PARAMETERS.items() if v["parameter_type"] == "biological"
}


def get_usgs_code(ea_parameter: str) -> str:
    """
    Get USGS parameter code for an EA parameter.

    Args:
        ea_parameter: EA parameter name

    Returns:
        USGS parameter code or None if not found
    """
    mapping = EA_TO_USGS_PARAMETERS.get(ea_parameter)
    return mapping["usgs_code"] if mapping else None


def get_column_name(ea_parameter: str) -> str:
    """
    Get USGS-compatible column name for an EA parameter.

    Args:
        ea_parameter: EA parameter name

    Returns:
        USGS-compatible column name
    """
    mapping = EA_TO_USGS_PARAMETERS.get(ea_parameter)
    return mapping["column_name"] if mapping else ea_parameter


def get_parameter_info(ea_parameter: str) -> dict:
    """
    Get complete parameter mapping information.

    Args:
        ea_parameter: EA parameter name

    Returns:
        Complete parameter mapping info or None
    """
    return EA_TO_USGS_PARAMETERS.get(ea_parameter)


def get_parameters_by_type(parameter_type: str) -> dict:
    """
    Get all parameters of a specific type.

    Args:
        parameter_type: Type of parameters ('physical', 'water_quality', 'nutrient', 'biological')

    Returns:
        Dictionary of parameters of the specified type
    """
    type_mappings = {
        "physical": PHYSICAL_PARAMETERS,
        "water_quality": WATER_QUALITY_PARAMETERS,
        "nutrient": NUTRIENT_PARAMETERS,
        "biological": BIOLOGICAL_PARAMETERS,
    }
    return type_mappings.get(parameter_type, {})


def get_parameters_by_priority(priority: int) -> dict:
    """
    Get all parameters of a specific priority level.

    Args:
        priority: Priority level (1=core, 2=extended, 3=specialized)

    Returns:
        Dictionary of parameters at the specified priority level
    """
    priority_mappings = {1: CORE_PARAMETERS, 2: EXTENDED_PARAMETERS, 3: SPECIALIZED_PARAMETERS}
    return priority_mappings.get(priority, {})


def create_parameter_mapping_dict() -> dict:
    """
    Create the legacy parameter mapping dictionary for backwards compatibility.

    Returns:
        Dictionary mapping EA parameters to USGS column names
    """
    return {ea_param: info["column_name"] for ea_param, info in EA_TO_USGS_PARAMETERS.items()}


# Summary information
PARAMETER_SUMMARY = {
    "total_parameters": len(EA_TO_USGS_PARAMETERS),
    "core_parameters": len(CORE_PARAMETERS),
    "extended_parameters": len(EXTENDED_PARAMETERS),
    "specialized_parameters": len(SPECIALIZED_PARAMETERS),
    "parameter_types": {
        "physical": len(PHYSICAL_PARAMETERS),
        "water_quality": len(WATER_QUALITY_PARAMETERS),
        "nutrient": len(NUTRIENT_PARAMETERS),
        "biological": len(BIOLOGICAL_PARAMETERS),
    },
}

# For backwards compatibility - the original parameter mapping
PARAMETER_MAPPING = create_parameter_mapping_dict()

if __name__ == "__main__":
    print(" EA to USGS Parameter Reference")
    print("=" * 50)
    print(f"Total Parameters: {PARAMETER_SUMMARY['total_parameters']}")
    print(f"Core Parameters: {PARAMETER_SUMMARY['core_parameters']}")
    print(f"Extended Parameters: {PARAMETER_SUMMARY['extended_parameters']}")
    print(f"Specialized Parameters: {PARAMETER_SUMMARY['specialized_parameters']}")
    print()

    print(" Parameters by Type:")
    for param_type, count in PARAMETER_SUMMARY["parameter_types"].items():
        print(f" {param_type}: {count}")
    print()

    print(" Core Parameters (Priority 1):")
    for ea_param, info in CORE_PARAMETERS.items():
        print(f" {ea_param} → {info['usgs_code']} ({info['column_name']})")
    print()

    print(" Extended Parameters (Priority 2):")
    for ea_param, info in EXTENDED_PARAMETERS.items():
        print(f" {ea_param} → {info['usgs_code']} ({info['column_name']})")
    print()

    print(" Specialized Parameters (Priority 3):")
    for ea_param, info in SPECIALIZED_PARAMETERS.items():
        print(f" {ea_param} → {info['usgs_code']} ({info['column_name']})")
