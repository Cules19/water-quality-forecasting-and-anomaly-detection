#!/usr/bin/env python3
"""
Centralized Parameter Registry
==============================

Single source of truth for all water quality parameter mappings between
USGS codes, EA names, and display formats.

This module provides:
- ParameterCode: Rich parameter metadata object
- ParameterType: Enum for parameter categories
- ParameterRegistry: Central registry with flexible querying

Usage:
    from core.utilities.parameter_registry import registry

    # Get parameter by USGS code
    param = registry.get_by_usgs_code('00010')
    print(param.ea_name)  # 'TEMPERATURE'

    # Get all water quality parameters
    wq_params = registry.get_by_type(ParameterType.WATER_QUALITY)

    # Get mapping dictionaries for backwards compatibility
    usgs_to_ea = registry.usgs_to_ea_map()
    ea_to_usgs = registry.ea_to_usgs_map()

Author: TSLR
Date: February 2026
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Parameter categories for water quality monitoring"""

    PHYSICAL = "physical"
    WATER_QUALITY = "water_quality"
    NUTRIENT = "nutrient"
    BIOLOGICAL = "biological"
    HYDROLOGICAL = "hydrological"
    METEOROLOGICAL = "meteorological"


@dataclass
class ParameterCode:
    """
    Complete metadata for a water quality parameter.

    Attributes:
        usgs_code: Primary USGS parameter code
        ea_name: Environment Agency parameter name (uppercase)
        display_name: Human-readable display name
        description: Full parameter description
        units: Measurement units
        column_name: Standard column name in DataFrames
        parameter_type: Category (physical, water_quality, etc.)
        priority: 1=core, 2=extended, 3=specialized
        alternative_codes: Other USGS codes for same parameter
        alternative_names: Alternative column names and identifiers
    """

    usgs_code: str
    ea_name: str
    display_name: str
    description: str
    units: str
    column_name: str
    parameter_type: ParameterType
    priority: int = 3
    alternative_codes: List[str] = field(default_factory=list)
    alternative_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate and normalize data"""
        if not self.usgs_code:
            raise ValueError("usgs_code is required")
        if not self.ea_name:
            raise ValueError("ea_name is required")
        if self.priority not in [1, 2, 3]:
            raise ValueError(f"priority must be 1, 2, or 3, got {self.priority}")

        # Ensure EA name is uppercase
        self.ea_name = self.ea_name.upper()

    def matches(self, identifier: str) -> bool:
        """
        Check if this parameter matches the given identifier.

        Args:
            identifier: USGS code, EA name, column name, or alternative

        Returns:
            True if identifier matches this parameter
        """
        identifier_upper = identifier.upper()

        # Check primary identifiers
        if identifier == self.usgs_code:
            return True
        if identifier_upper == self.ea_name:
            return True
        if identifier_upper == self.column_name.upper():
            return True
        if identifier_upper == self.display_name.upper():
            return True

        # Check alternatives
        if identifier in self.alternative_codes:
            return True
        if any(identifier_upper == alt.upper() for alt in self.alternative_names):
            return True

        return False

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "usgs_code": self.usgs_code,
            "ea_name": self.ea_name,
            "display_name": self.display_name,
            "description": self.description,
            "units": self.units,
            "column_name": self.column_name,
            "parameter_type": self.parameter_type.value,
            "priority": self.priority,
            "alternative_codes": self.alternative_codes,
            "alternative_names": self.alternative_names,
        }


class ParameterRegistry:
    """
    Central registry for all water quality parameters.

    This class maintains the single source of truth for parameter mappings
    and provides flexible querying methods.
    """

    def __init__(self):
        """Initialize empty registry"""
        self._parameters: Dict[str, ParameterCode] = {}
        self._usgs_code_index: Dict[str, str] = {}  # code -> primary_code
        self._ea_name_index: Dict[str, str] = {}  # name -> primary_code
        self._all_identifiers: Dict[str, str] = {}  # any_id -> primary_code

    def register(self, parameter: ParameterCode) -> None:
        """
        Register a parameter in the registry.

        Args:
            parameter: ParameterCode to register
        """
        code = parameter.usgs_code

        if code in self._parameters:
            logger.warning(f"Parameter {code} already registered, overwriting")

        # Store parameter
        self._parameters[code] = parameter

        # Build indices
        self._usgs_code_index[code] = code
        self._ea_name_index[parameter.ea_name.upper()] = code
        self._all_identifiers[code] = code
        self._all_identifiers[parameter.ea_name.upper()] = code
        self._all_identifiers[parameter.column_name.upper()] = code
        self._all_identifiers[parameter.display_name.upper()] = code

        # Index alternatives
        for alt_code in parameter.alternative_codes:
            self._usgs_code_index[alt_code] = code
            self._all_identifiers[alt_code] = code

        for alt_name in parameter.alternative_names:
            self._all_identifiers[alt_name.upper()] = code

    def get_by_usgs_code(self, code: str) -> Optional[ParameterCode]:
        """
        Get parameter by USGS code (primary or alternative).

        Args:
            code: USGS parameter code

        Returns:
            ParameterCode or None if not found
        """
        primary_code = self._usgs_code_index.get(code)
        if primary_code:
            return self._parameters.get(primary_code)
        return None

    def get_by_ea_name(self, name: str) -> Optional[ParameterCode]:
        """
        Get parameter by EA name.

        Args:
            name: EA parameter name (case insensitive)

        Returns:
            ParameterCode or None if not found
        """
        primary_code = self._ea_name_index.get(name.upper())
        if primary_code:
            return self._parameters.get(primary_code)
        return None

    def get_by_any_identifier(self, identifier: str) -> Optional[ParameterCode]:
        """
        Get parameter by any identifier (code, name, column name, etc.).

        Args:
            identifier: Any parameter identifier

        Returns:
            ParameterCode or None if not found
        """
        # Try exact match first
        primary_code = self._all_identifiers.get(identifier.upper())
        if primary_code:
            return self._parameters.get(primary_code)

        # Try case-sensitive code match
        primary_code = self._all_identifiers.get(identifier)
        if primary_code:
            return self._parameters.get(primary_code)

        return None

    def get_by_type(self, param_type: ParameterType) -> List[ParameterCode]:
        """
        Get all parameters of a specific type.

        Args:
            param_type: ParameterType enum value

        Returns:
            List of matching parameters
        """
        return [param for param in self._parameters.values() if param.parameter_type == param_type]

    def get_by_priority(self, priority: int) -> List[ParameterCode]:
        """
        Get all parameters with a specific priority level.

        Args:
            priority: Priority level (1=core, 2=extended, 3=specialized)

        Returns:
            List of matching parameters
        """
        return [param for param in self._parameters.values() if param.priority == priority]

    def get_core_parameters(self) -> List[ParameterCode]:
        """Get all core (priority 1) parameters"""
        return self.get_by_priority(1)

    def get_extended_parameters(self) -> List[ParameterCode]:
        """Get all extended (priority 2) parameters"""
        return self.get_by_priority(2)

    def get_specialized_parameters(self) -> List[ParameterCode]:
        """Get all specialized (priority 3) parameters"""
        return self.get_by_priority(3)

    def list_all(self) -> List[ParameterCode]:
        """Get list of all registered parameters"""
        return list(self._parameters.values())

    # Mapping methods for backwards compatibility

    def usgs_to_ea_map(self) -> Dict[str, str]:
        """
        Get USGS code to EA name mapping.

        Returns:
            Dict mapping USGS codes to EA names
        """
        result = {}
        for code, param in self._parameters.items():
            result[code] = param.ea_name
            # Include alternatives
            for alt_code in param.alternative_codes:
                result[alt_code] = param.ea_name
        return result

    def ea_to_usgs_map(self) -> Dict[str, str]:
        """
        Get EA name to USGS code mapping.

        Returns:
            Dict mapping EA names to primary USGS codes
        """
        return {param.ea_name: param.usgs_code for param in self._parameters.values()}

    def code_to_display_map(self) -> Dict[str, str]:
        """
        Get code/name to display name mapping.

        Returns:
            Dict mapping any identifier to display name
        """
        result = {}
        for param in self._parameters.values():
            # Primary mappings
            result[param.usgs_code] = param.display_name
            result[param.ea_name] = param.display_name
            result[param.column_name] = param.display_name

            # Alternative mappings
            for alt_code in param.alternative_codes:
                result[alt_code] = param.display_name
            for alt_name in param.alternative_names:
                result[alt_name] = param.display_name

        return result

    def type_to_codes_map(self) -> Dict[str, List[str]]:
        """
        Get parameter type to codes mapping (for usgs_utils compatibility).

        Returns:
            Dict mapping type names to lists of USGS codes
        """
        result = {}

        for param in self._parameters.values():
            # Use display name lowercase as key (matches usgs_utils pattern)
            key = param.display_name.lower().replace(" ", "_")

            if key not in result:
                result[key] = []

            # Add primary and alternative codes
            result[key].append(param.usgs_code)
            result[key].extend(param.alternative_codes)

        return result

    def get_statistics(self) -> Dict[str, any]:
        """Get registry statistics"""
        type_counts = {}
        for param_type in ParameterType:
            type_counts[param_type.value] = len(self.get_by_type(param_type))

        return {
            "total_parameters": len(self._parameters),
            "core_parameters": len(self.get_core_parameters()),
            "extended_parameters": len(self.get_extended_parameters()),
            "specialized_parameters": len(self.get_specialized_parameters()),
            "parameters_by_type": type_counts,
            "total_indexed_identifiers": len(self._all_identifiers),
        }

    def validate(self) -> List[str]:
        """
        Validate registry for consistency issues.

        Returns:
            List of warning messages
        """
        warnings = []

        # Check for duplicate EA names
        ea_names = {}
        for code, param in self._parameters.items():
            if param.ea_name in ea_names:
                warnings.append(
                    f"Duplicate EA name '{param.ea_name}' in codes "
                    f"{ea_names[param.ea_name]} and {code}"
                )
            else:
                ea_names[param.ea_name] = code

        # Check for missing required fields
        for code, param in self._parameters.items():
            if not param.units:
                warnings.append(f"Parameter {code} missing units")
            if not param.description:
                warnings.append(f"Parameter {code} missing description")

        return warnings


# =============================================================================
# Initialize Global Registry and Populate with Data
# =============================================================================


def _initialize_registry() -> ParameterRegistry:
    """
    Initialize and populate the global parameter registry.

    This function imports data from the existing parameters_ea_usgs_toolkit.py
    and creates ParameterCode objects for the registry.
    """
    reg = ParameterRegistry()

    # Import existing data
    try:
        from core.data.parameters_ea_usgs_toolkit import EA_TO_USGS_PARAMETERS

        # Convert to ParameterCode objects
        for ea_name, data in EA_TO_USGS_PARAMETERS.items():
            # Map parameter_type string to enum
            param_type_str = data.get("parameter_type", "physical")
            try:
                param_type = ParameterType(param_type_str)
            except ValueError:
                logger.warning(
                    f"Unknown parameter type '{param_type_str}' for {ea_name}, using PHYSICAL"
                )
                param_type = ParameterType.PHYSICAL

            # Create ParameterCode
            param = ParameterCode(
                usgs_code=data["usgs_code"],
                ea_name=ea_name,
                display_name=data.get("usgs_description", ea_name).split(",")[
                    0
                ],  # First part of description
                description=data.get("usgs_description", ""),
                units=data.get("units", ""),
                column_name=data.get("column_name", ea_name),
                parameter_type=param_type,
                priority=data.get("priority", 3),
                alternative_codes=[],  # To be populated from other sources
                alternative_names=[],  # To be populated from other sources
            )

            reg.register(param)

        logger.info(f"Initialized parameter registry with {len(EA_TO_USGS_PARAMETERS)} parameters")

    except ImportError as e:
        logger.error(f"Could not import EA_TO_USGS_PARAMETERS: {e}")
        logger.warning("Registry will be empty - parameters must be added manually")

    # Add alternative codes from usgs_utils PARAMETER_MAPPINGS
    try:
        from core.utilities.usgs_utils import PARAMETER_MAPPINGS

        for param_name, codes in PARAMETER_MAPPINGS.items():
            # Find primary code (first in list usually)
            if codes:
                primary_code = codes[0]
                param = reg.get_by_usgs_code(primary_code)

                if param:
                    # Add remaining codes as alternatives
                    for alt_code in codes[1:]:
                        if alt_code not in param.alternative_codes:
                            param.alternative_codes.append(alt_code)

        logger.info("Added alternative codes from usgs_utils")

    except ImportError:
        logger.warning("Could not import PARAMETER_MAPPINGS from usgs_utils")

    # Add alternative names from translate_parameters_toolkit
    try:
        from core.utilities.translate_parameters_toolkit import PARAMETER_NAMES

        for identifier, display_name in PARAMETER_NAMES.items():
            # Try to find the parameter
            param = reg.get_by_any_identifier(identifier)

            if param:
                # Add identifier as alternative if not already present
                if identifier not in param.alternative_names and identifier != param.usgs_code:
                    param.alternative_names.append(identifier)

        logger.info("Added alternative names from translate_parameters_toolkit")

    except ImportError:
        logger.warning("Could not import PARAMETER_NAMES from translate_parameters_toolkit")

    # Validate and report
    warnings = reg.validate()
    if warnings:
        logger.warning(f"Registry validation found {len(warnings)} issues:")
        for warning in warnings[:10]:  # Show first 10
            logger.warning(f"  - {warning}")

    return reg


# Global registry instance - initialized on first import
registry = _initialize_registry()


# Convenience functions for common operations
def get_parameter(identifier: str) -> Optional[ParameterCode]:
    """Get parameter by any identifier"""
    return registry.get_by_any_identifier(identifier)


def get_ea_name(usgs_code: str) -> Optional[str]:
    """Get EA name for a USGS code"""
    param = registry.get_by_usgs_code(usgs_code)
    return param.ea_name if param else None


def get_usgs_code(ea_name: str) -> Optional[str]:
    """Get USGS code for an EA name"""
    param = registry.get_by_ea_name(ea_name)
    return param.usgs_code if param else None


def get_display_name(identifier: str) -> Optional[str]:
    """Get display name for any identifier"""
    param = registry.get_by_any_identifier(identifier)
    return param.display_name if param else None


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("Parameter Registry Statistics")
    print("=" * 60)

    stats = registry.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("Example Queries")
    print("=" * 60)

    # Test lookups
    test_codes = ["00010", "00095", "63680", "00300", "00400"]
    for code in test_codes:
        param = registry.get_by_usgs_code(code)
        if param:
            print(f"\nUSGS Code {code}:")
            print(f"  EA Name: {param.ea_name}")
            print(f"  Display: {param.display_name}")
            print(f"  Type: {param.parameter_type.value}")
            print(f"  Units: {param.units}")
            print(f"  Priority: {param.priority}")
