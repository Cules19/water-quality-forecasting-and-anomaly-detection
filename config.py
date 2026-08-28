"""
Configuration file for project paths and server details.
Usage: from config import config

to load registry path, use:
    df = read
"""
import os
import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

# Common configuration and imports for parameter estimation
warnings.filterwarnings('ignore')


class Config:
    """Central configuration class for all project settings."""

    # Base directories - use relative paths from project root
    BASE_DIR = Path(__file__).parent.absolute()
    DATA_DIR = BASE_DIR / "data"  # In-repo; large dirs gitignored and tracked via DVC + Google Drive
    OUTPUT_DIR = BASE_DIR / "outputs"
    REGISTRY_PATH = BASE_DIR / "data" / "reference" / "sites_register.json"
    TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"


    # Data paths
    RAW_DATA_PATH = DATA_DIR / "raw"
    PROCESSED_DATA_PATH = DATA_DIR / "processed"
    RAW_DIR = str(RAW_DATA_PATH)
    CLEAN_DIR = str(DATA_DIR / "clean")
    CACHE_PATH = DATA_DIR / "training"

    # Output paths
    RESULTS_PATH = OUTPUT_DIR / "results"
    LOGS_PATH = OUTPUT_DIR / "logs"
    MODELS_PATH = OUTPUT_DIR / "models"
    REPORTS_PATH = OUTPUT_DIR / "reports"

    # Feature flags
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    VERBOSE = os.getenv("VERBOSE", "False").lower() == "true"

    @classmethod
    def setup_paths(cls):
        """Setup Python paths for imports"""
        # Add core tools directory to path
        tools_path = str(cls.BASE_DIR / "core" / "tools")
        if tools_path not in sys.path:
            sys.path.append(tools_path)

        # Add base directory to path
        base_path = str(cls.BASE_DIR)
        if base_path not in sys.path:
            sys.path.append(base_path)

    @classmethod
    def import_tools(cls):
        """Import database tools after setting up paths"""
        cls.setup_paths()

        try:
            from core.utilities.database_toolkit import (
                DatabaseQueryManager,
                DataQualityAssessment,
                FeatureEngineering,
                perform_database_query_analysis,
                prepare_analysis_dataset
            )
            return {
                'DatabaseQueryManager': DatabaseQueryManager,
                'DataQualityAssessment': DataQualityAssessment,
                'FeatureEngineering': FeatureEngineering,
                'perform_database_query_analysis': perform_database_query_analysis,
                'prepare_analysis_dataset': prepare_analysis_dataset
            }
        except ImportError as e:
            print(f"Warning: Could not import tools - {e}")
            return {}

    @classmethod
    def create_directories(cls):
        """Create all necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.OUTPUT_DIR,
            cls.RAW_DATA_PATH,
            cls.PROCESSED_DATA_PATH,
            cls.CACHE_PATH,
            cls.RESULTS_PATH,
            cls.LOGS_PATH,
            cls.MODELS_PATH,
            cls.REPORTS_PATH
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Initialize config and setup paths
config = Config()
config.setup_paths()

# Make tools available at module level for easy import
tools = config.import_tools()
