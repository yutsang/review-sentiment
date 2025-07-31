"""
Configuration loading utilities
"""
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str = "config/settings.json") -> Dict[str, Any]:
    """Load general settings configuration"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def load_app_config(config_path: str = "config/apps.json") -> Dict[str, Any]:
    """Load apps configuration"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Apps configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in apps configuration file: {e}")


def get_app_config(app_key: str, config_path: str = "config/apps.json") -> Optional[Dict[str, Any]]:
    """Get configuration for a specific app"""
    app_config = load_app_config(config_path)
    return app_config.get("apps", {}).get(app_key)


def list_available_apps(config_path: str = "config/apps.json") -> Dict[str, str]:
    """List all available apps with their names"""
    app_config = load_app_config(config_path)
    apps = app_config.get("apps", {})
    return {key: app["name"] for key, app in apps.items()}


def create_output_directory(output_dir: str) -> None:
    """Create output directory if it doesn't exist"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def ensure_directories() -> None:
    """Ensure all required directories exist"""
    directories = ["output", "logs", "config"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True) 