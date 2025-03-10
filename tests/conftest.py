import os
import sys
import pytest
import json
import tempfile
import shutil
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.utils import Config
from lib.TranslationProject import TranslationProject


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for a test project."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return Config(
        name="test_project",
        sourceFile="source.json",
        baseLanguage="en",
        languages=["en", "es", "fr"],
        keyColumn="key",
    )


@pytest.fixture
def mock_source_data():
    """Create mock source data for testing."""
    return {
        "greeting": "Hello",
        "farewell": "Goodbye",
        "welcome": "Welcome to our application",
    }


@pytest.fixture
def setup_test_project(temp_project_dir, mock_config, mock_source_data):
    """Set up a test project directory with source file and config."""
    # Create source file
    source_file = temp_project_dir / mock_config.sourceFile
    os.makedirs(temp_project_dir, exist_ok=True)

    with open(source_file, "w", encoding="utf-8") as f:
        json.dump(mock_source_data, f, ensure_ascii=False, indent=2)

    # Create config file
    config_file = temp_project_dir / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(mock_config.__dict__, f, ensure_ascii=False, indent=2)

    # Create language directories
    for lang in mock_config.languages:
        if lang != mock_config.baseLanguage:
            os.makedirs(temp_project_dir / lang, exist_ok=True)

    return temp_project_dir, mock_config, mock_source_data
