import pytest
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import and expose fixtures from utils.py
from tests.utils import *


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path"""
    return project_root


@pytest.fixture(scope="session")
def tests_path():
    """Return the tests directory path"""
    return project_root / "tests"
