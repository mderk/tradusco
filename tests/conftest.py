import os
import sys
import tempfile
from typing import Any, Generator
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import pytest
import shutil

from lib.TranslationProject import TranslationProject
from lib.TranslationTool import TranslationTool
from lib.utils import Config


# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for tests that will be automatically cleaned up."""
    temp_path = tempfile.mkdtemp(prefix="ai_translator_test_")
    yield Path(temp_path)
    # Clean up the temporary directory after tests
    shutil.rmtree(temp_path, ignore_errors=True)


# Import and expose fixtures from utils.py
@pytest.fixture
def mock_config() -> Config:
    """Create a mock configuration for testing."""
    return Config(
        name="test_project",
        sourceFile="translations.csv",
        languages=["en", "fr", "es"],
        baseLanguage="en",
        keyColumn="en",
    )


@pytest.fixture
def translation_project(mock_config: Config, temp_dir: Path) -> TranslationProject:
    """Create a TranslationProject instance for testing."""
    # We need to patch all file operations to prevent actual file system access
    with patch("pathlib.Path.exists", return_value=True), patch(
        "builtins.open", MagicMock()
    ):
        project_dir = temp_dir / "test_project_dir"
        project = TranslationProject(
            project_name="test_project",
            project_dir=project_dir,
            config=mock_config,
            dst_language="fr",
            prompt="Test prompt with {base_language}, {dst_language}, and {phrases_json}",
        )
        return project


def create_gemini_response_mock(content: str) -> MagicMock:
    """Create a mock for a Gemini AI response.

    Args:
        content: The content to include in the mock response

    Returns:
        A mock response object with the appropriate structure
    """
    mock_response = MagicMock()
    mock_response.text = content

    # Create a mock for the response.parts[0]
    mock_part = MagicMock()
    mock_part.text = content
    mock_response.parts = [mock_part]

    return mock_response


@pytest.fixture
def common_translation_project_patches() -> Generator[dict[str, MagicMock], None, None]:
    """Return a dictionary of common patches for TranslationProject."""
    with patch(
        "lib.TranslationProject.load_translations"
    ) as mock_load_translations, patch(
        "lib.TranslationProject.load_progress"
    ) as mock_load_progress, patch.object(
        TranslationTool, "process_batch", new_callable=AsyncMock
    ) as mock_process_batch, patch(
        "lib.TranslationProject.save_progress"
    ) as mock_save_progress, patch(
        "lib.TranslationProject.save_translations"
    ) as mock_save_translations, patch(
        "lib.TranslationProject.get_driver"
    ) as mock_get_driver, patch.dict(
        os.environ, {"GEMINI_API_KEY": "fake_api_key"}
    ), patch.object(
        Path, "exists", return_value=True
    ) as mock_path_exists:

        yield {
            "mock_load_translations": mock_load_translations,
            "mock_load_progress": mock_load_progress,
            "mock_process_batch": mock_process_batch,
            "mock_save_progress": mock_save_progress,
            "mock_save_translations": mock_save_translations,
            "mock_get_driver": mock_get_driver,
            "mock_path_exists": mock_path_exists,
        }


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path"""
    return project_root


@pytest.fixture(scope="session")
def tests_path():
    """Return the tests directory path"""
    return project_root / "tests"
