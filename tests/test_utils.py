import os
import sys
import pytest
import json
import asyncio
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.utils import (
    Config,
    load_config,
    load_progress,
    save_progress,
    load_translations,
    save_translations,
    load_context,
)


class TestUtilsFunctions:
    """Test suite for utility functions."""

    @pytest.mark.asyncio
    async def test_load_config(self, tmp_path):
        """Test loading configuration from a file."""
        # Create a test config file
        config_data = {
            "name": "test_project",
            "sourceFile": "source.json",
            "baseLanguage": "en",
            "languages": ["en", "es", "fr"],
            "keyColumn": "key",
        }

        config_file = tmp_path / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        # Load the config
        config = await load_config(config_file)

        # Verify the config
        assert isinstance(config, Config)
        assert config.name == "test_project"
        assert config.sourceFile == "source.json"
        assert config.baseLanguage == "en"
        assert config.languages == ["en", "es", "fr"]
        assert config.keyColumn == "key"

    @pytest.mark.asyncio
    async def test_load_and_save_progress(self, tmp_path):
        """Test loading and saving progress."""
        # Create test progress data
        progress_data = {"phrase1": "Hola", "phrase2": "Adiós", "phrase3": "Bienvenido"}

        progress_file = tmp_path / "progress.json"

        # Save the progress
        await save_progress(progress_file, progress_data)

        # Verify the file exists
        assert progress_file.exists()

        # Load the progress
        loaded_progress = await load_progress(progress_file)

        # Verify the loaded progress
        assert loaded_progress == progress_data

    @pytest.mark.asyncio
    async def test_load_progress_nonexistent_file(self, tmp_path):
        """Test loading progress from a nonexistent file."""
        # Define a path that doesn't exist
        progress_file = tmp_path / "nonexistent.json"

        # Load the progress (should return empty dict)
        loaded_progress = await load_progress(progress_file)

        # Verify the result
        assert loaded_progress == {}

    @pytest.mark.asyncio
    async def test_load_context(self, tmp_path):
        """Test loading context from a file."""
        # Create a test context file
        context_content = "This is test context for translation."
        context_file = tmp_path / "context.md"
        with open(context_file, "w", encoding="utf-8") as f:
            f.write(context_content)

        # Load the context from the project directory
        context_parts = await load_context(tmp_path)

        # Verify the loaded context
        assert len(context_parts) == 1
        assert context_parts[0] == context_content

        # Load the context from a specific file
        context_parts = await load_context(tmp_path, str(context_file))

        # Verify the loaded context
        assert len(context_parts) >= 1  # May have both project dir and specific file
        assert context_content in context_parts
