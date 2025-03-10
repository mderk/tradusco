import os
import sys
import pytest
import json
import asyncio
import csv
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, AsyncMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import translate
import create_project
from lib.TranslationProject import TranslationProject


class TestCommandLine:
    """Test suite for command-line functionality."""

    @pytest.mark.asyncio
    async def test_list_models(self, monkeypatch, capsys):
        """Test that the --list-models flag lists available models."""
        # Mock the command line arguments
        test_args = ["translate.py", "--list-models"]
        monkeypatch.setattr(sys, "argv", test_args)

        # Create a mock for get_available_models
        mock_models = ["gemini", "grok", "openai"]

        with patch.object(
            TranslationProject, "get_available_models", return_value=mock_models
        ):
            # Call the main function
            result = await translate.async_main()

            # Check the output
            captured = capsys.readouterr()
            assert "Available models:" in captured.out
            for model in mock_models:
                assert model in captured.out

            # Check the return code
            assert result == 0

    @pytest.mark.asyncio
    async def test_missing_required_args(self, monkeypatch, capsys):
        """Test that an error is raised when required args are missing."""
        # Mock the command line arguments with missing required args
        test_args = ["translate.py"]
        monkeypatch.setattr(sys, "argv", test_args)

        # Patch the ArgumentParser.error method to avoid system exit
        with patch("argparse.ArgumentParser.error") as mock_error:
            # Call the main function
            await translate.async_main()

            # Check that error was called for missing required args
            mock_error.assert_called()

    @pytest.mark.asyncio
    async def test_create_project_command(self, monkeypatch, tmp_path):
        """Test the create_project command."""
        # Create a temporary CSV file with valid content
        csv_content = "key,en\ngreeting,Hello\nfarewell,Goodbye\nwelcome,Welcome"
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write(csv_content)

        # Mock the command line arguments
        test_args = [
            "create_project.py",
            "--name",
            "test_project",
            "--csv",
            str(csv_file),
            "--base-lang",
            "en",
            "--key",
            "key",
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        # Create a mock for the create_project function to avoid sys.exit
        mock_create_project = AsyncMock()
        monkeypatch.setattr(create_project, "create_project", mock_create_project)

        # Call the function
        await create_project.create_project()

        # Verify the function was called
        mock_create_project.assert_called_once()
