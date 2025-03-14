import os
import sys
import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import patch
import csv
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.TranslationProject import TranslationProject
from lib.utils import Config, load_config
from tests.mock_llm_driver import MockLLMDriver


class TestTranslationProject:
    """Test suite for TranslationProject class."""

    @pytest.fixture
    def mock_llm_driver(self):
        """Create a mock LLM driver for testing."""
        return MockLLMDriver()

    @pytest.mark.asyncio
    async def test_create_project(self, setup_test_project):
        """Test TranslationProject.create method."""
        project_dir, mock_config, _ = setup_test_project

        # Create a real config file for the test
        config_path = project_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(mock_config.__dict__, f, ensure_ascii=False, indent=2)

        # Create a translation project using the project_path parameter
        project = await TranslationProject.create(
            project_name=mock_config.name, dst_language="es", project_path=project_dir
        )

        # Verify project attributes
        assert project.project_name == mock_config.name
        assert project.dst_language == "es"
        assert project.base_language == mock_config.baseLanguage
        assert project.project_path == project_dir

        # Verify project directories
        assert os.path.exists(project.progress_dir)

    @pytest.mark.asyncio
    async def test_get_available_models(self):
        """Test that get_available_models returns a list of models."""
        models = TranslationProject.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0

    @pytest.mark.asyncio
    async def test_count_tokens(self, mock_llm_driver):
        """Test token counting functionality with mock LLM driver."""
        text = "This is a test sentence for token counting."

        # Patch the get_driver function to return our mock
        with patch("lib.llm.get_driver", return_value=mock_llm_driver):
            token_count = TranslationProject.count_tokens(text)
            assert isinstance(token_count, int)
            assert token_count > 0

    @pytest.mark.asyncio
    async def test_load_context(self, setup_test_project):
        """Test loading context from a file."""
        project_dir, mock_config, _ = setup_test_project

        # Create a context file
        context_content = "This is test context for translation."
        context_file = project_dir / "context.md"
        with open(context_file, "w", encoding="utf-8") as f:
            f.write(context_content)

        # Create a config file
        config_path = project_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(mock_config.__dict__, f, ensure_ascii=False, indent=2)

        # Directly create a TranslationProject instance without using create
        project = TranslationProject(
            project_name=mock_config.name,
            project_path=project_dir,
            config=mock_config,
            dst_language="es",
            context_file=str(context_file),
        )

        # Load context
        context = await project._load_context()
        assert context_content in context

    @patch("lib.TranslationTool.TranslationTool.translate_standard")
    async def test_translate(self, mock_translate_standard_patch, mock_llm_driver):
        """Test translation process with mock driver"""
        # Create test directories
        project_dir = Path("test_project")
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(project_dir / "es", exist_ok=True)

        # Create test config
        config = Config(
            name="test_project",
            sourceFile="source.csv",
            baseLanguage="en",
            languages=["en", "es"],
            keyColumn="key",
        )

        # Create source file
        source_file = project_dir / "source.csv"
        with open(source_file, "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "en", "es"])
            writer.writerow(["greeting", "Hello", ""])
            writer.writerow(["farewell", "Goodbye", ""])
            writer.writerow(["welcome", "Welcome", ""])
            writer.writerow(["thanks", "Thank you", ""])

        # Create a test project
        project = TranslationProject(
            project_name="test_project",
            project_path=project_dir,
            config=config,
            dst_language="es",
            prompt="Translate from {base_language} to {dst_language}",
        )

        # Create mock translate_standard function
        async def mock_translate_standard(
            phrases: list[str],
            indices: list[int],
            translations: list[dict[str, str]],
            progress: dict[str, str],
            model: str,
            base_language: str,
            dst_language: str,
            prompt: str,
            context: Optional[str] = None,
            delay_seconds: float = 1.0,
            max_retries: int = 3,
        ) -> int:
            # Simulate translation
            for i, phrase in enumerate(phrases):
                translations[indices[i]][dst_language] = f"{phrase} (translated)"
                progress[phrase] = f"{phrase} (translated)"
            return len(phrases)

        # Patch the translation tool's translate_standard method and get_driver function
        with patch.object(
            project.translation_tool,
            "translate_standard",
            side_effect=mock_translate_standard,
        ), patch("lib.llm.get_driver", return_value=mock_llm_driver), patch(
            "lib.TranslationProject.get_driver", return_value=mock_llm_driver
        ):
            # Run translation
            await project.translate()

            # Verify translations were created
            with open(source_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                translations = list(reader)
                for row in translations:
                    if row["en"]:
                        assert (
                            "(translated)" in row["es"]
                        ), f"Translation missing for {row['key']}"

            # Verify progress was updated
            with open(project.progress_file, "r", encoding="utf-8") as f:
                progress = json.load(f)
                for key in ["Hello", "Goodbye", "Welcome", "Thank you"]:
                    assert key in progress, f"Progress missing for {key}"
                    assert (
                        "(translated)" in progress[key]
                    ), f"Progress translation missing for {key}"
