import os
import sys
import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import patch
import csv

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

        # Patch the create method to use our test directory
        with patch("lib.TranslationProject.Path") as mock_path:
            # Make Path return our test directory when called with the project path
            mock_path.return_value = project_dir
            # But for all other calls, use the real Path
            mock_path.side_effect = lambda p: (
                project_dir if p == f"projects/{mock_config.name}" else Path(p)
            )

            # Create a translation project
            project = await TranslationProject.create(
                project_name=mock_config.name, dst_language="es"
            )

            # Verify project attributes
            assert project.project_name == mock_config.name
            assert project.dst_language == "es"
            assert project.base_language == mock_config.baseLanguage

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
            project_dir=project_dir,
            config=mock_config,
            dst_language="es",
            context_file=str(context_file),
        )

        # Load context
        context = await project._load_context()
        assert context_content in context

    @pytest.mark.asyncio
    async def test_translate(self, setup_test_project, mock_llm_driver):
        """Test the translate method with mocked LLM responses."""
        project_dir, mock_config, source_data = setup_test_project

        # Create source file with translation data in CSV format
        source_file_path = project_dir / mock_config.sourceFile

        # First create a list of dictionaries with the correct format
        translations = []
        for key, phrase in source_data.items():
            # Include the key column and use the base language as the column name
            translation_row = {
                mock_config.keyColumn: key,
                mock_config.baseLanguage: phrase,
                "es": "",
            }
            translations.append(translation_row)

        with open(source_file_path, "w", encoding="utf-8") as f:
            # Create a CSV writer for dictionaries
            fieldnames = [mock_config.keyColumn, mock_config.baseLanguage, "es"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(translations)

        # Create config file
        config_path = project_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(mock_config.__dict__, f, ensure_ascii=False, indent=2)

        # Create a prompt file
        prompt_dir = project_dir / "prompts"
        os.makedirs(prompt_dir, exist_ok=True)
        prompt_file = prompt_dir / "translation.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(
                """Translate the following phrases from {base_language} to {dst_language}.

Phrases to translate:
{phrases_json}

{context}
{phrase_contexts}

Return the translations in JSON format with the original phrase as the key and the translation as the value.
Example:
```json
{
  "Hello": "Hola",
  "Goodbye": "Adiós"
}
```"""
            )

        # Directly create a TranslationProject instance without using create
        project = TranslationProject(
            project_name=mock_config.name,
            project_dir=project_dir,
            config=mock_config,
            dst_language="es",
            prompt_file=str(prompt_file),
        )

        # Setup our mocks more directly
        async def mock_process_batch(
            phrases, indices, translations, progress, *args, **kwargs
        ):
            """Mock the process_batch method to directly update translations"""
            # Update the translations and progress directly
            for i, phrase in enumerate(phrases):
                # Get the actual phrase from the translations data
                source_phrase = translations[indices[i]][mock_config.baseLanguage]

                # Set the expected translations based on the source phrases
                if source_phrase == "Hello":
                    translations[indices[i]]["es"] = "Hola"
                    progress[source_phrase] = "Hola"
                elif source_phrase == "Goodbye":
                    translations[indices[i]]["es"] = "Adiós"
                    progress[source_phrase] = "Adiós"
                elif source_phrase == "Welcome to our application":
                    translations[indices[i]]["es"] = "Bienvenido a nuestra aplicación"
                    progress[source_phrase] = "Bienvenido a nuestra aplicación"

            # Return the number of phrases translated
            return len(phrases)

        # Patch the translation tool's process_batch method
        with patch.object(
            project.translation_tool, "process_batch", side_effect=mock_process_batch
        ):
            # Run the translate method
            await project.translate(
                model="mock-model", delay_seconds=0, max_retries=1, batch_size=10
            )

            # Load the results to verify - now load as CSV
            translations = []
            with open(source_file_path, "r", encoding="utf-8") as f:
                csv_reader = csv.DictReader(f)
                translations = list(csv_reader)

            # Check that translations were applied
            # Find the row for each key and check its translation
            greeting_row = next(
                (t for t in translations if t[mock_config.keyColumn] == "greeting"),
                None,
            )
            farewell_row = next(
                (t for t in translations if t[mock_config.keyColumn] == "farewell"),
                None,
            )
            welcome_row = next(
                (t for t in translations if t[mock_config.keyColumn] == "welcome"), None
            )

            # Add null checks before accessing dictionary items
            assert greeting_row is not None, "Greeting row not found in translations"
            assert farewell_row is not None, "Farewell row not found in translations"
            assert welcome_row is not None, "Welcome row not found in translations"

            assert greeting_row["es"] == "Hola"
            assert farewell_row["es"] == "Adiós"
            assert welcome_row["es"] == "Bienvenido a nuestra aplicación"

            # Check progress file
            progress_file = project.progress_file
            with open(progress_file, "r", encoding="utf-8") as f:
                progress = json.load(f)

            assert progress.get("Hello") == "Hola"
            assert progress.get("Goodbye") == "Adiós"
            assert (
                progress.get("Welcome to our application")
                == "Bienvenido a nuestra aplicación"
            )
