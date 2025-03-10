import os
import sys
import pytest
import json
import csv
import asyncio
from pathlib import Path
from io import StringIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.utils import (
    load_translations,
    save_translations,
    load_config,
    load_progress,
    save_progress,
    Config,
)


class TestFileOperations:
    """
    Test suite for file operations using real files in temporary directories.
    This avoids mocking file operations to test actual functionality.
    """

    @pytest.mark.asyncio
    async def test_load_save_translations_csv(self, tmp_path):
        """Test loading and saving translations from/to a CSV file."""
        # Create a test CSV file with translations
        csv_file = tmp_path / "translations.csv"

        # Create test data
        fieldnames = ["key", "en", "es", "fr"]
        test_data = [
            {"key": "greeting", "en": "Hello", "es": "Hola", "fr": "Bonjour"},
            {"key": "farewell", "en": "Goodbye", "es": "Adiós", "fr": "Au revoir"},
            {"key": "welcome", "en": "Welcome", "es": "Bienvenido", "fr": "Bienvenue"},
        ]

        # Manually create the CSV file
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(test_data)

        # Test loading translations
        loaded_translations = await load_translations(csv_file)

        # Verify loaded data
        assert len(loaded_translations) == 3
        assert loaded_translations[0]["key"] == "greeting"
        assert loaded_translations[0]["en"] == "Hello"
        assert loaded_translations[0]["es"] == "Hola"

        # Modify the translations
        loaded_translations[1]["es"] = "Hasta luego"

        # Save the modified translations to a new file
        output_file = tmp_path / "modified_translations.csv"
        await save_translations(output_file, loaded_translations)

        # Verify the file exists
        assert output_file.exists()

        # Load the saved translations and verify changes
        saved_translations = await load_translations(output_file)
        assert len(saved_translations) == 3
        assert saved_translations[1]["key"] == "farewell"
        assert saved_translations[1]["en"] == "Goodbye"
        assert saved_translations[1]["es"] == "Hasta luego"

    @pytest.mark.asyncio
    async def test_load_translations_with_missing_columns(self, tmp_path):
        """Test loading translations with missing columns."""
        # Create a test CSV file with incomplete data
        csv_file = tmp_path / "incomplete.csv"

        # Create test data with missing columns in some rows
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            f.write("key,en,es,fr\n")
            f.write("greeting,Hello,Hola,Bonjour\n")
            f.write("farewell,Goodbye,,\n")  # Missing es and fr
            f.write("welcome,Welcome,Bienvenido,\n")  # Missing fr

        # Test loading translations
        loaded_translations = await load_translations(csv_file)

        # Verify loaded data handles missing values
        assert len(loaded_translations) == 3
        assert loaded_translations[1]["key"] == "farewell"
        assert loaded_translations[1]["en"] == "Goodbye"
        assert loaded_translations[1]["es"] == ""  # Empty string for missing value
        assert loaded_translations[1]["fr"] == ""  # Empty string for missing value

    @pytest.mark.asyncio
    async def test_load_translations_file_not_found(self, tmp_path):
        """Test loading translations from a nonexistent file."""
        # Define a path that doesn't exist
        nonexistent_file = tmp_path / "nonexistent.csv"

        # Verify that FileNotFoundError is raised
        with pytest.raises(FileNotFoundError):
            await load_translations(nonexistent_file)

    @pytest.mark.asyncio
    async def test_translations_with_special_characters(self, tmp_path):
        """Test handling special characters in translations."""
        # Create a test CSV file with special characters
        csv_file = tmp_path / "special_chars.csv"

        # Create test data with special characters
        test_data = [
            {
                "key": "special",
                "en": "Special characters: áéíóú",
                "es": "Caracteres especiales: ñçüß",
                "fr": "Caractères spéciaux: àâêîôû",
            },
            {
                "key": "quotes",
                "en": 'Text with "quotes"',
                "es": 'Texto con "comillas"',
                "fr": 'Texte avec "guillemets"',
            },
            {
                "key": "commas",
                "en": "Text with, commas",
                "es": "Texto con, comas",
                "fr": "Texte avec, virgules",
            },
        ]

        # Save the translations
        await save_translations(csv_file, test_data)

        # Load the translations
        loaded_translations = await load_translations(csv_file)

        # Verify special characters are preserved
        assert loaded_translations[0]["en"] == "Special characters: áéíóú"
        assert loaded_translations[0]["es"] == "Caracteres especiales: ñçüß"
        assert loaded_translations[1]["en"] == 'Text with "quotes"'
        assert loaded_translations[2]["en"] == "Text with, commas"

    @pytest.mark.asyncio
    async def test_integrated_project_setup(self, tmp_path):
        """
        Integration test for project setup with real files.
        This test simulates creating and configuring a project with real file operations.
        """
        # Create project structure
        project_dir = tmp_path / "test_project"
        os.makedirs(project_dir, exist_ok=True)

        # Create language directories
        for lang in ["es", "fr"]:
            os.makedirs(project_dir / lang, exist_ok=True)

        # Create config file
        config_data = {
            "name": "test_project",
            "sourceFile": "translations.csv",
            "baseLanguage": "en",
            "languages": ["en", "es", "fr"],
            "keyColumn": "key",
        }

        config_file = project_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        # Create initial translations file
        translations_file = project_dir / "translations.csv"
        test_data = [
            {"key": "greeting", "en": "Hello", "es": "", "fr": ""},
            {"key": "farewell", "en": "Goodbye", "es": "", "fr": ""},
        ]

        # Create the CSV file manually
        with open(translations_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["key", "en", "es", "fr"])
            writer.writeheader()
            writer.writerows(test_data)

        # Create progress file for Spanish
        es_progress_dir = project_dir / "es"
        es_progress_file = es_progress_dir / "progress.json"
        progress_data = {"Hello": "Hola"}

        with open(es_progress_file, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)

        # Load the config
        config = await load_config(config_file)
        assert isinstance(config, Config)
        assert config.name == "test_project"

        # Load translations
        translations = await load_translations(translations_file)
        assert len(translations) == 2
        assert translations[0]["en"] == "Hello"
        assert translations[0]["es"] == ""

        # Load progress
        progress = await load_progress(es_progress_file)
        assert progress["Hello"] == "Hola"

        # Update translations from progress
        translations[0]["es"] = progress["Hello"]

        # Save updated translations
        await save_translations(translations_file, translations)

        # Reload translations and verify updates
        updated_translations = await load_translations(translations_file)
        assert updated_translations[0]["es"] == "Hola"
