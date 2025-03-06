import json
import os
import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from lib.TranslationProject import TranslationProject


class TestTranslationProject(unittest.TestCase):
    """Tests for the TranslationProject class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a mock config
        self.mock_config = {
            "name": "test_project",
            "sourceFile": "translations.csv",
            "languages": ["en", "fr", "es"],
            "baseLanguage": "en",
            "keyColumn": "en",
        }

        # Create a mock prompt
        self.mock_prompt = "You are a translator. Translate from {base_language} to {dst_language}.\n{phrases_json}"

        # Create a patch for _load_config
        self.config_patcher = patch.object(TranslationProject, "_load_config")
        self.mock_load_config = self.config_patcher.start()
        self.mock_load_config.return_value = self.mock_config

        # Create a patch for _load_default_prompt
        self.prompt_patcher = patch.object(TranslationProject, "_load_default_prompt")
        self.mock_load_default_prompt = self.prompt_patcher.start()
        self.mock_load_default_prompt.return_value = self.mock_prompt

        # Create patches for file operations
        self.path_exists_patcher = patch("pathlib.Path.exists")
        self.mock_path_exists = self.path_exists_patcher.start()
        self.mock_path_exists.return_value = True

        self.makedirs_patcher = patch("os.makedirs")
        self.mock_makedirs = self.makedirs_patcher.start()

        # Initialize the project
        self.project = TranslationProject("test_project", "fr")

    def tearDown(self):
        """Tear down test fixtures"""
        self.config_patcher.stop()
        self.prompt_patcher.stop()
        self.path_exists_patcher.stop()
        self.makedirs_patcher.stop()

    def test_init(self):
        """Test initialization of TranslationProject"""
        self.assertEqual(self.project.project_name, "test_project")
        self.assertEqual(self.project.dst_language, "fr")
        self.assertEqual(self.project.base_language, "en")
        self.assertEqual(self.project.default_prompt, self.mock_prompt)

    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    def test_load_progress(self, mock_file):
        """Test loading progress from file"""
        result = self.project._load_progress()
        self.assertEqual(result, {"key": "value"})
        mock_file.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_progress(self, mock_json_dump, mock_file):
        """Test saving progress to file"""
        progress = {"phrase1": "translation1"}
        self.project._save_progress(progress)
        mock_file.assert_called_once()
        mock_json_dump.assert_called_once()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="en,fr\nHello,Bonjour\nWorld,Monde",
    )
    @patch("csv.DictReader")
    def test_load_translations(self, mock_dict_reader, mock_file):
        """Test loading translations from CSV"""
        mock_dict_reader.return_value = [
            {"en": "Hello", "fr": "Bonjour"},
            {"en": "World", "fr": "Monde"},
        ]
        result = self.project._load_translations()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["en"], "Hello")
        self.assertEqual(result[0]["fr"], "Bonjour")
        mock_file.assert_called_once()

    def test_create_batch_prompt(self):
        """Test creating a batch prompt"""
        phrases = ["Hello", "World"]
        result = self.project._create_batch_prompt(phrases)
        self.assertIn("en", result)
        self.assertIn("fr", result)
        self.assertIn(json.dumps(phrases, ensure_ascii=False, indent=2), result)

    def test_parse_batch_response_json_array(self):
        """Test parsing a batch response with JSON array"""
        response = '```json\n["Bonjour", "Monde"]\n```'
        original_phrases = ["Hello", "World"]
        result = self.project._parse_batch_response(response, original_phrases)
        self.assertEqual(result, {"Hello": "Bonjour", "World": "Monde"})

    def test_parse_batch_response_json_object(self):
        """Test parsing a batch response with JSON object"""
        response = '```json\n{"Hello": "Bonjour", "World": "Monde"}\n```'
        original_phrases = ["Hello", "World"]
        result = self.project._parse_batch_response(response, original_phrases)
        self.assertEqual(result, {"Hello": "Bonjour", "World": "Monde"})

    def test_parse_batch_response_fallback(self):
        """Test parsing a batch response with fallback to line-by-line"""
        response = "1. Bonjour\n2. Monde"
        original_phrases = ["Hello", "World"]
        result = self.project._parse_batch_response(response, original_phrases)
        self.assertEqual(result, {"Hello": "Bonjour", "World": "Monde"})

    @patch.object(TranslationProject, "_load_translations")
    @patch.object(TranslationProject, "_load_progress")
    @patch.object(TranslationProject, "_process_batch")
    @patch.object(TranslationProject, "_save_progress")
    @patch.object(TranslationProject, "_save_translations")
    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake_api_key"})
    @patch.object(Path, "exists", return_value=True)
    @patch.object(
        TranslationProject, "_load_default_prompt", return_value="Default prompt"
    )
    def test_translate(
        self,
        mock_load_default_prompt,
        mock_path_exists,
        mock_save_translations,
        mock_save_progress,
        mock_process_batch,
        mock_load_progress,
        mock_load_translations,
    ):
        """Test the translate method"""
        # Setup mocks
        mock_load_translations.return_value = [
            {"en": "Hello", "fr": ""},
            {"en": "World", "fr": "Monde"},
        ]
        mock_load_progress.return_value = {"World": "Monde"}

        # Mock get_llm directly in the test
        mock_driver = MagicMock()

        with patch(
            "lib.TranslationProject.get_llm", return_value=mock_driver
        ) as mock_get_llm:
            # Call the method
            self.project.translate(delay_seconds=0.1, max_retries=1, batch_size=10)

            # Verify the calls
            mock_load_translations.assert_called_once()
            mock_load_progress.assert_called_once()
            mock_get_llm.assert_called_once_with("gemini")
            mock_process_batch.assert_called_once()
            mock_save_progress.assert_called()
            mock_save_translations.assert_called()


if __name__ == "__main__":
    unittest.main()
