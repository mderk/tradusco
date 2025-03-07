import json
import os
import unittest
import asyncio
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from lib.TranslationProject import TranslationProject


class AsyncMock(MagicMock):
    """Helper class for mocking async methods"""

    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


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

        # Create patches for file operations
        self.path_exists_patcher = patch("pathlib.Path.exists")
        self.mock_path_exists = self.path_exists_patcher.start()
        self.mock_path_exists.return_value = True

        self.makedirs_patcher = patch("os.makedirs")
        self.mock_makedirs = self.makedirs_patcher.start()

        # Initialize the project directly with the mock config and prompt
        # This avoids the need for async initialization in tests
        self.project = TranslationProject(
            "test_project",
            "fr",
            config=self.mock_config,
            default_prompt=self.mock_prompt,
        )

        # Create an event loop for each test
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Tear down test fixtures"""
        self.path_exists_patcher.stop()
        self.makedirs_patcher.stop()

        # Close the event loop
        self.loop.close()
        asyncio.set_event_loop(None)

    def test_init(self):
        """Test initialization of TranslationProject"""
        self.assertEqual(self.project.project_name, "test_project")
        self.assertEqual(self.project.dst_language, "fr")
        self.assertEqual(self.project.base_language, "en")
        self.assertEqual(self.project.default_prompt, self.mock_prompt)

    @patch("aiofiles.open")
    def test_load_progress(self, mock_aiofiles_open):
        """Test loading progress from file"""
        # Setup the async mock
        mock_file = AsyncMock()
        mock_file.read.return_value = '{"key": "value"}'

        # Setup the context manager
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

        # Run the test
        result = self.loop.run_until_complete(self.project._load_progress())
        self.assertEqual(result, {"key": "value"})
        mock_aiofiles_open.assert_called_once()

    @patch("aiofiles.open")
    def test_save_progress(self, mock_aiofiles_open):
        """Test saving progress to file"""
        # Setup the async mock
        mock_file = AsyncMock()

        # Setup the context manager
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

        # Run the test
        progress = {"phrase1": "translation1"}
        self.loop.run_until_complete(self.project._save_progress(progress))
        mock_aiofiles_open.assert_called_once()
        mock_file.write.assert_called_once()

    @patch("aiofiles.open")
    def test_load_translations(self, mock_aiofiles_open):
        """Test loading translations from CSV"""
        # Setup the async mock
        mock_file = AsyncMock()
        mock_file.read.return_value = "en,fr\nHello,Bonjour\nWorld,Monde"

        # Setup the context manager
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

        # Run the test
        result = self.loop.run_until_complete(self.project._load_translations())
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["en"], "Hello")
        self.assertEqual(result[0]["fr"], "Bonjour")
        mock_aiofiles_open.assert_called_once()

    def test_create_batch_prompt(self):
        """Test creating a batch prompt"""
        # Mock _load_custom_prompt to return empty string
        self.project._load_custom_prompt = AsyncMock(return_value="")

        phrases = ["Hello", "World"]
        result = self.loop.run_until_complete(
            self.project._create_batch_prompt(phrases)
        )
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
    def test_translate(
        self,
        mock_path_exists,
        mock_save_translations,
        mock_save_progress,
        mock_process_batch,
        mock_load_progress,
        mock_load_translations,
    ):
        """Test the translate method"""
        # Setup data to be returned by mocks
        translations_data = [
            {"en": "Hello", "fr": ""},
            {"en": "World", "fr": "Monde"},
        ]

        progress_data = {"World": "Monde"}

        # Create async mocks
        async def mock_load_translations_impl():
            return translations_data

        async def mock_load_progress_impl():
            return progress_data

        async def mock_process_batch_impl(*args, **kwargs):
            return None

        async def mock_save_progress_impl(*args, **kwargs):
            return None

        async def mock_save_translations_impl(*args, **kwargs):
            return None

        # Assign the async implementations to the mocks
        mock_load_translations.side_effect = mock_load_translations_impl
        mock_load_progress.side_effect = mock_load_progress_impl
        mock_process_batch.side_effect = mock_process_batch_impl
        mock_save_progress.side_effect = mock_save_progress_impl
        mock_save_translations.side_effect = mock_save_translations_impl

        # Mock get_driver
        mock_driver = MagicMock()

        with patch("lib.TranslationProject.get_driver", return_value=mock_driver):
            # Call the method
            self.loop.run_until_complete(
                self.project.translate(delay_seconds=0.1, max_retries=1, batch_size=10)
            )

            # Verify the calls
            mock_load_translations.assert_called_once()
            mock_load_progress.assert_called_once()
            mock_process_batch.assert_called_once()
            mock_save_progress.assert_called()
            mock_save_translations.assert_called()


if __name__ == "__main__":
    unittest.main()
