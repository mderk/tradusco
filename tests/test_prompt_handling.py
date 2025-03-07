import unittest
import asyncio
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

from lib.TranslationProject import TranslationProject


class AsyncMock(MagicMock):
    """Helper class for mocking async methods"""

    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


class TestPromptHandling(unittest.TestCase):
    """Tests for prompt loading and handling functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a mock config
        self.mock_config = {
            "name": "test_project",
            "sourceFile": "translations.csv",
            "languages": ["en", "fr"],
            "baseLanguage": "en",
            "keyColumn": "en",
        }

        # Create patches to avoid actual file operations
        self.path_exists_patcher = patch("pathlib.Path.exists")
        self.mock_path_exists = self.path_exists_patcher.start()
        self.mock_path_exists.return_value = True

        self.makedirs_patcher = patch("os.makedirs")
        self.mock_makedirs = self.makedirs_patcher.start()

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

    @patch("aiofiles.open")
    @patch.object(Path, "exists", return_value=True)
    def test_load_default_prompt(self, mock_exists, mock_aiofiles_open):
        """Test loading the default prompt"""
        # Setup the async mock
        mock_file = AsyncMock()
        mock_file.read.return_value = "Default prompt content"

        # Setup the context manager
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

        with patch.object(Path, "parent", return_value=Path("/fake/path")):
            # Create project with mock config
            project = TranslationProject("test_project", "fr", config=self.mock_config)
            result = self.loop.run_until_complete(
                project._load_default_prompt("test_prompt.txt")
            )
            self.assertEqual(result, "Default prompt content")

    @patch("aiofiles.open")
    @patch("os.path.exists", return_value=True)
    def test_load_custom_prompt_with_valid_file(self, mock_exists, mock_aiofiles_open):
        """Test loading a custom prompt with a valid file path"""
        # Setup the async mock
        mock_file = AsyncMock()
        mock_file.read.return_value = "Custom prompt content"

        # Setup the context manager
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

        # Create project with mock config
        project = TranslationProject("test_project", "fr", config=self.mock_config)
        result = self.loop.run_until_complete(
            project._load_custom_prompt("custom_prompt.txt")
        )
        self.assertEqual(result, "Custom prompt content")

    @patch.object(Path, "exists", return_value=True)
    def test_load_custom_prompt_with_none(self, mock_path_exists):
        """Test loading a custom prompt with None as the file path"""
        # Create project with mock config
        project = TranslationProject("test_project", "fr", config=self.mock_config)
        result = self.loop.run_until_complete(project._load_custom_prompt(None))
        self.assertEqual(result, "")

    @patch.object(Path, "exists", return_value=True)
    @patch("os.path.exists", return_value=False)
    def test_load_custom_prompt_with_nonexistent_file(
        self, mock_os_exists, mock_path_exists
    ):
        """Test loading a custom prompt with a nonexistent file path"""
        # Create project with mock config
        project = TranslationProject("test_project", "fr", config=self.mock_config)
        result = self.loop.run_until_complete(
            project._load_custom_prompt("nonexistent.txt")
        )
        self.assertEqual(result, "")
        mock_os_exists.assert_called_once_with("nonexistent.txt")

    def test_custom_prompt_precedence(self):
        """Test that custom prompts take precedence over default prompts"""
        # Create project with mock config and default prompt
        project = TranslationProject(
            "test_project",
            "fr",
            config=self.mock_config,
            default_prompt="Default prompt",
        )

        # Mock the _load_custom_prompt method
        project._load_custom_prompt = AsyncMock(return_value="Custom prompt")

        # Create a batch prompt
        phrases = ["Hello", "World"]
        translations = [
            {"en": "Hello", "fr": "", "context": "greeting"},
            {"en": "World", "fr": "", "context": "place"},
        ]
        indices = [0, 1]
        result = self.loop.run_until_complete(
            project._create_batch_prompt(phrases, translations, indices)
        )

        # Verify custom prompt was used
        project._load_custom_prompt.assert_called_once()
        self.assertIn("Custom prompt", result)

    def test_default_prompt_fallback(self):
        """Test fallback to default prompt when custom prompt is not available"""
        # Create project with mock config and default prompt
        project = TranslationProject(
            "test_project",
            "fr",
            config=self.mock_config,
            default_prompt="Default prompt",
        )

        # Mock the _load_custom_prompt method to return empty string
        project._load_custom_prompt = AsyncMock(return_value="")

        # Create a batch prompt
        phrases = ["Hello", "World"]
        translations = [
            {"en": "Hello", "fr": "", "context": "greeting"},
            {"en": "World", "fr": "", "context": "place"},
        ]
        indices = [0, 1]
        result = self.loop.run_until_complete(
            project._create_batch_prompt(phrases, translations, indices)
        )

        # Verify default prompt was used
        project._load_custom_prompt.assert_called_once()
        self.assertIn("Default prompt", result)


if __name__ == "__main__":
    unittest.main()
