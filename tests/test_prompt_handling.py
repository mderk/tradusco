import unittest
import asyncio
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

from lib.TranslationProject import TranslationProject, PromptManager


class AsyncMock(MagicMock):
    """Helper class for mocking async methods"""

    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


class TestPromptManager(unittest.IsolatedAsyncioTestCase):
    """Tests for the PromptManager class"""

    def setUp(self):
        """Set up test fixtures"""
        self.project_dir = Path("/fake/project")
        self.manager = PromptManager(self.project_dir)

    async def test_validate_prompt(self):
        """Test prompt validation"""
        # Test valid translation prompt
        valid_prompt = """Translate from {base_language} to {dst_language}.
        Phrases: {phrases_json}"""
        is_valid, error = self.manager._validate_prompt(
            "translation", valid_prompt, strict=True
        )
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

        # Test invalid translation prompt (missing variable)
        invalid_prompt = "Translate from {base_language} to {dst_language}."
        is_valid, error = self.manager._validate_prompt(
            "translation", invalid_prompt, strict=True
        )
        self.assertFalse(is_valid)
        self.assertIn("phrases_json", error)

        # Test non-strict validation
        is_valid, error = self.manager._validate_prompt(
            "translation", invalid_prompt, strict=False
        )
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

        # Test empty prompt
        is_valid, error = self.manager._validate_prompt("translation", "", strict=False)
        self.assertFalse(is_valid)
        self.assertEqual(error, "Empty prompt template")

    @patch("aiofiles.open")
    @patch("pathlib.Path.exists")
    @patch("os.path.exists")
    async def test_prompt_caching(
        self, mock_os_exists, mock_path_exists, mock_aiofiles_open
    ):
        """Test prompt caching behavior"""
        # Setup mock file
        mock_os_exists.return_value = True
        mock_path_exists.return_value = True
        mock_file = AsyncMock()
        mock_file.read.return_value = "Cached prompt content"
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

        # First load should read from file
        result1 = await self.manager.load_prompt("test", validate=False, use_cache=True)
        self.assertEqual(result1, "Cached prompt content")
        mock_aiofiles_open.assert_called_once()

        # Second load should use cache
        result2 = await self.manager.load_prompt("test", validate=False, use_cache=True)
        self.assertEqual(result2, "Cached prompt content")
        mock_aiofiles_open.assert_called_once()  # Should not be called again

        # Clear cache and load again
        self.manager.clear_cache()
        result3 = await self.manager.load_prompt("test", validate=False, use_cache=True)
        self.assertEqual(result3, "Cached prompt content")
        self.assertEqual(mock_aiofiles_open.call_count, 2)  # Should be called again

    @patch("aiofiles.open")
    @patch("pathlib.Path.exists")
    async def test_load_prompt_with_validation(self, mock_exists, mock_aiofiles_open):
        """Test loading prompts with validation"""
        # Setup mock file with invalid prompt
        mock_exists.return_value = True
        mock_file = AsyncMock()
        mock_file.read.return_value = "Invalid prompt without variables"
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

        # Test loading with strict validation
        result = await self.manager.load_prompt(
            "translation", validate=True, strict_validation=True
        )
        self.assertEqual(result, "")  # Should fail validation and return empty string

        # Test loading with non-strict validation
        result = await self.manager.load_prompt(
            "translation", validate=True, strict_validation=False
        )
        self.assertEqual(
            result, "Invalid prompt without variables"
        )  # Should accept the prompt

        # Test loading without validation
        result = await self.manager.load_prompt("translation", validate=False)
        self.assertEqual(
            result, "Invalid prompt without variables"
        )  # Should return content as is

    def test_format_prompt(self):
        """Test prompt formatting"""
        template = "Hello {name}, welcome to {place}!"

        # Test successful formatting
        result = self.manager.format_prompt(template, name="John", place="Earth")
        self.assertEqual(result, "Hello John, welcome to Earth!")

        # Test missing variable
        result = self.manager.format_prompt(template, name="John")
        self.assertEqual(result, template)  # Should return original template

        # Test invalid format
        invalid_template = "Hello {name:invalid}"
        result = self.manager.format_prompt(invalid_template, name="John")
        self.assertEqual(result, invalid_template)  # Should return original template


class TestPromptHandling(unittest.TestCase):
    """Tests for prompt handling in TranslationProject"""

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

    @patch("lib.TranslationProject.PromptManager.load_prompt")
    def test_create_batch_prompt(self, mock_load_prompt):
        """Test creating a batch prompt with custom prompt"""
        # Setup mock prompt manager
        mock_load_prompt.return_value = "Custom prompt"

        # Create project with mock config
        project = TranslationProject(
            "test_project",
            "fr",
            config=self.mock_config,
            default_prompt="Default prompt",
        )

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
        mock_load_prompt.assert_called_once_with(
            "translation",
            project.prompt_file,
            validate=True,
            strict_validation=True,
        )
        self.assertIn("Custom prompt", result)

    @patch("lib.TranslationProject.PromptManager.load_prompt")
    def test_create_batch_prompt_with_default(self, mock_load_prompt):
        """Test creating a batch prompt with default prompt"""
        # Setup mock prompt manager to return empty string
        mock_load_prompt.return_value = ""

        # Create project with mock config and default prompt
        project = TranslationProject(
            "test_project",
            "fr",
            config=self.mock_config,
            default_prompt="Default prompt",
        )

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
        mock_load_prompt.assert_called_once_with(
            "translation",
            project.prompt_file,
            validate=True,
            strict_validation=True,
        )
        self.assertIn("Default prompt", result)

    @patch("lib.TranslationProject.PromptManager.load_prompt")
    def test_fix_invalid_json(self, mock_load_prompt):
        """Test fixing invalid JSON"""
        # Setup mock prompt manager
        mock_load_prompt.return_value = "Fix this JSON: {invalid_json}"

        # Create project with mock config
        project = TranslationProject("test_project", "fr", config=self.mock_config)

        # Mock the driver
        mock_driver = MagicMock()
        mock_driver.translate_async = AsyncMock(return_value='{"fixed": "json"}')

        # Test fixing invalid JSON
        result = self.loop.run_until_complete(
            project._fix_invalid_json('{"broken": "json"', mock_driver)
        )

        # Verify prompt was loaded and used correctly
        mock_load_prompt.assert_called_once_with("json_fix")
        self.assertEqual(result, '{"fixed": "json"}')


if __name__ == "__main__":
    unittest.main()
