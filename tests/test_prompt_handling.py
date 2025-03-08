import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock

from lib.PromptManager import PromptManager
from lib.TranslationProject import TranslationProject


@pytest.fixture
def prompt_manager():
    """Create a PromptManager instance for testing"""
    test_dir = Path("test_dir")
    return PromptManager(test_dir)


@pytest.fixture
def test_config():
    """Create test configuration for TranslationProject"""
    return {
        "sourceFile": "test.csv",
        "languages": ["en", "es", "fr"],
        "baseLanguage": "en",
    }


@pytest.fixture
def translation_project(test_config):
    """Create a TranslationProject instance for testing"""
    return TranslationProject(
        "test_project", "es", config=test_config, default_prompt=""
    )


class TestPromptManager:
    """Tests for the PromptManager class"""

    @pytest.mark.asyncio
    async def test_validate_prompt(self, prompt_manager):
        """Test prompt validation"""
        # Test valid translation prompt
        valid_prompt = """Translate from {base_language} to {dst_language}.
        Phrases: {phrases_json}"""
        is_valid, error = prompt_manager._validate_prompt(
            "translation", valid_prompt, strict=True
        )
        assert is_valid
        assert error == ""

        # Test invalid translation prompt (missing variable)
        invalid_prompt = "Translate from {base_language} to {dst_language}."
        is_valid, error = prompt_manager._validate_prompt(
            "translation", invalid_prompt, strict=True
        )
        assert not is_valid
        assert "phrases_json" in error

        # Test non-strict validation
        is_valid, error = prompt_manager._validate_prompt(
            "translation", invalid_prompt, strict=False
        )
        assert is_valid
        # With non-strict validation, it will still return a warning about missing variables
        assert "Missing required variables" in error

        # Test empty prompt
        is_valid, error = prompt_manager._validate_prompt(
            "translation", "", strict=False
        )
        assert not is_valid
        assert error == "Empty prompt template"

    @patch("aiofiles.open")
    @patch("pathlib.Path.exists")
    @patch("os.path.exists")
    @pytest.mark.asyncio
    async def test_prompt_caching(
        self, mock_os_exists, mock_path_exists, mock_aiofiles_open, prompt_manager
    ):
        """Test that prompts are cached after loading"""
        # Setup mocks
        mock_os_exists.return_value = True
        mock_path_exists.return_value = True

        # Create mock file
        mock_file = AsyncMock()
        mock_file.read.return_value = "Test prompt with {variable}"
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

        # First load should read from file
        result1 = await prompt_manager.load_prompt("test_type")
        assert result1 == "Test prompt with {variable}"
        mock_aiofiles_open.assert_called_once()

        # Reset mock to verify it's not called again
        mock_aiofiles_open.reset_mock()

        # Second load should use cached value
        result2 = await prompt_manager.load_prompt("test_type")
        assert result2 == "Test prompt with {variable}"
        mock_aiofiles_open.assert_not_called()

    @patch("aiofiles.open")
    @patch("pathlib.Path.exists")
    @pytest.mark.asyncio
    async def test_load_prompt_with_validation(
        self, mock_exists, mock_aiofiles_open, prompt_manager
    ):
        """Test loading prompts with validation"""
        # Setup mocks
        mock_exists.return_value = True

        # Clear cache to ensure we're not getting cached values
        prompt_manager.clear_cache()

        # Create mock file
        mock_file = AsyncMock()
        mock_file.read.return_value = (
            "Valid prompt with {base_language} and {dst_language} and {phrases_json}"
        )
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

        # Load with validation
        result = await prompt_manager.load_prompt("translation", validate=True)
        assert (
            result
            == "Valid prompt with {base_language} and {dst_language} and {phrases_json}"
        )

        # Test with an invalid prompt
        # Clear cache again and reset the mock file
        prompt_manager.clear_cache()
        mock_file.read.return_value = "Invalid prompt with {unknown_var}"

        # When validating strictly with invalid prompt, the implementation returns empty string
        # We need to mock the path to check multiple files
        result = await prompt_manager.load_prompt(
            "translation", validate=True, strict_validation=True
        )
        assert result == ""

    def test_format_prompt(self, prompt_manager):
        """Test formatting prompts with variables"""
        # Test basic formatting
        template = "Hello, {name}! Today is {day}."

        # The format_prompt method takes template and kwargs, not a dict
        result = prompt_manager.format_prompt(template, name="World", day="Monday")
        assert result == "Hello, World! Today is Monday."

        # Test with missing values
        template = "Hello, {name}! Missing {missing}."
        result = prompt_manager.format_prompt(template, name="World")
        # The method returns the original template when missing variables
        assert result == template

        # Test with invalid format
        template = "Hello, {name! Invalid."
        result = prompt_manager.format_prompt(template, name="World")
        assert result == template


class TestPromptHandling:
    """Tests for prompt handling in TranslationProject"""

    @patch("lib.TranslationProject.PromptManager.load_prompt")
    def test_create_batch_prompt(self, mock_load_prompt, translation_project):
        """Test creating a batch prompt"""
        # Mock the prompt loading
        mock_custom_prompt = (
            "Custom prompt with {base_language}, {dst_language}, and {phrases_json}"
        )
        mock_load_prompt.return_value = mock_custom_prompt

        # Create data for the test
        phrases = ["Hello", "World"]
        translations = [
            {"en": "Hello", "es": "", "fr": "Bonjour"},
            {"en": "World", "es": "", "fr": "Monde"},
        ]
        indices = [0, 1]

        # Create the batch prompt
        result = asyncio.run(
            translation_project._create_batch_prompt(phrases, translations, indices)
        )

        # Verify the prompt contains expected elements
        assert "Custom prompt with" in result
        assert "en" in result  # Source language
        assert "es" in result  # Target language
        assert "Hello" in result  # First phrase
        assert "World" in result  # Second phrase

    @patch("lib.TranslationProject.PromptManager.load_prompt")
    def test_create_batch_prompt_with_default(
        self, mock_load_prompt, translation_project
    ):
        """Test creating a batch prompt with default prompt"""
        # Mock that custom prompt loading returns empty string
        mock_load_prompt.return_value = ""

        # Set default prompt
        default_prompt = (
            "Default prompt with {base_language}, {dst_language}, and {phrases_json}"
        )
        translation_project.default_prompt = default_prompt

        # Create data for the test
        phrases = ["Hello", "World"]
        translations = [
            {"en": "Hello", "es": "", "fr": "Bonjour"},
            {"en": "World", "es": "", "fr": "Monde"},
        ]
        indices = [0, 1]

        # Create the batch prompt
        result = asyncio.run(
            translation_project._create_batch_prompt(phrases, translations, indices)
        )

        # Verify the prompt contains expected elements
        assert "Default prompt" in result
        assert "en" in result  # Source language
        assert "es" in result  # Target language
        assert "Hello" in result  # First phrase
        assert "World" in result  # Second phrase

    @patch("lib.TranslationProject.PromptManager.load_prompt")
    def test_fix_invalid_json(self, mock_load_prompt, translation_project):
        """Test fixing invalid JSON with a fix prompt"""
        # Mock the prompt loading
        fix_prompt = "Fix this JSON: {invalid_json}"
        mock_load_prompt.return_value = fix_prompt

        # Mock the LLM driver
        mock_driver = AsyncMock()
        mock_driver.translate_async.return_value = '{"fixed": "json"}'

        # Test fixing invalid JSON
        invalid_json = "{invalid: json}"
        result = asyncio.run(
            translation_project._fix_invalid_json(invalid_json, mock_driver)
        )

        # Verify result
        assert result == '{"fixed": "json"}'
        # Check that the correct prompt type is loaded
        mock_load_prompt.assert_called_once_with("json_fix")
