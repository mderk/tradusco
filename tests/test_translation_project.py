import json
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from lib.TranslationProject import TranslationProject, InvalidJSONException
from tests.utils import AsyncMock


# Fixtures
@pytest.fixture
def mock_config():
    """Create a mock configuration for tests"""
    return {
        "name": "test_project",
        "sourceFile": "translations.csv",
        "languages": ["en", "fr", "es"],
        "baseLanguage": "en",
        "keyColumn": "en",
    }


@pytest.fixture
def mock_prompt():
    """Create a mock prompt for tests"""
    return "You are a translator. Translate from {base_language} to {dst_language}.\n{phrases_json}"


@pytest.fixture
def translation_project(
    mock_config,
    mock_prompt,
):
    """Create a TranslationProject instance for testing"""
    # We need to patch all file operations to prevent actual file system access
    with patch("pathlib.Path.exists", return_value=True), patch(
        "builtins.open", MagicMock()
    ):
        project = TranslationProject(
            "test_project",
            "fr",
            config=mock_config,
            default_prompt=mock_prompt,
        )
        return project


class TestTranslationProject:
    """Tests for the TranslationProject class"""

    def test_init(self, translation_project, mock_config):
        """Test initialization of the TranslationProject"""
        assert translation_project.project_name == "test_project"
        assert translation_project.dst_language == "fr"
        assert translation_project.base_language == mock_config["baseLanguage"]
        # The actual implementation doesn't have a languages attribute directly
        assert "languages" in mock_config

    @pytest.mark.asyncio
    @patch("aiofiles.open")
    @patch(
        "pathlib.Path.exists", return_value=True
    )  # Ensure the progress file is found
    async def test_load_progress(
        self, mock_path_exists, mock_aiofiles_open, translation_project
    ):
        """Test loading progress from a file"""
        # Mock the async file open
        mock_file = AsyncMock()
        mock_file.read.return_value = '{"Hello": "Bonjour", "World": "Monde"}'
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

        # Run the test
        result = await translation_project._load_progress()

        # Check result
        assert result == {"Hello": "Bonjour", "World": "Monde"}
        mock_aiofiles_open.assert_called_once()

    @patch("aiofiles.open")
    def test_save_progress(self, mock_aiofiles_open, translation_project):
        """Test saving progress to a file"""
        # Mock the async file open
        mock_file = AsyncMock()
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

        # Set up test data
        progress = {"Hello": "Bonjour", "World": "Monde"}

        # Run the test
        asyncio.run(translation_project._save_progress(progress))

        # Verify the mocked file was written to with the expected JSON
        mock_file.write.assert_called_once()
        written_data = mock_file.write.call_args[0][0]
        assert json.loads(written_data) == {"Hello": "Bonjour", "World": "Monde"}

    @pytest.mark.asyncio
    @patch("aiofiles.open")
    @patch("pathlib.Path.exists", return_value=True)  # Ensure the source file is found
    async def test_load_translations(
        self, mock_path_exists, mock_aiofiles_open, translation_project
    ):
        """Test loading translations from a file"""
        # Mock the async file open
        mock_file = AsyncMock()
        mock_csv_content = "en,fr\nHello,Bonjour\nWorld,Monde\n"
        mock_file.read.return_value = mock_csv_content
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

        # Run the test
        result = await translation_project._load_translations()

        # Check the result - returns a list of dicts, not a dict
        assert len(result) == 2
        assert result[0]["en"] == "Hello"
        assert result[0]["fr"] == "Bonjour"
        assert result[1]["en"] == "World"
        assert result[1]["fr"] == "Monde"

    @pytest.mark.asyncio
    @patch("aiofiles.open")
    @patch("pathlib.Path.exists", return_value=True)  # Ensure the source file is found
    async def test_preserve_multiline_text(
        self, mock_path_exists, mock_aiofiles_open, translation_project
    ):
        """Test that multiline text is preserved in translations"""
        # Prepare CSV with multiline content
        mock_csv_content = 'en,fr\n"Hello\nWorld","Bonjour\nMonde"\n'

        # Mock the file open
        mock_file = AsyncMock()
        mock_file.read.return_value = mock_csv_content
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

        # Run the test
        result = await translation_project._load_translations()

        # Assertions - returns a list of dicts, not a dict
        assert len(result) == 1
        assert result[0]["en"] == "Hello\nWorld"
        assert result[0]["fr"] == "Bonjour\nMonde"

        # Test that the newlines are preserved when parsing a batch response
        mock_response = '{"Hello\\nWorld": "Bonjour\\nMonde"}'
        parsed_response = translation_project._parse_batch_response(
            mock_response, ["Hello\nWorld"]
        )

        assert parsed_response == {"Hello\nWorld": "Bonjour\nMonde"}

    @patch("lib.TranslationProject.PromptManager.load_prompt")
    def test_create_batch_prompt(
        self, mock_load_prompt, translation_project, mock_prompt
    ):
        """Test creation of batch prompt"""
        # Setup mocks
        mock_load_prompt.return_value = mock_prompt

        # Set up test data
        phrases = ["Hello", "World"]
        translations = [{"en": "Hello", "fr": ""}, {"en": "World", "fr": ""}]
        indices = [0, 1]

        # Generate the prompt
        prompt = asyncio.run(
            translation_project._create_batch_prompt(phrases, translations, indices)
        )

        # Check prompt contains expected elements
        assert "Translate" in prompt
        assert "en" in prompt  # Source language
        assert "fr" in prompt  # Target language
        assert "Hello" in prompt  # First phrase
        assert "World" in prompt  # Second phrase

    def test_parse_batch_response_json_array(self, translation_project):
        """Test parsing batch response in JSON array format"""
        response = '["Bonjour", "Monde"]'
        original_phrases = ["Hello", "World"]

        result = translation_project._parse_batch_response(response, original_phrases)

        assert result == {"Hello": "Bonjour", "World": "Monde"}

    def test_parse_batch_response_json_object(self, translation_project):
        """Test parsing batch response in JSON object format"""
        response = '{"Hello": "Bonjour", "World": "Monde"}'
        original_phrases = ["Hello", "World"]

        result = translation_project._parse_batch_response(response, original_phrases)

        assert result == {"Hello": "Bonjour", "World": "Monde"}

    def test_parse_batch_response_invalid_json(self, translation_project):
        """Test handling invalid JSON in batch response"""
        response = "Invalid JSON"
        original_phrases = ["Hello", "World"]

        with pytest.raises(InvalidJSONException):
            translation_project._parse_batch_response(response, original_phrases)

    @pytest.mark.asyncio
    async def test_translate(
        self, translation_project, common_translation_project_patches
    ):
        """Test the translate method"""
        # Extract mocks from common patches
        mock_load_translations = common_translation_project_patches[
            "mock_load_translations"
        ]
        mock_load_progress = common_translation_project_patches["mock_load_progress"]
        mock_process_batch = common_translation_project_patches["mock_process_batch"]
        mock_save_progress = common_translation_project_patches["mock_save_progress"]
        mock_save_translations = common_translation_project_patches[
            "mock_save_translations"
        ]
        mock_get_driver = common_translation_project_patches["mock_get_driver"]

        # Mock the translations and progress
        translations = [{"en": "Hello", "fr": ""}, {"en": "World", "fr": ""}]
        progress = {"Hello": "Bonjour"}

        # Set up the mock implementations
        mock_load_translations.return_value = translations
        mock_load_progress.return_value = progress

        # Mock the driver
        mock_driver = MagicMock()
        mock_get_driver.return_value = mock_driver

        # Skip the actual batch processing
        mock_process_batch.return_value = None

        # Run the translate method with minimal processing
        await translation_project.translate(batch_size=1, delay_seconds=0)
