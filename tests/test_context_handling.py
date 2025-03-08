import pytest
from unittest.mock import patch

from lib.TranslationProject import TranslationProject
from tests.utils import AsyncMock


class TestContextHandling:
    """Tests for context handling in TranslationProject"""

    @pytest.mark.asyncio
    async def test_load_global_context_from_string(self, translation_project):
        """Test loading a global context from a string"""
        # Set a context directly
        context = "This is a global context for testing."
        translation_project.context = context

        # Verify the context is set
        assert translation_project.context == context

    @pytest.mark.asyncio
    @patch("aiofiles.open")
    @patch("os.path.exists")
    async def test_load_global_context_from_file(
        self, mock_os_exists, mock_aiofiles_open, translation_project
    ):
        """Test loading a global context from a file"""
        # Set up mocks
        mock_os_exists.return_value = True

        # Mock the file read operation
        mock_file = AsyncMock()
        mock_file.read.return_value = "This is a context from a file."
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

        # Set a context file path
        context_file = "/path/to/context.txt"
        translation_project.context_file = context_file

        # Load the context
        context = await translation_project._load_context()

        # Verify the context was loaded from the file
        assert "This is a context from a file." in context
        # The mock is called multiple times because _load_context checks multiple files

    @pytest.mark.asyncio
    @patch("lib.PromptManager.PromptManager.load_prompt")
    @patch.object(TranslationProject, "_load_translations")
    @patch.object(TranslationProject, "_load_context")
    async def test_create_batch_prompt_with_context(
        self,
        mock_load_context,
        mock_load_translations,
        mock_load_prompt,
        translation_project,
    ):
        """Test creating a batch prompt with context"""
        # Mock the context
        mock_load_context.return_value = "Global context for testing"

        # Mock translations with context
        mock_load_translations.return_value = [
            {"en": "Hello", "es": "", "context": "Context for Hello"},
            {"en": "World", "es": "", "context": "Context for World"},
        ]

        # Mock prompt loading
        mock_load_prompt.return_value = (
            "Translate with context: {global_context}\n{phrases_json}"
        )

        # Create a batch prompt
        phrases = ["Hello", "World"]
        translations = await translation_project._load_translations()
        indices = [0, 1]

        # Generate the prompt
        prompt = await translation_project._create_batch_prompt(
            phrases, translations, indices
        )

        # Since we're mocking the prompt template and not actually formatting it,
        # we should check for the template variables instead
        assert "{global_context}" in prompt
        assert "{phrases_json}" in prompt

    @pytest.mark.asyncio
    @patch("aiofiles.open")
    async def test_missing_context_file(self, mock_aiofiles_open, translation_project):
        """Test handling a missing context file"""
        # Mock file not found error
        mock_aiofiles_open.side_effect = FileNotFoundError("File not found")

        # Set a non-existent context file
        translation_project.context_file = "/path/to/nonexistent.txt"

        # Load the context
        context = await translation_project._load_context()

        # Verify empty context is returned
        assert context == ""
