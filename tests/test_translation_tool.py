import os
import sys
import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.TranslationTool import TranslationTool, Input
from lib.PromptManager import PromptManager
from lib.storage.base import StorageAdapter
from tests.mock_llm_driver import MockLLMDriver


# Mock storage adapter for testing
class MockStorageAdapter(StorageAdapter):
    def __init__(self):
        self.context_file = None
        self.prompt_file = None
        self.prompts = {}

    def set_context_file(self, context_file):
        self.context_file = context_file

    def set_prompt_file(self, prompt_file):
        self.prompt_file = prompt_file

    async def load_config(self, project_id):
        return MagicMock()

    async def load_progress(self, project_id, language):
        return {}

    async def save_progress(self, project_id, language, progress):
        pass

    async def load_translations(self, project_id):
        return []

    async def save_translations(self, project_id, translations):
        pass

    async def load_context(self, project_id):
        return []

    async def load_prompt(self, project_id, prompt_type):
        if prompt_type in self.prompts:
            return self.prompts[prompt_type]
        return ""


class TestTranslationTool:
    """Test suite for TranslationTool class."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage adapter for testing."""
        storage = MockStorageAdapter()
        storage.prompts = {
            "translation": "Translate from {base_language} to {dst_language}: {phrases_json}",
            "json_fix": "Fix this invalid JSON: {invalid_json}",
            "output_format": "Return a JSON array of translations.",
        }
        return storage

    @pytest.fixture
    def prompt_manager(self, mock_storage):
        """Create a PromptManager instance for testing."""
        return PromptManager(mock_storage, "test_project")

    @pytest.fixture
    def translation_tool(self, prompt_manager):
        """Create a TranslationTool instance with a PromptManager."""
        return TranslationTool(prompt_manager)

    @pytest.fixture
    def mock_llm_driver(self):
        """Create a mock LLM driver for testing."""
        return MockLLMDriver()

    @pytest.mark.asyncio
    async def test_create_prompt(self, translation_tool, prompt_manager):
        """Test creating a batch prompt."""
        # Prepare test data using the new format (list of tuples with phrase and context)
        phrases = [
            ("Hello", "Greeting"),
            ("Goodbye", "Farewell"),
            ("Welcome", "Greeting someone arriving"),
        ]
        base_language = "en"
        dst_language = "es"
        # Use template with correct placeholders
        prompt = "Translate the following phrases from {base_language} to {dst_language}.\n\nPhrases to translate:\n{phrases_json}\n{context}"
        context = "These are common greetings."

        # Call the method with updated parameters
        result = await translation_tool.create_prompt(
            phrases=phrases,
            base_language=base_language,
            dst_language=dst_language,
            prompt=prompt,
            context=context,
        )

        # Verify the result contains expected content
        assert isinstance(result, str)
        assert f"from {base_language.upper()} to {dst_language.upper()}" in result
        assert "Hello" in result
        assert "Goodbye" in result
        assert "Welcome" in result
        # The context should be included
        assert "common greetings" in result

    @pytest.mark.asyncio
    async def test_translate_standard(self, translation_tool, mock_llm_driver):
        """Test processing a batch of translations with the mock LLM driver."""
        # Prepare test data using the new format
        phrases = [("Hello", None), ("Goodbye", None), ("Welcome", None)]
        base_language = "en"
        dst_language = "es"
        prompt = "Translate the following phrases from {base_language} to {dst_language}.\n\nPhrases to translate:\n{phrases_json}"

        # Ensure mock_llm_driver is properly registered
        with patch(
            "lib.TranslationTool.get_driver", return_value=mock_llm_driver
        ), patch("lib.llm.get_driver", return_value=mock_llm_driver):

            # Set up a specific response pattern for this test
            mock_llm_driver.register_response(
                r"Translate.*from EN to ES",
                """```json
                ["Hola", "Adiós", "Bienvenido"]
                ```""",
            )

            # Set up mock translate_async method
            mock_llm_driver.translate_async = AsyncMock(
                return_value="""```json
                ["Hola", "Adiós", "Bienvenido"]
                ```"""
            )

            # Call the method
            result = await translation_tool.translate_standard(
                phrases=phrases,
                model="mock-model",
                base_language=base_language,
                dst_language=dst_language,
                prompt=prompt,
            )

            # Verify the result is a dictionary mapping phrases to translations
            assert isinstance(result, dict)
            assert result.get("Hello") == "Hola"
            assert result.get("Goodbye") == "Adiós"
            assert result.get("Welcome") == "Bienvenido"
            assert len(result) == 3
