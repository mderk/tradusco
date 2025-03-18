import os
import sys
import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.TranslationTool import TranslationTool
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
        # Prepare test data
        phrases = ["Hello", "Goodbye", "Welcome"]
        translations = [
            {"key": "phrase1", "en": "Hello", "context": "Greeting"},
            {"key": "phrase2", "en": "Goodbye", "context": "Farewell"},
            {"key": "phrase3", "en": "Welcome", "context": "Greeting someone arriving"},
        ]
        indices = [0, 1, 2]
        base_language = "en"
        dst_language = "es"
        # Use the template with correct placeholders including phrase_contexts
        prompt = "Translate the following phrases from {base_language} to {dst_language}.\n\nPhrases to translate:\n{phrases_json}\n{context}\n{phrase_contexts}"
        context = "These are common greetings."

        # Call the method with real components
        result = await translation_tool.create_prompt(
            phrases=phrases,
            translations=translations,
            indices=indices,
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
        # The context should now be included
        assert "common greetings" in result

        # Check that phrase contexts are included
        # The contexts are stored as a JSON object
        assert '"Hello": "Greeting"' in result or '"Hello":"Greeting"' in result
        assert "Farewell" in result
        assert "Greeting someone arriving" in result

    @pytest.mark.asyncio
    async def test_translate_standard(self, translation_tool, mock_llm_driver):
        """Test processing a batch of translations with the mock LLM driver."""
        # Prepare test data
        phrases = ["Hello", "Goodbye", "Welcome"]
        translations = [
            {"key": "phrase1", "en": "Hello"},
            {"key": "phrase2", "en": "Goodbye"},
            {"key": "phrase3", "en": "Welcome"},
        ]
        indices = [0, 1, 2]
        progress = {}
        base_language = "en"
        dst_language = "es"
        prompt = "Translate the following phrases from {base_language} to {dst_language}.\n\nPhrases to translate:\n{phrases_json}"

        # Ensure mock_llm_driver is properly registered in the lib.llm module
        with patch(
            "lib.TranslationTool.get_driver", return_value=mock_llm_driver
        ), patch("lib.llm.get_driver", return_value=mock_llm_driver):

            # Set up a specific response pattern for this test
            mock_llm_driver.register_response(
                r"Translate.*from EN to ES",
                """```json
                {
                    "translations": [
                        "Hola",
                        "Adiós",
                        "Bienvenido"
                    ]
                }
                ```""",
            )

            # Call the method
            result = await translation_tool.translate_standard(
                phrases=phrases,
                indices=indices,
                translations=translations,
                progress=progress,
                model="mock-model",
                base_language=base_language,
                dst_language=dst_language,
                prompt=prompt,
            )

            # Verify the result
            assert result == 3
            assert translations[0]["es"] == "Hola"
            assert translations[1]["es"] == "Adiós"
            assert translations[2]["es"] == "Bienvenido"
            assert progress.get("Hello") == "Hola"
