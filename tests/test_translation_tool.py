import os
import sys
import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.TranslationTool import TranslationTool, InvalidJSONException
from lib.PromptManager import PromptManager
from tests.mock_llm_driver import MockLLMDriver


class TestTranslationTool:
    """Test suite for TranslationTool class."""

    @pytest.fixture
    def prompt_manager(self):
        """Create a real PromptManager instance for testing."""
        # Use the actual project root directory
        project_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return PromptManager(project_dir)

    @pytest.fixture
    def translation_tool(self, prompt_manager):
        """Create a TranslationTool instance with a real PromptManager."""
        return TranslationTool(prompt_manager)

    @pytest.fixture
    def mock_llm_driver(self):
        """Create a mock LLM driver for testing."""
        return MockLLMDriver()

    def test_get_available_models(self, translation_tool):
        """Test that get_available_models returns a list of models."""
        models = translation_tool.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0

    @pytest.mark.asyncio
    async def test_create_batch_prompt(self, translation_tool, prompt_manager):
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
        result = await translation_tool.create_batch_prompt(
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
    async def test_fix_invalid_json(self, translation_tool, mock_llm_driver):
        """Test fixing invalid JSON with the mock LLM driver."""
        invalid_json = '{"Hello": "Hola", "Goodbye": "Adiós", Welcome: "Bienvenido"}'

        # Patch the get_driver function to return our mock
        with patch("lib.llm.get_driver", return_value=mock_llm_driver):
            # Call the method
            fixed_json = await translation_tool.fix_invalid_json(
                invalid_json, mock_llm_driver
            )

            # Verify the result
            assert isinstance(fixed_json, str)
            assert '"Welcome"' in fixed_json  # The quotes should be fixed

            # Try to parse the fixed JSON to ensure it's valid
            parsed = json.loads(fixed_json)
            assert parsed["Hello"] == "Hola"
            assert parsed["Welcome"] == "Bienvenido"

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
