import os
import sys
import pytest
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.TranslationTool import TranslationTool
from lib.PromptManager import PromptManager
from lib.storage.base import StorageAdapter
from lib.llm import get_driver, get_available_models


# Mock storage adapter for integration testing
class IntegrationTestStorageAdapter(StorageAdapter):
    def __init__(self):
        self.context_file = None
        self.prompt_file = None
        self.prompts = {}
        self.project_dir = Path(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

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
        # For integration tests, we still want to load real prompts from the file system
        prompts_dir = self.project_dir / "prompts"
        prompt_paths = [
            prompts_dir / f"{prompt_type}.txt",
            prompts_dir / prompt_type / "prompt.txt",
        ]

        for path in prompt_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return content

        # Return custom prompt if set
        if prompt_type in self.prompts:
            return self.prompts[prompt_type]

        return ""


# Mark these tests as integration tests so they can be skipped by default
# Run with: pytest tests/test_integration_translation_methods.py -v
# Note: To run tests with openrouter-grok-3-beta, you need an OpenRouter API key
# set in your environment as OPENROUTER_API_KEY
@pytest.mark.integration
class TestIntegrationTranslationMethods:
    """Integration tests for different translation methods using appropriate models for each method."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage adapter for testing."""
        return IntegrationTestStorageAdapter()

    @pytest.fixture
    def prompt_manager(self, mock_storage):
        """Create a real PromptManager instance for testing."""
        return PromptManager(mock_storage, "test_project")

    @pytest.fixture
    def translation_tool(self, prompt_manager):
        """Create a TranslationTool instance with a real PromptManager."""
        return TranslationTool(prompt_manager)

    @pytest.fixture
    def test_data(self):
        """Create test data for translation tests."""
        return {
            "phrases": [
                ("Hello world", "Greeting"),
                ("Goodbye", ""),
                ("Thank you", ""),
                ("How are you?", ""),
                ("What is your name?", ""),
            ],
            "translations": [
                {"en": "Hello world", "es": ""},
                {"en": "Goodbye", "es": ""},
                {"en": "Thank you", "es": ""},
                {"en": "How are you?", "es": ""},
                {"en": "What is your name?", "es": ""},
            ],
            "progress": {},
            "base_language": "en",
            "dst_language": "es",
        }

    @pytest.fixture
    def translation_params(self):
        """Define parameters for translation tests."""
        return {
            # Select an appropriate model for each method based on capability
            "standard_model": "gemini",  # or "gpt-3.5-turbo"
            "structured_model": "gemini",  # or "gpt-3.5-turbo" or "claude-3-opus-20240229"
            "function_model": "openrouter-gemini-2.0-flash-lite-preview-02-05",  # Use OpenRouter for function calling
            "delay_seconds": 1.0,
            "max_retries": 2,
            "verbose": True,  # Set to true to see more detailed output
        }

    @pytest.mark.asyncio
    async def test_load_prompt(self, translation_tool, prompt_manager, mock_storage):
        """Test that we can load a valid prompt for translation."""
        # Set a specific test prompt
        test_prompt = "You are translating from {base_language} to {dst_language}:\n{phrases_json}"
        mock_storage.prompts["translation"] = test_prompt

        # Load the prompt using the prompt manager
        prompt = await prompt_manager.load_prompt("translation")
        assert "base_language" in prompt
        assert "dst_language" in prompt
        assert "phrases_json" in prompt

    @pytest.mark.asyncio
    async def test_standard_method(
        self, translation_tool, test_data, translation_params
    ):
        """Test standard translation method using the standard model."""
        # Skip test if the model is not available
        model = translation_params["standard_model"]
        if model not in get_available_models():
            pytest.skip(f"Model {model} not available")

        # Clone test data to avoid modifying the fixture
        translations = [dict(item) for item in test_data["translations"]]
        progress = dict(test_data["progress"])

        # Load the real prompt from the PromptManager
        prompt = await translation_tool.prompt_manager.load_prompt("translation")

        # Run the translation
        translated = await translation_tool.translate_standard(
            test_data["phrases"],
            model,
            test_data["base_language"],
            test_data["dst_language"],
            prompt,
            None,
            translation_params["delay_seconds"],
            translation_params["max_retries"],
        )

        # Check that we got translations
        assert translated, "No translations were produced"

        # Update translations and progress with results
        for i, phrase_data in enumerate(test_data["phrases"]):
            phrase = phrase_data[0]
            translation = translated.get(phrase)
            if translation:
                translations[i]["es"] = translation
                progress[phrase] = translation

        # Print translations if verbose
        if translation_params["verbose"]:
            print("\nTranslations from standard method:")
            for i, translation in enumerate(translations):
                print(f"{test_data['phrases'][i][0]} -> {translation['es']}")

        # Verify translations
        for i, phrase_data in enumerate(test_data["phrases"]):
            phrase = phrase_data[0]
            assert translations[i]["es"], f"No translation for '{phrase}'"
            assert progress[phrase], f"Translation not added to progress for '{phrase}'"

    @pytest.mark.asyncio
    async def test_structured_method(
        self, translation_tool, test_data, translation_params
    ):
        """Test structured output translation method using the structured model."""
        # Skip test if the model is not available
        model = translation_params["structured_model"]
        if model not in get_available_models():
            pytest.skip(f"Model {model} not available")

        # Check if the selected model supports structured output
        driver = get_driver(model)
        if not hasattr(driver, "translate_structured_async"):
            pytest.skip(f"Model {model} does not support structured output")

        # Clone test data to avoid modifying the fixture
        translations = [dict(item) for item in test_data["translations"]]
        progress = dict(test_data["progress"])

        prompt = await translation_tool.prompt_manager.load_prompt("translation")

        # Run the translation
        translated = await translation_tool.translate_structured(
            test_data["phrases"],
            model,
            test_data["base_language"],
            test_data["dst_language"],
            prompt,
            None,  # context
            translation_params["delay_seconds"],
            translation_params["max_retries"],
        )

        # Check that we got translations
        assert translated, "No translations were produced"

        # Update translations and progress with results
        translation_count = 0
        for phrase, translation in translated.items():
            for i, phrase_data in enumerate(test_data["phrases"]):
                if phrase_data[0] == phrase:
                    translations[i]["es"] = translation
                    progress[phrase] = translation
                    translation_count += 1

        # Check that we got translations
        assert translation_count > 0, "No translations were produced"

        # Print translations if verbose
        if translation_params["verbose"]:
            print("\nTranslations from structured method:")
            for i, translation in enumerate(translations):
                print(f"{test_data['phrases'][i][0]} -> {translation['es']}")

        # Verify translations
        for i, phrase_data in enumerate(test_data["phrases"]):
            phrase = phrase_data[0]
            assert translations[i]["es"], f"No translation for '{phrase}'"
            assert progress[phrase], f"Translation not added to progress for '{phrase}'"

    @pytest.mark.asyncio
    async def test_function_method(
        self, translation_tool, test_data, translation_params
    ):
        """Test function call translation method using the function model."""
        # Skip test if the model is not available
        model = translation_params["function_model"]
        if model not in get_available_models():
            pytest.skip(f"Model {model} not available")

        # Check if the selected model supports function calling
        driver = get_driver(model)

        # Skip this test if the model doesn't support function calling
        if (
            not hasattr(driver, "translate_function_async")
            or not driver.supports_function_calling
        ):
            pytest.skip(f"Model {model} does not support function calling")

        # Clone test data to avoid modifying the fixture
        translations = [dict(item) for item in test_data["translations"]]
        progress = dict(test_data["progress"])

        prompt = await translation_tool.prompt_manager.load_prompt("translation")

        # Run the translation
        translated = await translation_tool.translate_function(
            test_data["phrases"],
            model,
            test_data["base_language"],
            test_data["dst_language"],
            prompt,
            None,  # context
            translation_params["delay_seconds"],
            translation_params["max_retries"],
        )

        # Check that we got translations
        assert translated, "No translations were produced"

        # Update translations and progress with results
        translation_count = 0
        for phrase, translation in translated.items():
            for i, phrase_data in enumerate(test_data["phrases"]):
                if phrase_data[0] == phrase:
                    translations[i]["es"] = translation
                    progress[phrase] = translation
                    translation_count += 1

        # Check that we got translations
        assert translation_count > 0, "No translations were produced"

        # Print translations if verbose
        if translation_params["verbose"]:
            print("\nTranslations from function method:")
            for i, translation in enumerate(translations):
                print(f"{test_data['phrases'][i][0]} -> {translation['es']}")

        # Verify translations
        for i, phrase_data in enumerate(test_data["phrases"]):
            phrase = phrase_data[0]
            assert translations[i]["es"], f"No translation for '{phrase}'"
            assert progress[phrase], f"Translation not added to progress for '{phrase}'"


if __name__ == "__main__":
    # call tests from current module
    pytest.main(["-v", "-s", "-k", "integration", __file__])
