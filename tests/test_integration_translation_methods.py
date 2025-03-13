import os
import sys
import pytest
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.TranslationTool import TranslationTool
from lib.PromptManager import PromptManager
from lib.llm import get_driver, get_available_models


# Mark these tests as integration tests so they can be skipped by default
# Run with: pytest tests/test_integration_translation_methods.py -v
# Note: To run tests with openrouter-grok-3-beta, you need an OpenRouter API key
# set in your environment as OPENROUTER_API_KEY
@pytest.mark.integration
class TestIntegrationTranslationMethods:
    """Integration tests for different translation methods using appropriate models for each method."""

    @pytest.fixture
    def prompt_manager(self):
        """Create a real PromptManager instance for testing."""
        project_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return PromptManager(project_dir)

    @pytest.fixture
    def translation_tool(self, prompt_manager):
        """Create a TranslationTool instance with a real PromptManager."""
        return TranslationTool(prompt_manager)

    @pytest.fixture
    def test_data(self):
        """Create test data for translation tests."""
        phrases = ["Hello", "Thank you", "How are you?"]
        translations = [
            {"key": "greeting", "en": "Hello", "context": "Casual greeting"},
            {"key": "thanks", "en": "Thank you", "context": "Expression of gratitude"},
            {
                "key": "inquiry",
                "en": "How are you?",
                "context": "Asking about wellbeing",
            },
        ]
        indices = [0, 1, 2]
        progress: Dict[str, str] = {}

        return {
            "phrases": phrases,
            "translations": translations,
            "indices": indices,
            "progress": progress,
        }

    @pytest.fixture
    def translation_params(self):
        """Common parameters for translation tests."""
        return {
            "model": "gemini",
            "base_language": "en",
            "dst_language": "es",
            "context": "These are casual conversational phrases.",
            "delay_seconds": 1.0,
            "max_retries": 2,
        }

    @pytest.mark.asyncio
    async def test_load_prompt(self, translation_tool, prompt_manager):
        """Test loading the translation prompt from the prompt manager."""
        prompt = await prompt_manager.load_prompt("translation")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "{base_language}" in prompt
        assert "{dst_language}" in prompt
        assert "{phrases_json}" in prompt

    @pytest.mark.asyncio
    async def test_standard_method(
        self, translation_tool, test_data, translation_params
    ):
        """Test translation using the standard method."""
        # Prepare the translation prompt
        prompt = await translation_tool.prompt_manager.load_prompt("translation")

        # Process a batch using the standard method
        result = await translation_tool.translate_standard(
            phrases=test_data["phrases"],
            indices=test_data["indices"],
            translations=test_data["translations"],
            progress=test_data["progress"],
            prompt=prompt,
            **translation_params,
        )

        # Validate the results
        assert result > 0  # Some phrases should be translated
        assert test_data["translations"][0]["es"] is not None
        assert test_data["translations"][1]["es"] is not None
        assert test_data["translations"][2]["es"] is not None

        # Print the translations for debugging
        print("\nStandard Method Translations:")
        for i, phrase in enumerate(test_data["phrases"]):
            print(f"  {phrase} -> {test_data['translations'][i]['es']}")

    @pytest.mark.asyncio
    async def test_structured_method(
        self, translation_tool, test_data, translation_params
    ):
        """Test translation using the structured output method."""

        prompt = await translation_tool.prompt_manager.load_prompt("translation")

        structured_params = translation_params.copy()
        structured_params["model"] = "gemini"

        try:
            # Process a batch using the structured method
            result = await translation_tool.translate_structured(
                phrases=test_data["phrases"],
                indices=test_data["indices"],
                translations=test_data["translations"],
                progress=test_data["progress"],
                prompt=prompt,
                **structured_params,
            )

            # If we get here without an exception, validate the results
            assert result > 0, "No translations were obtained from structured output"
            assert (
                test_data["translations"][0]["es"] is not None
            ), "First translation result is missing"
            assert (
                test_data["translations"][1]["es"] is not None
            ), "Second translation result is missing"
            assert (
                test_data["translations"][2]["es"] is not None
            ), "Third translation result is missing"

            # Print the translations for debugging
            print("\nStructured Method Translations:")
            for i, phrase in enumerate(test_data["phrases"]):
                print(f"  {phrase} -> {test_data['translations'][i]['es']}")

        except Exception as e:
            pytest.fail(f"Error in structured output test: {e}")

    @pytest.mark.asyncio
    async def test_function_method(
        self, translation_tool, test_data, translation_params
    ):
        """Test translation using the function calling metho."""

        prompt = await translation_tool.prompt_manager.load_prompt("translation")

        function_params = translation_params.copy()
        function_params["model"] = "openrouter-gemini-2.0-flash-lite-preview-02-05"

        result = await translation_tool.translate_function(
            phrases=test_data["phrases"],
            indices=test_data["indices"],
            translations=test_data["translations"],
            progress=test_data["progress"],
            prompt=prompt,
            **function_params,
        )

        # Validate the results
        assert result > 0  # Some phrases should be translated
        assert test_data["translations"][0]["es"] is not None
        assert test_data["translations"][1]["es"] is not None
        assert test_data["translations"][2]["es"] is not None

        # Print the translations for debugging
        print("\nFunction Method Translations:")
        for i, phrase in enumerate(test_data["phrases"]):
            print(f"  {phrase} -> {test_data['translations'][i]['es']}")

    @pytest.mark.asyncio
    async def test_compare_methods(self, translation_tool, prompt_manager):
        """Test and compare all three translation methods side by side using appropriate models for each method."""

        # Common test parameters
        base_language = "en"
        dst_language = "fr"  # Using French for variety
        test_phrases = ["Good morning", "I love programming", "See you tomorrow"]
        context = (
            "These are common phrases used in conversation."  # Added global context
        )

        # Prepare fresh data for each test to avoid cross-contamination
        def create_test_data():
            translations = [
                {
                    "key": f"phrase{i}",
                    "en": phrase,
                    "fr": "",
                    "context": "Common conversational phrase",
                }  # Added context
                for i, phrase in enumerate(test_phrases)
            ]
            indices = list(range(len(test_phrases)))
            progress = {}
            return translations, indices, progress

        # Load the prompt
        prompt = await prompt_manager.load_prompt("translation")

        # Run all three methods in parallel
        standard_data = create_test_data()
        structured_data = create_test_data()
        function_data = create_test_data()

        # Prepare the tasks to run concurrently
        tasks = [
            translation_tool.translate_standard(
                phrases=test_phrases,
                indices=standard_data[1],
                translations=standard_data[0],
                progress=standard_data[2],
                model="gemini",
                base_language=base_language,
                dst_language=dst_language,
                prompt=prompt,
                context=context,  # Added context parameter
                delay_seconds=1.0,
                max_retries=2,
            ),
            translation_tool.translate_structured(
                phrases=test_phrases,
                indices=structured_data[1],
                translations=structured_data[0],
                progress=structured_data[2],
                model="gemini",
                base_language=base_language,
                dst_language=dst_language,
                prompt=prompt,
                context=context,  # Added context parameter
                delay_seconds=1.0,
                max_retries=2,
            ),
            translation_tool.translate_function(
                phrases=test_phrases,
                indices=function_data[1],
                translations=function_data[0],
                progress=function_data[2],
                model="openrouter-gemini-2.0-flash-lite-preview-02-05",
                base_language=base_language,
                dst_language=dst_language,
                prompt=prompt,
                context=context,  # Added context parameter
                delay_seconds=1.0,
                max_retries=2,
            ),
        ]

        # Run the standard and function methods concurrently
        try:
            results = await asyncio.gather(*tasks)
            # Print comparison of results (all three methods)

            print(
                "\nTranslation Method Comparison (English to French with appropriate models):"
            )
            print(
                f"{'Original':<20} | {'Standard':<25} | {'Structured':<25} | {'Function':<25}"
            )
            print("-" * 100)

            for i, phrase in enumerate(test_phrases):
                standard_translation = standard_data[0][i].get(dst_language, "N/A")
                structured_translation = structured_data[0][i].get(dst_language, "N/A")
                function_translation = function_data[0][i].get(dst_language, "N/A")

                print(
                    f"{phrase:<20} | {standard_translation:<25} | {structured_translation:<25} | {function_translation:<25}"
                )

        except Exception as e:
            pytest.fail(f"Unexpected error in comparison test: {e}")


if __name__ == "__main__":
    # call tests from current module
    pytest.main(["-v", "-s", "-k", "integration", __file__])
