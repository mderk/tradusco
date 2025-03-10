import json
import re
from typing import Optional
from .llm import BaseDriver, get_driver, get_available_models
from .PromptManager import PromptManager


class InvalidJSONException(Exception):
    """Exception raised when invalid JSON is encountered during parsing."""

    def __init__(self, message: str, json_str: str):
        self.message = message
        self.json_str = json_str
        super().__init__(f"{message}: {json_str[:100]}...")


class TranslationTool:
    """
    A class for handling the translation functionality.

    This class is responsible for:
    1. Creating translation prompts
    2. Parsing translation responses
    3. Fixing invalid JSON responses
    4. Processing batches of translations
    """

    def __init__(self, prompt_manager: PromptManager):
        """
        Initialize the TranslationTool.

        Args:
            prompt_manager: PromptManager instance for loading and formatting prompts
        """
        self.prompt_manager = prompt_manager

    @staticmethod
    def get_available_models() -> list[str]:
        """Get a list of available translation models"""
        return get_available_models()

    async def create_batch_prompt(
        self,
        phrases: list[str],
        translations: list[dict[str, str]],
        indices: list[int],
        base_language: str,
        dst_language: str,
        prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """Create a prompt for batch translation using JSON format"""
        # Create a list of phrases and a separate context mapping
        phrases_to_translate = []
        phrase_contexts = {}

        for i, phrase in enumerate(phrases):
            phrases_to_translate.append(phrase)
            phrase_context = translations[indices[i]].get("context", "")
            if phrase_context:  # Only include phrases that have context
                phrase_contexts[phrase] = phrase_context

        # Encode phrases and contexts as JSON
        phrases_json = json.dumps(phrases_to_translate, ensure_ascii=False, indent=2)
        contexts_json = (
            json.dumps(phrase_contexts, ensure_ascii=False, indent=2)
            if phrase_contexts
            else ""
        )

        prompt_template = ""

        if prompt:
            valid, error = self.prompt_manager.validate_prompt(
                "translation", prompt, strict=True
            )
            if valid:
                prompt_template = prompt
            else:
                print(f"Warning: {error}")

        if not prompt_template:
            prompt_template = await self.prompt_manager.load_prompt(
                "translation",
                prompt_file,
                validate=True,
                strict_validation=True,  # Only enforce required variables when actually translating
            )

        # Add global context if provided
        context_section = (
            f"\nGlobal Translation Context:\n{context}\n" if context else ""
        )

        # Add phrase-specific contexts section if any exist
        phrase_contexts_section = (
            f"\nPhrase-specific contexts:\n{contexts_json}\n" if contexts_json else ""
        )

        # Format the prompt template with the required variables
        return self.prompt_manager.format_prompt(
            prompt_template,
            base_language=base_language,
            dst_language=dst_language,
            phrases_json=phrases_json,
            context=context_section,
            phrase_contexts=phrase_contexts_section,
        )

    async def fix_invalid_json(self, invalid_json: str, driver: BaseDriver) -> str:
        """
        Attempt to fix invalid JSON by sending it back to the LLM.

        Args:
            invalid_json: The invalid JSON string
            driver: The LLM driver to use for fixing

        Returns:
            Corrected JSON string or original string if correction failed
        """
        # Load the JSON fix prompt template
        prompt_template = await self.prompt_manager.load_prompt("json_fix")

        # Fill in the invalid JSON
        prompt = prompt_template.replace("{invalid_json}", invalid_json)

        try:
            # Send the prompt to the LLM
            corrected_json = await driver.translate_async(
                prompt, delay_seconds=1.0, max_retries=2
            )

            # Extract JSON from the response
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", corrected_json)
            if json_match:
                return json_match.group(1)

            # If no code blocks, try to find a JSON array or object directly
            json_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", corrected_json)
            if json_match:
                return json_match.group(1)

            # If no recognizable JSON pattern, return the full response
            return corrected_json

        except Exception as e:
            print(f"Error fixing JSON: {e}")
            return invalid_json  # Return original if fixing failed

    def parse_batch_response(
        self,
        response: str,
        original_phrases: list[str],
    ) -> dict[str, str]:
        """Parse the batch translation response from JSON format"""
        translations = {}

        # Extract potential JSON content
        # Try to find JSON content between code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no code blocks, try to find a JSON array or object directly
            json_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", response)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Use the entire response as a last resort
                json_str = response

        try:
            # Parse the JSON content
            translated_items = json.loads(json_str)

            # Handle different possible JSON formats
            if isinstance(translated_items, list):
                # If it's a list of translations in the same order as original phrases
                for i, translation in enumerate(translated_items):
                    if i < len(original_phrases):
                        if isinstance(translation, str):
                            translations[original_phrases[i]] = translation
                        elif isinstance(translation, dict):
                            # Handle both new and old format responses
                            if "translation" in translation:
                                translations[original_phrases[i]] = translation[
                                    "translation"
                                ]
                            elif "text" in translation:
                                translations[original_phrases[i]] = translation["text"]
            elif isinstance(translated_items, dict):
                # If it's a dictionary mapping original phrases to translations
                for original, translation in translated_items.items():
                    # Try to match by the original phrase
                    if original in original_phrases:
                        translations[original] = translation
                    # Try to match by the phrase in "text" field
                    elif isinstance(translation, dict) and "text" in translation:
                        original_text = translation.get("text", "")
                        if original_text in original_phrases:
                            translations[original_text] = translation["translation"]
                    # Also try to match by index if the keys are numeric
                    elif original.isdigit() and (int(original) - 1) < len(
                        original_phrases
                    ):
                        translations[original_phrases[int(original) - 1]] = translation

        except Exception as e:
            # If JSON parsing fails, raise InvalidJSONException
            raise InvalidJSONException(f"Error parsing JSON response: {e}", json_str)

        return translations

    def update_translations_from_batch(
        self,
        batch_translations: dict[str, str],
        phrases: list[str],
        indices: list[int],
        translations: list[dict[str, str]],
        progress: dict[str, str],
        dst_language: str,
    ) -> int:
        """
        Update translations and progress based on batch results

        Args:
            batch_translations: Dictionary mapping phrases to translations
            phrases: List of original phrases being translated
            indices: Indices of phrases in the translations list
            translations: Full translations data structure
            progress: Progress dictionary
            dst_language: Destination language code

        Returns:
            Number of phrases successfully translated
        """
        updates_count = 0
        for i, phrase in enumerate(phrases):
            if phrase in batch_translations:
                translation = batch_translations[phrase]
                translations[indices[i]][dst_language] = translation
                progress[phrase] = translation
                print(f"Translated: {phrase} -> {translation}")
                updates_count += 1
            else:
                print(f"Warning: No translation found for '{phrase}'")
        return updates_count

    async def process_batch(
        self,
        phrases: list[str],
        indices: list[int],
        translations: list[dict[str, str]],
        progress: dict[str, str],
        model: str,
        base_language: str,
        dst_language: str,
        prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        context: Optional[str] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
    ) -> int:
        """
        Process a batch of phrases for translation

        Args:
            phrases: List of phrases to translate
            indices: Indices of phrases in the translations list
            translations: Full translations data structure
            progress: Progress dictionary
            model: Model to use
            base_language: Source language code
            dst_language: Destination language code
            prompt: Optional custom prompt text
            prompt_file: Optional path to prompt file
            context: Optional context text
            delay_seconds: Delay between LLM calls
            max_retries: Maximum retries for failed calls

        Returns:
            Number of phrases successfully translated
        """
        if not phrases:
            return 0

        print(f"Translating batch of {len(phrases)} phrases...")

        # Create the batch prompt with translations data for context
        prompt_text = await self.create_batch_prompt(
            phrases,
            translations,
            indices,
            base_language,
            dst_language,
            prompt,
            prompt_file,
            context,
        )

        update_count = 0
        try:
            # Initialize the LLM driver with the specified model
            driver = get_driver(model)

            # Send the batch to the LLM using the async translate method
            response = await driver.translate_async(
                prompt_text, delay_seconds, max_retries
            )

            try:
                # Parse the response
                batch_translations = self.parse_batch_response(response, phrases)

                # Update translations and progress
                update_count = self.update_translations_from_batch(
                    batch_translations,
                    phrases,
                    indices,
                    translations,
                    progress,
                    dst_language,
                )

            except InvalidJSONException as e:
                print(f"Invalid JSON detected: {e.message}")
                print("Attempting to fix invalid JSON using LLM...")

                # Try to fix the JSON
                fixed_json_str = await self.fix_invalid_json(e.json_str, driver)

                # Try parsing again with the fixed JSON
                try:
                    # Create a new response with the fixed JSON
                    fixed_response = response.replace(e.json_str, fixed_json_str)

                    # Parse the fixed response
                    batch_translations = self.parse_batch_response(
                        fixed_response, phrases
                    )

                    # Update translations and progress
                    update_count = self.update_translations_from_batch(
                        batch_translations,
                        phrases,
                        indices,
                        translations,
                        progress,
                        dst_language,
                    )

                    if update_count > 0:
                        print("Successfully fixed and parsed JSON!")
                    else:
                        print(
                            "Fixed JSON parsing succeeded but no translations were found."
                        )

                except Exception as e2:
                    print(f"Failed to process fixed JSON: {e2}")
                    print("Skipping this batch due to unrecoverable JSON error.")

        except Exception as e:
            print(f"Error translating batch: {e}")
            print("Failed to translate batch. Skipping these phrases.")

        return update_count
