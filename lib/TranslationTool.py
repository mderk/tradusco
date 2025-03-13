import json
import re
import os
from typing import Optional, Union
from .llm import BaseDriver, get_driver, get_available_models
from .PromptManager import PromptManager

DEBUG = os.environ.get("TRADUSCO_DEBUG")


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
        prompt: str,
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
            prompt,
            base_language=base_language.upper(),
            dst_language=dst_language.upper(),
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
            return self.extract_json_from_response(corrected_json)

        except Exception as e:
            if DEBUG:
                print(f"Error fixing JSON: {e}")
            return invalid_json  # Return original if fixing failed

    async def setup_batch_translation(
        self,
        phrases: list[str],
        translations: list[dict[str, str]],
        indices: list[int],
        model: str,
        base_language: str,
        dst_language: str,
        prompt: str,
        context: Optional[str] = None,
    ) -> tuple[Optional[BaseDriver], str]:
        """
        Set up common components for batch translation process.

        Args:
            phrases: List of phrases to translate
            translations: Full translations list
            indices: Indices of phrases in the translations list
            model: LLM model to use
            base_language: Source language
            dst_language: Target language
            prompt: Translation prompt
            context: Optional context for translation

        Returns:
            Tuple of (driver, batch_prompt) or (None, "") if driver initialization failed
        """
        # Get the LLM driver
        driver = get_driver(model)
        if not driver:
            if DEBUG:
                print(f"Warning: Could not get driver for model {model}")
            return None, ""

        # Create the batch prompt
        batch_prompt = await self.create_batch_prompt(
            phrases=phrases,
            translations=translations,
            indices=indices,
            base_language=base_language,
            dst_language=dst_language,
            prompt=prompt,
            context=context,
        )

        return driver, batch_prompt

    def extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON content from a response string using multiple approaches.

        Args:
            response: The string response that may contain JSON

        Returns:
            The extracted JSON string or the original response if no JSON pattern is found
        """
        # Approach 1: Extract potential JSON content from code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if json_match:
            return json_match.group(1)

        # Approach 2: If no code blocks, try to find a JSON array or object directly
        json_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", response)
        if json_match:
            return json_match.group(1)

        # Use the entire response as a last resort
        return response

    def update_translations_from_list(
        self,
        translations_list: list[str],
        phrases: list[str],
        indices: list[int],
        translations: list[dict[str, str]],
        progress: dict[str, str],
        dst_language: str,
    ) -> int:
        """
        Update translations from a list of translations (in the same order as phrases).

        Args:
            translations_list: List of translations
            phrases: List of original phrases
            indices: Indices of phrases in the full translations list
            translations: Full translations list to update
            progress: Progress dictionary to update
            dst_language: Target language

        Returns:
            Number of translations updated
        """
        update_count = 0

        for i, translation in enumerate(translations_list):
            if (
                i < len(phrases) and translation.strip()
            ):  # Only update if we have a non-empty translation
                translations[indices[i]][dst_language] = translation
                progress[phrases[i]] = translation
                update_count += 1
                if DEBUG:
                    print(f"Translated: {phrases[i]} -> {translation}")
            elif i < len(phrases) and DEBUG:
                print(f"Warning: Empty translation for '{phrases[i]}'")

        return update_count

    async def setup_translation(
        self,
        phrases: list[str],
        indices: list[int],
        translations: list[dict[str, str]],
        model: str,
        base_language: str,
        dst_language: str,
        prompt: str,
        context: Optional[str] = None,
        method_name: str = "standard",
    ) -> tuple[Optional[BaseDriver], str]:
        """
        Base method for processing batches of translations.
        Handles common setup and error handling logic.

        Args:
            phrases: List of phrases to translate
            indices: Indices of phrases in the translations list
            translations: Full translations list
            model: LLM model to use
            base_language: Source language
            dst_language: Target language
            prompt: Translation prompt
            context: Optional context for translation
            method_name: Name of the translation method being used (for logging)

        Returns:
            Tuple of (driver, batch_prompt) or (None, "") if setup failed
        """
        if DEBUG:
            print(
                f"Translating batch of {len(phrases)} phrases using {method_name} method..."
            )

        # Get the LLM driver
        driver, batch_prompt = await self.setup_batch_translation(
            phrases=phrases,
            translations=translations,
            indices=indices,
            model=model,
            base_language=base_language,
            dst_language=dst_language,
            prompt=prompt,
            context=context,
        )
        if not driver:
            return None, ""

        # Load the output format instructions
        if method_name == "standard":
            try:
                output_format = await self.prompt_manager.load_prompt("output_format")
            except Exception as e:
                if DEBUG:
                    print(f"Warning: Could not load output format instructions: {e}")
                output_format = ""

            # Add output format instructions if available
            if output_format:
                batch_prompt += f"\n\n{output_format}"

        return driver, batch_prompt

    def handle_response_format(
        self,
        response: Union[str, dict, list],
        phrases: list[str],
        indices: list[int],
        translations: list[dict[str, str]],
        progress: dict[str, str],
        dst_language: str,
    ) -> int:
        """
        Handle translation response format and update translations.
        Expects either a list of translations or a dict with a translations array.

        Args:
            response: The response from the translation service
            phrases: List of original phrases
            indices: Indices of phrases in the translations list
            translations: Full translations list to update
            progress: Progress dictionary to update
            dst_language: Target language

        Returns:
            Number of translations updated
        """
        if isinstance(response, str):
            # First extract JSON from code blocks if present
            json_str = self.extract_json_from_response(response)
            # Then try to parse it as JSON
            try:
                parsed_response = json.loads(json_str)
                return self.handle_response_format(
                    parsed_response,
                    phrases,
                    indices,
                    translations,
                    progress,
                    dst_language,
                )
            except json.JSONDecodeError:
                if DEBUG:
                    print("Invalid JSON response received")
                return 0

        # Handle list of translations
        if isinstance(response, list):
            return self.update_translations_from_list(
                translations_list=response,
                phrases=phrases,
                indices=indices,
                translations=translations,
                progress=progress,
                dst_language=dst_language,
            )

        # Handle dict with translations array
        if (
            isinstance(response, dict)
            and "translations" in response
            and isinstance(response["translations"], list)
        ):
            return self.update_translations_from_list(
                translations_list=response["translations"],
                phrases=phrases,
                indices=indices,
                translations=translations,
                progress=progress,
                dst_language=dst_language,
            )

        if DEBUG:
            print(f"Unexpected response format: {response}")
        return 0

    async def translate_standard(
        self,
        phrases: list[str],
        indices: list[int],
        translations: list[dict[str, str]],
        progress: dict[str, str],
        model: str,
        base_language: str,
        dst_language: str,
        prompt: str,
        context: Optional[str] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
    ) -> int:
        """Process a batch of phrases for translation"""
        # Get common setup
        driver, batch_prompt = await self.setup_translation(
            phrases=phrases,
            indices=indices,
            translations=translations,
            model=model,
            base_language=base_language,
            dst_language=dst_language,
            prompt=prompt,
            context=context,
            method_name="standard",
        )
        if not driver:
            return 0

        # Get the translation response
        try:
            response = await driver.translate_async(
                batch_prompt, delay_seconds=delay_seconds, max_retries=max_retries
            )
        except Exception as e:
            if DEBUG:
                print(f"Error processing batch: {e}")
                print("Skipping this batch...")
            return 0

        # Parse and handle the response using the same method as structured and function calls
        try:
            return self.handle_response_format(
                response,
                phrases,
                indices,
                translations,
                progress,
                dst_language,
            )
        except Exception as e:
            if DEBUG:
                print(f"Error processing batch: {e}")
            print("Skipping this batch...")
            return 0

    async def translate_structured(
        self,
        phrases: list[str],
        indices: list[int],
        translations: list[dict[str, str]],
        progress: dict[str, str],
        model: str,
        base_language: str,
        dst_language: str,
        prompt: str,
        context: Optional[str] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
    ) -> int:
        """Process a batch of phrases using structured output"""
        # Get common setup
        driver, batch_prompt = await self.setup_translation(
            phrases=phrases,
            indices=indices,
            translations=translations,
            model=model,
            base_language=base_language,
            dst_language=dst_language,
            prompt=prompt,
            context=context,
            method_name="structured",
        )
        if not driver:
            return 0

        # Get the translation response using structured output
        try:
            if DEBUG:
                print(f"DEBUG: Calling translate_structured_async for model {model}")
            response = await driver.translate_structured_async(
                batch_prompt,
                delay_seconds=delay_seconds,
                max_retries=max_retries,
            )
            if DEBUG:
                print(
                    f"DEBUG: Response type: {type(response)}, value: {repr(response)[:200]}"
                )

            return self.handle_response_format(
                response,
                phrases,
                indices,
                translations,
                progress,
                dst_language,
            )
        except Exception as e:
            if DEBUG:
                print(f"Error from structured output call: {e}")
                print(f"Failed to process batch using structured output: {e}")
            return 0

    async def translate_function(
        self,
        phrases: list[str],
        indices: list[int],
        translations: list[dict[str, str]],
        progress: dict[str, str],
        model: str,
        base_language: str,
        dst_language: str,
        prompt: str,
        context: Optional[str] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
    ) -> int:
        """Process a batch of phrases using function calling"""
        # Get common setup
        driver, batch_prompt = await self.setup_translation(
            phrases=phrases,
            indices=indices,
            translations=translations,
            model=model,
            base_language=base_language,
            dst_language=dst_language,
            prompt=prompt,
            context=context,
            method_name="function",
        )
        if not driver:
            return 0

        # Get the translation response using function calling
        try:
            response = await driver.translate_function_async(
                prompt=batch_prompt,
                delay_seconds=delay_seconds,
                max_retries=max_retries,
            )

            # Handle the response
            if isinstance(response, dict) and "arguments" in response:
                try:
                    # The arguments might be a JSON string that needs parsing
                    if isinstance(response["arguments"], str):
                        args = json.loads(response["arguments"])
                    else:
                        args = response["arguments"]

                    return self.handle_response_format(
                        args,
                        phrases,
                        indices,
                        translations,
                        progress,
                        dst_language,
                    )
                except Exception as e:
                    if DEBUG:
                        print(f"Unexpected function arguments format: {e}")
                    return 0
            else:
                if DEBUG:
                    print(f"Unexpected response format: {response}")
                return 0
        except Exception as e:
            if DEBUG:
                print(f"Error from function call: {e}")
                print("Function call translation failed")
            return 0
