import json
import re
import os
from typing import Optional, Union
from .llm import BaseDriver, get_driver, get_available_models
from .PromptManager import PromptManager

DEBUG = os.environ.get("TRADUSCO_DEBUG")


class TranslationTool:
    """
    A class for handling the translation functionality.

    This class is responsible for:
    1. Creating translation prompts
    2. Parsing translation responses
    3. Processing translations
    """

    def __init__(self, prompt_manager: PromptManager):
        """
        Initialize the TranslationTool.

        Args:
            prompt_manager: PromptManager instance for loading and formatting prompts
        """
        self.prompt_manager = prompt_manager

    async def create_prompt(
        self,
        phrases: list[tuple[str, str | None]],
        base_language: str,
        dst_language: str,
        prompt: str,
        context: Optional[str] = None,
    ) -> str:
        """Create a prompt for translation using JSON format"""
        # Create a list of phrases and a separate context mapping
        phrases_to_translate = []
        phrase_contexts = {}

        for phrase, context in phrases:
            phrases_to_translate.append(phrase)
            if context:  # Only include phrases that have context
                phrase_contexts[phrase] = context

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

    def merge_translations(
        self,
        translations_list: list[str],
        phrases: list[tuple[str, str | None]],
    ) -> dict[str, str]:
        """
        Update translations from a list of translations (in the same order as phrases).

        Args:
            translations_list: List of translations
            phrases: List of original phrases

        Returns:
            mapping of phrases to translations
        """

        result = {}

        for i, translation in enumerate(translations_list):
            if (
                i < len(phrases) and translation.strip()
            ):  # Only update if we have a non-empty translation
                result[phrases[i][0]] = translation

                if DEBUG:
                    print(f"Translated: {phrases[i]} -> {translation}")
            elif i < len(phrases) and DEBUG:
                print(f"Warning: Empty translation for '{phrases[i]}'")

        return result

    async def setup(
        self,
        phrases: list[tuple[str, str | None]],
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
            phrases: List of [phrase, context] tuples
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
        driver = get_driver(model)
        if not driver:
            if DEBUG:
                print(f"Warning: Could not get driver for model {model}")
            return None, ""

        # Create the batch prompt
        batch_prompt = await self.create_prompt(
            phrases=phrases,
            base_language=base_language,
            dst_language=dst_language,
            prompt=prompt,
            context=context,
        )

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

    def handle_response(
        self,
        response: Union[str, dict, list],
        phrases: list[tuple[str, str | None]],
    ) -> dict[str, str] | None:
        """
        Handle translation response format and update translations.
        Expects either a list of translations or a dict with a translations array.

        Args:
            response: The response from the translation service
            phrases: List of original phrases

        Returns:
            Mapping of phrases to translations
        """
        if isinstance(response, str):
            # First extract JSON from code blocks if present
            json_str = self.extract_json_from_response(response)
            # Then try to parse it as JSON
            try:
                parsed_response = json.loads(json_str)
                return self.merge_translations(
                    translations_list=parsed_response,
                    phrases=phrases,
                )
            except json.JSONDecodeError:
                if DEBUG:
                    print("Invalid JSON response received")
                return None

        # Handle list of translations
        if isinstance(response, list):
            return self.merge_translations(
                translations_list=response,
                phrases=phrases,
            )

        # Handle dict with translations array
        if (
            isinstance(response, dict)
            and "translations" in response
            and isinstance(response["translations"], list)
        ):
            return self.merge_translations(
                translations_list=response["translations"],
                phrases=phrases,
            )

        if DEBUG:
            print(f"Unexpected response format: {response}")
        return None

    async def translate_standard(
        self,
        phrases: list[tuple[str, str | None]],
        model: str,
        base_language: str,
        dst_language: str,
        prompt: str,
        context: Optional[str] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
    ) -> dict[str, str] | None:
        """Process a batch of phrases for translation"""
        # Get common setup
        driver, batch_prompt = await self.setup(
            phrases=phrases,
            model=model,
            base_language=base_language,
            dst_language=dst_language,
            prompt=prompt,
            context=context,
            method_name="standard",
        )
        if not driver:
            if DEBUG:
                print("Skipping this batch...")
            return None

        # Get the translation response
        try:
            response = await driver.translate_async(
                batch_prompt, delay_seconds=delay_seconds, max_retries=max_retries
            )
        except Exception as e:
            if DEBUG:
                print(f"Error processing batch: {e}")
                print("Skipping this batch...")
            return None

        # Parse and handle the response using the same method as structured and function calls
        try:
            return self.handle_response(
                response,
                phrases,
            )
        except Exception as e:
            if DEBUG:
                print(f"Error processing batch: {e}")
            print("Skipping this batch...")
            return None

    async def translate_structured(
        self,
        phrases: list[tuple[str, str | None]],
        model: str,
        base_language: str,
        dst_language: str,
        prompt: str,
        context: Optional[str] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
    ) -> dict[str, str] | None:
        """Process a batch of phrases using structured output"""
        # Get common setup
        driver, batch_prompt = await self.setup(
            phrases=phrases,
            model=model,
            base_language=base_language,
            dst_language=dst_language,
            prompt=prompt,
            context=context,
            method_name="structured",
        )
        if not driver:
            if DEBUG:
                print("Skipping this batch...")
            return None

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

            return self.handle_response(
                response,
                phrases,
            )
        except Exception as e:
            if DEBUG:
                print(f"Error from structured output call: {e}")
                print(f"Failed to process batch using structured output: {e}")
            return None

    async def translate_function(
        self,
        phrases: list[tuple[str, str | None]],
        model: str,
        base_language: str,
        dst_language: str,
        prompt: str,
        context: Optional[str] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
    ) -> dict[str, str] | None:
        """Process a batch of phrases using function calling"""
        # Get common setup
        driver, batch_prompt = await self.setup(
            phrases=phrases,
            model=model,
            base_language=base_language,
            dst_language=dst_language,
            prompt=prompt,
            context=context,
            method_name="function",
        )
        if not driver:
            if DEBUG:
                print("Skipping this batch...")
            return None

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

                    return self.handle_response(
                        args,
                        phrases,
                    )
                except Exception as e:
                    if DEBUG:
                        print(f"Unexpected function arguments format: {e}")
                    return None
            else:
                if DEBUG:
                    print(f"Unexpected response format: {response}")
                return None
        except Exception as e:
            if DEBUG:
                print(f"Error from function call: {e}")
                print("Function call translation failed")
            return None
