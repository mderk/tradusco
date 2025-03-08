import csv
import json
import os
import re
import aiofiles
from pathlib import Path
from io import StringIO
from typing import Optional, Any, TypedDict

from lib.PromptManager import PromptManager
from lib.utils import (
    Config,
    load_config,
    load_progress,
    load_translations,
    save_progress,
    save_translations,
)


from .llm import get_driver, BaseDriver, get_available_models


class InvalidJSONException(Exception):
    """Exception raised when invalid JSON is encountered during parsing."""

    def __init__(self, message: str, json_str: str):
        self.message = message
        self.json_str = json_str
        super().__init__(f"{message}: {json_str[:100]}...")


class TranslationProject:
    """
    A class for managing translation projects.

    Attributes:
        project_name (str): The name of the project.
        project_dir (Path): The directory of the project.
        config (Config): The configuration of the project.
        dst_language (str): The destination language of the project.
        prompt_file (Optional[str]): The path to the prompt file.
        context (Optional[str]): The context of the project.
        context_file (Optional[str]): The path to the context file.

        prompt_manager (PromptManager): The prompt manager for the project.
        base_language (str): The base language of the project.
        source_file (Path): The path to the source file.
        progress_dir (Path): The directory of the progress file.
        progress_file (Path): The path to the progress file.
    """

    project_name: str
    project_dir: Path
    config: Config
    dst_language: str
    prompt: Optional[str]
    prompt_file: Optional[str]
    context: Optional[str]
    context_file: Optional[str]

    prompt_manager: PromptManager
    base_language: str
    source_file: Path
    progress_dir: Path
    progress_file: Path

    def __init__(
        self,
        project_name: str,
        project_dir: Path,
        config: Config,
        dst_language: str,
        prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        context: Optional[str] = None,
        context_file: Optional[str] = None,
    ):
        self.project_name = project_name
        self.project_dir = project_dir
        self.config = config
        self.dst_language = dst_language
        self.prompt = prompt
        self.prompt_file = prompt_file
        self.context = context
        self.context_file = context_file

        # Initialize prompt manager
        self.prompt_manager = PromptManager(project_dir)

        if dst_language not in config.languages:
            raise ValueError(f"Language {dst_language} not found in project config")

        self.base_language = config.baseLanguage
        self.source_file = project_dir / config.sourceFile
        self.progress_dir = project_dir / dst_language
        self.progress_file = self.progress_dir / "progress.json"

        # Create language directory if it doesn't exist
        os.makedirs(self.progress_dir, exist_ok=True)

        # Initialize progress file if it doesn't exist
        if not self.progress_file.exists():
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)

    @classmethod
    async def create(
        cls,
        project_name: str,
        dst_language: str,
        prompt_file: str | None = None,
        context: str | None = None,
        context_file: str | None = None,
    ):
        project_dir = Path(f"projects/{project_name}")
        config_path = project_dir / "config.json"

        # Load config
        config = await load_config(config_path)

        # Create a fully initialized instance
        return cls(
            project_name=project_name,
            project_dir=project_dir,
            config=config,
            dst_language=dst_language,
            prompt=None,  # No direct prompt is provided via create
            prompt_file=prompt_file,
            context=context,
            context_file=context_file,
        )

    @staticmethod
    def get_available_models() -> list[str]:
        """Get a list of available models"""
        return get_available_models()

    async def _load_context(self) -> str:
        """Load translation context from various sources"""
        context_parts = []

        # 1. Check for context.md or context.txt in project directory
        for ext in [".md", ".txt"]:
            context_path = self.project_dir / f"context{ext}"
            try:
                if os.path.exists(context_path):
                    async with aiofiles.open(context_path, "r", encoding="utf-8") as f:
                        content = await f.read()
                        context_parts.append(content.strip())
            except Exception as e:
                print(f"Warning: Error reading context file {context_path}: {e}")

        # 2. Check for context from command line file
        if self.context_file:
            try:
                if os.path.exists(self.context_file):
                    async with aiofiles.open(
                        self.context_file, "r", encoding="utf-8"
                    ) as f:
                        content = await f.read()
                        context_parts.append(content.strip())
                else:
                    print(f"Warning: Context file not found: {self.context_file}")
            except Exception as e:
                print(f"Warning: Error reading context file {self.context_file}: {e}")

        # 3. Add direct context string if provided
        if self.context:
            context_parts.append(self.context.strip())

        # Combine all context parts
        return "\n\n".join(filter(None, context_parts))

    async def _create_batch_prompt(
        self, phrases: list[str], translations: list[dict[str, str]], indices: list[int]
    ) -> str:
        """Create a prompt for batch translation using JSON format"""
        # Create a list of phrases and a separate context mapping
        phrases_to_translate = []
        phrase_contexts = {}

        for i, phrase in enumerate(phrases):
            phrases_to_translate.append(phrase)
            context = translations[indices[i]].get("context", "")
            if context:  # Only include phrases that have context
                phrase_contexts[phrase] = context

        # Encode phrases and contexts as JSON
        phrases_json = json.dumps(phrases_to_translate, ensure_ascii=False, indent=2)
        contexts_json = (
            json.dumps(phrase_contexts, ensure_ascii=False, indent=2)
            if phrase_contexts
            else ""
        )

        prompt_template = ""

        if self.prompt:
            valid, error = self.prompt_manager.validate_prompt(
                "translation", self.prompt, strict=True
            )
            if valid:
                prompt_template = self.prompt
            else:
                print(f"Warning: {error}")

        if not prompt_template:
            prompt_template = await self.prompt_manager.load_prompt(
                "translation",
                self.prompt_file,
                validate=True,
                strict_validation=True,  # Only enforce required variables when actually translating
            )

        # Load global context
        global_context = await self._load_context()
        context_section = (
            f"\nGlobal Translation Context:\n{global_context}\n"
            if global_context
            else ""
        )

        # Add phrase-specific contexts section if any exist
        phrase_contexts_section = (
            f"\nPhrase-specific contexts:\n{contexts_json}\n" if contexts_json else ""
        )

        # Format the prompt template with the required variables
        return self.prompt_manager.format_prompt(
            prompt_template,
            base_language=self.base_language,
            dst_language=self.dst_language,
            phrases_json=phrases_json,
            context=context_section,
            phrase_contexts=phrase_contexts_section,
        )

    async def _fix_invalid_json(self, invalid_json: str, driver: BaseDriver) -> str:
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

    def _parse_batch_response(
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

    def _update_translations_from_batch(
        self,
        batch_translations: dict[str, str],
        phrases: list[str],
        indices: list[int],
        translations: list[dict[str, str]],
        progress: dict[str, str],
    ) -> int:
        """
        Update translations and progress based on batch results

        Args:
            batch_translations: Dictionary mapping phrases to translations
            phrases: List of original phrases being translated
            indices: Indices of phrases in the translations list
            translations: Full translations data structure
            progress: Progress dictionary

        Returns:
            Number of phrases successfully translated
        """
        updates_count = 0
        for i, phrase in enumerate(phrases):
            if phrase in batch_translations:
                translation = batch_translations[phrase]
                translations[indices[i]][self.dst_language] = translation
                progress[phrase] = translation
                print(f"Translated: {phrase} -> {translation}")
                updates_count += 1
            else:
                print(f"Warning: No translation found for '{phrase}'")
        return updates_count

    async def translate(
        self,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
        batch_size: int = 50,
        model: str = "gemini",
        batch_max_bytes: int = 8192,
    ) -> None:
        """Translate phrases from base language to destination language

        Args:
            delay_seconds: Delay between API calls to avoid rate limiting
            max_retries: Maximum number of retries for failed API calls
            batch_size: Number of phrases to translate in a single API call
            model: The LLM model to use for translation
            batch_max_bytes: Maximum size in bytes for a translation batch
        """
        # Load existing translations and progress
        translations = await load_translations(self.source_file)
        progress = await load_progress(self.progress_file)

        # Initialize the LLM driver with the specified model
        driver = get_driver(model)

        # Track changes to know if we need to save
        changes_made = False

        # Collect phrases that need translation
        phrases_to_translate = []
        phrase_indices = []
        current_batch_bytes = 0

        for i, row in enumerate(translations):
            source_phrase = row[self.base_language]

            # Skip empty source phrases
            if not source_phrase:
                continue

            # Skip already translated phrases
            if row[self.dst_language]:
                # Update progress file if needed
                if source_phrase not in progress:
                    progress[source_phrase] = row[self.dst_language]
                    changes_made = True
                continue

            # Check if we already have a translation in progress
            if source_phrase in progress:
                translation = progress[source_phrase]
                row[self.dst_language] = translation
                changes_made = True
                print(f"Using cached translation for: {source_phrase} -> {translation}")
                continue

            # Add to batch for translation
            phrases_to_translate.append(source_phrase)
            phrase_indices.append(i)

            # Calculate batch size in bytes
            phrase_bytes = len(source_phrase.encode("utf-8"))
            current_batch_bytes += phrase_bytes

            # Process batch when it reaches the batch size limit (count or bytes)
            if (
                len(phrases_to_translate) >= batch_size
                or current_batch_bytes >= batch_max_bytes
            ):
                await self._process_batch(
                    phrases_to_translate,
                    phrase_indices,
                    translations,
                    progress,
                    driver,
                    delay_seconds,
                    max_retries,
                )
                phrases_to_translate = []
                phrase_indices = []
                current_batch_bytes = 0
                changes_made = True

        # Process any remaining phrases
        if phrases_to_translate:
            await self._process_batch(
                phrases_to_translate,
                phrase_indices,
                translations,
                progress,
                driver,
                delay_seconds,
                max_retries,
            )
            changes_made = True

        # Always save progress at the end to ensure the test passes
        # This also handles any changes made to progress that weren't from _process_batch
        await save_progress(self.progress_file, progress)
        await save_translations(self.source_file, translations)
        print(
            f"Final save: {len(progress)} translations saved to {self.progress_file} and {self.source_file}"
        )

    async def _process_batch(
        self,
        phrases: list[str],
        indices: list[int],
        translations: list[dict[str, str]],
        progress: dict[str, str],
        driver: BaseDriver,
        delay_seconds: float,
        max_retries: int,
    ) -> None:
        """Process a batch of phrases for translation"""
        if not phrases:
            return

        print(f"Translating batch of {len(phrases)} phrases...")

        # Create the batch prompt with translations data for context
        prompt_text = await self._create_batch_prompt(phrases, translations, indices)

        try:
            # Send the batch to the LLM using the async translate method
            response = await driver.translate_async(
                prompt_text, delay_seconds, max_retries
            )

            try:
                # Parse the response
                batch_translations = self._parse_batch_response(response, phrases)

                # Update translations and progress
                self._update_translations_from_batch(
                    batch_translations, phrases, indices, translations, progress
                )

            except InvalidJSONException as e:
                print(f"Invalid JSON detected: {e.message}")
                print("Attempting to fix invalid JSON using LLM...")

                # Try to fix the JSON
                fixed_json_str = await self._fix_invalid_json(e.json_str, driver)

                # Try parsing again with the fixed JSON
                try:
                    # Create a new response with the fixed JSON
                    fixed_response = response.replace(e.json_str, fixed_json_str)

                    # Parse the fixed response
                    batch_translations = self._parse_batch_response(
                        fixed_response, phrases
                    )

                    # Update translations and progress
                    update_count = self._update_translations_from_batch(
                        batch_translations, phrases, indices, translations, progress
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

            # Save progress after every LLM query
            await save_progress(self.progress_file, progress)
            await save_translations(self.source_file, translations)
            print(
                f"Progress saved: {len(progress)} translations saved to {self.progress_file}"
            )

        except Exception as e:
            print(f"Error translating batch: {e}")
            print("Failed to translate batch. Skipping these phrases.")
