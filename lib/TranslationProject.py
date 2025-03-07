import csv
import json
import os
import re
import aiofiles
from pathlib import Path
from io import StringIO
from typing import Optional, Any


from .llm import get_driver, BaseDriver, get_available_models


class TranslationProject:
    def __init__(
        self,
        project_name: str,
        dst_language: str,
        prompt_file: Optional[str] = None,
        context: Optional[str] = None,
        context_file: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        default_prompt: str = "",
    ):
        self.project_name = project_name
        self.dst_language = dst_language
        self.project_dir = Path(f"projects/{project_name}")
        self.config_path = self.project_dir / "config.json"
        self.prompt_file = prompt_file
        self.context = context
        self.context_file = context_file

        # These will be loaded by create method
        self.config = config or {}
        self.default_prompt = default_prompt

        if self.config and self.dst_language not in self.config["languages"]:
            raise ValueError(f"Language {dst_language} not found in project config")

        if self.config:
            self.base_language = self.config["baseLanguage"]
            self.source_file = self.project_dir / self.config["sourceFile"]
            self.progress_dir = self.project_dir / self.dst_language
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
        """Async factory method to create and initialize a TranslationProject"""
        # Create a minimal instance first
        instance = cls(project_name, dst_language, prompt_file, context, context_file)

        # Load config
        config = await instance._load_config()

        # Load default prompt
        default_prompt = await instance._load_default_prompt("prompt.txt")

        # Create a fully initialized instance
        return cls(
            project_name,
            dst_language,
            prompt_file,
            context,
            context_file,
            config=config,
            default_prompt=default_prompt,
        )

    @staticmethod
    def get_available_models() -> list[str]:
        """Get a list of available models"""
        return get_available_models()

    async def _load_config(self) -> dict:
        """Load the project configuration from config.json"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        async with aiofiles.open(self.config_path, "r", encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)

    async def _load_progress(self) -> dict[str, str]:
        """Load the translation progress from progress.json"""
        if not self.progress_file.exists():
            return {}

        async with aiofiles.open(self.progress_file, "r", encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)

    async def _save_progress(self, progress: dict[str, str]) -> None:
        """Save the translation progress to progress.json"""
        async with aiofiles.open(self.progress_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(progress, ensure_ascii=False, indent=2))

    async def _load_translations(self) -> list[dict[str, str]]:
        """Load translations from the CSV file"""
        if not self.source_file.exists():
            raise FileNotFoundError(f"Source file not found: {self.source_file}")

        async with aiofiles.open(
            self.source_file, "r", newline="", encoding="utf-8"
        ) as f:
            content = await f.read()
            # Use StringIO to properly handle CSV with potential multiline fields
            csv_file = StringIO(content)
            reader = csv.DictReader(csv_file)
            return list(reader)

    async def _save_translations(self, translations: list[dict[str, str]]) -> None:
        """Save translations to the CSV file"""
        if not translations:
            return

        fieldnames = list(translations[0].keys())

        # Use StringIO to write CSV to a string
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(translations)

        content = output.getvalue()

        async with aiofiles.open(
            self.source_file, "w", newline="", encoding="utf-8"
        ) as f:
            await f.write(content)

    async def _load_default_prompt(self, prompt_filename: str) -> str:
        """Load a default prompt from the prompts directory"""
        prompt_path = Path(__file__).parent.parent / "prompts" / prompt_filename
        if not prompt_path.exists():
            print(f"Warning: Default prompt file {prompt_filename} not found.")
            return ""

        async with aiofiles.open(prompt_path, "r", encoding="utf-8") as f:
            content = await f.read()
            return content.strip()

    async def _load_custom_prompt(self, prompt_file: str | None) -> str:
        """Load a custom prompt from the specified file path"""
        if not prompt_file:
            return ""

        if not os.path.exists(prompt_file):
            print(
                f"Warning: Custom prompt file {prompt_file} not found. Using default prompt."
            )
            return ""

        async with aiofiles.open(prompt_file, "r", encoding="utf-8") as f:
            content = await f.read()
            return content.strip()

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

        # Try to load custom prompt first, fall back to default
        custom_prompt = await self._load_custom_prompt(self.prompt_file)
        prompt_template = custom_prompt or self.default_prompt

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
        return prompt_template.format(
            base_language=self.base_language,
            dst_language=self.dst_language,
            phrases_json=phrases_json,
            context=context_section,
            phrase_contexts=phrase_contexts_section,
        )

    def _parse_batch_response(
        self, response: str, original_phrases: list[str]
    ) -> dict[str, str]:
        """Parse the batch translation response from JSON format"""
        translations = {}

        try:
            # Try to extract JSON from the response
            # First, look for JSON content between triple backticks

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
            print(f"Error parsing JSON response: {e}")
            print("Falling back to line-by-line parsing...")

            # Fallback to the original line-by-line parsing method
            lines = response.strip().split("\n")
            valid_lines = [
                line.strip()
                for line in lines
                if line.strip() and line.strip()[0].isdigit()
            ]

            for i, line in enumerate(valid_lines):
                if i >= len(original_phrases):
                    break

                # Extract the translation part (after the number and dot)
                parts = line.split(".", 1)
                if len(parts) < 2:
                    continue

                translation = parts[1].strip()
                original = original_phrases[i]
                translations[original] = translation

        return translations

    async def translate(
        self,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
        batch_size: int = 50,
        model: str = "gemini",
    ) -> None:
        """Translate phrases from base language to destination language

        Args:
            delay_seconds: Delay between API calls to avoid rate limiting
            max_retries: Maximum number of retries for failed API calls
            batch_size: Number of phrases to translate in a single API call
            model: The LLM model to use for translation
        """
        # Load existing translations and progress
        translations = await self._load_translations()
        progress = await self._load_progress()

        # Initialize the LLM driver with the specified model
        driver = get_driver(model)

        # Track changes to know if we need to save
        changes_made = False

        # Collect phrases that need translation
        phrases_to_translate = []
        phrase_indices = []

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

            # Process batch when it reaches the batch size
            if len(phrases_to_translate) >= batch_size:
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
        await self._save_progress(progress)
        await self._save_translations(translations)
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

            # Parse the response
            batch_translations = self._parse_batch_response(response, phrases)

            # Update translations and progress
            for i, phrase in enumerate(phrases):
                if phrase in batch_translations:
                    translation = batch_translations[phrase]
                    translations[indices[i]][self.dst_language] = translation
                    progress[phrase] = translation
                    print(f"Translated: {phrase} -> {translation}")
                else:
                    print(f"Warning: No translation found for '{phrase}'")

            # Save progress after every LLM query
            await self._save_progress(progress)
            await self._save_translations(translations)
            print(
                f"Progress saved: {len(progress)} translations saved to {self.progress_file}"
            )

        except Exception as e:
            print(f"Error translating batch: {e}")
            print("Failed to translate batch. Skipping these phrases.")
