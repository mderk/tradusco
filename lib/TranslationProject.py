import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from .llm import get_llm, BaseDriver


class TranslationProject:
    def __init__(
        self,
        project_name: str,
        dst_language: str,
        batch_prompt_file: str | None = None,
        single_prompt_file: str | None = None,
    ):
        self.project_name = project_name
        self.dst_language = dst_language
        self.project_dir = Path(f"projects/{project_name}")
        self.config_path = self.project_dir / "config.json"
        self.config = self._load_config()

        # Set prompt file paths
        self.batch_prompt_file = batch_prompt_file
        self.single_prompt_file = single_prompt_file

        # Load default prompts
        self.default_batch_prompt = self._load_default_prompt("batch_prompt.txt")
        self.default_single_prompt = self._load_default_prompt("single_prompt.txt")

        if self.dst_language not in self.config["languages"]:
            raise ValueError(f"Language {dst_language} not found in project config")

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

    def _load_config(self) -> dict:
        """Load the project configuration from config.json"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_progress(self) -> dict[str, str]:
        """Load the translation progress from progress.json"""
        if not self.progress_file.exists():
            return {}

        with open(self.progress_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_progress(self, progress: dict[str, str]) -> None:
        """Save the translation progress to progress.json"""
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def _load_translations(self) -> list[dict[str, str]]:
        """Load translations from the CSV file"""
        if not self.source_file.exists():
            raise FileNotFoundError(f"Source file not found: {self.source_file}")

        with open(self.source_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _save_translations(self, translations: list[dict[str, str]]) -> None:
        """Save translations to the CSV file"""
        if not translations:
            return

        fieldnames = list(translations[0].keys())

        with open(self.source_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(translations)

    def _load_default_prompt(self, prompt_filename: str) -> str:
        """Load a default prompt from the prompts directory"""
        prompt_path = Path(__file__).parent / "prompts" / prompt_filename
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Warning: Default prompt file {prompt_filename} not found.")
            return ""

    def _load_custom_prompt(self, prompt_file: str | None) -> str:
        """Load a custom prompt from the specified file path"""
        if not prompt_file:
            return ""

        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            print(
                f"Warning: Custom prompt file {prompt_file} not found. Using default prompt."
            )
            return ""

    def _create_batch_prompt(self, phrases: list[str]) -> str:
        """Create a prompt for batch translation"""
        phrases_text = "\n".join(
            [f"{i+1}. {phrase}" for i, phrase in enumerate(phrases)]
        )

        # Try to load custom prompt first, fall back to default
        prompt_template = (
            self._load_custom_prompt(self.batch_prompt_file)
            or self.default_batch_prompt
        )

        # Format the prompt template with the required variables
        return prompt_template.format(
            base_language=self.base_language,
            dst_language=self.dst_language,
            phrases_text=phrases_text,
        )

    def _parse_batch_response(
        self, response: str, original_phrases: list[str]
    ) -> dict[str, str]:
        """Parse the batch translation response into a dictionary"""
        translations = {}
        lines = response.strip().split("\n")

        # Filter out empty lines and ensure each line starts with a number
        valid_lines = [
            line.strip() for line in lines if line.strip() and line.strip()[0].isdigit()
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

    def translate(
        self, delay_seconds: float = 1.0, max_retries: int = 3, batch_size: int = 50
    ) -> None:
        """Translate phrases from base language to destination language

        Args:
            delay_seconds: Delay between API calls to avoid rate limiting
            max_retries: Maximum number of retries for failed API calls
            batch_size: Number of phrases to translate in a single API call
        """
        # Load existing translations and progress
        translations = self._load_translations()
        progress = self._load_progress()

        # Initialize the Gemini driver
        driver = get_llm("gemini")

        # Track changes to know if we need to save
        changes_made = False

        # Save progress periodically
        last_save_time = time.time()
        save_interval = 60  # Save every 60 seconds

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
                self._process_batch(
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

                # Save progress periodically
                current_time = time.time()
                if current_time - last_save_time > save_interval:
                    self._save_progress(progress)
                    self._save_translations(translations)
                    print(f"Periodic save: {len(progress)} translations saved")
                    last_save_time = current_time

        # Process any remaining phrases
        if phrases_to_translate:
            self._process_batch(
                phrases_to_translate,
                phrase_indices,
                translations,
                progress,
                driver,
                delay_seconds,
                max_retries,
            )
            changes_made = True

        # Save changes if any were made
        if changes_made:
            self._save_progress(progress)
            self._save_translations(translations)
            print(
                f"Saved {len(progress)} translations to {self.progress_file} and {self.source_file}"
            )

    def _process_batch(
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

        # Create the batch prompt
        prompt_text = self._create_batch_prompt(phrases)

        try:
            # Send the batch to the LLM using the GeminiDriver
            response = driver.translate_batch(prompt_text, delay_seconds, max_retries)

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
        except Exception as e:
            print(f"Error translating batch: {e}")
            # Try to translate phrases individually as fallback
            print("Falling back to individual translation...")
            self._process_individual_fallback(
                phrases, indices, translations, progress, driver, delay_seconds
            )

    def _process_individual_fallback(
        self,
        phrases: list[str],
        indices: list[int],
        translations: list[dict[str, str]],
        progress: dict[str, str],
        driver: BaseDriver,
        delay_seconds: float,
    ) -> None:
        """Process phrases individually as a fallback when batch processing fails"""
        for i, phrase in enumerate(phrases):
            try:
                print(f"Individual translation: {phrase}")

                # Try to load custom prompt first, fall back to default
                prompt_template = (
                    self._load_custom_prompt(self.single_prompt_file)
                    or self.default_single_prompt
                )

                # Format the prompt template with the required variables
                prompt = prompt_template.format(
                    base_language=self.base_language,
                    dst_language=self.dst_language,
                    phrase=phrase,
                )

                # Send to LLM using the GeminiDriver
                response = driver.translate_single(prompt, delay_seconds)

                # Clean up response (remove quotes, etc.)
                translation = response.strip().strip("\"'").strip()

                # Update translations and progress
                translations[indices[i]][self.dst_language] = translation
                progress[phrase] = translation

                print(f"Translated: {phrase} -> {translation}")
            except Exception as e:
                print(f"Error translating '{phrase}' individually: {e}")
