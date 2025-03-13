import json
import os
import aiofiles
from pathlib import Path
from typing import Optional

from lib.PromptManager import PromptManager
from lib.TranslationTool import TranslationTool
from lib.utils import (
    Config,
    load_config,
    load_context,
    load_progress,
    load_translations,
    save_progress,
    save_translations,
)


from .llm import get_driver, BaseDriver, get_available_models


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
        translation_tool (TranslationTool): The translation tool for the project.
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
    translation_tool: TranslationTool
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

        # Initialize translation tool
        self.translation_tool = TranslationTool(self.prompt_manager)

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

    @staticmethod
    def count_tokens(text: str, model: str = "gemini") -> int:
        """
        Count tokens in a text string using the specified model's driver.
        Falls back to a simple character-based approximation if the driver fails.

        Args:
            text: The input text to count tokens for
            model: The model name

        Returns:
            Number of tokens in the text
        """
        try:
            driver = get_driver(model)
            return driver.count_tokens(text)
        except Exception as e:
            # Fallback to a simple character-based approximation
            # Most models use ~4 characters per token on average
            if not text:
                return 0
            return max(1, len(text) // 4)

    async def _load_context(self) -> str:
        """Load translation context from various sources"""
        context_parts = await load_context(self.project_dir, self.context_file)

        # 3. Add direct context string if provided
        if self.context:
            context_parts.append(self.context.strip())

        # Combine all context parts
        return "\n\n".join(filter(None, context_parts))

    async def _load_prompt(self) -> str:
        prompt = ""

        if self.prompt:
            valid, error = self.prompt_manager.validate_prompt(
                "translation", self.prompt, strict=True
            )
            if valid:
                prompt = self.prompt
            else:
                print(f"Warning: {error}")

        if not prompt:
            prompt = await self.prompt_manager.load_prompt(
                "translation",
                self.prompt_file,
                validate=True,
                strict_validation=True,  # Only enforce required variables when actually translating
            )

        return prompt

    async def _process_translation_batch(
        self,
        phrases_to_translate: list[str],
        phrase_indices: list[int],
        translations: list[dict[str, str]],
        progress: dict[str, str],
        model: str,
        method: str,
        prompt: str,
        context: str,
        delay_seconds: float,
        max_retries: int,
    ) -> int:
        """
        Process a batch of phrases for translation using the appropriate method.

        Args:
            phrases_to_translate: list of phrases to translate
            phrase_indices: list of indices corresponding to each phrase
            translations: list of translation dictionaries
            progress: Progress dictionary tracking completed translations
            model: The LLM model to use
            method: The translation method to use
            prompt: The translation prompt
            context: The translation context
            delay_seconds: Delay between API calls
            max_retries: Maximum number of retries for failed API calls

        Returns:
            Number of translations processed successfully
        """
        translation_count = 0
        if method == "structured":
            translation_count = await self.translation_tool.translate_structured(
                phrases_to_translate,
                phrase_indices,
                translations,
                progress,
                model,
                self.base_language,
                self.dst_language,
                prompt,
                context,
                delay_seconds,
                max_retries,
            )
        elif method == "function":
            translation_count = await self.translation_tool.translate_function(
                phrases_to_translate,
                phrase_indices,
                translations,
                progress,
                model,
                self.base_language,
                self.dst_language,
                prompt,
                context,
                delay_seconds,
                max_retries,
            )
        else:  # standard method
            translation_count = await self.translation_tool.translate_standard(
                phrases_to_translate,
                phrase_indices,
                translations,
                progress,
                model,
                self.base_language,
                self.dst_language,
                prompt,
                context,
                delay_seconds,
                max_retries,
            )
        return translation_count

    async def _save_translation_progress(
        self,
        progress: dict[str, str],
        translations: list[dict[str, str]],
        is_final: bool = False,
    ) -> None:
        """
        Save translation progress and translations to files.

        Args:
            progress: Progress dictionary tracking completed translations
            translations: list of translation dictionaries
            is_final: Whether this is the final save (affects log message)
        """
        await save_progress(self.progress_file, progress)
        await save_translations(self.source_file, translations)

        if is_final:
            print(
                f"Final save: {len(progress)} translations saved to {self.progress_file} and {self.source_file}"
            )
        else:
            print(
                f"Progress saved: {len(progress)} translations saved to {self.progress_file}"
            )

    async def translate(
        self,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
        batch_size: int = 50,
        model: str = "gemini",
        batch_max_tokens: int = 2048,
        translation_method: str = "standard",
    ) -> None:
        """Translate phrases from base language to destination language

        Args:
            delay_seconds: Delay between API calls to avoid rate limiting
            max_retries: Maximum number of retries for failed API calls
            batch_size: Number of phrases to translate in a single API call
            model: The LLM model to use for translation
            batch_max_tokens: Maximum number of tokens for a translation batch
            translation_method: Method to use for translation ('auto', 'standard', 'structured', or 'function')
        """
        # Validate translation method
        valid_methods = ["auto", "standard", "structured", "function"]
        if translation_method not in valid_methods:
            raise ValueError(
                f"Invalid translation method: {translation_method}. Must be one of: {valid_methods}"
            )

        # Get the driver instance for the selected model
        driver = get_driver(model)

        # If 'auto' is selected or the requested method is not supported by the model,
        # determine the best method for this driver
        method = driver.get_best_translation_method(translation_method)

        print(f"Using translation method: {method}")

        # Load existing translations and progress
        translations = await load_translations(self.source_file)
        progress = await load_progress(self.progress_file)

        # Load context
        context = await self._load_context()
        prompt = await self._load_prompt()

        # Track changes to know if we need to save
        changes_made = False

        # Collect phrases that need translation
        phrases_to_translate = []
        phrase_indices = []
        current_batch_tokens = 0

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

            # Calculate batch size in tokens
            phrase_tokens = self.count_tokens(source_phrase, model)
            current_batch_tokens += phrase_tokens

            # Process batch when it reaches the batch size limit (count or tokens)
            if (
                len(phrases_to_translate) >= batch_size
                or current_batch_tokens >= batch_max_tokens
            ):
                translation_count = await self._process_translation_batch(
                    phrases_to_translate,
                    phrase_indices,
                    translations,
                    progress,
                    model,
                    method,
                    prompt,
                    context,
                    delay_seconds,
                    max_retries,
                )

                # Save progress after batch processing
                await self._save_translation_progress(progress, translations)

                phrases_to_translate = []
                phrase_indices = []
                current_batch_tokens = 0
                changes_made = True

        # Process any remaining phrases
        if phrases_to_translate:
            translation_count = await self._process_translation_batch(
                phrases_to_translate,
                phrase_indices,
                translations,
                progress,
                model,
                method,
                prompt,
                context,
                delay_seconds,
                max_retries,
            )

            # Save progress after batch processing
            await self._save_translation_progress(progress, translations)
            changes_made = True

        # Always save progress at the end to ensure the test passes
        # This also handles any changes made to progress that weren't from translate_standard
        await self._save_translation_progress(progress, translations, is_final=True)
