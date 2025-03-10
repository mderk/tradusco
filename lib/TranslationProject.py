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

        # Load context
        context = await self._load_context()

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
                await self.translation_tool.process_batch(
                    phrases_to_translate,
                    phrase_indices,
                    translations,
                    progress,
                    model,
                    self.base_language,
                    self.dst_language,
                    self.prompt,
                    self.prompt_file,
                    context,
                    delay_seconds,
                    max_retries,
                )

                # Save progress after batch processing
                await save_progress(self.progress_file, progress)
                await save_translations(self.source_file, translations)
                print(
                    f"Progress saved: {len(progress)} translations saved to {self.progress_file}"
                )

                phrases_to_translate = []
                phrase_indices = []
                current_batch_bytes = 0
                changes_made = True

        # Process any remaining phrases
        if phrases_to_translate:
            await self.translation_tool.process_batch(
                phrases_to_translate,
                phrase_indices,
                translations,
                progress,
                model,
                self.base_language,
                self.dst_language,
                self.prompt,
                self.prompt_file,
                context,
                delay_seconds,
                max_retries,
            )

            # Save progress after batch processing
            await save_progress(self.progress_file, progress)
            await save_translations(self.source_file, translations)
            print(
                f"Progress saved: {len(progress)} translations saved to {self.progress_file}"
            )

            changes_made = True

        # Always save progress at the end to ensure the test passes
        # This also handles any changes made to progress that weren't from process_batch
        await save_progress(self.progress_file, progress)
        await save_translations(self.source_file, translations)
        print(
            f"Final save: {len(progress)} translations saved to {self.progress_file} and {self.source_file}"
        )

    # Test adapter methods for backward compatibility

    async def _create_batch_prompt(
        self, phrases: list[str], translations: list[dict[str, str]], indices: list[int]
    ) -> str:
        """Test adapter for backward compatibility"""
        context = await self._load_context()
        return await self.translation_tool.create_batch_prompt(
            phrases,
            translations,
            indices,
            self.base_language,
            self.dst_language,
            self.prompt,
            self.prompt_file,
            context,
        )

    async def _fix_invalid_json(self, invalid_json: str, driver: BaseDriver) -> str:
        """Test adapter for backward compatibility"""
        return await self.translation_tool.fix_invalid_json(invalid_json, driver)

    def _parse_batch_response(
        self,
        response: str,
        original_phrases: list[str],
    ) -> dict[str, str]:
        """Test adapter for backward compatibility"""
        return self.translation_tool.parse_batch_response(response, original_phrases)

    def _update_translations_from_batch(
        self,
        batch_translations: dict[str, str],
        phrases: list[str],
        indices: list[int],
        translations: list[dict[str, str]],
        progress: dict[str, str],
    ) -> int:
        """Test adapter for backward compatibility"""
        return self.translation_tool.update_translations_from_batch(
            batch_translations,
            phrases,
            indices,
            translations,
            progress,
            self.dst_language,
        )

    async def _process_batch(
        self,
        phrases: list[str],
        indices: list[int],
        translations: list[dict[str, str]],
        progress: dict[str, str],
        model: str,
        delay_seconds: float,
        max_retries: int,
    ) -> None:
        """Test adapter for backward compatibility"""
        context = await self._load_context()
        await self.translation_tool.process_batch(
            phrases,
            indices,
            translations,
            progress,
            model,
            self.base_language,
            self.dst_language,
            self.prompt,
            self.prompt_file,
            context,
            delay_seconds,
            max_retries,
        )
