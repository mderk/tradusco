import asyncio
import time
from typing import Optional

from lib.failure_reporting import (
    batch_error_kind_to_category,
    FailureCategory,
    make_failure_record,
)
from lib.envelope import BatchEnvelope, EnvelopeBuilder
from lib.PromptManager import PromptManager
from lib.TranslationTool import (
    BatchErrorInfo,
    BatchTranslationError,
    classify_llm_error,
    language_ref,
    LanguageRef,
    TranslationTool,
)
from lib.utils import Config
from lib.storage.base import StorageAdapter


from .llm import get_driver, get_available_models


class TranslationProject:
    """
    A class for managing translation projects.

    Attributes:
        project_id (str): The unique identifier of the project.
        config (Config): The configuration of the project.
        dst_languages (list[str]): The destination languages of the project.
        prompt (Optional[str]): Direct prompt string if provided.
        context (Optional[str]): Direct context string if provided.

        storage (StorageAdapter): The storage adapter for data persistence.
        prompt_manager (PromptManager): The prompt manager for the project.
        translation_tool (TranslationTool): The translation tool for the project.
        base_language (str): The base language of the project.
    """

    project_id: str
    config: Config
    dst_languages: list[str]
    language_refs: list[LanguageRef]
    reference_languages: list[str]
    prompt: Optional[str]
    context: Optional[str]

    storage: StorageAdapter
    prompt_manager: PromptManager
    translation_tool: TranslationTool
    base_language: str

    def __init__(
        self,
        project_id: str,
        config: Config,
        dst_languages: list[str],
        storage: StorageAdapter,
        prompt: Optional[str] = None,
        context: Optional[str] = None,
        reference_languages: list[str] | None = None,
    ):
        self.project_id = project_id
        self.config = config
        if not dst_languages:
            raise ValueError("At least one destination language is required")
        self.dst_languages = dst_languages
        self.language_refs = [language_ref(language) for language in dst_languages]
        self.reference_languages = list(reference_languages or [])
        self.storage = storage
        self.prompt = prompt
        self.context = context

        # Initialize prompt manager with storage adapter
        self.prompt_manager = PromptManager(storage, project_id)
        self.translation_tool = TranslationTool(self.prompt_manager)

        for language in dst_languages:
            if language not in config.languages:
                raise ValueError(f"Language {language} not found in project config")
        for language in self.reference_languages:
            if language not in config.languages:
                raise ValueError(
                    f"Reference language {language} not found in project config"
                )
        overlap = set(dst_languages) & set(self.reference_languages)
        if overlap:
            raise ValueError(
                "Reference languages cannot be target languages: "
                + ", ".join(sorted(overlap))
            )

        self.base_language = config.baseLanguage

    @classmethod
    async def create(
        cls,
        project_name: str,
        dst_languages: list[str],
        storage: StorageAdapter,
        context: str | None = None,
        reference_languages: list[str] | None = None,
    ):
        # Load config using storage adapter
        config = await storage.load_config(project_name)

        # Create a fully initialized instance
        return cls(
            project_id=project_name,
            config=config,
            dst_languages=dst_languages,
            storage=storage,
            prompt=None,  # No direct prompt is provided via create
            context=context,
            reference_languages=reference_languages,
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
        except Exception:
            # Fallback to a simple character-based approximation
            # Most models use ~4 characters per token on average
            if not text:
                return 0
            return max(1, len(text) // 4)

    async def _load_context(self) -> str:
        """Load translation context from various sources"""
        context_parts = await self.storage.load_context(self.project_id)
        for language in self.dst_languages:
            parts = await self.storage.load_context(self.project_id, language)
            if parts:
                context_parts.append(f"[{language}]\n" + "\n\n".join(parts))

        # Add direct context string if provided
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
                validate=True,
                strict_validation=True,  # Only enforce required variables when actually translating
            )

        return prompt

    async def _record_failure(
        self,
        *,
        language: str,
        model: str,
        phrase: str,
        category: FailureCategory,
        message: str,
        method: str | None = None,
    ) -> None:
        record = make_failure_record(
            model=model,
            phrase=phrase,
            category=category,
            message=message,
            method=method,
        )
        await self.storage.append_failure(
            self.project_id,
            language,
            record,
        )

    async def _record_batch_failures(
        self,
        phrases: list[tuple[str, str | None]],
        phrase_languages: dict[str, set[str]],
        *,
        model: str,
        method: str,
        info: BatchErrorInfo,
    ) -> None:
        category = batch_error_kind_to_category(info.kind)
        for phrase, _ctx in phrases:
            for language in phrase_languages[phrase]:
                await self._record_failure(
                    language=language,
                    model=model,
                    phrase=phrase,
                    category=category,
                    message=info.message,
                    method=method,
                )

    def _has_missing_phrases(
        self,
        translations: list[dict[str, str]],
        progress: dict[str, dict[str, str]],
        regenerate: bool,
        editorial: dict[str, dict[str, str]] | None = None,
        only_keys: set[str] | None = None,
    ) -> bool:
        for language in self.dst_languages:
            for row in translations:
                source_phrase = row.get(self.base_language) or ""
                if not source_phrase:
                    continue
                if only_keys is not None and source_phrase not in only_keys:
                    continue
                if regenerate and (editorial or {}).get(language, {}).get(source_phrase):
                    continue
                existing_translation = row.get(language) or ""
                if existing_translation and not regenerate:
                    ok, _ = self.translation_tool.validate_translation_text(
                        existing_translation
                    )
                    if ok:
                        continue
                if source_phrase in progress[language] and not regenerate:
                    translation = progress[language][source_phrase]
                    ok, _ = self.translation_tool.validate_translation_text(translation)
                    if ok:
                        continue
                return True
        return False

    async def _apply_translated_batch(
        self,
        translated: dict[str, dict[str, str]],
        *,
        phrase_indices: dict[str, int],
        phrase_languages: dict[str, set[str]],
        progress: dict[str, dict[str, str]],
        translations: list[dict[str, str]],
        model: str,
        method: str,
    ) -> None:
        for language in self.dst_languages:
            language_translations = translated.get(language)
            for phrase, needed_languages in phrase_languages.items():
                if language not in needed_languages:
                    continue
                translation = (
                    language_translations.get(phrase) if language_translations else None
                )
                if translation is None:
                    await self._record_failure(
                        language=language,
                        model=model,
                        phrase=phrase,
                        category="parse_error",
                        message="Missing or invalid translation block",
                        method=method,
                    )
                    continue
                ok, reason = self.translation_tool.validate_placeholders(
                    phrase, translation
                )
                if not ok:
                    print(
                        "Warning: Skipping translation due to placeholder/tag mismatch for: "
                        f"{phrase} ({language})\n{reason}"
                    )
                    await self._record_failure(
                        language=language,
                        model=model,
                        phrase=phrase,
                        category="placeholder_mismatch",
                        message=reason or "placeholder/tag mismatch",
                        method=method,
                    )
                    continue
                ok, reason = self.translation_tool.validate_translation_text(
                    translation
                )
                if not ok:
                    print(
                        "Warning: Skipping translation due to invalid translation text for: "
                        f"{phrase} ({language})\n{reason}"
                    )
                    await self._record_failure(
                        language=language,
                        model=model,
                        phrase=phrase,
                        category="invalid_artifact_rejected",
                        message=reason or "invalid translation text",
                        method=method,
                    )
                    continue
                progress[language][phrase] = translation
                translations[phrase_indices[phrase]][language] = translation

    async def _process_translation_batch(
        self,
        phrases_to_translate: list[tuple[str, str | None]],
        phrase_languages: dict[str, set[str]],
        model: str,
        method: str,
        prompt: str,
        context: str,
        delay_seconds: float,
        max_retries: int,
        request_timeout: float,
        batch_input: BatchEnvelope,
        fallback_model: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> dict[str, dict[str, str]] | None:
        """
        Process a batch of phrases for translation using the appropriate method.

        Args:
            phrases_to_translate: list of tuples of (phrase, context)
            model: The LLM model to use
            method: The translation method to use
            prompt: The translation prompt
            context: The translation context
            delay_seconds: Delay between API calls
            max_retries: Maximum number of retries for failed API calls

        Returns:
            Dictionary of translations
        """

        def _summarize_error(msg: str, limit: int = 220) -> str:
            s = (msg or "").strip().replace("\n", " ")
            return s if len(s) <= limit else (s[: limit - 3] + "...")

        async def _finish_failure(
            info: BatchErrorInfo, *, failed_model: str, failed_method: str
        ) -> None:
            if raise_on_error:
                raise BatchTranslationError(info)
            await self._record_batch_failures(
                phrases_to_translate,
                phrase_languages,
                model=failed_model,
                method=failed_method,
                info=info,
            )

        async def _run_once(
            *, run_model: str, run_method: str
        ) -> dict[str, dict[str, str]] | None:
            if run_method == "structured":
                return await self.translation_tool.translate_structured(
                    phrases_to_translate,
                    run_model,
                    self.base_language,
                    self.language_refs,
                    prompt,
                    context,
                    delay_seconds,
                    max_retries,
                    raise_on_error=True,
                    batch_input=batch_input,
                )
            if run_method == "function":
                return await self.translation_tool.translate_function(
                    phrases_to_translate,
                    run_model,
                    self.base_language,
                    self.language_refs,
                    prompt,
                    context,
                    delay_seconds,
                    max_retries,
                    raise_on_error=True,
                    batch_input=batch_input,
                )
            return await self.translation_tool.translate_standard(
                phrases_to_translate,
                run_model,
                self.base_language,
                self.language_refs,
                prompt,
                context,
                delay_seconds,
                max_retries,
                raise_on_error=True,
                batch_input=batch_input,
            )

        async def _run_timed(
            *, run_model: str, run_method: str
        ) -> dict[str, dict[str, str]] | None:
            try:
                return await asyncio.wait_for(
                    _run_once(run_model=run_model, run_method=run_method),
                    timeout=request_timeout,
                )
            except TimeoutError as error:
                raise BatchTranslationError(
                    BatchErrorInfo(
                        kind="timeout",
                        message=f"Model request exceeded {request_timeout:g} seconds",
                    )
                ) from error

        primary_info: BatchErrorInfo | None = None
        try:
            return await _run_timed(run_model=model, run_method=method)
        except BatchTranslationError as e:
            primary_info = e.info
            print(
                "Batch failed "
                f"(kind={primary_info.kind}, status={primary_info.status_code}) "
                f"model={model} method={method}: {_summarize_error(primary_info.message)}"
            )
        except Exception as e:
            primary_info = classify_llm_error(e)
            print(
                "Batch failed "
                f"(kind={primary_info.kind}, status={primary_info.status_code}) "
                f"model={model} method={method}: {_summarize_error(primary_info.message)}"
            )

        if not fallback_model:
            if primary_info is not None:
                await _finish_failure(
                    primary_info, failed_model=model, failed_method=method
                )
            return None

        # Try once with fallback model, using its own best method (auto).
        try:
            fb_driver = get_driver(fallback_model)
            fb_method = fb_driver.get_best_translation_method("auto")
        except Exception as e:
            info = classify_llm_error(e)
            print(
                "Fallback setup failed "
                f"(kind={info.kind}, status={info.status_code}) "
                f"fallback_model={fallback_model}: {_summarize_error(info.message)}"
            )
            await _finish_failure(
                info, failed_model=fallback_model, failed_method="auto"
            )
            return None

        print(f"Retrying batch with fallback model={fallback_model} method={fb_method}")
        try:
            return await _run_timed(run_model=fallback_model, run_method=fb_method)
        except BatchTranslationError as e:
            info = e.info
            print(
                "Fallback batch failed "
                f"(kind={info.kind}, status={info.status_code}) "
                f"model={fallback_model} method={fb_method}: {_summarize_error(info.message)}"
            )
            await _finish_failure(
                info, failed_model=fallback_model, failed_method=fb_method
            )
            return None
        except Exception as e:
            info = classify_llm_error(e)
            print(
                "Fallback batch failed "
                f"(kind={info.kind}, status={info.status_code}) "
                f"model={fallback_model} method={fb_method}: {_summarize_error(info.message)}"
            )
            await _finish_failure(
                info, failed_model=fallback_model, failed_method=fb_method
            )
            return None

    async def _save_translation_progress(
        self,
        progress: dict[str, dict[str, str]],
        translations: list[dict[str, str]],
        is_final: bool = False,
        csv_corrections: Optional[dict[str, set[str]]] = None,
    ) -> None:
        """
        Save translation progress and translations to storage.

        Args:
            progress: Progress dictionary tracking completed translations
            translations: list of translation dictionaries
            is_final: Whether this is the final save (affects log message)
            csv_corrections: Keys whose value was taken from a valid CSV cell and
                must overwrite stale translation memory (manual edits in the CSV).
        """
        for language in self.dst_languages:
            await self.storage.save_progress(
                self.project_id,
                language,
                progress[language],
                overwrite_keys=(csv_corrections or {}).get(language),
            )
        await self.storage.save_translations(self.project_id, translations)

        count = sum(len(language_progress) for language_progress in progress.values())
        if is_final:
            print(f"Final save: {count} translations saved")
        else:
            print(f"Progress saved: {count} translations saved")

    async def _translate_pass(
        self,
        *,
        model: str,
        method: str,
        delay_seconds: float,
        max_retries: int,
        batch_size: int,
        batch_max_input_tokens: int,
        regenerate: bool,
        editorial: dict[str, dict[str, str]] | None = None,
        only_keys: set[str] | None = None,
        fallback_model: Optional[str],
        driver,
        request_timeout: float = 120.0,
    ) -> None:
        translations = await self.storage.load_translations(self.project_id)
        envelope_builder = EnvelopeBuilder(
            await self.storage.load_glossary(self.project_id),
            translations,
            self.base_language,
            self.dst_languages,
            self.reference_languages,
        )
        progress = {
            language: await self.storage.load_progress(self.project_id, language)
            for language in self.dst_languages
        }
        context = await self._load_context()
        prompt = await self._load_prompt()

        phrases_to_translate: list[tuple[str, str | None]] = []
        phrase_indices: dict[str, int] = {}
        phrase_languages: dict[str, set[str]] = {}
        csv_corrections: dict[str, set[str]] = {
            language: set() for language in self.dst_languages
        }
        sent_batch = False
        batches_started = 0
        phrases_attempted = 0
        pass_started = time.monotonic()

        def split_batch(
            phrases: list[tuple[str, str | None]],
            indices: dict[str, int],
            languages: dict[str, set[str]],
        ) -> tuple[
            tuple[list[tuple[str, str | None]], dict[str, int], dict[str, set[str]]],
            tuple[list[tuple[str, str | None]], dict[str, int], dict[str, set[str]]],
        ]:
            midpoint = len(phrases) // 2

            def part(items: list[tuple[str, str | None]]):
                keys = {phrase for phrase, _context in items}
                return (
                    items,
                    {key: value for key, value in indices.items() if key in keys},
                    {key: value for key, value in languages.items() if key in keys},
                )

            return part(phrases[:midpoint]), part(phrases[midpoint:])

        async def run_batch(
            phrases: list[tuple[str, str | None]],
            indices: dict[str, int],
            languages: dict[str, set[str]],
        ) -> None:
            nonlocal sent_batch, batches_started, phrases_attempted
            batch_input = envelope_builder.build(phrases, indices)
            batch_prompt = await self.translation_tool.create_batch_prompt(
                phrases,
                self.base_language,
                self.language_refs,
                prompt,
                context,
                method,
                batch_input,
            )
            if (
                batch_prompt
                and len(phrases) > 1
                and self.count_tokens(batch_prompt, model) > batch_max_input_tokens
            ):
                left, right = split_batch(phrases, indices, languages)
                await run_batch(*left)
                await run_batch(*right)
                return

            if envelope_builder.last_omitted_glossary:
                omitted = ", ".join(
                    f"{term} ({count})"
                    for term, count in envelope_builder.last_omitted_glossary
                )
                print(f"Glossary entries omitted by 20-entry prompt cap: {omitted}")

            if sent_batch:
                await driver.wait(delay_seconds)
            sent_batch = True
            batches_started += 1
            phrases_attempted += len(phrases)
            print(
                f"Batch {batches_started}: phrases={len(phrases)}, "
                f"attempted={phrases_attempted}, elapsed={time.monotonic() - pass_started:.1f}s"
            )

            try:
                translated = await self._process_translation_batch(
                    phrases,
                    languages,
                    model,
                    method,
                    prompt,
                    context,
                    delay_seconds,
                    max_retries,
                    request_timeout,
                    batch_input,
                    fallback_model=fallback_model,
                    raise_on_error=True,
                )
            except BatchTranslationError as error:
                if error.info.kind == "model_error" and len(phrases) > 1:
                    left, right = split_batch(phrases, indices, languages)
                    await run_batch(*left)
                    await run_batch(*right)
                    return
                await self._record_batch_failures(
                    phrases,
                    languages,
                    model=model,
                    method=method,
                    info=error.info,
                )
                print(
                    f"Batch {batches_started} finished: failed, "
                    f"elapsed={time.monotonic() - pass_started:.1f}s"
                )
                return

            if translated is not None:
                await self._apply_translated_batch(
                    translated,
                    phrase_indices=indices,
                    phrase_languages=languages,
                    progress=progress,
                    translations=translations,
                    model=model,
                    method=method,
                )
            print(
                f"Batch {batches_started} finished: "
                f"{'translated' if translated is not None else 'empty'}, "
                f"elapsed={time.monotonic() - pass_started:.1f}s"
            )

        async def flush_batch() -> None:
            nonlocal phrases_to_translate, phrase_indices, phrase_languages
            await run_batch(
                phrases_to_translate,
                phrase_indices,
                phrase_languages,
            )

            await self._save_translation_progress(
                progress, translations, csv_corrections=csv_corrections
            )
            phrases_to_translate = []
            phrase_indices = {}
            phrase_languages = {}

        for i, row in enumerate(translations):
            source_phrase = row[self.base_language]
            if not source_phrase:
                continue
            if only_keys is not None and source_phrase not in only_keys:
                continue

            missing_languages: set[str] = set()
            for language in self.dst_languages:
                editorial_value = (editorial or {}).get(language, {}).get(source_phrase)
                if regenerate and editorial_value:
                    row[language] = editorial_value
                    progress[language][source_phrase] = editorial_value
                    continue
                existing_translation = row.get(language) or ""
                if existing_translation and not regenerate:
                    ok, _ = self.translation_tool.validate_translation_text(
                        existing_translation
                    )
                    if ok:
                        if (
                            progress[language].get(source_phrase)
                            != existing_translation
                        ):
                            progress[language][source_phrase] = existing_translation
                            csv_corrections[language].add(source_phrase)
                        continue
                    row[language] = ""

                if source_phrase in progress[language] and not regenerate:
                    translation = progress[language][source_phrase]
                    ok, _ = self.translation_tool.validate_translation_text(translation)
                    if not ok:
                        del progress[language][source_phrase]
                    else:
                        row[language] = translation
                        print(
                            f"Using cached translation for {language}: "
                            f"{source_phrase} -> {translation}"
                        )
                        continue
                missing_languages.add(language)

            if not missing_languages:
                continue

            phrase_context = row.get("context") or ""
            language_contexts = [
                f"[{language}] {row.get(f'context_{language}')}"
                for language in missing_languages
                if row.get(f"context_{language}")
            ]
            if language_contexts:
                phrase_context = "; ".join(
                    filter(None, [phrase_context, *language_contexts])
                )

            phrases_to_translate.append((source_phrase, phrase_context))
            phrase_indices[source_phrase] = i
            phrase_languages[source_phrase] = missing_languages

            if len(phrases_to_translate) >= batch_size:
                await flush_batch()

        if phrases_to_translate:
            await flush_batch()

        await self._save_translation_progress(
            progress,
            translations,
            is_final=True,
            csv_corrections=csv_corrections,
        )

    async def translate(
        self,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
        request_timeout: float = 120.0,
        batch_size: int = 50,
        model: str = "gemini",
        batch_max_input_tokens: int = 65536,
        translation_method: str = "standard",
        regenerate: bool = False,
        fallback_model: Optional[str] = None,
        only_keys: set[str] | None = None,
    ) -> None:
        """Translate phrases from base language to destination language

        Args:
            delay_seconds: Delay between API calls to avoid rate limiting
            max_retries: Maximum number of retries for failed API calls
            batch_size: Number of phrases to translate in a single API call
            model: The LLM model to use for translation
            batch_max_input_tokens: Maximum assembled prompt tokens for a batch
            translation_method: Method to use for translation ('auto', 'standard', 'structured', or 'function')
            regenerate: If True, ignore existing translations and progress, and re-translate all phrases.
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
        editorial = await self.storage.load_editorial(self.project_id)

        await self._translate_pass(
            model=model,
            method=method,
            delay_seconds=delay_seconds,
            max_retries=max_retries,
            request_timeout=request_timeout,
            batch_size=batch_size,
            batch_max_input_tokens=batch_max_input_tokens,
            regenerate=regenerate,
            editorial=editorial,
            only_keys=only_keys,
            fallback_model=fallback_model,
            driver=driver,
        )

        if not fallback_model:
            return

        translations = await self.storage.load_translations(self.project_id)
        progress = {
            language: await self.storage.load_progress(self.project_id, language)
            for language in self.dst_languages
        }
        editorial = await self.storage.load_editorial(self.project_id)
        if not self._has_missing_phrases(
            translations, progress, regenerate, editorial, only_keys
        ):
            return

        fb_driver = get_driver(fallback_model)
        fb_method = fb_driver.get_best_translation_method("auto")
        print(
            f"Gap-filling pass with fallback model={fallback_model} method={fb_method}"
        )
        await self._translate_pass(
            model=fallback_model,
            method=fb_method,
            delay_seconds=delay_seconds,
            max_retries=max_retries,
            request_timeout=request_timeout,
            batch_size=batch_size,
            batch_max_input_tokens=batch_max_input_tokens,
            regenerate=regenerate,
            editorial=editorial,
            only_keys=only_keys,
            fallback_model=None,
            driver=fb_driver,
        )
