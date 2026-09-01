import json
import re
import os
import ast
from dataclasses import dataclass
from typing import Annotated, Optional, Union

from pydantic import BaseModel, Field, field_serializer
from .envelope import BatchEnvelope, PhraseEnvelope
from .llm import BaseDriver, get_driver
from .PromptManager import PromptManager
from .utils import (
    placeholders_match as _placeholders_match,
    validate_translation_text as _validate_translation_text,
)

DEBUG = os.environ.get("TRADUSCO_DEBUG")


@dataclass(frozen=True)
class BatchErrorInfo:
    """
    Minimal batch-level error classification used for fallback routing.

    We intentionally keep this coarse-grained:
    - auth_error
    - rate_limit
    - blocked (policy / content filter)
    - model_error (everything else)
    """

    kind: str
    message: str
    status_code: int | None = None


class BatchTranslationError(Exception):
    def __init__(self, info: BatchErrorInfo):
        super().__init__(info.message)
        self.info = info


def _extract_status_code(e: Exception) -> int | None:
    for attr in ("status_code", "status"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            return v

    resp = getattr(e, "response", None)
    if resp is not None:
        v = getattr(resp, "status_code", None)
        if isinstance(v, int):
            return v

    return None


def classify_llm_error(e: Exception) -> BatchErrorInfo:
    """
    Best-effort classification for "why did this batch fail?".

    This is intentionally simple and based on:
    - HTTP status codes when available
    - common substrings in provider/SDK error messages
    """
    status = _extract_status_code(e)
    msg = str(e)
    text = msg.lower()
    cls = e.__class__.__name__.lower()

    def _has(*needles: str) -> bool:
        return any(n in text for n in needles)

    # Provider policy / content filtering
    if _has(
        "content_filter",
        "content filter",
        "content policy",
        "policy",
        "safety",
        "blocked",
        "refused",
        "refuse",
        "moderation",
    ):
        return BatchErrorInfo(kind="blocked", message=msg, status_code=status)

    # Auth / permissions
    if status in (401, 403) or _has(
        "unauthorized",
        "authentication",
        "invalid api key",
        "api key",
        "permission denied",
        "forbidden",
        "environment variable not set",
        "not set",
    ):
        return BatchErrorInfo(kind="auth_error", message=msg, status_code=status)

    # Rate limiting / quotas
    if status == 429 or _has(
        "rate limit",
        "quota",
        "too many requests",
        "resourceexhausted",
    ) or ("ratelimit" in cls):
        return BatchErrorInfo(kind="rate_limit", message=msg, status_code=status)

    return BatchErrorInfo(kind="model_error", message=msg, status_code=status)


@dataclass(frozen=True)
class LanguageRef:
    code: str
    name: str


LANGUAGE_NAMES = {
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "es-ES": "Spanish (Spain)",
    "es-419": "Spanish (Latin America)",
    "es-LA": "Spanish (Latin America)",
    "pt-BR": "Portuguese (Brazil)",
    "pt-PT": "Portuguese (Portugal)",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
}


def language_ref(code: str) -> LanguageRef:
    return LanguageRef(
        code=code,
        name=LANGUAGE_NAMES.get(
            code, LANGUAGE_NAMES.get(code.split("-", 1)[0], code)
        ),
    )


class Input(BaseModel):
    """
    Input format for the translation prompt.
    """

    base_language: str
    dst_languages: list[LanguageRef]
    context: str
    phrases: Annotated[
        BatchEnvelope,
        Field(description="Batch glossary and phrase translation envelopes"),
    ]

    @field_serializer("dst_languages")
    def serialize_dst_languages(self, languages: list[LanguageRef]) -> str:
        return ", ".join(f"{language.code} ({language.name})" for language in languages)


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

    def validate_placeholders(self, source: str, translation: str) -> tuple[bool, str]:
        """
        Ensure translation preserves placeholders and Lingui tags.

        This prevents breaking runtime interpolation like `{num}` / `{name}` and
        rich-text tags like `<0>...</0>`.
        """
        return _placeholders_match(source, translation)

    def validate_translation_text(self, translation: str) -> tuple[bool, str]:
        """
        Reject obvious model-output artifacts that can slip through parsing.

        Delegates to the shared ``lib.utils`` implementation so the live path,
        the storage layer and the offline audit all agree on what counts as a
        valid translation.
        """
        return _validate_translation_text(translation)

    async def create_prompt(
        self,
        phrases: list[tuple[str, str | None]],
        base_language: str,
        dst_languages: list[LanguageRef],
        prompt: str,
        context: Optional[str] = None,
        batch_input: BatchEnvelope | None = None,
    ) -> str | None:
        """Create a prompt for translation using JSON format"""
        # Create a list of phrases and a separate context mapping

        # Add global context if provided
        context_section = (
            f"\nGlobal Translation Context:\n{context}\n" if context else ""
        )
        data = Input(
            base_language=base_language.upper(),
            dst_languages=dst_languages,
            context=context_section,
            phrases=batch_input
            or BatchEnvelope(
                phrases=[
                    PhraseEnvelope(phrase=phrase, context=phrase_context)
                    for phrase, phrase_context in phrases
                ]
            ),
        )

        # Format the prompt template with the required variables
        return self.prompt_manager.format_prompt(prompt, data)

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
        translations_list: list[object],
        phrases: list[tuple[str, str | None]],
        dst_languages: list[LanguageRef],
    ) -> dict[str, dict[str, str]] | None:
        """Align valid per-language blocks with the input phrases."""
        requested = {language.code for language in dst_languages}
        result: dict[str, dict[str, str]] = {}
        for block in translations_list:
            if not isinstance(block, dict):
                continue
            language = block.get("language")
            values = block.get("translations")
            if (
                not isinstance(language, str)
                or language not in requested
                or not isinstance(values, list)
                or len(values) != len(phrases)
                or not all(isinstance(value, str) for value in values)
            ):
                continue
            result[str(language)] = {
                phrase[0]: translation
                for phrase, translation in zip(phrases, values)
                if translation.strip()
            }
        return result or None

    async def setup(
        self,
        phrases: list[tuple[str, str | None]],
        model: str,
        base_language: str,
        dst_languages: list[LanguageRef],
        prompt: str,
        context: Optional[str] = None,
        method_name: str = "standard",
        batch_input: BatchEnvelope | None = None,
    ) -> tuple[Optional[BaseDriver], str]:
        """
        Base method for processing batches of translations.
        Handles common setup and error handling logic.

        Args:
            phrases: List of [phrase, context] tuples
            model: LLM model to use
            base_language: Source language
            dst_languages: Target languages
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

        batch_prompt = await self.create_batch_prompt(
            phrases=phrases,
            base_language=base_language,
            dst_languages=dst_languages,
            prompt=prompt,
            context=context,
            method_name=method_name,
            batch_input=batch_input,
        )
        if not batch_prompt:
            if DEBUG:
                print(f"Warning: Could not create batch prompt for model {model}")
            return None, ""

        if DEBUG:
            print(batch_prompt)

        return driver, batch_prompt

    async def create_batch_prompt(
        self,
        phrases: list[tuple[str, str | None]],
        base_language: str,
        dst_languages: list[LanguageRef],
        prompt: str,
        context: Optional[str] = None,
        method_name: str = "standard",
        batch_input: BatchEnvelope | None = None,
    ) -> str | None:
        """Build exactly the prompt passed to the driver."""
        batch_prompt = await self.create_prompt(
            phrases=phrases,
            base_language=base_language,
            dst_languages=dst_languages,
            prompt=prompt,
            context=context,
            batch_input=batch_input,
        )
        if not batch_prompt:
            return None

        if method_name == "standard":
            try:
                output_format = await self.prompt_manager.load_prompt("output_format")
            except Exception as e:
                if DEBUG:
                    print(f"Warning: Could not load output format instructions: {e}")
                output_format = ""

            if output_format:
                batch_prompt += f"\n\n{output_format}"

        return batch_prompt

    def handle_response(
        self,
        response: Union[str, dict, list],
        phrases: list[tuple[str, str | None]],
        dst_languages: list[LanguageRef],
    ) -> dict[str, dict[str, str]] | None:
        """Parse the common multilingual response without coupling language failures."""
        if isinstance(response, str):
            json_str = self.extract_json_from_response(response)
            try:
                response = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    response = ast.literal_eval(json_str)
                except Exception:
                    return None

        blocks = response.get("translations") if isinstance(response, dict) else response
        if isinstance(blocks, list):
            return self.merge_translations(blocks, phrases, dst_languages)

        if DEBUG:
            print(f"Unexpected response format: {response}")
        return None

    async def translate_standard(
        self,
        phrases: list[tuple[str, str | None]],
        model: str,
        base_language: str,
        dst_languages: list[LanguageRef],
        prompt: str,
        context: Optional[str] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
        raise_on_error: bool = False,
        batch_input: BatchEnvelope | None = None,
    ) -> dict[str, dict[str, str]] | None:
        """Process a batch of phrases for translation"""
        # Get common setup
        try:
            driver, batch_prompt = await self.setup(
                phrases=phrases,
                model=model,
                base_language=base_language,
                dst_languages=dst_languages,
                prompt=prompt,
                context=context,
                method_name="standard",
                batch_input=batch_input,
            )
        except Exception as e:
            if raise_on_error:
                raise BatchTranslationError(classify_llm_error(e)) from e
            if DEBUG:
                print(f"Error processing batch: {e}")
                print("Skipping this batch...")
            return None
        if not driver:
            if DEBUG:
                print("Skipping this batch...")
            if raise_on_error:
                raise BatchTranslationError(
                    BatchErrorInfo(
                        kind="model_error",
                        message="No driver available for model",
                    )
                )
            return None

        # Get the translation response
        try:
            response = await driver.translate_async(
                batch_prompt, delay_seconds=delay_seconds, max_retries=max_retries
            )
        except Exception as e:
            if raise_on_error:
                raise BatchTranslationError(classify_llm_error(e)) from e
            if DEBUG:
                print(f"Error processing batch: {e}")
                print("Skipping this batch...")
            return None

        # Parse and handle the response using the same method as structured and function calls
        try:
            handled = self.handle_response(
                response,
                phrases,
                dst_languages,
            )
            if handled is None and raise_on_error:
                raise BatchTranslationError(
                    BatchErrorInfo(
                        kind="model_error",
                        message="Failed to parse/align translations from standard response",
                    )
                )
            return handled
        except Exception as e:
            if raise_on_error:
                raise BatchTranslationError(classify_llm_error(e)) from e
            if DEBUG:
                print(f"Error processing batch: {e}")
            print("Skipping this batch...")
            return None

    async def translate_structured(
        self,
        phrases: list[tuple[str, str | None]],
        model: str,
        base_language: str,
        dst_languages: list[LanguageRef],
        prompt: str,
        context: Optional[str] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
        raise_on_error: bool = False,
        batch_input: BatchEnvelope | None = None,
    ) -> dict[str, dict[str, str]] | None:
        """Process a batch of phrases using structured output"""
        # Get common setup
        try:
            driver, batch_prompt = await self.setup(
                phrases=phrases,
                model=model,
                base_language=base_language,
                dst_languages=dst_languages,
                prompt=prompt,
                context=context,
                method_name="structured",
                batch_input=batch_input,
            )
        except Exception as e:
            if raise_on_error:
                raise BatchTranslationError(classify_llm_error(e)) from e
            if DEBUG:
                print(f"Error processing batch: {e}")
                print("Skipping this batch...")
            return None
        if not driver:
            if DEBUG:
                print("Skipping this batch...")
            if raise_on_error:
                raise BatchTranslationError(
                    BatchErrorInfo(
                        kind="model_error",
                        message="No driver available for model",
                    )
                )
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

            handled = self.handle_response(
                response,
                phrases,
                dst_languages,
            )
            if handled is None and raise_on_error:
                raise BatchTranslationError(
                    BatchErrorInfo(
                        kind="model_error",
                        message="Failed to parse/align translations from structured response",
                    )
                )
            return handled
        except Exception as e:
            if raise_on_error:
                raise BatchTranslationError(classify_llm_error(e)) from e
            if DEBUG:
                print(f"Error from structured output call: {e}")
                print(f"Failed to process batch using structured output: {e}")
            return None

    async def translate_function(
        self,
        phrases: list[tuple[str, str | None]],
        model: str,
        base_language: str,
        dst_languages: list[LanguageRef],
        prompt: str,
        context: Optional[str] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
        raise_on_error: bool = False,
        batch_input: BatchEnvelope | None = None,
    ) -> dict[str, dict[str, str]] | None:
        """Process a batch of phrases using function calling"""
        # Get common setup
        try:
            driver, batch_prompt = await self.setup(
                phrases=phrases,
                model=model,
                base_language=base_language,
                dst_languages=dst_languages,
                prompt=prompt,
                context=context,
                method_name="function",
                batch_input=batch_input,
            )
        except Exception as e:
            if raise_on_error:
                raise BatchTranslationError(classify_llm_error(e)) from e
            if DEBUG:
                print(f"Error processing batch: {e}")
                print("Skipping this batch...")
            return None
        if not driver:
            if DEBUG:
                print("Skipping this batch...")
            if raise_on_error:
                raise BatchTranslationError(
                    BatchErrorInfo(
                        kind="model_error",
                        message="No driver available for model",
                    )
                )
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

                    handled = self.handle_response(
                        args,
                        phrases,
                        dst_languages,
                    )
                    if handled is None and raise_on_error:
                        raise BatchTranslationError(
                            BatchErrorInfo(
                                kind="model_error",
                                message="Failed to parse/align translations from function-call response",
                            )
                        )
                    return handled
                except Exception as e:
                    if raise_on_error:
                        raise BatchTranslationError(classify_llm_error(e)) from e
                    if DEBUG:
                        print(f"Unexpected function arguments format: {e}")
                    return None
            else:
                if DEBUG:
                    print(f"Unexpected response format: {response}")
                if raise_on_error:
                    raise BatchTranslationError(
                        BatchErrorInfo(
                            kind="model_error",
                            message="Unexpected function-call response format",
                        )
                    )
                return None
        except Exception as e:
            if raise_on_error:
                raise BatchTranslationError(classify_llm_error(e)) from e
            if DEBUG:
                print(f"Error from function call: {e}")
                print("Function call translation failed")
            return None
