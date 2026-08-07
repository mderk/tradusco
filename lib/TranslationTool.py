import json
import re
import os
import ast
from dataclasses import dataclass
from typing import Annotated, Optional, Union

from pydantic import BaseModel, Field
from .llm import BaseDriver, get_driver
from .PromptManager import PromptManager
from .utils import (
    looks_like_json_artifact as _looks_like_json_artifact,
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


class Input(BaseModel):
    """
    Input format for the translation prompt.
    """

    base_language: str
    dst_language: str
    context: str
    phrases: Annotated[
        list[tuple[str, str | None]],
        Field(
            description="List of phrases to translate, each element is a tuple of the phrase and its context (optional)"
        ),
    ]


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
        dst_language: str,
        prompt: str,
        context: Optional[str] = None,
    ) -> str | None:
        """Create a prompt for translation using JSON format"""
        # Create a list of phrases and a separate context mapping

        # Add global context if provided
        context_section = (
            f"\nGlobal Translation Context:\n{context}\n" if context else ""
        )
        data = Input(
            base_language=base_language.upper(),
            dst_language=dst_language.upper(),
            context=context_section,
            phrases=phrases,
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

        if DEBUG:
            print("Translated", len(translations_list), translations_list)

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
        if not batch_prompt:
            if DEBUG:
                print(f"Warning: Could not create batch prompt for model {model}")
            return None, ""

        if DEBUG:
            print(batch_prompt)

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
                translations_list = None
                if isinstance(parsed_response, dict):
                    if "translations" in parsed_response:  # type: ignore
                        translations_list = parsed_response["translations"]
                elif isinstance(parsed_response, list):
                    translations_list = parsed_response

                if translations_list:
                    return self.merge_translations(
                        translations_list=translations_list,
                        phrases=phrases,
                    )
                else:
                    if DEBUG:
                        print("Invalid JSON response received")
                    return None
            except json.JSONDecodeError:
                # Some providers/models occasionally emit “JSON-like” output that isn't
                # strict JSON (single quotes, trailing commas, etc). Try a safe
                # Python literal parse, then fall back to line-based parsing.
                try:
                    parsed_response = ast.literal_eval(json_str)
                except Exception:
                    parsed_response = None

                translations_list = None
                if isinstance(parsed_response, dict) and "translations" in parsed_response:
                    translations_list = parsed_response.get("translations")
                elif isinstance(parsed_response, list):
                    translations_list = parsed_response

                if isinstance(translations_list, list):
                    return self.merge_translations(
                        translations_list=[str(x) for x in translations_list],
                        phrases=phrases,
                    )

                # Last resort: accept plain line output (one translation per line).
                cleaned = re.sub(r"```(?:json)?|```", "", response)
                lines: list[str] = []
                for line in cleaned.splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    # Strip simple list markers / numbering.
                    s = re.sub(r"^\s*[-*•]\s+", "", s)
                    s = re.sub(r"^\s*\d+\s*[\).\:-]\s+", "", s)
                    s = s.strip()
                    if not s:
                        continue
                    # Skip JSON scaffolding lines (brackets / the translations key).
                    if _looks_like_json_artifact(s):
                        continue
                    # If the model returned the full dict literally, skip it here.
                    if s.startswith("{") and s.endswith("}"):
                        continue

                    # If this is a JSON string literal, decode it safely.
                    if s.startswith('"') and (s.endswith('"') or s.endswith('",')):
                        try:
                            decoded = json.loads(s[:-1] if s.endswith('",') else s)
                            lines.append(str(decoded))
                            continue
                        except Exception:
                            pass
                    if s.startswith("'") and (s.endswith("'") or s.endswith("',")):
                        try:
                            decoded = ast.literal_eval(s[:-1] if s.endswith("',") else s)
                            lines.append(str(decoded))
                            continue
                        except Exception:
                            pass

                    lines.append(s)

                # Positional merge is only safe when line and phrase counts match.
                # Drop a leading prelude (more lines than phrases), but if the
                # counts still disagree, refuse rather than silently misalign and
                # persist shifted translations — the batch is retried instead.
                if len(lines) > len(phrases):
                    lines = lines[-len(phrases) :]
                if lines and len(lines) == len(phrases):
                    return self.merge_translations(
                        translations_list=lines,
                        phrases=phrases,
                    )

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
        raise_on_error: bool = False,
    ) -> dict[str, str] | None:
        """Process a batch of phrases for translation"""
        # Get common setup
        try:
            driver, batch_prompt = await self.setup(
                phrases=phrases,
                model=model,
                base_language=base_language,
                dst_language=dst_language,
                prompt=prompt,
                context=context,
                method_name="standard",
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
        dst_language: str,
        prompt: str,
        context: Optional[str] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
        raise_on_error: bool = False,
    ) -> dict[str, str] | None:
        """Process a batch of phrases using structured output"""
        # Get common setup
        try:
            driver, batch_prompt = await self.setup(
                phrases=phrases,
                model=model,
                base_language=base_language,
                dst_language=dst_language,
                prompt=prompt,
                context=context,
                method_name="structured",
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
        dst_language: str,
        prompt: str,
        context: Optional[str] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
        raise_on_error: bool = False,
    ) -> dict[str, str] | None:
        """Process a batch of phrases using function calling"""
        # Get common setup
        try:
            driver, batch_prompt = await self.setup(
                phrases=phrases,
                model=model,
                base_language=base_language,
                dst_language=dst_language,
                prompt=prompt,
                context=context,
                method_name="function",
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
