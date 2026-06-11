from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from typing import Optional
import os
import asyncio
from ..BaseDriver import BaseDriver, DEBUG


class GeminiDriver(BaseDriver):
    """
    Driver class for interacting with Google's Gemini LLM.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Gemini driver.

        Args:
            model: The Gemini model to use
            api_key: API key for Gemini. If None, will try to get from environment variable
        """
        super().__init__(model, api_key)

        # Get API key from parameter or environment variables
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. Please check your .env file."
            )

        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(model=model, api_key=SecretStr(self.api_key))

        # Set capability flags based on model version
        if "2." in model:
            # Gemini 2.0 supports structured output
            self.supports_structured_output = True
            self.supports_function_calling = False
            self.preferred_method = "structured"
        else:
            # Older models have limited support
            self.supports_structured_output = False
            self.supports_function_calling = False
            self.preferred_method = "standard"

    def _convert_type(self, type_str: str) -> str:
        """
        Convert Python/JSON Schema types to OpenAPI 3.0 types.

        Args:
            type_str: The type string to convert

        Returns:
            str: The converted type string
        """
        type_mapping = {
            "string": "string",
            "str": "string",
            "integer": "integer",
            "int": "integer",
            "number": "number",
            "float": "number",
            "boolean": "boolean",
            "bool": "boolean",
            "array": "array",
            "list": "array",
            "object": "object",
            "dict": "object",
        }
        return type_mapping.get(type_str.lower(), "string")

    def get_structured_output_schema(self) -> dict:
        """
        Get the schema for structured output translation optimized for Gemini.

        Returns:
            JSON schema for structured output in Gemini's format
        """
        # Keep the structured output contract in one place (Pydantic). Gemini's
        # GenAI SDK can transform JSON Schemas (including $defs/$ref) for
        # compatibility.
        from lib.contracts.translation_contracts import TranslationsResponse

        schema: dict = TranslationsResponse.model_json_schema()
        # Gemini-specific: preserve property ordering when possible.
        schema["propertyOrdering"] = ["translations", "failures"]
        return schema

    async def translate_structured_async(
        self,
        prompt: str,
        output_schema: Optional[dict] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
    ) -> dict:
        """
        Send a request to the LLM asynchronously and get a structured response.
        Uses Gemini tool-calling + Pydantic parsing for structured output.

        Args:
            prompt: The formatted prompt to send to the model
            output_schema: JSON schema defining the expected output structure (optional)
            delay_seconds: Delay between retries to avoid rate limiting
            max_retries: Maximum number of retries for failed API calls

        Returns:
            The model's response as a structured dictionary
        """
        if output_schema is None:
            output_schema = self.get_structured_output_schema()

        # This version of `langchain-google-genai` implements structured output
        # via tool calling + output parsers. Passing a Pydantic model gives us
        # typed parsing (instead of a list of tool calls).
        from lib.contracts.translation_contracts import TranslationsResponse

        include_raw = bool(DEBUG)
        structured = self.llm.with_structured_output(
            TranslationsResponse, include_raw=include_raw
        )

        for retry in range(max_retries):
            try:
                # Add delay before retries (but not before the first attempt)
                if retry > 0:
                    wait_time = delay_seconds * (2 ** (retry - 1))
                    if DEBUG:
                        print(
                            f"Retrying in {wait_time:.1f} seconds (attempt {retry+1}/{max_retries})..."
                        )
                    await asyncio.sleep(wait_time)

                if DEBUG:
                    print(f"\nUsing structured output with schema: {output_schema}")

                result = await structured.ainvoke(prompt)
                parsed = (
                    result.get("parsed")
                    if include_raw and isinstance(result, dict)
                    else result
                )

                if parsed is None:
                    raise TypeError("Structured output returned no parsed result")

                if isinstance(parsed, dict):
                    return parsed

                # Pydantic v2
                model_dump = getattr(parsed, "model_dump", None)
                if callable(model_dump):
                    dumped = model_dump()
                    if isinstance(dumped, dict):
                        return dumped

                # Pydantic v1
                as_dict = getattr(parsed, "dict", None)
                if callable(as_dict):
                    dumped = as_dict()
                    if isinstance(dumped, dict):
                        return dumped

                raise TypeError(f"Unexpected structured output type: {type(parsed)}")
            except Exception as e:
                if DEBUG:
                    print(
                        f"Error in {self.model} structured output call (attempt {retry+1}/{max_retries}): {e}"
                    )
                if retry == max_retries - 1:
                    raise Exception(
                        f"Failed to get structured output after {max_retries} attempts: {e}"
                    )

        raise Exception(f"Failed to get structured output after {max_retries} attempts")
