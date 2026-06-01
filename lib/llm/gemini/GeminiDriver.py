from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from typing import Any, Optional, Dict, List
import os
import time
import json
import re
import asyncio
from ..BaseDriver import BaseDriver, DEBUG


class GeminiDriver(BaseDriver):
    """
    Driver class for interacting with Google's Gemini LLM.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash-preview-04-17",
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
        # Create our Gemini-specific schema for the structured output
        response_schema = {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "description": "Array of translations from source language to target language in the same order as input phrases",
                    "items": {
                        "type": "string",
                        "description": "Translated text in target language",
                    },
                }
            },
            "required": ["translations"],
            "propertyOrdering": [
                "translations"
            ],  # Gemini-specific for consistent property ordering
        }

        return response_schema

    async def translate_structured_async(
        self,
        prompt: str,
        output_schema: Optional[dict] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
    ) -> dict:
        """
        Send a request to the LLM asynchronously and get a structured response.
        Uses Gemini's native structured output capability.

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

        for retry in range(max_retries):
            try:
                # Add delay before retries (but not before the first attempt)
                if retry > 0:
                    # Apply exponential backoff for retries
                    wait_time = delay_seconds * (2 ** (retry - 1))
                    if DEBUG:
                        print(
                            f"Retrying in {wait_time:.1f} seconds (attempt {retry+1}/{max_retries})..."
                        )
                    await asyncio.sleep(wait_time)

                # Debug output to see what we're doing
                if DEBUG:
                    print(f"\nUsing structured output with schema: {output_schema}")

                # Use Gemini's native structured output
                response = await self.llm.ainvoke(
                    prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": output_schema,
                    },
                )

                # Extract and parse the response
                content = str(
                    response.content if hasattr(response, "content") else response
                )

                if DEBUG:
                    print(f"Raw response: {content}")

                # First, try to extract JSON from markdown code blocks if present
                json_block_match = re.search(
                    r"```(?:json)?\s*([\s\S]*?)\s*```", content
                )

                if json_block_match:
                    extracted_json = json_block_match.group(1).strip()
                    if DEBUG:
                        print(f"Extracted JSON from markdown: {extracted_json}")
                    try:
                        result = json.loads(extracted_json)

                        # Handle the case where we get a valid JSON but not in the expected format
                        if "translations" not in result and output_schema.get(
                            "properties", {}
                        ).get("translations"):
                            # If the response is just an array, assume it's translations
                            if isinstance(result, list):
                                result = {"translations": result}
                            # If it's a string, wrap it in an array
                            elif isinstance(result, str):
                                result = {"translations": [result]}

                        return result
                    except json.JSONDecodeError:
                        # If we can't parse the extracted content, continue to next parsing attempt
                        if DEBUG:
                            print(
                                f"Failed to parse extracted JSON content, trying full content"
                            )

                # If no code blocks or parsing the extracted content failed, try parsing the full content
                try:
                    # Parse the JSON response
                    result = json.loads(content)

                    # Handle the case where we get a valid JSON but not in the expected format
                    if "translations" not in result and output_schema.get(
                        "properties", {}
                    ).get("translations"):
                        # If the response is just an array, assume it's translations
                        if isinstance(result, list):
                            result = {"translations": result}
                        # If it's a string, wrap it in an array
                        elif isinstance(result, str):
                            result = {"translations": [result]}

                    return result

                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Failed to parse LLM response as JSON: {e}\nContent: {content}"
                    ) from e

            except Exception as e:
                if DEBUG:
                    print(
                        f"Error in {self.model} structured output call (attempt {retry+1}/{max_retries}): {e}"
                    )
                # No need to sleep here since we'll sleep at the start of the next iteration
                # if we're not on the last retry
                if retry == max_retries - 1:
                    raise Exception(
                        f"Failed to get structured output after {max_retries} attempts: {e}"
                    )

        raise Exception(f"Failed to get structured output after {max_retries} attempts")
