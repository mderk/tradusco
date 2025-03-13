from abc import ABC, abstractmethod
from typing import Optional, Any
import asyncio
import os

import tiktoken

# Set debug flag from environment variable
DEBUG = os.environ.get("TRADUSCO_DEBUG")


class BaseDriver(ABC):
    """
    Abstract base class for LLM drivers.
    All LLM driver implementations should inherit from this class.
    """

    @abstractmethod
    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize the LLM driver.

        Args:
            model: The model to use
            api_key: API key for the service. If None, will try to get from environment variable
        """
        self.model = model
        self.llm: Any = None  # Will be initialized by subclasses

        # Capability flags - will be set by subclasses
        self.supports_structured_output = False
        self.supports_function_calling = False
        self.preferred_method = (
            "standard"  # Can be 'standard', 'structured', or 'function'
        )

    def get_structured_output_schema(self) -> dict:
        """
        Get the schema for structured output translation.
        Can be overridden by subclasses to provide model-specific schemas.

        Returns:
            JSON schema for structured output
        """
        return {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "description": f"Array of translations from source language to target language in the same order as input phrases",
                    "items": {
                        "type": "string",
                        "description": f"Translated text in target language",
                    },
                }
            },
            "required": ["translations"],
        }

    def get_function_schema(self) -> dict:
        """
        Get the schema for function calling translation.
        Can be overridden by subclasses to provide model-specific schemas.

        Returns:
            Function schema for translation
        """
        return {
            "name": "translations",
            "description": f"Translated phrases from source language to target language",
            "parameters": {
                "type": "object",
                "properties": {
                    "translations": {
                        "type": "array",
                        "description": f"Array of translations from source language to target language in the same order as input phrases",
                        "items": {
                            "type": "string",
                            "description": f"Translated text in target language",
                        },
                    }
                },
                "required": ["translations"],
            },
        }

    async def translate_async(
        self, prompt: str, delay_seconds: float = 1.0, max_retries: int = 3
    ) -> str:
        """
        Send a batch translation request to the LLM asynchronously.
        Args:
            prompt: The formatted prompt to send to the model
            delay_seconds: Delay between retries to avoid rate limiting
            max_retries: Maximum number of retries for failed API calls

        Returns:
            The model's response content as a string

        Raises:
            Exception: If all retry attempts fail
        """

        for retry in range(max_retries):
            try:
                # Send the batch to the LLM
                response = await self.llm.ainvoke(prompt)

                # Add delay to avoid rate limiting
                await asyncio.sleep(delay_seconds)

                # Ensure we return a string
                return str(response.content)
            except Exception as e:
                if DEBUG:
                    print(
                        f"Error in {self.model} API call (attempt {retry+1}/{max_retries}): {e}"
                    )
                if retry < max_retries - 1:
                    # Exponential backoff
                    wait_time = delay_seconds * (2**retry)
                    if DEBUG:
                        print(f"Retrying in {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(
                        f"Failed to translate after {max_retries} attempts: {e}"
                    )
        # This should never be reached due to the raise in the else clause above,
        # but adding it to satisfy the linter
        raise Exception(f"Failed to translate after {max_retries} attempts")

    @staticmethod
    def count_tokens(text: str) -> int:
        try:
            # Use cl100k_base encoding which is used by GPT-3.5 and GPT-4
            # This is a good general-purpose tokenizer for most models
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception as e:
            # Check debug flag from global variable
            if DEBUG:
                print(
                    f"Warning: Tiktoken failed: {e}, using character-based approximation"
                )
            # Simple character-based approximation (4 chars ~= 1 token)
            return max(1, len(text) // 4) if text else 0

    async def translate_structured_async(
        self,
        prompt: str,
        output_schema: Optional[dict] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
    ) -> dict:
        """
        Send a request to the LLM asynchronously and get a structured response.

        Args:
            prompt: The formatted prompt to send to the model
            output_schema: JSON schema defining the expected output structure
            delay_seconds: Delay between retries to avoid rate limiting
            max_retries: Maximum number of retries for failed API calls

        Returns:
            The model's response as a structured dictionary

        Raises:
            Exception: If all retry attempts fail
        """
        if output_schema is None:
            output_schema = self.get_structured_output_schema()
        for retry in range(max_retries):
            try:
                # Standard approach for models that support response_format parameter
                response = await self.llm.ainvoke(
                    prompt,
                    response_format={
                        "type": "json_object",
                        "schema": output_schema,
                    },
                )

                # Add delay to avoid rate limiting
                await asyncio.sleep(delay_seconds)

                # Return the structured output
                if hasattr(response, "content"):
                    # If it's a string, parse it
                    if isinstance(response.content, str):
                        import json

                        return json.loads(response.content)
                    # If it's already a dict, return it
                    elif isinstance(response.content, dict):
                        return response.content

                # Fallback for other response types
                if not isinstance(response, dict):
                    import json

                    try:
                        # If it has content attribute, maybe it's a BaseMessage or similar
                        if hasattr(response, "content"):
                            if isinstance(response.content, dict):
                                return response.content
                            elif isinstance(response.content, str):
                                try:
                                    return json.loads(response.content)
                                except:
                                    return {"result": response.content}
                        # Last resort: convert to string and try to parse
                        try:
                            return json.loads(str(response))
                        except:
                            return {"result": str(response)}
                    except:
                        # If all conversion attempts fail, return a simple dict with the response
                        return {"result": str(response)}
                return response

            except Exception as e:
                if DEBUG:
                    print(
                        f"Error in {self.model} structured output API call (attempt {retry+1}/{max_retries}): {e}"
                    )
                if retry < max_retries - 1:
                    # Exponential backoff
                    wait_time = delay_seconds * (2**retry)
                    if DEBUG:
                        print(f"Retrying in {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(
                        f"Failed to get structured output after {max_retries} attempts: {e}"
                    )

        # This should never be reached due to the raise in the else clause above
        raise Exception(f"Failed to get structured output after {max_retries} attempts")

    async def translate_function_async(
        self,
        prompt: str,
        functions: Optional[list[dict]] = None,
        function_name: Optional[str] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
    ) -> dict:
        """
        Send a request to the LLM asynchronously that can call a specified function.

        Args:
            prompt: The formatted prompt to send to the model
            functions: List of function definitions in OpenAI format
            function_name: Optional name of function to force call (None means model decides)
            delay_seconds: Delay between retries to avoid rate limiting
            max_retries: Maximum number of retries for failed API calls

        Returns:
            The model's function call parameters as a dictionary

        Raises:
            Exception: If all retry attempts fail
        """
        if functions is None:
            fn = self.get_function_schema()
            functions = [fn]
            function_name = fn["name"]

        import re
        import json

        for retry in range(max_retries):
            try:
                # Add delay before retries (but not before the first attempt)
                if retry > 0:
                    # Exponential backoff
                    wait_time = delay_seconds * (2 ** (retry - 1))
                    if DEBUG:
                        print(
                            f"Retrying function call in {wait_time:.1f} seconds (attempt {retry+1}/{max_retries})..."
                        )
                    await asyncio.sleep(wait_time)

                # Standard approach for models that support function calling
                tools = [{"type": "function", "function": func} for func in functions]

                response = await self.llm.ainvoke(
                    prompt,
                    tools=tools,
                    tool_choice=(
                        {"type": "function", "function": {"name": function_name}}
                        if function_name
                        else "auto"
                    ),
                )

                # Print response type for debugging
                if DEBUG:
                    print(f"Function call response type: {type(response)}")

                # Extract function call information based on response format
                # Try to handle all the different formats that might be returned

                # 1. Standard OpenAI format with tool_calls that have function attributes
                if hasattr(response, "tool_calls") and response.tool_calls:
                    # Check the type of the tool_calls items
                    tool_call = response.tool_calls[0]

                    # OpenRouter dictionary format
                    if isinstance(tool_call, dict):
                        if "name" in tool_call and (
                            "args" in tool_call or "arguments" in tool_call
                        ):
                            # Get arguments from either 'args' or 'arguments' key
                            arguments = tool_call.get("args") or tool_call.get(
                                "arguments"
                            )
                            return {
                                "name": tool_call["name"],
                                "arguments": arguments,
                            }
                    # OpenAI object format
                    elif hasattr(tool_call, "function"):
                        return {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }

                # 2. Response with direct content attribute that might contain the answer
                if hasattr(response, "content") and response.content:
                    content = response.content

                    # Debug output
                    if DEBUG:
                        print(f"Response content preview: {str(content)[:200]}")

                    # If content is a dict, it might be the arguments
                    if isinstance(content, dict):
                        return {
                            "name": function_name or "translations",
                            "arguments": content,
                        }

                    # If content is a string, try to extract translations
                    elif isinstance(content, str):
                        # First try to extract JSON from a code block if present
                        json_match = re.search(
                            r"```(?:json)?\s*([\s\S]*?)\s*```", content
                        )
                        json_content = json_match.group(1) if json_match else content

                        # Try to parse as JSON
                        try:
                            content_json = json.loads(json_content)
                            return {
                                "name": function_name or "translations",
                                "arguments": content_json,
                            }
                        except:
                            # If JSON parsing failed, try to extract translations line by line
                            # This helps with models that don't properly format JSON but still give translations
                            lines = content.strip().split("\n")
                            translations = []

                            # Extract meaningful lines (skipping code block markers, etc.)
                            for line in lines:
                                line = line.strip()
                                if line and not line.startswith(("```", "{", "}")):
                                    # Clean up the line (remove numbers, quotes, etc.)
                                    cleaned_line = re.sub(r"^\d+\.\s*", "", line)
                                    cleaned_line = re.sub(
                                        r'^["\']|["\']$', "", cleaned_line
                                    )
                                    if cleaned_line:
                                        translations.append(cleaned_line)

                            # If we found enough translations, use them
                            if len(translations) >= 3:  # Assuming 3 phrases is common
                                return {
                                    "name": function_name or "translations",
                                    "arguments": {"translations": translations[:3]},
                                }

                            # If we still have no translations, return the content as a single translation
                            # This may not be ideal but prevents returning nothing
                            return {
                                "name": function_name or "translations",
                                "arguments": {"translations": [content]},
                            }

                # 3. Response is already a dict (common with some wrappers)
                if isinstance(response, dict):
                    # If already in the right format, return it
                    if "name" in response and "arguments" in response:
                        return response

                    # If it has translations directly, wrap it
                    if "translations" in response:
                        return {
                            "name": function_name or "translations",
                            "arguments": response,
                        }

                    # Fallback - wrap the entire dict as arguments
                    return {
                        "name": function_name or "translations",
                        "arguments": response,
                    }

                # 4. Last resort - convert response to string and wrap
                return {
                    "name": function_name or "translations",
                    "arguments": {"translations": [str(response)]},
                }

            except Exception as e:
                if DEBUG:
                    print(
                        f"Error in {self.model} function call API (attempt {retry+1}/{max_retries}): {e}"
                    )
                # Don't need to sleep here since we'll sleep at the beginning of the next iteration
                if retry == max_retries - 1:
                    raise Exception(
                        f"Failed to call function after {max_retries} attempts: {e}"
                    )

        # This should never be reached due to the raise in the else clause above
        raise Exception(f"Failed to call function after {max_retries} attempts")

    def get_best_translation_method(self, requested_method: str = "auto") -> str:
        """
        Determine the best translation method to use based on driver capabilities
        and requested method.

        Args:
            requested_method: Requested method, can be 'auto', 'standard', 'structured', or 'function'

        Returns:
            The best method to use: 'standard', 'structured', or 'function'
        """
        # If a specific method is requested (not 'auto'), try to honor it if supported
        if requested_method != "auto":
            if requested_method == "function" and self.supports_function_calling:
                return "function"
            elif requested_method == "structured" and self.supports_structured_output:
                return "structured"
            elif requested_method == "standard":
                return "standard"

            # If the requested method is not supported, warn the user
            if DEBUG:
                print(
                    f"Warning: Model {self.model} does not support {requested_method} method."
                )
                print(f"Falling back to {self.preferred_method} method.")

        # If 'auto' is requested or the requested method is not supported,
        # use the model's preferred method if available
        if self.preferred_method == "function" and self.supports_function_calling:
            return "function"
        elif self.preferred_method == "structured" and self.supports_structured_output:
            return "structured"

        # Default to standard method if nothing else is supported
        return "standard"
