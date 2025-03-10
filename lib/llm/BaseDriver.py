from abc import ABC, abstractmethod
from typing import Optional, Any
import time
import asyncio

import tiktoken


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

    def translate(
        self, prompt: str, delay_seconds: float = 1.0, max_retries: int = 3
    ) -> str:
        """
        Send a batch translation request to the LLM.

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
                response = self.llm.invoke(prompt)

                # Add delay to avoid rate limiting
                time.sleep(delay_seconds)

                # Ensure we return a string
                return str(response.content)
            except Exception as e:
                print(
                    f"Error in {self.model} API call (attempt {retry+1}/{max_retries}): {e}"
                )
                if retry < max_retries - 1:
                    # Exponential backoff
                    wait_time = delay_seconds * (2**retry)
                    print(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    raise Exception(
                        f"Failed to translate after {max_retries} attempts: {e}"
                    )

        # This should never be reached due to the raise in the else clause above,
        # but adding it to satisfy the linter
        raise Exception(f"Failed to translate after {max_retries} attempts")

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
                print(
                    f"Error in {self.model} API call (attempt {retry+1}/{max_retries}): {e}"
                )
                if retry < max_retries - 1:
                    # Exponential backoff
                    wait_time = delay_seconds * (2**retry)
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
            print(f"Warning: Tiktoken failed: {e}, using character-based approximation")
            # Simple character-based approximation (4 chars ~= 1 token)
            return max(1, len(text) // 4) if text else 0
