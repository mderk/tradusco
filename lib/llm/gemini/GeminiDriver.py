from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from typing import Any, Optional
import os
import time
from ..BaseDriver import BaseDriver


class GeminiDriver(BaseDriver):
    """
    Driver class for interacting with Google's Gemini LLM.
    """

    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        """
        Initialize the Gemini driver.

        Args:
            model: The Gemini model to use
            api_key: API key for Gemini. If None, will try to get from environment variable
        """
        # Get API key from parameter or environment variables
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. Please check your .env file."
            )

        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(model=model, api_key=SecretStr(self.api_key))
        self.model = model

    def translate_batch(
        self, prompt: str, delay_seconds: float = 1.0, max_retries: int = 3
    ) -> str:
        """
        Send a batch translation request to the Gemini model.

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
                    f"Error in Gemini API call (attempt {retry+1}/{max_retries}): {e}"
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

    def translate_single(self, prompt: str, delay_seconds: float = 1.0) -> str:
        """
        Send a single translation request to the Gemini model.

        Args:
            prompt: The formatted prompt to send to the model
            delay_seconds: Delay after the request to avoid rate limiting

        Returns:
            The model's response content as a string

        Raises:
            Exception: If the API call fails
        """
        try:
            # Send to LLM
            response = self.llm.invoke(prompt)

            # Add delay to avoid rate limiting
            time.sleep(delay_seconds)

            # Ensure we return a string
            return str(response.content)
        except Exception as e:
            raise Exception(f"Failed to translate: {e}")
