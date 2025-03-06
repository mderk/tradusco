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

    def __init__(self, model: str = "gemini-2.0-flash", api_key: Optional[str] = None):
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
