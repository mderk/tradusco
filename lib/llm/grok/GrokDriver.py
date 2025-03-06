import os
import time
from typing import Optional
from pydantic import SecretStr
from langchain_xai import ChatXAI
from ..BaseDriver import BaseDriver


class GrokDriver(BaseDriver):
    """
    Driver class for interacting with xAI's Grok LLM using the LangChain integration.
    """

    def __init__(self, model: str = "grok-2-1212", api_key: Optional[str] = None):
        """
        Initialize the Grok driver.

        Args:
            model: The Grok model to use
            api_key: API key for Grok. If None, will try to get from environment variable
        """
        super().__init__(model, api_key)

        # Get API key from parameter or environment variables
        self.api_key = api_key or os.environ.get("GROK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROK_API_KEY environment variable not set. Please check your .env file."
            )

        # Initialize the LLM
        self.llm = ChatXAI(model=model, api_key=SecretStr(self.api_key))
