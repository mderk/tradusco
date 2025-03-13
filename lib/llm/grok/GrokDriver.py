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

    def __init__(
        self,
        model: str = "grok-2-1212",
        api_key: Optional[str] = None,
    ):
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

        # Initialize the LLM - pass parameters according to API requirements
        self.llm = ChatXAI(api_key=SecretStr(self.api_key))
        self.llm.model = (
            model  # Set model as attribute if it's not accepted as a parameter
        )

        # Set capability flags based on model version
        if "grok-3" in model.lower() or "grok-beta" in model.lower():
            # Grok-3 and Beta support structured output and function calling
            self.supports_structured_output = True
            self.supports_function_calling = True
            self.preferred_method = (
                "structured"  # Default to structured output for Grok
            )
        elif "grok-2" in model.lower():
            # Grok-2 has more limited support
            self.supports_structured_output = True
            self.supports_function_calling = False
            self.preferred_method = "structured"
        else:
            # Other models - assume basic support to be safe
            self.supports_structured_output = False
            self.supports_function_calling = False
            self.preferred_method = "standard"
