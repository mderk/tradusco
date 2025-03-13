import os
from typing import Optional

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from ..BaseDriver import BaseDriver


class OpenAIDriver(BaseDriver):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(model, api_key)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. Please check your .env file."
            )

        self.llm = ChatOpenAI(
            model=model, api_key=SecretStr(self.api_key), base_url=base_url
        )

        # Set capability flags - OpenAI models support both structured output and function calling
        self.supports_structured_output = True
        self.supports_function_calling = True
        self.preferred_method = (
            "function"  # Function calling generally works best for OpenAI models
        )
