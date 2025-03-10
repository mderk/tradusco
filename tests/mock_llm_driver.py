import sys
import os
import json
import re
from typing import Optional, Any, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.llm.BaseDriver import BaseDriver


class MockResponse:
    """Mock LangChain response object that mimics the structure of a real LLM response."""

    def __init__(self, content: str):
        self.content = content


class MockLLMDriver(BaseDriver):
    """
    A mock LLM driver for testing that returns predefined responses.
    This avoids making actual API calls while still testing the real parsing logic.
    """

    def __init__(self, model: str = "mock-model", api_key: Optional[str] = None):
        """Initialize the mock LLM driver with a model name."""
        self.model = model
        self.llm = self  # Set self as the LLM to handle invoke calls

        # Dictionary of prompt patterns to responses
        self.response_map = {}

        # Default responses for common prompt types
        self._setup_default_responses()

    def _setup_default_responses(self):
        """Set up default responses for common prompt types."""
        # Default translation response
        translation_response = json.dumps(
            {"Hello": "Hola", "Goodbye": "Adiós", "Welcome": "Bienvenido"}
        )
        self.register_response(
            r"Translate.*English to Spanish", f"```json\n{translation_response}\n```"
        )

        # Default JSON fix response
        invalid_json = r'{"Hello": "Hola", "Goodbye": "Adiós", Welcome: "Bienvenido"}'
        fixed_json = '{"Hello": "Hola", "Goodbye": "Adiós", "Welcome": "Bienvenido"}'
        self.register_response(
            rf".*{re.escape(invalid_json)}.*", f"```json\n{fixed_json}\n```"
        )

    def register_response(self, prompt_pattern: str, response: str):
        """
        Register a response for a given prompt pattern.

        Args:
            prompt_pattern: Regex pattern to match against prompts
            response: Response text to return when the pattern matches
        """
        self.response_map[prompt_pattern] = response

    def _find_matching_response(self, prompt: str) -> str:
        """Find a matching response for the given prompt."""
        for pattern, response in self.response_map.items():
            if re.search(pattern, prompt, re.DOTALL):
                return response

        # Default response if no pattern matches
        return '{"default": "response"}'

    def invoke(self, prompt: str) -> MockResponse:
        """
        Mock invoke method that returns a predefined response based on the prompt.

        Args:
            prompt: The prompt to respond to

        Returns:
            A MockResponse object containing the predefined response
        """
        response_text = self._find_matching_response(prompt)
        return MockResponse(response_text)

    async def ainvoke(self, prompt: str) -> MockResponse:
        """
        Async version of the invoke method.

        Args:
            prompt: The prompt to respond to

        Returns:
            A MockResponse object containing the predefined response
        """
        return self.invoke(prompt)

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Mock token counting function.

        Args:
            text: The text to count tokens for

        Returns:
            A simple approximation of the token count
        """
        if not text:
            return 0
        # Simple approximation: ~4 characters per token
        return max(1, len(text) // 4)
