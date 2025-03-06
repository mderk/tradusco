from abc import ABC, abstractmethod
from typing import Optional


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
        pass

    @abstractmethod
    def translate_batch(
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
        pass

    @abstractmethod
    def translate_single(self, prompt: str, delay_seconds: float = 1.0) -> str:
        """
        Send a single translation request to the LLM.

        Args:
            prompt: The formatted prompt to send to the model
            delay_seconds: Delay after the request to avoid rate limiting

        Returns:
            The model's response content as a string

        Raises:
            Exception: If the API call fails
        """
        pass
