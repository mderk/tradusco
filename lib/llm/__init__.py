import os

from .BaseDriver import BaseDriver
from .gemini import GeminiDriver
from .grok import GrokDriver
from .openai import OpenAIDriver

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


def _openrouter_driver(openrouter_model_id: str) -> OpenAIDriver:
    """
    Create an OpenAI-compatible driver pointing at OpenRouter.

    We intentionally default capability flags to "standard" to avoid relying on
    tool calling / response_format support across different OpenRouter models.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. Please check your environment."
        )

    driver = OpenAIDriver(
        model=openrouter_model_id,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    # Be conservative by default. Users can still request other methods,
    # but auto mode should work broadly.
    driver.supports_structured_output = False
    driver.supports_function_calling = False
    driver.preferred_method = "standard"
    return driver


drivers = {
    "gemini": GeminiDriver,
    "grok": GrokDriver,
    "openai": OpenAIDriver,
    # Explicit OpenRouter aliases (stable names)
    "openrouter-gemini-2.0-flash-lite-preview-02-05": lambda: _openrouter_driver(
        "google/gemini-2.0-flash-lite-preview-02-05:free"
    ),
    "openrouter-grok-2-1212": lambda: _openrouter_driver("x-ai/grok-2-1212"),
    "openrouter-grok-3-beta": lambda: _openrouter_driver("x-ai/grok-beta"),
    "openrouter-deepseek-r1-free": lambda: _openrouter_driver("deepseek/deepseek-r1:free"),
}


def get_available_models() -> list[str]:
    return list(drivers.keys())


def get_driver(model: str) -> BaseDriver:
    if model in drivers:
        return drivers[model]()

    # Allow raw OpenRouter model IDs, e.g. "google/gemini-2.5-flash".
    if "/" in model:
        return _openrouter_driver(model)

    raise ValueError(f"Unsupported model: {model}")


__all__ = [
    "BaseDriver",
    "GeminiDriver",
    "GrokDriver",
    "OpenAIDriver",
    "get_driver",
    "get_available_models",
]
