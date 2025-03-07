import os

from .BaseDriver import BaseDriver
from .gemini import GeminiDriver
from .grok import GrokDriver
from .openai import OpenAIDriver

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

drivers = {
    "gemini": GeminiDriver,
    "grok": GrokDriver,
    "openai": OpenAIDriver,
    "openrouter-gemini-2.0-flash-lite-preview-02-05": lambda: OpenAIDriver(
        model="google/gemini-2.0-flash-lite-preview-02-05:free",
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    ),
    "openrouter-grok-2-1212": lambda: OpenAIDriver(
        model="x-ai/grok-2-1212",
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    ),
    "openrouter-grok-3-beta": lambda: OpenAIDriver(
        model="x-ai/grok-beta",
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    ),
    "openrouter-deepseek-r1-free": lambda: OpenAIDriver(
        model="deepseek/deepseek-r1:free",
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    ),
}


def get_available_models() -> list[str]:
    return list(drivers.keys())


def get_driver(model: str) -> BaseDriver:
    if model in drivers:
        return drivers[model]()
    else:
        raise ValueError(f"Unsupported model: {model}")


__all__ = [
    "BaseDriver",
    "GeminiDriver",
    "GrokDriver",
    "OpenAIDriver",
    "get_driver",
    "get_available_models",
]
