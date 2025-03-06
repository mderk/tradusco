from .BaseDriver import BaseDriver
from .gemini import GeminiDriver


def get_llm(model: str) -> BaseDriver:
    if model == "gemini":
        return GeminiDriver()
    else:
        raise ValueError(f"Unsupported model: {model}")


__all__ = ["BaseDriver", "GeminiDriver", "get_llm"]
