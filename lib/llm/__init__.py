from .BaseDriver import BaseDriver
from .gemini import GeminiDriver
from .grok import GrokDriver


def get_llm(model: str) -> BaseDriver:
    if model == "gemini":
        return GeminiDriver()
    elif model == "grok":
        return GrokDriver()
    else:
        raise ValueError(f"Unsupported model: {model}")


__all__ = ["BaseDriver", "GeminiDriver", "GrokDriver", "get_llm"]
