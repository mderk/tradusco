import os
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from collections.abc import Callable

from .BaseDriver import BaseDriver
from .gemini import GeminiDriver
from .grok import GrokDriver
from .openai import OpenAIDriver

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_OPENROUTER_SUPPORTED_PARAMETERS_BY_MODEL: dict[str, set[str]] | None = None


def _fetch_openrouter_supported_parameters_by_model() -> dict[str, set[str]]:
    """
    Fetch OpenRouter model capability metadata once per process.

    OpenRouter exposes this endpoint publicly. If it cannot be reached (e.g. some
    test/CI environments), we fall back to conservative defaults.
    """
    global _OPENROUTER_SUPPORTED_PARAMETERS_BY_MODEL
    if _OPENROUTER_SUPPORTED_PARAMETERS_BY_MODEL is not None:
        return _OPENROUTER_SUPPORTED_PARAMETERS_BY_MODEL

    url = f"{OPENROUTER_BASE_URL}/models"
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=10) as resp:  # noqa: S310 (public API)
            payload = json.loads(resp.read().decode("utf-8"))

        mapping: dict[str, set[str]] = {}
        for model in payload.get("data", []):
            model_id = model.get("id")
            supported = model.get("supported_parameters") or []
            if not isinstance(model_id, str) or not isinstance(supported, list):
                continue
            mapping[model_id] = {p for p in supported if isinstance(p, str)}

        _OPENROUTER_SUPPORTED_PARAMETERS_BY_MODEL = mapping
        return mapping
    except (HTTPError, URLError, TimeoutError, ValueError):
        _OPENROUTER_SUPPORTED_PARAMETERS_BY_MODEL = {}
        return _OPENROUTER_SUPPORTED_PARAMETERS_BY_MODEL
    except Exception:
        _OPENROUTER_SUPPORTED_PARAMETERS_BY_MODEL = {}
        return _OPENROUTER_SUPPORTED_PARAMETERS_BY_MODEL


def _openrouter_supported_parameters(model_id: str) -> set[str]:
    return _fetch_openrouter_supported_parameters_by_model().get(model_id, set())


def _openrouter_driver(openrouter_model_id: str) -> OpenAIDriver:
    """
    Create an OpenAI-compatible driver pointing at OpenRouter.

    Capability flags are derived from the OpenRouter Models API (`supported_parameters`).
    If capability probing fails, we fall back to conservative defaults.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. Please check your environment."
        )

    driver = OpenAIDriver(
        model=openrouter_model_id,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )

    supported = _openrouter_supported_parameters(openrouter_model_id)

    # Structured Outputs mode:
    # - auto (default): enable only if OpenRouter reports `structured_outputs`
    # - force: enable regardless of metadata (useful for testing)
    # - off: disable regardless of metadata
    so_mode = (os.environ.get("TRADUSCO_OPENROUTER_SO") or "auto").strip().lower()
    so_supported = "structured_outputs" in supported

    # Tool calling support requires both tools and tool_choice (we force the tool call).
    tools_supported = ("tools" in supported) and ("tool_choice" in supported)

    supports_structured_output = (
        True
        if so_mode == "force"
        else (False if so_mode == "off" else so_supported)
    )

    driver.supports_structured_output = supports_structured_output
    driver.supports_function_calling = tools_supported
    driver.preferred_method = (
        "structured"
        if supports_structured_output
        else ("function" if tools_supported else "standard")
    )
    return driver


drivers: dict[str, Callable[[], BaseDriver]] = {
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
