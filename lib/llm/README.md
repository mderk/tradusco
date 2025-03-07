# LLM Drivers

This directory contains the LLM driver architecture for the Tradusco project.

## Overview

The LLM driver architecture is designed to provide a consistent interface for interacting with different LLM providers. The architecture consists of:

-   `BaseDriver`: An abstract base class that defines the interface for all LLM drivers
-   Concrete driver implementations for specific LLM providers (e.g., `GeminiDriver`, `GrokDriver`, `OpenAIDriver`)
-   Factory functions (`get_driver` and `get_available_models`) to create and list the appropriate drivers

## Available Drivers

The following drivers are currently implemented:

-   `GeminiDriver`: For Google's Gemini models
-   `GrokDriver`: For xAI's Grok models
-   `OpenAIDriver`: For OpenAI models and OpenRouter-based models

## Adding a New Driver

To add a new LLM driver:

1. Create a new directory under `lib/llm/` for your provider (e.g., `anthropic/`)
2. Create a new driver class that inherits from `BaseDriver` (e.g., `AnthropicDriver`)
3. Implement all the required methods from `BaseDriver`
4. Update the `drivers` dictionary in `__init__.py` to support your new driver

## Driver Interface

All drivers must implement the following methods:

-   `__init__(model: str, api_key: Optional[str] = None)`: Initialize the driver with a model name and optional API key
-   `translate(prompt: str, delay_seconds: float = 1.0, max_retries: int = 3) -> str`: Send a translation request
-   `translate_async(prompt: str, delay_seconds: float = 1.0, max_retries: int = 3) -> str`: Send an asynchronous translation request

## Factory Functions

The module provides two factory functions:

-   `get_driver(model: str) -> BaseDriver`: Create a driver instance for the specified model
-   `get_available_models() -> list[str]`: Get a list of all available models

## Example Usage

```python
from lib.llm import get_driver, get_available_models

# List all available models
models = get_available_models()
print(f"Available models: {models}")

# Get a driver for a specific model
driver = get_driver("gemini")

# Use the driver to translate text
result = driver.translate("Translate this text from English to French: Hello, world!")
print(result)

# Or use the async version
import asyncio

async def translate_async():
    result = await driver.translate_async("Translate this text from English to Spanish: Hello, world!")
    print(result)

asyncio.run(translate_async())
```
