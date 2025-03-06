# LLM Driver Architecture

This directory contains the LLM driver architecture for the AI Translator project.

## Overview

The LLM driver architecture is designed to provide a consistent interface for interacting with different LLM providers. The architecture consists of:

-   `BaseDriver`: An abstract base class that defines the interface for all LLM drivers
-   Concrete driver implementations for specific LLM providers (e.g., `GeminiDriver`)
-   A factory function (`get_llm`) to create the appropriate driver based on configuration

## Adding a New Driver

To add a new LLM driver:

1. Create a new directory under `ai_translator/lib/llm/` for your provider (e.g., `openai/`)
2. Create a new driver class that inherits from `BaseDriver` (e.g., `OpenAIDriver`)
3. Implement all the required methods from `BaseDriver`
4. Update the `get_llm` function in `__init__.py` to support your new driver

## Driver Interface

All drivers must implement the following methods:

-   `__init__(model: str, api_key: Optional[str] = None)`: Initialize the driver with a model name and optional API key
-   `translate(prompt: str, delay_seconds: float = 1.0, max_retries: int = 3) -> str`: Send a batch translation request

## Example Usage

```python
from ai_translator.lib.llm import get_llm

# Get a driver for a specific model
driver = get_llm("gemini")

# Use the driver to translate text
result = driver.translate("Translate this text from English to French: Hello, world!")
print(result)
```
