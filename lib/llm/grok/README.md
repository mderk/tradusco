# Grok Driver for Tradusco

This module provides integration with xAI's Grok language model for the Tradusco project using the official LangChain xAI integration.

## Setup

1. Obtain a Grok API key from xAI.
2. Set the API key in your environment variables:
    ```
    export GROK_API_KEY="your-api-key-here"
    ```
    Or add it to your `.env` file:
    ```
    GROK_API_KEY=your-api-key-here
    ```
3. Ensure you have the required packages installed:
    ```
    pip install langchain-xai
    ```

## Usage

To use the Grok driver in your translation project:

```python
from lib.llm import get_driver

# Initialize the Grok driver
driver = get_driver("grok")

# Or initialize it directly
from lib.llm.grok import GrokDriver
driver = GrokDriver(model="grok-2-1212")  # Default model

# Use the async version for translation
import asyncio

async def translate_async():
    response = await driver.translate_async("Translate this text to Spanish: Hello world")
    print(response)

asyncio.run(translate_async())
```

## Available Models

For the most up-to-date list of available models, refer to the [xAI documentation](https://platform.xai.org/docs/models).

## API Reference

The Grok driver implements the BaseDriver interface with the following methods:

-   `__init__(model="grok-2-1212", api_key=None)`: Initialize the driver with a model name and optional API key
-   `translate_async(prompt, delay_seconds=1.0, max_retries=3)`: Send an asynchronous translation request
-   `translate_structured_async(prompt, output_schema, delay_seconds=1.0, max_retries=3)`: Send an asynchronous request for structured output
-   `translate_function_async(prompt, functions, function_name=None, delay_seconds=1.0, max_retries=3)`: Send an asynchronous request that can call a function

## Implementation Details

This driver uses the LangChain xAI integration (`langchain-xai`) to interact with the Grok API. The integration provides a clean interface to the Grok models and handles the API communication details.

## Notes

Make sure your API key has the necessary permissions for the models you want to use. For more information about the xAI API and available models, visit the [xAI Platform documentation](https://platform.xai.org/docs).
