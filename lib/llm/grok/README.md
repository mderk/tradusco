# Grok Driver for AI Translator

This module provides integration with xAI's Grok language model for the AI Translator project using the official LangChain xAI integration.

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
from ai_translator.lib.llm import get_llm

# Initialize the Grok driver
driver = get_llm("grok")

# Or initialize it directly
from ai_translator.lib.llm.grok import GrokDriver
driver = GrokDriver(model="grok-2-1212")  # Default model

# Use the driver for translation
response = driver.translate_single("Translate this text to French: Hello world")
print(response)
```

## Available Models

-   `grok-2-1212` (default)
-   `grok-2-vision-1212` (for vision capabilities)
-   `grok-beta`
-   `grok-vision-beta`

For the most up-to-date list of available models, refer to the [xAI documentation](https://platform.xai.org/docs/models).

## API Reference

The Grok driver implements the BaseDriver interface with the following methods:

-   `translate_batch(prompt, delay_seconds=1.0, max_retries=3)`: Send a batch translation request
-   `translate_single(prompt, delay_seconds=1.0)`: Send a single translation request

## Implementation Details

This driver uses the LangChain xAI integration (`langchain-xai`) to interact with the Grok API. The integration provides a clean interface to the Grok models and handles the API communication details.

## Notes

Make sure your API key has the necessary permissions for the models you want to use. For more information about the xAI API and available models, visit the [xAI Platform documentation](https://platform.xai.org/docs).
