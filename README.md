# Tradusco

A Python utility for translating phrases using LLMs via Langchain.

## Project Structure

The utility works with projects that follow this structure:

-   `projects/[project_name]/config.json` - Project configuration
-   `projects/[project_name]/translations.csv` - Source and destination translations
-   `projects/[project_name]/[language]/progress.json` - Translation progress for each language

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set up your API keys in a `.env` file:

```
GEMINI_API_KEY = "your_api_key_here"
GEMINI_PROJECT_ID = "your_project_id_here"
OPENAI_API_KEY = "your_api_key_here"
GROK_API_KEY = "your_api_key_here"
OPENROUTER_API_KEY = "your_api_key_here"
```

## Usage

Run the translator with the following command:

```bash
python translate.py -p PROJECT_NAME -l LANGUAGE_CODE [-m MODEL] [-d DELAY] [-r RETRIES] [-b BATCH_SIZE] [--prompt PROMPT_FILE] [--list-models]
```

### Arguments

-   `-p, --project`: Name of the project folder in the `projects` directory
-   `-l, --lang`: Destination language code (must be defined in the project's config.json)
-   `-m, --model`: Model to use for translation (default: "gemini")
-   `-d, --delay`: Delay between API calls in seconds (default: 1.0)
-   `-r, --retries`: Maximum number of retries for failed API calls (default: 3)
-   `-b, --batch-size`: Number of phrases to translate in a single API call (default: 50)
-   `--prompt`: Path to a custom translation prompt file
-   `--list-models`: List available models and exit

### Examples

```bash
# Translate project "myproject" to Russian
python translate.py -p myproject -l ru

# Use a specific model with custom delay and batch size
python translate.py -p myproject -l fr -m openai -d 2.0 -b 20

# Use a custom prompt file
python translate.py -p myproject -l de --prompt custom_prompts/my_prompt.txt

# List available models
python translate.py --list-models
```

## Custom Prompts

The translator supports custom prompt templates for translations.
Default prompts are stored in the `prompts` directory, but you can provide your own prompt file using the
`--prompt` command-line argument.

Prompt templates use Python's string formatting syntax with the following variables:

-   `{base_language}` - The source language
-   `{dst_language}` - The destination language
-   `{phrases_json}` - The JSON array of phrases to translate

## Project Configuration

Each project should have a `config.json` file with the following structure:

```json
{
    "name": "project_name",
    "sourceFile": "translations.csv",
    "languages": ["en", "ru", "de", "fr", "es", "it", "tr", "zh", "ja"],
    "baseLanguage": "en",
    "keyColumn": "en"
}
```

## How It Works

1. The utility reads the project configuration and source translations
2. For each phrase in the base language, it checks if a translation already exists
3. If no translation exists, it collects phrases into batches for efficient translation
4. It sends batches of phrases to LLM for translation, reducing API calls
5. The translations are saved to both the CSV file and the language-specific progress.json file
6. Translations are cached to avoid redundant API calls
7. The utility implements rate limiting and retries to handle API quotas

## Core Classes

### TranslationProject

The main class that handles the translation process. Key methods:

-   `async create(project_name, dst_language, prompt_file=None)`: Factory method to create a new instance
-   `async translate(delay_seconds=1.0, max_retries=3, batch_size=50, model="gemini")`: Translate missing phrases
-   `get_available_models()`: Static method to get a list of available models

### LLM Drivers

The project uses a driver architecture for interacting with different LLM providers:

-   `BaseDriver`: Abstract base class that defines the interface for all LLM drivers
-   `GeminiDriver`, `GrokDriver`, `OpenAIDriver`: Concrete implementations for specific providers
-   `get_driver(model)`: Factory function to create the appropriate driver

## Batch Processing

The utility processes phrases in batches to improve efficiency and reduce API calls. Benefits include:

-   **Reduced API Costs**: Fewer API calls for the same number of translations
-   **Faster Processing**: Translating multiple phrases at once is more efficient
-   **Rate Limit Management**: Better handling of API rate limits
-   **Consistent Format**: All translations use the same JSON-based format, even for single phrases
-   **Improved Handling of Multiline Strings**: JSON encoding preserves line breaks and special characters

You can adjust the batch size with the `-b` or `--batch-size` parameter.

## Testing Your Setup

You can verify that your environment is properly set up by running:

```bash
python test_setup.py
```

This will check:

1. If the required environment variables are set
2. If the required packages are installed
3. If the connection to the LLM APIs works

## Running Tests

The project includes a comprehensive test suite to ensure functionality works as expected. To run the tests:

```bash
python -m tests.run_tests
```

Or you can run the test script directly:

```bash
./tests/run_tests.py
```

The test suite includes:

1. **TranslationProject Tests**: Tests for the main TranslationProject class functionality
2. **Batch Response Parsing Tests**: Tests for JSON parsing of LLM responses
3. **Prompt Handling Tests**: Tests for loading and handling prompt templates

If you want to run a specific test file, you can use:

```bash
python -m unittest tests.test_translation_project
python -m unittest tests.test_batch_response_parsing
python -m unittest tests.test_prompt_handling
```
