# Tradusco

A Python utility for translating texts using LLMs.

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set up your API keys in a `.env` file:

```
GEMINI_API_KEY = "your_api_key_here"
OPENAI_API_KEY = "your_api_key_here"
GROK_API_KEY = "your_api_key_here"
OPENROUTER_API_KEY = "your_api_key_here"
```

## Creating a New Project

You can create a new translation project using the `create_project.py` script. This script sets up the necessary directory structure and configuration files based on a CSV file containing your translations.

### Usage

```bash
python create_project.py --path PROJECT_PATH --csv CSV_PATH --base-lang BASE_LANGUAGE --key KEY_COLUMN
```

Or using the short options:

```bash
python create_project.py -p PROJECT_PATH -c CSV_PATH -b BASE_LANGUAGE -k KEY_COLUMN
```

### Arguments

-   `--path`, `-p` (required): Path where the project will be created (the directory name will be used as the project name)
-   `--csv`, `-c` (required): Path to the CSV file containing translations
-   `--base-lang`, `-b` (required): Base language code (e.g., "en" for English)
-   `--key`, `-k` (required): Column name containing translation keys
-   `--ignore-columns`, `-i` (optional): Comma-separated list of column names to ignore (default: "context")

### Examples

```bash
# Create a new project with all required parameters
python create_project.py --path projects/myproject --csv data/translations.csv --base-lang en --key id

# Same using short options
python create_project.py -p projects/myproject -c data/translations.csv -b en -k id

# Create project in a custom location
python create_project.py -p /path/to/custom/project -c data/translations.csv -b en -k id

# Ignore specific columns
python create_project.py -p projects/myproject -c data/translations.csv -b en -k id -i context,notes,comments
```

### What It Does

1. Creates the project directory at the specified path
2. Reads and validates the CSV file
3. Creates a `config.json` file with project settings
4. Creates subdirectories for each language found in the CSV
5. Copies the source CSV file to the project directory

### CSV File Format

The CSV file should have:

-   A column for translation keys (specified by the `--key` parameter)
-   Language code columns for each supported language
-   Each row contains the translation key and corresponding translations

Example CSV format:

```csv
id,en,fr,es
welcome_message,Welcome,Bienvenue,Bienvenido
goodbye_message,Goodbye,Au revoir,Adiós
```

### Project Structure

The utility works with projects that follow this structure:

-   `projects/[project_name]/config.json` - Project configuration
-   `projects/[project_name]/translations.csv` - Source and destination translations
-   `projects/[project_name]/[language]/progress.json` - Translation progress for each language

### Project Configuration

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

## Translating Phrases

Run the translator with the following command:

```bash
python translate.py -p PROJECT_PATH -l LANGUAGE_CODE [-m MODEL] [-d DELAY] [-r RETRIES] [-b BATCH_SIZE] [--batch-max-tokens MAX_TOKENS] [--prompt PROMPT_FILE] [--context CONTEXT] [--context-file CONTEXT_FILE] [--method METHOD] [--list-models]
```

### Arguments

-   `-p, --project`: Path to the project directory (either absolute or relative path)
-   `-l, --lang`: Destination language code (must be defined in the project's config.json)
-   `-m, --model`: Model to use for translation (default: "gemini")
-   `-d, --delay`: Delay between API calls in seconds (default: 1.0)
-   `-r, --retries`: Maximum number of retries for failed API calls (default: 3)
-   `-b, --batch-size`: Number of phrases to translate in a single API call (default: 50)
-   `--batch-max-tokens`: Maximum number of tokens in a translation batch (default: 2048)
-   `--prompt`: Path to a custom translation prompt file
-   `--context`: Translation context as a text string to guide the translation style and tone
-   `--context-file`: Path to a file containing translation context
-   `--method`: Translation method to use: auto (automatic selection based on model capabilities), standard (prompt-based), structured (JSON output), or function (function calling) (default: "auto")
-   `--list-models`: List available models and exit

### Examples

```bash
# Translate project in the "projects/myproject" directory to Russian
python translate.py -p projects/myproject -l ru

# Using an absolute path
python translate.py -p /path/to/my/project -l fr -m openai

# Using a relative path
python translate.py -p ./custom_projects/myproject -l de -b 30

# Use a specific model with custom delay and batch size
python translate.py -p projects/myproject -l fr -m openai -d 2.0 -b 20

# Set both batch size and maximum batch tokens
python translate.py -p projects/myproject -l de -b 30 --batch-max-tokens 16384

# Use a custom prompt file
python translate.py -p projects/myproject -l de --prompt custom_prompts/my_prompt.txt

# Use structured output method (JSON schema)
python translate.py -p projects/myproject -l es --method structured

# Use function calling method
python translate.py -p projects/myproject -l it --method function

# Use automatic method selection (recommended)
python translate.py -p projects/myproject -l fr --method auto

# List available models
python translate.py --list-models
```

### Custom Prompts

The translator supports custom prompt templates for translations.
Default prompts are stored in the `prompts` directory, but you can provide your own prompt file using the
`--prompt` command-line argument.

Prompt templates use Python's string formatting syntax with the following variables:

-   `{base_language}` - The source language
-   `{dst_language}` - The destination language
-   `{phrases_json}` - The JSON array of phrases to translate
-   `{context}` - Global translation context (if any)
-   `{phrase_contexts}` - Individual phrase contexts (if any)

### Translation Contexts

The translator supports both global and phrase-specific contexts to improve translation accuracy. You can provide context in several ways:

#### Global Context

1. **Command-line argument**:

    ```bash
    python translate.py -p myproject -l es --context "This is a videogame translation with casual tone"
    ```

2. **Context file via command-line**:

    ```bash
    python translate.py -p myproject -l es --context-file path/to/context.txt
    ```

3. **Project-level context file**:
   Create either `context.md` or `context.txt` in your project directory:
    ```
    projects/
      myproject/
        context.md  # or context.txt
        config.json
        translations.csv
    ```

All global context sources are combined if multiple are provided.

#### Phrase-specific Context

You can add context for individual phrases by including a "context" column in your translations CSV file:

```csv
en,es,context
Hello,Hola,"Formal business setting"
Goodbye,,"Casual conversation between friends"
Welcome,,"Greeting at hotel entrance"
```

The context column provides specific instructions or background for translating individual phrases. This is particularly useful when:

-   The same word needs different translations based on context
-   There are cultural nuances to consider
-   The phrase has a specific tone or style requirement
-   Technical terms need specific domain context

#### How Context is Used

1. **Global Context**: Applied to all translations in the batch. Useful for:

    - Setting overall tone (formal/casual)
    - Defining domain (technical/medical/legal)
    - Specifying target audience
    - General cultural considerations

2. **Phrase-specific Context**: Applied only to individual phrases. Useful for:
    - Word sense disambiguation
    - Specific tone requirements
    - Cultural adaptations
    - Technical term clarification

The LLM receives both types of context in a structured format, ensuring accurate and contextually appropriate translations.

### How It Works

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

### Batch Processing

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

The project includes a comprehensive test suite to ensure functionality works as expected. To run the tests using pytest:

```bash
# Run all tests (excluding integration tests)
pytest

# Run tests with verbose output
pytest -v

# Include integration tests (will make real API calls)
pytest --run-integration
```

Integration tests are excluded from normal test runs by default to avoid unnecessary API usage and costs. This is configured in the `pytest.ini` file using markers.

The test suite includes:

1. **TranslationProject Tests**: Tests for the main TranslationProject class functionality
2. **Translation Tool Tests**: Tests for the translation tool and API interaction
3. **File Operation Tests**: Tests for CSV and JSON file handling
4. **Prompt Handling Tests**: Tests for loading and handling prompt templates

If you want to run a specific test file or test, you can use pytest with more specific targeting:

```bash
# Run a specific test file
pytest tests/test_translation_project.py -v

# Run a specific test class
pytest tests/test_translation_project.py::TestTranslationProject -v

# Run a specific test method
pytest tests/test_translation_project.py::TestTranslationProject::test_translate -v

# Run tests matching a specific keyword
pytest -k "translate" -v
```

### Integration Tests

The project also includes integration tests that make real API calls to test the translator against actual LLM services. These tests are **excluded from normal test runs** to avoid unnecessary API usage and costs.

#### Running Integration Tests

```bash
# Use the provided script (recommended)
./tests/run_integration_tests.sh

# Run all integration tests directly with pytest
pytest -k "integration" -v

# Run specific integration tests file
pytest -k "integration" tests/test_integration_translation_methods.py -v

# Run a specific integration test
pytest -k "integration" tests/test_integration_translation_methods.py::TestIntegrationTranslationMethods::test_standard_method -v
```

#### What Integration Tests Verify

The integration tests verify that all three translation methods work with real LLM APIs:

1. **Standard Method**: Uses basic prompt formatting and parses the JSON response
2. **Structured Method**: Uses the structured output API for more reliable JSON responses
3. **Function Method**: Uses function calling to guide the response format

The tests also include a comparison test that runs all three methods on the same inputs and displays a side-by-side comparison of the results.

#### Requirements for Integration Tests

To run integration tests, you need:

1. An active Gemini API key in your `.env` file
2. An active OpenRouter API key in your `.env` file (for specific models)
3. Internet connectivity to make API calls
