# AI Translator Utility

A Python utility for translating phrases using LLMs (Gemini 2 Flash) via Langchain.

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

3. Set up your Google API key in a `.env` file:

```
GEMINI_API_KEY = "your_api_key_here"
GEMINI_PROJECT_ID = "your_project_id_here"
```

## Usage

Run the translator with the following command:

```bash
python translate.py -p PROJECT_NAME -l LANGUAGE_CODE [-d DELAY] [-r RETRIES] [-b BATCH_SIZE] [--batch-prompt BATCH_PROMPT_FILE] [--single-prompt SINGLE_PROMPT_FILE]
```

Where:

-   `PROJECT_NAME` is the name of the project folder in the `projects` directory
-   `LANGUAGE_CODE` is the destination language code (must be defined in the project's config.json)
-   `DELAY` (optional) is the delay between API calls in seconds (default: 1.0)
-   `RETRIES` (optional) is the maximum number of retries for failed API calls (default: 3)
-   `BATCH_SIZE` (optional) is the number of phrases to translate in a single API call (default: 50)
-   `BATCH_PROMPT_FILE` (optional) is the path to a custom batch translation prompt file
-   `SINGLE_PROMPT_FILE` (optional) is the path to a custom single phrase translation prompt file

Example:

```bash
python translate.py -p booty -l ru -d 2.0 -r 5 -b 20 --batch-prompt custom_prompts/my_batch_prompt.txt
```

## Custom Prompts

The translator supports custom prompt templates for both batch translations and individual phrase translations.
Default prompts are stored in the `prompts` directory, but you can provide your own prompt files using the
`--batch-prompt` and `--single-prompt` command-line arguments.

Prompt templates use Python's string formatting syntax with the following variables:

For batch prompts:

-   `{base_language}` - The source language
-   `{dst_language}` - The destination language
-   `{phrases_text}` - The numbered list of phrases to translate

For single prompts:

-   `{base_language}` - The source language
-   `{dst_language}` - The destination language
-   `{phrase}` - The individual phrase to translate

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
4. It sends batches of phrases to Gemini 2 Flash for translation, reducing API calls
5. The translations are saved to both the CSV file and the language-specific progress.json file
6. Translations are cached to avoid redundant API calls
7. The utility implements rate limiting and retries to handle API quotas
8. If batch translation fails, it falls back to individual translation as a recovery mechanism

## Batch Processing

The utility processes phrases in batches to improve efficiency and reduce API calls. Benefits include:

-   **Reduced API Costs**: Fewer API calls for the same number of translations
-   **Faster Processing**: Translating multiple phrases at once is more efficient
-   **Rate Limit Management**: Better handling of API rate limits
-   **Fallback Mechanism**: If batch translation fails, individual translation is used as a fallback

You can adjust the batch size with the `-b` or `--batch-size` parameter.

## Testing Your Setup

You can verify that your environment is properly set up by running:

```bash
python test_setup.py
```

This will check:

1. If the required environment variables are set
2. If the required packages are installed
3. If the connection to the Gemini API works
