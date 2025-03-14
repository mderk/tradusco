#!/usr/bin/env python3
import argparse
import asyncio
from pathlib import Path

from lib.TranslationProject import TranslationProject

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "Warning: python-dotenv not installed. Environment variables must be set manually."
    )


async def async_main():
    parser = argparse.ArgumentParser(
        description="Tradusco - Translation Utility using LLMs"
    )

    # List models option
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )

    # Project and language arguments
    parser.add_argument("-p", "--project", help="Path to the project directory")
    parser.add_argument("-l", "--lang", help="Destination language code")
    parser.add_argument(
        "-m",
        "--model",
        default="gemini",
        help="Model to use for translation (default: gemini)",
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)",
    )
    parser.add_argument(
        "-r",
        "--retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed API calls (default: 3)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=50,
        help="Number of phrases to translate in one batch (default: 50)",
    )
    parser.add_argument(
        "--batch-max-tokens",
        type=int,
        default=2048,  # 2048 tokens is a reasonable default for most LLMs
        help="Maximum number of tokens for a translation batch (default: 2048)",
    )
    parser.add_argument(
        "--prompt",
        help="Path to a custom translation prompt file",
    )
    # Add context arguments
    parser.add_argument(
        "--context",
        help="Translation context as a text string",
    )
    parser.add_argument(
        "--context-file",
        help="Path to a file containing translation context",
    )
    # Add translation method argument
    parser.add_argument(
        "--method",
        choices=["standard", "structured", "function", "auto"],
        default="auto",
        help="Translation method to use: auto (recommended), standard (prompt-based), structured (JSON output), or function (function calling) (default: auto)",
    )

    args = parser.parse_args()

    try:
        # If --list-models is specified, list models and exit
        if args.list_models:
            print("Available models:")
            for model in TranslationProject.get_available_models():
                print(f"- {model}")
            return 0

        # For translation, project and lang are required
        if not args.project or not args.lang:
            parser.error("--project and --lang are required for translation")

        # Validate the model
        available_models = TranslationProject.get_available_models()
        if args.model not in available_models:
            parser.error(
                f"Invalid model: {args.model}. Use --list-models to see available models."
            )

        # Create a Path object from the project path
        project_path = Path(args.project).resolve()

        # Get the project name from the directory name
        project_name = project_path.name

        # Check if the project directory exists
        if not project_path.exists() or not project_path.is_dir():
            parser.error(f"Project directory does not exist: {project_path}")

        # Check if config.json exists in the project directory
        if not (project_path / "config.json").exists():
            parser.error(f"config.json not found in project directory: {project_path}")

        # Create and initialize the translator asynchronously
        translator = await TranslationProject.create(
            project_name=project_name,
            dst_language=args.lang,
            prompt_file=args.prompt,
            context=args.context,
            context_file=args.context_file,
            project_path=project_path,
        )

        await translator.translate(
            delay_seconds=args.delay,
            max_retries=args.retries,
            batch_size=args.batch_size,
            model=args.model,
            batch_max_tokens=args.batch_max_tokens,
            translation_method=args.method,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


def main():
    return asyncio.run(async_main())


if __name__ == "__main__":
    exit(main())
