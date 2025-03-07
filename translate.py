#!/usr/bin/env python3
import argparse
import asyncio

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
    parser.add_argument("-p", "--project", help="Project name")
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
        "--batch-max-bytes",
        type=int,
        default=8192,  # 8KB is a reasonable default for most LLMs
        help="Maximum size in bytes for a translation batch (default: 8192)",
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

        # Create and initialize the translator asynchronously
        translator = await TranslationProject.create(
            args.project,
            args.lang,
            prompt_file=args.prompt,
            context=args.context,
            context_file=args.context_file,
        )

        await translator.translate(
            delay_seconds=args.delay,
            max_retries=args.retries,
            batch_size=args.batch_size,
            model=args.model,
            batch_max_bytes=args.batch_max_bytes,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


def main():
    return asyncio.run(async_main())


if __name__ == "__main__":
    exit(main())
