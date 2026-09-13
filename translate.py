#!/usr/bin/env python3
import argparse
import asyncio
import os
from pathlib import Path

# Load environment variables from .env file (must happen before importing drivers)
try:
    from dotenv import load_dotenv

    # Do not override env passed by the caller project/process.
    load_dotenv(override=False)
except ImportError:
    print(
        "Warning: python-dotenv not installed. Environment variables must be set manually."
    )

from lib.TranslationProject import TranslationProject
from lib.storage.filesystem import FileSystemStorageAdapter

DEBUG = os.environ.get("TRADUSCO_DEBUG")


def _parse_languages(raw: str) -> list[str]:
    return list(
        dict.fromkeys(
            language.strip() for language in raw.split(",") if language.strip()
        )
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
    parser.add_argument(
        "-l", "--lang", help="Comma-separated destination language codes"
    )
    parser.add_argument(
        "--reference-langs",
        default="",
        help="Comma-separated reviewed translation columns used only for disambiguation",
    )
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
        "--request-timeout",
        type=float,
        default=120.0,
        help="Maximum seconds for one model request including SDK retries (default: 120).",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=50,
        help="Number of phrases to translate in one batch (default: 50)",
    )
    parser.add_argument(
        "--batch-max-input-tokens",
        type=int,
        default=65536,
        help="Maximum assembled input tokens for a translation batch (default: 65536)",
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
    # Add storage adapter argument
    parser.add_argument(
        "--storage",
        choices=["filesystem"],
        default="filesystem",
        help="Storage adapter to use (default: filesystem)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    # Add regenerate argument
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate all translations, ignoring existing ones and saved progress.",
    )
    parser.add_argument(
        "--fallback-model",
        help=(
            "Optional fallback model. Retries each failed batch once, then runs a "
            "gap-filling pass for any still-missing/invalid cells (without --regenerate)."
        ),
    )

    args = parser.parse_args()

    if args.debug:
        os.environ["TRADUSCO_DEBUG"] = "true"

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
            # Allow raw OpenRouter model IDs (e.g. "google/gemini-2.5-flash")
            # when OPENROUTER_API_KEY is present.
            if "/" in args.model and os.environ.get("OPENROUTER_API_KEY"):
                pass
            else:
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

        # Create storage adapter
        if args.storage == "filesystem":
            storage = FileSystemStorageAdapter(
                project_path,
                prompt_file=args.prompt,
                context_file=args.context_file,
            )
            dst_languages = _parse_languages(args.lang)
            reference_languages = _parse_languages(args.reference_langs)
            if not dst_languages:
                parser.error("--lang must contain at least one language code")
            config = await storage.load_config(project_name)
            invalid_languages = [
                language
                for language in dst_languages
                if language not in config.languages
            ]
            if invalid_languages:
                parser.error(
                    f"Invalid language(s): {', '.join(invalid_languages)}. "
                    f"Available languages: {', '.join(config.languages)}"
                )
            invalid_references = [
                language
                for language in reference_languages
                if language not in config.languages
            ]
            if invalid_references:
                parser.error(
                    f"Invalid reference language(s): {', '.join(invalid_references)}. "
                    f"Available languages: {', '.join(config.languages)}"
                )
            overlap = set(dst_languages) & set(reference_languages)
            if overlap:
                parser.error(
                    "Reference languages cannot be target languages: "
                    + ", ".join(sorted(overlap))
                )
            storage.set_active_languages(dst_languages)
            storage.set_overwrite_active_language(bool(args.regenerate))
        else:
            parser.error(f"Invalid storage adapter: {args.storage}")

        # Create and initialize the translator asynchronously
        translator = await TranslationProject.create(
            project_name=project_name,
            dst_languages=dst_languages,
            context=args.context,
            storage=storage,
            reference_languages=reference_languages,
        )

        await translator.translate(
            delay_seconds=args.delay,
            max_retries=args.retries,
            request_timeout=args.request_timeout,
            batch_size=args.batch_size,
            model=args.model,
            batch_max_input_tokens=args.batch_max_input_tokens,
            translation_method=args.method,
            regenerate=args.regenerate,
            fallback_model=args.fallback_model,
        )
    except Exception as e:
        print(f"Error: {e}")
        if DEBUG:
            raise e
        return 1

    return 0


def main():
    return asyncio.run(async_main())


if __name__ == "__main__":
    exit(main())
