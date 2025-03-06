#!/usr/bin/env python3
import argparse

from lib.TranslationProject import TranslationProject

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "Warning: python-dotenv not installed. Environment variables must be set manually."
    )


def main():
    parser = argparse.ArgumentParser(description="AI Translation Utility using LLMs")
    parser.add_argument("-p", "--project", required=True, help="Project name")
    parser.add_argument("-l", "--lang", required=True, help="Destination language code")
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
        help="Number of phrases to translate in a single API call (default: 50)",
    )
    parser.add_argument(
        "--prompt",
        help="Path to a custom translation prompt file",
    )

    args = parser.parse_args()

    try:
        translator = TranslationProject(
            args.project,
            args.lang,
            prompt_file=args.prompt,
        )
        translator.translate(
            delay_seconds=args.delay,
            max_retries=args.retries,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
