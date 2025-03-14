#!/usr/bin/env python3

import os
import sys
import json
import csv
import shutil
import argparse
from pathlib import Path


async def create_project():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create a new translation project")
    parser.add_argument(
        "--path",
        "-p",
        required=True,
        help="Project path (directory where project will be created)",
    )
    parser.add_argument("--csv", "-c", required=True, help="Path to CSV file")
    parser.add_argument("--base-lang", "-b", required=True, help="Base language code")
    parser.add_argument(
        "--key", "-k", required=True, help="Column name containing translation keys"
    )
    parser.add_argument(
        "--ignore-columns",
        "-i",
        help="Column names to ignore",
        default="context",
        type=lambda x: x.split(","),
        required=False,
    )

    # Parse arguments
    args = parser.parse_args()

    project_path = args.path
    csv_path = args.csv
    base_lang = args.base_lang
    key_column = args.key
    ignore_columns = args.ignore_columns

    # Define project directory
    project_dir = Path(project_path).resolve()
    project_name = project_dir.name

    try:
        # Resolve the CSV path relative to current working directory
        resolved_csv_path = Path(csv_path).resolve()
        print(f"Trying to read CSV from: {resolved_csv_path}")

        # Read and parse CSV file
        with open(resolved_csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            records = list(reader)

            if not records:
                print("CSV file is empty", file=sys.stderr)
                sys.exit(1)

            # Validate key column exists
            if key_column not in records[0]:
                print(
                    f'Key column "{key_column}" not found in CSV file', file=sys.stderr
                )
                print(
                    f'Available columns: {", ".join(records[0].keys())}',
                    file=sys.stderr,
                )
                sys.exit(1)

            # Get language codes from CSV headers, excluding the key column
            languages = [
                col
                for col in records[0].keys()
                if col != key_column
                and (col not in ignore_columns if ignore_columns else True)
                or (key_column == base_lang and col == base_lang)
            ]

            # Validate base language exists in CSV
            if base_lang not in languages:
                print(
                    f'Base language "{base_lang}" not found in CSV file',
                    file=sys.stderr,
                )
                print(f'Available languages: {", ".join(languages)}', file=sys.stderr)
                sys.exit(1)

            # Create project directory if it doesn't exist
            os.makedirs(project_dir, exist_ok=True)

            project_exists = os.path.exists(project_dir / "config.json")
            action_word = "updated" if project_exists else "created"

            # Create config file
            config = {
                "name": project_name,
                "sourceFile": os.path.basename(csv_path),
                "languages": languages,
                "baseLanguage": base_lang,
                "keyColumn": key_column,
            }

            with open(
                project_dir / "config.json", "w", encoding="utf-8"
            ) as config_file:
                json.dump(config, config_file, indent=2)

            # Create language directories
            for lang in languages:
                os.makedirs(project_dir / lang, exist_ok=True)

            # Copy CSV file to project directory
            shutil.copy2(resolved_csv_path, project_dir / os.path.basename(csv_path))

            print(
                f'Project "{project_name}" {action_word} successfully at {project_dir}'
            )
            print(f'Languages detected: {", ".join(languages)}')
            print(f"Base language: {base_lang}")
            print(f'Using "{key_column}" as translation key column')

    except FileNotFoundError:
        print(f"CSV file not found: {resolved_csv_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error creating/updating project: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import asyncio

    asyncio.run(create_project())
