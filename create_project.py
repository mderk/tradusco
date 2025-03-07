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
    parser.add_argument("--name", "-n", required=True, help="Project name")
    parser.add_argument("--csv", "-c", required=True, help="Path to CSV file")
    parser.add_argument("--base-lang", "-b", required=True, help="Base language code")
    parser.add_argument(
        "--key", "-k", required=True, help="Column name containing translation keys"
    )

    # Parse arguments
    args = parser.parse_args()

    project_name = args.name
    csv_path = args.csv
    base_lang = args.base_lang
    key_column = args.key

    # Define project directories
    projects_dir = Path(os.getcwd()) / "projects"
    project_dir = projects_dir / project_name

    # Ensure projects directory exists
    os.makedirs(projects_dir, exist_ok=True)

    # Check if project already exists
    if project_dir.exists():
        print(f'Project "{project_name}" already exists', file=sys.stderr)
        sys.exit(1)

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
                if col != key_column or (key_column == base_lang and col == base_lang)
            ]

            # Validate base language exists in CSV
            if base_lang not in languages:
                print(
                    f'Base language "{base_lang}" not found in CSV file',
                    file=sys.stderr,
                )
                print(f'Available languages: {", ".join(languages)}', file=sys.stderr)
                sys.exit(1)

            # Create project directory
            os.makedirs(project_dir)

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
                os.makedirs(project_dir / lang)

            # Copy CSV file to project directory
            shutil.copy2(resolved_csv_path, project_dir / os.path.basename(csv_path))

            print(f'Project "{project_name}" created successfully')
            print(f'Languages detected: {", ".join(languages)}')
            print(f"Base language: {base_lang}")
            print(f'Using "{key_column}" as translation key column')

    except FileNotFoundError:
        print(f"CSV file not found: {resolved_csv_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error creating project: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import asyncio

    asyncio.run(create_project())
