import csv
from io import StringIO
import json
from pathlib import Path
from typing import TypedDict
import aiofiles
from pydantic import BaseModel


class Config(BaseModel):
    name: str
    sourceFile: str
    languages: list[str]
    baseLanguage: str
    keyColumn: str
    ...


async def load_config(config_path: Path) -> Config:
    """Load the project configuration from config.json"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    async with aiofiles.open(config_path, "r", encoding="utf-8") as f:
        content = await f.read()
        return Config(**json.loads(content))


async def load_progress(progress_file: Path) -> dict[str, str]:
    """Load the translation progress from progress.json"""
    if not progress_file.exists():
        return {}

    async with aiofiles.open(progress_file, "r", encoding="utf-8") as f:
        content = await f.read()
        return json.loads(content)


async def save_progress(progress_file: Path, progress: dict[str, str]) -> None:
    """Save the translation progress to progress.json"""
    async with aiofiles.open(progress_file, "w", encoding="utf-8") as f:
        await f.write(json.dumps(progress, ensure_ascii=False, indent=2))


async def load_translations(source_file: Path) -> list[dict[str, str]]:
    """Load translations from the CSV file"""
    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")

    async with aiofiles.open(source_file, "r", newline="", encoding="utf-8") as f:
        content = await f.read()
        # Use StringIO to properly handle CSV with potential multiline fields
        csv_file = StringIO(content)
        reader = csv.DictReader(csv_file)
        return list(reader)


async def save_translations(
    output_file: Path, translations: list[dict[str, str]]
) -> None:
    """Save translations to the CSV file"""
    if not translations:
        return

    fieldnames = list(translations[0].keys())

    # Use StringIO to write CSV to a string
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(translations)

    content = output.getvalue()

    async with aiofiles.open(output_file, "w", newline="", encoding="utf-8") as f:
        await f.write(content)
