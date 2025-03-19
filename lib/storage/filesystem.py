"""
File system implementation of the storage adapter.
"""

import csv
import json
import os
from io import StringIO
from pathlib import Path
from typing import Optional, List, Dict

import aiofiles

from .base import StorageAdapter
from lib.utils import Config


class FileSystemStorageAdapter(StorageAdapter):
    """
    File system implementation of the storage adapter.
    Stores data in the local file system using the original project structure.
    """

    project_path: Path

    def __init__(
        self,
        project_path: Path,
        context_file: Optional[str] = None,
        prompt_file: Optional[str] = None,
    ):
        """
        Initialize the file system storage adapter.

        Args:
            base_path: Base path for all projects
        """
        self.project_path = project_path
        self.context_file = context_file
        self.prompt_file = prompt_file

    def set_context_file(self, context_file: Optional[str]) -> None:
        """Set the context file path"""
        self.context_file = context_file

    def set_prompt_file(self, prompt_file: Optional[str]) -> None:
        """Set the prompt file path"""
        self.prompt_file = prompt_file

    def _get_config_path(self) -> Path:
        """Get the config file path"""
        return self.project_path / "config.json"

    def _get_progress_path(self, language: str) -> Path:
        """Get the progress file path for a language"""
        return self.project_path / language / "progress.json"

    def _get_translations_path(self, config: Optional[Config] = None) -> Path:
        """Get the translations file path"""
        if config:
            return self.project_path / config.sourceFile
        return self.project_path / "translations.csv"

    async def load_config(self, project_id: str) -> Config:
        """Load project configuration from config.json"""
        config_path = self._get_config_path()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        async with aiofiles.open(config_path, "r", encoding="utf-8") as f:
            content = await f.read()
            return Config(**json.loads(content))

    async def load_progress(self, project_id: str, language: str) -> Dict[str, str]:
        """Load translation progress from progress.json"""
        progress_path = self._get_progress_path(language)
        if not progress_path.exists():
            return {}

        async with aiofiles.open(progress_path, "r", encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)

    async def save_progress(
        self, project_id: str, language: str, progress: Dict[str, str]
    ) -> None:
        """Save translation progress to progress.json"""
        progress_path = self._get_progress_path(language)

        # Create language directory if it doesn't exist
        os.makedirs(progress_path.parent, exist_ok=True)

        async with aiofiles.open(progress_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(progress, ensure_ascii=False, indent=2))

    async def load_translations(self, project_id: str) -> List[Dict[str, str]]:
        """Load translations from the CSV file"""
        config = await self.load_config(project_id)
        source_file = self._get_translations_path(config)

        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")

        async with aiofiles.open(source_file, "r", newline="", encoding="utf-8") as f:
            content = await f.read()
            csv_file = StringIO(content)
            reader = csv.DictReader(csv_file)
            return list(reader)

    async def save_translations(
        self, project_id: str, translations: List[Dict[str, str]]
    ) -> None:
        """Save translations to the CSV file"""
        if not translations:
            return

        config = await self.load_config(project_id)
        output_file = self._get_translations_path(config)

        fieldnames = list(translations[0].keys())
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(translations)

        content = output.getvalue()

        async with aiofiles.open(output_file, "w", newline="", encoding="utf-8") as f:
            await f.write(content)

    async def load_context(self, project_id: str) -> List[str]:
        """Load translation context from various sources"""
        context_parts = []

        # 1. Check for context.md or context.txt in project directory
        for ext in [".md", ".txt"]:
            context_path = self.project_path / f"context{ext}"
            try:
                if os.path.exists(context_path):
                    async with aiofiles.open(context_path, "r", encoding="utf-8") as f:
                        content = await f.read()
                        context_parts.append(content.strip())
            except Exception as e:
                print(f"Warning: Error reading context file {context_path}: {e}")

        # 2. Check for context from command line file
        if self.context_file:
            try:
                if os.path.exists(self.context_file):
                    async with aiofiles.open(
                        self.context_file, "r", encoding="utf-8"
                    ) as f:
                        content = await f.read()
                        context_parts.append(content.strip())
            except Exception as e:
                print(f"Warning: Error reading context file {self.context_file}: {e}")

        return context_parts

    async def load_prompt(self, project_id: str, prompt_type: str) -> str:
        """Load translation prompt from file"""

        # First try the provided prompt file
        if self.prompt_file:
            try:
                async with aiofiles.open(self.prompt_file, "r", encoding="utf-8") as f:
                    return await f.read()
            except Exception as e:
                print(f"Warning: Error reading prompt file {self.prompt_file}: {e}")

        # Then try the default prompt file in the project
        prompt_path = self.project_path / "prompts" / f"{prompt_type}.txt"
        try:
            if prompt_path.exists():
                async with aiofiles.open(prompt_path, "r", encoding="utf-8") as f:
                    return await f.read()
        except Exception as e:
            print(f"Warning: Error reading prompt file {prompt_path}: {e}")

        return ""
