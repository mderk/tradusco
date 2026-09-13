"""
File system implementation of the storage adapter.
"""

import csv
import json
import os
import tempfile
from io import StringIO
from pathlib import Path
from typing import Optional, List, Dict

import aiofiles

from .base import StorageAdapter
from lib.utils import Config, is_valid_translation


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
        self.active_languages: list[str] = []
        self.overwrite_active_language: bool = False
        self.force_translation_keys: dict[str, set[str]] = {}

    def set_context_file(self, context_file: Optional[str]) -> None:
        """Set the context file path"""
        self.context_file = context_file

    def set_prompt_file(self, prompt_file: Optional[str]) -> None:
        """Set the prompt file path"""
        self.prompt_file = prompt_file

    def set_active_language(self, language: Optional[str]) -> None:
        """Set the destination language for the current translation run (optional)."""
        self.set_active_languages([language] if language else [])

    def set_active_languages(self, languages: list[str]) -> None:
        """Set destination languages for the current translation run."""
        self.active_languages = list(languages)

    def set_overwrite_active_language(self, enabled: bool) -> None:
        """Allow overwriting non-empty cells for the active language (e.g. --regenerate)."""
        self.overwrite_active_language = enabled

    def set_force_translation_keys(self, keys: dict[str, set[str]]) -> None:
        self.force_translation_keys = keys

    def _get_config_path(self) -> Path:
        """Get the config file path"""
        return self.project_path / "config.json"

    def _get_progress_path(self, language: str) -> Path:
        """Get the progress file path for a language"""
        return self.project_path / language / "progress.json"

    def _get_failures_path(self, language: str) -> Path:
        """Get the append-only failure log path for a language."""
        return self.project_path / language / "failures.jsonl"

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

    async def load_editorial(self, project_id: str) -> dict[str, dict[str, str]]:
        path = self.project_path / "editorial.json"
        if not path.exists():
            return {}
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            value = json.loads(await f.read())
        return value if isinstance(value, dict) else {}

    async def save_progress(
        self,
        project_id: str,
        language: str,
        progress: Dict[str, str],
        overwrite_keys: Optional[set] = None,
    ) -> None:
        """Save translation progress to progress.json.

        This file is translation memory and is treated as primary state.
        When multiple processes accidentally run for the same language, a stale
        process should not be able to truncate the file. We therefore:
        - take an exclusive lock
        - merge with the latest on-disk progress (union, never delete)
        - write atomically

        ``overwrite_keys`` lists keys whose incoming value is authoritative and
        must replace the on-disk value even if the latter is valid (e.g. a manual
        correction made directly in the CSV). All other keys keep the
        conservative "prefer a valid on-disk value" policy.
        """
        progress_path = self._get_progress_path(language)
        progress_path.parent.mkdir(parents=True, exist_ok=True)

        lock_path = progress_path.with_name(f".{progress_path.name}.lock")
        try:
            import fcntl  # type: ignore
        except Exception:
            fcntl = None  # type: ignore

        def _load_progress(path: Path) -> dict[str, str]:
            if not path.exists():
                return {}
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
            return data if isinstance(data, dict) else {}

        def _looks_invalid_translation_value(v: object) -> bool:
            # Empty or JSON scaffolding -> not a usable translation.
            return not is_valid_translation(v)

        def _write_progress_atomic(path: Path, data: dict[str, str]) -> None:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=str(path.parent),
                prefix=f".{path.name}.tmp.",
            ) as tmp:
                tmp.write(json.dumps(data, ensure_ascii=False, indent=2))
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, path)

        with lock_path.open("a", encoding="utf-8") as lock_fp:
            if fcntl is not None:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            try:
                current = _load_progress(progress_path)
                merged = dict(current)

                overwrite = bool(self.overwrite_active_language)
                force_keys = overwrite_keys or set()
                for k, v in (progress or {}).items():
                    if v is None:
                        continue
                    sv = str(v)
                    if not sv.strip():
                        continue
                    # `overwrite` (--regenerate) forces every value; `force_keys`
                    # forces specific authoritative corrections (e.g. edited in CSV).
                    if overwrite or k in force_keys:
                        merged[k] = sv
                        continue
                    current_val = merged.get(k)
                    if (k not in merged) or _looks_invalid_translation_value(current_val):
                        merged[k] = sv

                _write_progress_atomic(progress_path, merged)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

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
        """Save translations to the CSV file.

        Multiple translation runs (one per destination locale) may execute in parallel.
        To prevent overwriting each other's work, we take an exclusive lock and merge
        updates into the latest on-disk CSV. When active languages are set, we preserve
        other locales' columns from disk and apply updates only for those languages.
        """
        if not translations:
            return

        config = await self.load_config(project_id)
        output_file = self._get_translations_path(config)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        def _is_empty_cell(v: object) -> bool:
            return v is None or v == ""

        def _looks_invalid_cell(v: object) -> bool:
            # Empty or JSON scaffolding -> not a usable translation cell.
            return not is_valid_translation(v)

        def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
            if not path.exists():
                return [], []
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows: list[dict[str, str]] = list(reader)
                return list(reader.fieldnames or []), rows

        def _write_csv_atomic(
            path: Path, fieldnames: list[str], rows: list[dict[str, str]]
        ) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                delete=False,
                dir=str(path.parent),
                prefix=f".{path.name}.tmp.",
            ) as tmp:
                writer = csv.DictWriter(
                    tmp, fieldnames=fieldnames, lineterminator="\n"
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: (row.get(k) or "") for k in fieldnames})
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, path)

        lock_path = output_file.with_name(f".{output_file.name}.lock")
        try:
            import fcntl  # type: ignore
        except Exception:
            fcntl = None  # type: ignore

        with lock_path.open("a", encoding="utf-8") as lock_fp:
            if fcntl is not None:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            try:
                current_fieldnames, current_rows = _read_csv(output_file)

                incoming_fieldnames = list(translations[0].keys())
                fieldnames = (
                    list(current_fieldnames)
                    if current_fieldnames
                    else list(incoming_fieldnames)
                )
                for k in incoming_fieldnames:
                    if k not in fieldnames:
                        fieldnames.append(k)

                # When active languages are set, keep other columns from disk to avoid
                # losing updates from other concurrent language processes.
                if current_rows and self.active_languages:
                    merged_rows: list[dict[str, str]] = []
                    common_len = min(len(current_rows), len(translations))

                    for i in range(common_len):
                        disk_row = current_rows[i]
                        incoming_row = translations[i]
                        merged = dict(disk_row)

                        # Fill any missing cells from incoming snapshot (e.g. metadata),
                        # but prefer disk for non-empty values.
                        for col in fieldnames:
                            if _looks_invalid_cell(
                                merged.get(col)
                            ) and not _looks_invalid_cell(incoming_row.get(col)):
                                merged[col] = str(incoming_row.get(col) or "")

                        # If regenerating, allow overwriting non-empty cells for the
                        # active language, but never overwrite with an empty value.
                        if self.overwrite_active_language:
                            for language in self.active_languages:
                                incoming_lang_val = incoming_row.get(language)
                                if not _is_empty_cell(incoming_lang_val):
                                    merged[language] = str(incoming_lang_val or "")

                        source = str(incoming_row.get(config.baseLanguage) or "")
                        for language in self.active_languages:
                            if source in self.force_translation_keys.get(language, set()):
                                merged[language] = str(incoming_row.get(language) or "")

                        merged_rows.append(merged)

                    # If row counts diverge, keep trailing disk rows and/or append new.
                    if len(current_rows) > common_len:
                        merged_rows.extend(current_rows[common_len:])
                    if len(translations) > common_len:
                        for row in translations[common_len:]:
                            merged_rows.append(
                                {k: (row.get(k) or "") for k in fieldnames}
                            )
                else:
                    # Default behavior (single process): overwrite with the provided list.
                    merged_rows = [
                        {k: (row.get(k) or "") for k in fieldnames}
                        for row in translations
                    ]

                _write_csv_atomic(output_file, fieldnames, merged_rows)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

    async def load_context(
        self, project_id: str, language: Optional[str] = None
    ) -> List[str]:
        """Load shared context, or one language's context when specified."""
        context_parts = []

        context_dir = self.project_path / language if language else self.project_path
        for ext in [".md", ".txt"]:
            context_path = context_dir / f"context{ext}"
            try:
                if os.path.exists(context_path):
                    async with aiofiles.open(context_path, "r", encoding="utf-8") as f:
                        content = await f.read()
                        context_parts.append(content.strip())
            except Exception as e:
                print(f"Warning: Error reading context file {context_path}: {e}")

        if language is None and self.context_file:
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

    async def load_glossary(self, project_id: str) -> dict[str, object]:
        """Load optional ``<project>/glossary.json``."""
        path = self.project_path / "glossary.json"
        if not path.exists():
            return {}
        async with aiofiles.open(path, "r", encoding="utf-8") as glossary_file:
            data = json.loads(await glossary_file.read())
        if not isinstance(data, dict):
            raise ValueError(f"Glossary must contain a JSON object: {path}")
        return data

    async def append_failure(
        self,
        project_id: str,
        language: str,
        record: dict[str, str | None],
    ) -> None:
        """Append a failure record to ``<lang>/failures.jsonl``."""
        failures_path = self._get_failures_path(language)
        failures_path.parent.mkdir(parents=True, exist_ok=True)

        lock_path = failures_path.with_name(f".{failures_path.name}.lock")
        try:
            import fcntl  # type: ignore
        except Exception:
            fcntl = None  # type: ignore

        line = json.dumps(record, ensure_ascii=False) + "\n"
        with lock_path.open("a", encoding="utf-8") as lock_fp:
            if fcntl is not None:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            try:
                with failures_path.open("a", encoding="utf-8") as out_fp:
                    out_fp.write(line)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

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
