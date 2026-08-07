import os
import sys
import pytest
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.utils import Config
from lib.storage.filesystem import FileSystemStorageAdapter


class TestUtilsFunctions:
    """Test suite for utility functions."""

    @pytest.mark.asyncio
    async def test_load_config(self, tmp_path):
        """Test loading configuration from a file."""
        # Create a test config file
        config_data = {
            "name": "test_project",
            "sourceFile": "source.json",
            "baseLanguage": "en",
            "languages": ["en", "es", "fr"],
            "keyColumn": "key",
        }

        # Create project directory
        project_dir = tmp_path / "test_project"
        os.makedirs(project_dir, exist_ok=True)

        config_file = project_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        # Create a storage adapter
        storage = FileSystemStorageAdapter(project_dir)

        # Load the config
        config = await storage.load_config("test_project")

        # Verify the config
        assert isinstance(config, Config)
        assert config.name == "test_project"
        assert config.sourceFile == "source.json"
        assert config.baseLanguage == "en"
        assert config.languages == ["en", "es", "fr"]
        assert config.keyColumn == "key"

    @pytest.mark.asyncio
    async def test_load_and_save_progress(self, tmp_path):
        """Test loading and saving progress."""
        # Create project directory and language subdirectory
        project_dir = tmp_path / "test_project"
        lang_dir = project_dir / "es"
        os.makedirs(lang_dir, exist_ok=True)

        # Create test progress data
        progress_data = {"phrase1": "Hola", "phrase2": "Adiós", "phrase3": "Bienvenido"}

        # Create a storage adapter
        storage = FileSystemStorageAdapter(project_dir)

        # Save the progress
        await storage.save_progress("test_project", "es", progress_data)

        # Verify the file exists
        progress_file = lang_dir / "progress.json"
        assert progress_file.exists()

        # Load the progress
        loaded_progress = await storage.load_progress("test_project", "es")

        # Verify the loaded progress
        assert loaded_progress == progress_data

    @pytest.mark.asyncio
    async def test_append_failure_writes_jsonl(self, tmp_path):
        project_dir = tmp_path / "test_project"
        lang_dir = project_dir / "es"
        lang_dir.mkdir(parents=True)

        storage = FileSystemStorageAdapter(project_dir)
        record = {
            "ts": "2026-01-01T00:00:00+00:00",
            "model": "test-model",
            "method": "standard",
            "phrase": "Hello",
            "category": "parse_error",
            "message": "failed to parse",
        }
        await storage.append_failure("test_project", "es", record)

        failures_file = lang_dir / "failures.jsonl"
        assert failures_file.exists()
        lines = failures_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert '"phrase": "Hello"' in lines[0]

    @pytest.mark.asyncio
    async def test_save_progress_merge_policy(self, tmp_path):
        """save_progress merges with disk: it never regresses a valid value to an
        empty/invalid one, but `overwrite_keys` forces authoritative corrections."""
        project_dir = tmp_path / "test_project"
        lang_dir = project_dir / "es"
        os.makedirs(lang_dir, exist_ok=True)

        storage = FileSystemStorageAdapter(project_dir)
        await storage.save_progress("test_project", "es", {"Hello": "Hola-OLD"})

        # Without overwrite_keys: a valid on-disk value is preserved (no regression),
        # and unrelated keys are merged in (union, never delete).
        await storage.save_progress(
            "test_project",
            "es",
            {"Hello": "Hola-FIXED", "Bye": "Adios"},
        )
        loaded = await storage.load_progress("test_project", "es")
        assert loaded["Hello"] == "Hola-OLD"
        assert loaded["Bye"] == "Adios"

        # With overwrite_keys: the correction replaces the existing valid value.
        await storage.save_progress(
            "test_project",
            "es",
            {"Hello": "Hola-FIXED"},
            overwrite_keys={"Hello"},
        )
        loaded = await storage.load_progress("test_project", "es")
        assert loaded["Hello"] == "Hola-FIXED"
        assert loaded["Bye"] == "Adios"  # untouched key survives

    @pytest.mark.asyncio
    async def test_load_progress_nonexistent_file(self, tmp_path):
        """Test loading progress from a nonexistent file."""
        # Create a project directory without progress file
        project_dir = tmp_path / "test_project"
        lang_dir = project_dir / "fr"
        os.makedirs(lang_dir, exist_ok=True)

        # Create a storage adapter
        storage = FileSystemStorageAdapter(project_dir)

        # Load the progress (should return empty dict)
        loaded_progress = await storage.load_progress("test_project", "fr")

        # Verify the result
        assert loaded_progress == {}

    @pytest.mark.asyncio
    async def test_load_context(self, tmp_path):
        """Test loading context from a file."""
        # Create project directory
        project_dir = tmp_path / "test_project"
        os.makedirs(project_dir, exist_ok=True)

        # Create a test context file
        context_content = "This is test context for translation."
        context_file = project_dir / "context.md"
        with open(context_file, "w", encoding="utf-8") as f:
            f.write(context_content)

        # Create a storage adapter
        storage = FileSystemStorageAdapter(project_dir)

        # Test without specific context file (uses default in project dir)
        context_parts = await storage.load_context("test_project", "es")

        # Verify the loaded context
        assert len(context_parts) == 1
        assert context_parts[0] == context_content

        # Create a specific context file outside project dir
        specific_context = "This is specific context for project."
        specific_file = tmp_path / "specific_context.md"
        with open(specific_file, "w", encoding="utf-8") as f:
            f.write(specific_context)

        # Set the specific context file
        storage.set_context_file(str(specific_file))

        # Load context with specified file
        context_parts = await storage.load_context("test_project", "es")

        # Verify the loaded context includes the specific file
        assert len(context_parts) >= 1
        assert specific_context in context_parts
