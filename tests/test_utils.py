import os
import sys
import pytest
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.utils import (
    Config,
    display_width,
    is_abbreviation,
    length_within_limit,
    measurable_text,
    placeholders_match,
)
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
        context_parts = await storage.load_context("test_project")

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
        context_parts = await storage.load_context("test_project")

        # Verify the loaded context includes the specific file
        assert len(context_parts) >= 1
        assert specific_context in context_parts

        language_dir = project_dir / "es"
        language_dir.mkdir()
        (language_dir / "context.txt").write_text("Spanish rules", encoding="utf-8")

        assert await storage.load_context("test_project", "es") == ["Spanish rules"]


class TestLengthCheck:
    """UI labels must stay short enough for the widget that shows them."""

    def test_prose_is_exempt(self):
        long_ru = "Я совершенно не понимаю, что здесь происходит."
        ok, _ = length_within_limit("I’m lost.", long_ru)
        assert ok

    def test_placeholders_do_not_count(self):
        # `{level}` renders identically in both languages.
        assert display_width(measurable_text("Lvl {level}")) == 3

    def test_abbreviation_must_stay_short(self):
        ok, reason = length_within_limit("Lvl {level}", "Уровень {level}")
        assert not ok and "abbreviation" in reason
        ok, _ = length_within_limit("Lvl {level}", "Ур. {level}")
        assert ok

    def test_ordinary_short_word_is_not_an_abbreviation(self):
        # "Ash" is a name, not a shortened form: katakana is allowed to be wider.
        assert not is_abbreviation("Ash")
        ok, _ = length_within_limit("Ash", "アッシュ")
        assert ok

    def test_cjk_width_counts_double(self):
        assert display_width("生命值") == 6
        ok, reason = length_within_limit("HP", "生命值")
        assert not ok and "abbreviation" in reason

    def test_label_growth_is_capped(self):
        ok, reason = length_within_limit("Headshot", "Πυροβολισμός στο Κεφάλι")
        assert not ok and "label" in reason
        ok, _ = length_within_limit("Headshot", "Выстрел в голову")
        assert ok

    def test_substituted_string_is_caught(self):
        ok, reason = length_within_limit("Charm", "1. 消耗低质量物品以锻造所需物品。")
        assert not ok and "label" in reason

    def test_per_project_options_apply(self):
        long_de = "Widerstandsfähigkeit"
        assert not length_within_limit("Fortitude", long_de)[0]
        assert length_within_limit("Fortitude", long_de, {"maxRatio": 2.5})[0]


class TestPlaceholdersMatch:
    """Runtime placeholders must survive translation; ICU branches may not."""

    def test_plain_placeholders_must_match_exactly(self):
        assert placeholders_match("Hi {name}", "Привет, {name}")[0]
        ok, reason = placeholders_match("Hi {name}", "Привет")
        assert not ok and "curly placeholders mismatch" in reason
        ok, _ = placeholders_match("Hi {name}", "Привет, {Name}")
        assert not ok
        # Double braces and `{n, number}` are still compared literally.
        assert placeholders_match("{{name}} x", "{{name}} у")[0]
        assert not placeholders_match("{n, number} x", "{n} у")[0]

    def test_lingui_tags_must_match(self):
        assert placeholders_match("<0>Buy</0>", "<0>Купить</0>")[0]
        assert not placeholders_match("<0>Buy</0>", "Купить")[0]

    def test_plural_branches_may_differ_between_languages(self):
        en = "Up to {max, plural, one {# file} other {# files}}, {mb} MB each."
        ru = (
            "До {max, plural, one {# файла} few {# файлов} many {# файлов}"
            " other {# файла}}, по {mb} МБ."
        )
        assert placeholders_match(en, ru) == (True, "")
        ja = "最大{max, plural, other {#個のファイル}}、各{mb} MB。"
        assert placeholders_match(en, ja)[0]

    def test_plural_needs_other_and_known_categories(self):
        en = "{n, plural, one {# file} other {# files}}"
        ok, reason = placeholders_match(en, "{n, plural, one {# файл} few {# файла}}")
        assert not ok and "missing `other`" in reason
        ok, reason = placeholders_match(en, "{n, plural, some {# файл} other {# файла}}")
        assert not ok and "unknown plural categories" in reason
        ok, reason = placeholders_match(en, "{n, plural, one {файл} other {файлы}}")
        assert not ok and "nested placeholders" in reason

    def test_exact_value_branches_are_kept(self):
        en = "{n, plural, =0 {No files} one {# file} other {# files}}"
        assert placeholders_match(en, "{n, plural, =0 {Нет файлов} one {# файл} other {# файлов}}")[0]
        ok, reason = placeholders_match(en, "{n, plural, one {# файл} other {# файлов}}")
        assert not ok and "exact-value branches differ" in reason

    def test_select_keys_are_fixed_by_source(self):
        en = "{g, select, male {He} female {She} other {They}}"
        assert placeholders_match(en, "{g, select, male {Он} female {Она} other {Они}}")[0]
        ok, reason = placeholders_match(en, "{g, select, male {Он} other {Они}}")
        assert not ok and "select keys differ" in reason

    def test_nested_placeholders_inside_branches(self):
        en = "{n, plural, one {{name} has # item} other {{name} has # items}}"
        assert placeholders_match(en, "{n, plural, one {У {name} # предмет} other {У {name} # предметов}}")[0]
        assert not placeholders_match(en, "{n, plural, one {# предмет} other {# предметов}}")[0]

    def test_dropped_argument_is_a_mismatch(self):
        en = "{n, plural, one {# file} other {# files}}"
        ok, reason = placeholders_match(en, "файлы")
        assert not ok and "{n, plural}" in reason

    def test_unbalanced_braces_are_literal_text(self):
        assert placeholders_match("a {b", "а {б")[0]
        assert not placeholders_match("a {b}", "а {б")[0]

    def test_measurable_text_uses_other_branch(self):
        en = "{n, plural, one {# file} other {# files}} left"
        assert measurable_text(en) == "0 files left"
