import os
import sys
import csv
import json
import pytest
from unittest.mock import patch, AsyncMock
from typing import Optional, List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.TranslationProject import (
    DEFAULT_OUTPUT_TOKEN_RATIO,
    TranslationProject,
    output_token_ratio,
)
from lib.TranslationTool import BatchErrorInfo, BatchTranslationError, LanguageRef
from lib.utils import Config
from lib.storage.base import StorageAdapter
from lib.storage.filesystem import FileSystemStorageAdapter
from tests.mock_llm_driver import MockLLMDriver


# Mock storage adapter for testing
class MockStorageAdapter(StorageAdapter):
    def __init__(self):
        self.config = Config(
            name="test_project",
            sourceFile="source.csv",
            baseLanguage="en",
            languages=["en", "es", "fr", "ko"],
            keyColumn="key",
        )
        self.translations = []
        self.context_strings = []
        self.language_contexts: Dict[str, List[str]] = {}
        self.prompts = {}
        self.context_file = None
        self.prompt_file = None
        self.progress: Dict[str, str] = {}
        self.progresses: Dict[str, Dict[str, str]] = {}
        self.progress_overwrite_keys: set = set()
        self.failures: List[Dict[str, Any]] = []

    async def append_failure(
        self, project_id: str, language: str, record: dict[str, str | None]
    ) -> None:
        self.failures.append({**record, "language": language})

    async def load_config(self, project_id: str) -> Config:
        return self.config

    async def load_progress(self, project_id: str, language: str) -> Dict[str, str]:
        return dict(self.progresses.get(language, self.progress))

    async def save_progress(
        self,
        project_id: str,
        language: str,
        progress: Dict[str, str],
        overwrite_keys=None,
    ) -> None:
        self.progresses[language] = dict(progress)
        if language == "es":
            self.progress = dict(progress)
        self.progress_overwrite_keys = set(overwrite_keys or set())

    async def load_translations(self, project_id: str) -> List[Dict[str, Any]]:
        return self.translations

    async def save_translations(
        self, project_id: str, translations: List[Dict[str, Any]]
    ) -> None:
        self.translations = translations

    async def load_context(
        self, project_id: str, language: Optional[str] = None
    ) -> List[str]:
        return (
            self.context_strings
            if language is None
            else self.language_contexts.get(language, [])
        )

    async def load_prompt(self, project_id: str, prompt_type: str) -> str:
        if prompt_type in self.prompts:
            return self.prompts[prompt_type]
        return ""


@pytest.mark.asyncio
class TestTranslationProject:
    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage adapter for testing."""
        storage = MockStorageAdapter()

        # Add some default translations
        storage.translations = [
            {"en": "Hello", "fr": "Bonjour", "es": ""},
            {"en": "Goodbye", "fr": "Au revoir", "es": ""},
            {"en": "Thank you", "fr": "Merci", "es": ""},
        ]

        # Add some default context as strings
        storage.context_strings = [
            "Context 1: This is for greetings and introductions.",
            "Context 2: This is for farewells and exits.",
        ]

        # Add default prompts
        storage.prompts = {
            "translation": "Translate from {base_language} to {dst_languages}: {phrases_json}",
            "output_format": "Valid JSON format",
            "json_fix": "Fix this JSON: {broken_json}",
        }

        return storage

    @pytest.fixture
    def mock_llm_driver(self):
        """Create a mock LLM driver for testing."""
        return MockLLMDriver()

    @pytest.mark.asyncio
    async def test_create_project(self, mock_storage):
        """Test project creation."""
        config = await mock_storage.load_config("test_project")

        project = TranslationProject(
            project_id="test_project",
            config=config,
            dst_languages=["es"],
            storage=mock_storage,
        )

        assert project.project_id == "test_project"
        assert project.base_language == "en"
        assert project.dst_languages == ["es"]
        assert project.storage is mock_storage
        assert project.config == config

        # Test prompt manager creation
        assert project.prompt_manager is not None
        assert project.prompt_manager.storage is mock_storage
        assert project.prompt_manager.project_id == "test_project"

        # Test translation tool creation
        assert project.translation_tool is not None

    @pytest.mark.asyncio
    async def test_get_available_models(self, mock_storage):
        """Test getting available models."""
        # Create a test project
        config = await mock_storage.load_config("test_project")
        project = TranslationProject(
            project_id="test_project",
            config=config,
            dst_languages=["es"],
            storage=mock_storage,
        )

        # Mock the get_available_models function
        with patch(
            "lib.TranslationProject.get_available_models",
            return_value=["gemini", "gpt-3.5-turbo"],
        ):
            models = project.get_available_models()
            assert len(models) > 0
            assert isinstance(models, list)
            # Check for common models
            assert "gemini" in models

    @pytest.mark.asyncio
    async def test_count_tokens(self, mock_storage):
        """Test token counting method."""
        # Create a test project
        config = await mock_storage.load_config("test_project")
        project = TranslationProject(
            project_id="test_project",
            config=config,
            dst_languages=["es"],
            storage=mock_storage,
        )

        # Count tokens in a string
        token_count = project.count_tokens("This is a test")
        assert token_count > 0
        assert isinstance(token_count, int)

    @patch("lib.TranslationProject.TranslationProject._process_translation_batch")
    async def test_output_budget_isolates_long_phrase_but_keeps_short_labels_together(
        self, process_batch_mock, mock_storage, mock_llm_driver
    ):
        languages = [
            "fr",
            "ru",
            "it",
            "de",
            "es",
            "es-la",
            "ja",
            "ko",
            "pl",
            "pt-br",
            "pt-pt",
            "zh-cn",
            "tr",
            "uk",
            "th",
            "cs",
            "hu",
            "vi",
            "ro",
            "ar",
        ]
        mock_storage.config.languages = ["en", *languages]
        short_phrases = [f"Label {index}" for index in range(100)]
        long_phrase = "Long dialogue"
        tail_phrase = "Tail"
        mock_storage.translations = [
            {"en": phrase, **dict.fromkeys(languages, "")}
            for phrase in [*short_phrases, long_phrase, tail_phrase]
        ]
        project = TranslationProject(
            project_id="test_project",
            config=mock_storage.config,
            dst_languages=languages,
            storage=mock_storage,
        )
        project.count_tokens = lambda text, model="gemini": (
            200 if text.startswith(long_phrase) else 1
        )
        process_batch_mock.return_value = None

        await project._translate_pass(
            model="test-model",
            method="standard",
            delay_seconds=0,
            max_retries=0,
            batch_size=200,
            batch_max_output_tokens=8192,
            regenerate=False,
            fallback_model=None,
            driver=mock_llm_driver,
        )

        batches = [
            [phrase for phrase, _context in call.args[0]]
            for call in process_batch_mock.await_args_list
        ]
        assert batches == [short_phrases, [long_phrase], [tail_phrase]]

    @patch("lib.TranslationProject.TranslationProject._process_translation_batch")
    async def test_one_language_keeps_input_equivalent_batch_boundary(
        self, process_batch_mock, mock_storage, mock_llm_driver
    ):
        phrases = [f"Phrase {index}" for index in range(6)]
        mock_storage.translations = [{"en": phrase, "es": ""} for phrase in phrases]
        project = TranslationProject(
            project_id="test_project",
            config=mock_storage.config,
            dst_languages=["es"],
            storage=mock_storage,
        )
        project.count_tokens = lambda text, model="gemini": 1
        process_batch_mock.return_value = None

        await project._translate_pass(
            model="test-model",
            method="standard",
            delay_seconds=0,
            max_retries=0,
            batch_size=50,
            batch_max_output_tokens=10,
            regenerate=False,
            fallback_model=None,
            driver=mock_llm_driver,
        )

        assert [
            len(call.args[0]) for call in process_batch_mock.await_args_list
        ] == [5, 1]

    async def test_unknown_language_uses_largest_measured_ratio(self):
        assert output_token_ratio(["unknown"]) == DEFAULT_OUTPUT_TOKEN_RATIO

    @pytest.mark.asyncio
    async def test_load_context(self, mock_storage):
        """Test loading context."""
        # Create a test project
        config = await mock_storage.load_config("test_project")
        project = TranslationProject(
            project_id="test_project",
            config=config,
            dst_languages=["es"],
            storage=mock_storage,
        )

        # Load context
        context = await project._load_context()

        # Verify context was loaded correctly
        assert len(context) > 0
        assert "Context 1" in context
        assert "Context 2" in context

    @pytest.mark.asyncio
    async def test_load_context_keeps_shared_context_single(self, mock_storage):
        config = await mock_storage.load_config("test_project")
        mock_storage.context_strings = ["Shared rules"]
        mock_storage.language_contexts = {
            "es": ["Spanish rules"],
            "fr": ["French rules"],
        }
        project = TranslationProject(
            project_id="test_project",
            config=config,
            dst_languages=["es", "fr"],
            storage=mock_storage,
        )

        assert await project._load_context() == (
            "Shared rules\n\n[es]\nSpanish rules\n\n[fr]\nFrench rules"
        )

    @patch("lib.TranslationTool.TranslationTool.translate_standard")
    @patch("lib.llm.get_driver")
    @patch("lib.TranslationProject.get_driver")
    async def test_translate(
        self,
        project_get_driver_mock,
        llm_get_driver_mock,
        mock_translate_standard_patch,
        mock_llm_driver,
        mock_storage,
    ):
        """Test translation process with mock driver"""

        # Configure the mock translate_standard method
        async def mock_translate_standard(
            phrases: list[tuple[str, str | None]],
            model: str,
            base_language: str,
            dst_languages: list[LanguageRef],
            prompt: str,
            context: Optional[str] = None,
            delay_seconds: float = 1.0,
            max_retries: int = 3,
            raise_on_error: bool = False,
        ) -> dict[str, dict[str, str]]:
            # Simulate translation
            progress = {}
            for i, (phrase, context) in enumerate(phrases):
                progress[phrase] = f"{phrase} (translated)"
            return {"es": progress}

        # Set up the mocks
        mock_translate_standard_patch.side_effect = mock_translate_standard
        llm_get_driver_mock.return_value = mock_llm_driver
        project_get_driver_mock.return_value = mock_llm_driver

        # Create a config
        config = await mock_storage.load_config("test_project")

        # Create a test project
        project = TranslationProject(
            project_id="test_project",
            config=config,
            dst_languages=["es"],
            storage=mock_storage,
            prompt="Translate from {base_language} to {dst_languages}",
        )

        # Run translation with mock driver
        await project.translate(model="test_model")

        # Verify the driver was called
        assert llm_get_driver_mock.called or project_get_driver_mock.called

        # Verify translate_standard was called
        assert mock_translate_standard_patch.called

        # Verify translations were updated
        translations = await mock_storage.load_translations("test_project")
        assert len(translations) > 0
        assert "es" in translations[0]
        assert "(translated)" in translations[0]["es"]

    @patch("lib.llm.get_driver")
    @patch("lib.TranslationProject.get_driver")
    async def test_manual_csv_correction_synced_to_progress(
        self,
        project_get_driver_mock,
        llm_get_driver_mock,
        mock_llm_driver,
        mock_storage,
    ):
        """A valid CSV cell that differs from stale progress must overwrite it.

        Reproduces editing a translation directly in the CSV: on the next run the
        phrase is already translated (so it is not re-sent to the model), and the
        corrected value must be promoted into translation memory as an
        authoritative override rather than being shadowed by the stale entry.
        """
        llm_get_driver_mock.return_value = mock_llm_driver
        project_get_driver_mock.return_value = mock_llm_driver

        # All destination cells are filled and valid; "Hello" was corrected by hand
        # in the CSV to a value that differs from the (stale) progress memory.
        mock_storage.translations = [
            {"en": "Hello", "es": "Hola-FIXED"},
            {"en": "Bye", "es": "Adios"},
        ]
        mock_storage.progress = {"Hello": "Hola-OLD", "Bye": "Adios"}

        config = await mock_storage.load_config("test_project")
        project = TranslationProject(
            project_id="test_project",
            config=config,
            dst_languages=["es"],
            storage=mock_storage,
            prompt="Translate from {base_language} to {dst_languages}: {phrases_json}",
        )

        await project.translate(model="test_model")

        # The manual correction is promoted into memory and flagged as override.
        assert mock_storage.progress["Hello"] == "Hola-FIXED"
        assert "Hello" in mock_storage.progress_overwrite_keys
        # An unchanged, in-sync cell is not flagged as a correction.
        assert "Bye" not in mock_storage.progress_overwrite_keys
        # The CSV keeps the corrected value (the phrase was not retranslated).
        saved = await mock_storage.load_translations("test_project")
        assert saved[0]["es"] == "Hola-FIXED"

    @patch("lib.TranslationTool.TranslationTool.translate_standard")
    @patch("lib.llm.get_driver")
    @patch("lib.TranslationProject.get_driver")
    async def test_batch_fallback_retries_with_fallback_model(
        self,
        project_get_driver_mock,
        llm_get_driver_mock,
        mock_translate_standard_patch,
        mock_llm_driver,
        mock_storage,
    ):
        """When the primary batch fails, retry once with the fallback model."""
        calls: list[str] = []

        async def mock_translate_standard(
            phrases: list[tuple[str, str | None]],
            model: str,
            base_language: str,
            dst_languages: list[LanguageRef],
            prompt: str,
            context: Optional[str] = None,
            delay_seconds: float = 1.0,
            max_retries: int = 3,
            raise_on_error: bool = False,
        ) -> dict[str, dict[str, str]]:
            calls.append(model)
            if model == "primary-model":
                raise BatchTranslationError(
                    BatchErrorInfo(kind="rate_limit", message="429 too many requests")
                )
            return {"es": {phrase: f"{phrase} (fb)" for phrase, _ctx in phrases}}

        mock_translate_standard_patch.side_effect = mock_translate_standard
        llm_get_driver_mock.return_value = mock_llm_driver
        project_get_driver_mock.return_value = mock_llm_driver

        config = await mock_storage.load_config("test_project")
        project = TranslationProject(
            project_id="test_project",
            config=config,
            dst_languages=["es"],
            storage=mock_storage,
            prompt="Translate from {base_language} to {dst_languages}",
        )

        await project.translate(
            model="primary-model",
            fallback_model="fallback-model",
        )

        assert calls == ["primary-model", "fallback-model"]
        assert mock_storage.progress["Hello"] == "Hello (fb)"

    @patch("lib.TranslationTool.TranslationTool.translate_standard")
    @patch("lib.llm.get_driver")
    @patch("lib.TranslationProject.get_driver")
    async def test_batch_failure_is_logged_to_failures(
        self,
        project_get_driver_mock,
        llm_get_driver_mock,
        mock_translate_standard_patch,
        mock_llm_driver,
        mock_storage,
    ):
        """When primary and fallback batches fail, record one failure per phrase."""
        async def mock_translate_standard(
            phrases: list[tuple[str, str | None]],
            model: str,
            base_language: str,
            dst_languages: list[LanguageRef],
            prompt: str,
            context: Optional[str] = None,
            delay_seconds: float = 1.0,
            max_retries: int = 3,
            raise_on_error: bool = False,
        ) -> dict[str, dict[str, str]]:
            raise BatchTranslationError(
                BatchErrorInfo(kind="blocked", message="content policy")
            )

        mock_translate_standard_patch.side_effect = mock_translate_standard
        llm_get_driver_mock.return_value = mock_llm_driver
        project_get_driver_mock.return_value = mock_llm_driver

        config = await mock_storage.load_config("test_project")
        project = TranslationProject(
            project_id="test_project",
            config=config,
            dst_languages=["es"],
            storage=mock_storage,
            prompt="Translate from {base_language} to {dst_languages}",
        )

        await project.translate(
            model="primary-model",
            fallback_model="fallback-model",
        )

        assert len(mock_storage.failures) == 6
        assert all(r["category"] == "refusal" for r in mock_storage.failures)
        assert {r["phrase"] for r in mock_storage.failures} == {
            "Hello",
            "Goodbye",
            "Thank you",
        }

    @patch("lib.TranslationProject.TranslationProject._translate_pass", new_callable=AsyncMock)
    @patch("lib.llm.get_driver")
    @patch("lib.TranslationProject.get_driver")
    async def test_gap_filling_pass_runs_when_cells_still_missing(
        self,
        project_get_driver_mock,
        llm_get_driver_mock,
        translate_pass_mock,
        mock_llm_driver,
        mock_storage,
    ):
        """After the primary pass, run a second pass with the fallback model for gaps."""
        llm_get_driver_mock.return_value = mock_llm_driver
        project_get_driver_mock.return_value = mock_llm_driver

        config = await mock_storage.load_config("test_project")
        project = TranslationProject(
            project_id="test_project",
            config=config,
            dst_languages=["es"],
            storage=mock_storage,
        )

        await project.translate(model="primary-model", fallback_model="fallback-model")

        assert translate_pass_mock.await_count == 2
        first_call = translate_pass_mock.await_args_list[0].kwargs
        second_call = translate_pass_mock.await_args_list[1].kwargs
        assert first_call["model"] == "primary-model"
        assert first_call["fallback_model"] == "fallback-model"
        assert second_call["model"] == "fallback-model"
        assert second_call["fallback_model"] is None

    @patch("lib.TranslationTool.TranslationTool.translate_standard")
    @patch("lib.llm.get_driver")
    @patch("lib.TranslationProject.get_driver")
    async def test_validation_failures_are_logged_to_failures(
        self,
        project_get_driver_mock,
        llm_get_driver_mock,
        mock_translate_standard_patch,
        mock_llm_driver,
        mock_storage,
    ):
        """Placeholder mismatches and JSON artifacts are recorded in failures.jsonl."""
        mock_storage.translations = [
            {"en": "Hello {name}", "es": ""},
            {"en": "Goodbye", "es": ""},
            {"en": "Thank you", "es": ""},
        ]

        async def mock_translate_standard(
            phrases: list[tuple[str, str | None]],
            model: str,
            base_language: str,
            dst_languages: list[LanguageRef],
            prompt: str,
            context: Optional[str] = None,
            delay_seconds: float = 1.0,
            max_retries: int = 3,
            raise_on_error: bool = False,
        ) -> dict[str, dict[str, str]]:
            return {
                "es": {
                    "Hello {name}": "Hola",
                    "Goodbye": "{",
                    "Thank you": "Gracias",
                }
            }

        mock_translate_standard_patch.side_effect = mock_translate_standard
        llm_get_driver_mock.return_value = mock_llm_driver
        project_get_driver_mock.return_value = mock_llm_driver

        config = await mock_storage.load_config("test_project")
        project = TranslationProject(
            project_id="test_project",
            config=config,
            dst_languages=["es"],
            storage=mock_storage,
            prompt="Translate from {base_language} to {dst_languages}",
        )

        await project.translate(model="primary-model")

        assert mock_storage.progress["Thank you"] == "Gracias"
        assert "Hello {name}" not in mock_storage.progress
        assert "Goodbye" not in mock_storage.progress

        by_phrase = {r["phrase"]: r for r in mock_storage.failures}
        assert by_phrase["Hello {name}"]["category"] == "placeholder_mismatch"
        assert by_phrase["Goodbye"]["category"] == "invalid_artifact_rejected"

    @patch("lib.TranslationTool.TranslationTool.translate_standard")
    @patch("lib.llm.get_driver")
    @patch("lib.TranslationProject.get_driver")
    async def test_gap_pass_fills_remaining_cells(
        self,
        project_get_driver_mock,
        llm_get_driver_mock,
        mock_translate_standard_patch,
        mock_llm_driver,
        mock_storage,
    ):
        """Gap-filling pass translates phrases the primary model left missing."""
        mock_storage.translations = [
            {"en": "Hello", "es": ""},
            {"en": "Goodbye", "es": ""},
        ]

        async def mock_translate_standard(
            phrases: list[tuple[str, str | None]],
            model: str,
            base_language: str,
            dst_languages: list[LanguageRef],
            prompt: str,
            context: Optional[str] = None,
            delay_seconds: float = 1.0,
            max_retries: int = 3,
            raise_on_error: bool = False,
        ) -> dict[str, dict[str, str]]:
            if model == "primary-model":
                return {"es": {"Hello": "Hola"}}
            return {"es": {phrase: f"{phrase}-fb" for phrase, _ctx in phrases}}

        mock_translate_standard_patch.side_effect = mock_translate_standard
        llm_get_driver_mock.return_value = mock_llm_driver
        project_get_driver_mock.return_value = mock_llm_driver

        config = await mock_storage.load_config("test_project")
        project = TranslationProject(
            project_id="test_project",
            config=config,
            dst_languages=["es"],
            storage=mock_storage,
            prompt="Translate from {base_language} to {dst_languages}",
        )

        await project.translate(
            model="primary-model",
            fallback_model="fallback-model",
        )

        assert mock_storage.progress["Hello"] == "Hola"
        assert mock_storage.progress["Goodbye"] == "Goodbye-fb"

    @patch("lib.TranslationTool.TranslationTool.translate_standard")
    @patch("lib.llm.get_driver")
    @patch("lib.TranslationProject.get_driver")
    async def test_one_request_updates_three_languages_and_isolates_failure(
        self,
        project_get_driver_mock,
        llm_get_driver_mock,
        mock_translate_standard_patch,
        mock_llm_driver,
        mock_storage,
    ):
        mock_storage.translations = [
            {"en": "Hello", "es": "", "fr": "Déjà relu", "ko": ""},
            {"en": "Goodbye", "es": "", "fr": "", "ko": ""},
        ]

        mock_translate_standard_patch.return_value = {
            "es": {"Hello": "Hola", "Goodbye": "Adiós"},
            "ko": {"Hello": "안녕하세요", "Goodbye": "안녕히 가세요"},
        }
        llm_get_driver_mock.return_value = mock_llm_driver
        project_get_driver_mock.return_value = mock_llm_driver
        config = await mock_storage.load_config("test_project")
        project = TranslationProject(
            project_id="test_project",
            config=config,
            dst_languages=["es", "fr", "ko"],
            storage=mock_storage,
            prompt="Translate from {base_language} to {dst_languages}: {phrases_json}",
        )

        await project.translate(model="test_model")

        mock_translate_standard_patch.assert_awaited_once()
        assert mock_storage.progresses["es"] == {
            "Hello": "Hola",
            "Goodbye": "Adiós",
        }
        assert mock_storage.progresses["ko"] == {
            "Hello": "안녕하세요",
            "Goodbye": "안녕히 가세요",
        }
        assert mock_storage.progresses["fr"] == {"Hello": "Déjà relu"}
        assert mock_storage.translations[0]["fr"] == "Déjà relu"
        assert mock_storage.translations[1]["fr"] == ""
        assert {failure["language"] for failure in mock_storage.failures} == {"fr"}

    @patch("lib.TranslationTool.TranslationTool.translate_standard")
    @patch("lib.TranslationProject.get_driver")
    async def test_multilanguage_run_writes_progress_per_language(
        self,
        project_get_driver_mock,
        mock_translate_standard_patch,
        mock_llm_driver,
        tmp_path,
    ):
        project_path = tmp_path / "multilang_project"
        project_path.mkdir()
        (project_path / "config.json").write_text(
            json.dumps(
                {
                    "name": project_path.name,
                    "sourceFile": "translations.csv",
                    "baseLanguage": "en",
                    "languages": ["en", "es", "fr", "ko"],
                    "keyColumn": "key",
                }
            ),
            encoding="utf-8",
        )
        with (project_path / "translations.csv").open(
            "w", encoding="utf-8", newline=""
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["key", "en", "es", "fr", "ko"])
            writer.writeheader()
            writer.writerow({"key": "hello", "en": "Hello", "es": "", "fr": "", "ko": ""})

        storage = FileSystemStorageAdapter(project_path)
        storage.set_active_languages(["es", "fr", "ko"])
        project_get_driver_mock.return_value = mock_llm_driver
        mock_translate_standard_patch.return_value = {
            "es": {"Hello": "Hola"},
            "ko": {"Hello": "안녕하세요"},
        }
        project = await TranslationProject.create(
            project_name=project_path.name,
            dst_languages=["es", "fr", "ko"],
            storage=storage,
        )

        await project.translate(model="test_model")

        mock_translate_standard_patch.assert_awaited_once()
        assert json.loads((project_path / "es" / "progress.json").read_text()) == {
            "Hello": "Hola"
        }
        assert json.loads((project_path / "fr" / "progress.json").read_text()) == {}
        assert json.loads((project_path / "ko" / "progress.json").read_text()) == {
            "Hello": "안녕하세요"
        }
        rows = await storage.load_translations(project_path.name)
        assert rows[0] | {"fr": ""} == {
            "key": "hello",
            "en": "Hello",
            "es": "Hola",
            "fr": "",
            "ko": "안녕하세요",
        }

    @pytest.mark.asyncio
    async def test_has_missing_phrases_false_when_all_cells_valid(self, mock_storage):
        """Gap detection returns false when CSV and progress are complete."""
        mock_storage.translations = [
            {"en": "Hello", "es": "Hola"},
            {"en": "Goodbye", "es": "Adios"},
        ]
        config = await mock_storage.load_config("test_project")
        project = TranslationProject(
            project_id="test_project",
            config=config,
            dst_languages=["es"],
            storage=mock_storage,
        )

        assert (
            project._has_missing_phrases(
                mock_storage.translations,
                {"es": {"Hello": "Hola", "Goodbye": "Adios"}},
                regenerate=False,
            )
            is False
        )

    @patch("lib.TranslationProject.TranslationProject._has_missing_phrases", return_value=False)
    @patch("lib.TranslationProject.TranslationProject._translate_pass", new_callable=AsyncMock)
    @patch("lib.llm.get_driver")
    @patch("lib.TranslationProject.get_driver")
    async def test_gap_pass_skipped_when_nothing_missing(
        self,
        project_get_driver_mock,
        llm_get_driver_mock,
        translate_pass_mock,
        _has_missing_mock,
        mock_llm_driver,
        mock_storage,
    ):
        """translate() must not schedule a gap pass when no cells are missing."""
        llm_get_driver_mock.return_value = mock_llm_driver
        project_get_driver_mock.return_value = mock_llm_driver

        config = await mock_storage.load_config("test_project")
        project = TranslationProject(
            project_id="test_project",
            config=config,
            dst_languages=["es"],
            storage=mock_storage,
        )

        await project.translate(
            model="primary-model",
            fallback_model="fallback-model",
        )

        assert translate_pass_mock.await_count == 1
        _has_missing_mock.assert_called_once()
