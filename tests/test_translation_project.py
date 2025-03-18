import os
import sys
import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import csv
from typing import Optional, List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.TranslationProject import TranslationProject
from lib.utils import Config
from lib.storage.base import StorageAdapter
from tests.mock_llm_driver import MockLLMDriver


# Mock storage adapter for testing
class MockStorageAdapter(StorageAdapter):
    def __init__(self):
        self.config = Config(
            name="test_project",
            sourceFile="source.csv",
            baseLanguage="en",
            languages=["en", "es", "fr"],
            keyColumn="key",
        )
        self.translations = []
        self.context_strings = []
        self.prompts = {}
        self.context_file = None
        self.prompt_file = None

    def set_context_file(self, context_file):
        self.context_file = context_file

    def set_prompt_file(self, prompt_file):
        self.prompt_file = prompt_file

    async def load_config(self, project_id: str) -> Config:
        return self.config

    async def load_progress(self, project_id: str, language: str) -> Dict[str, str]:
        return {}

    async def save_progress(
        self, project_id: str, language: str, progress: Dict[str, str]
    ) -> None:
        pass

    async def load_translations(self, project_id: str) -> List[Dict[str, Any]]:
        return self.translations

    async def save_translations(
        self, project_id: str, translations: List[Dict[str, Any]]
    ) -> None:
        self.translations = translations

    async def load_context(self, project_id: str) -> List[str]:
        return self.context_strings

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
            "translation": "Translate from {base_language} to {dst_language}: {phrases_json}",
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
            dst_language="es",
            storage=mock_storage,
        )

        assert project.project_id == "test_project"
        assert project.base_language == "en"
        assert project.dst_language == "es"
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
            dst_language="es",
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
            dst_language="es",
            storage=mock_storage,
        )

        # Count tokens in a string
        token_count = project.count_tokens("This is a test")
        assert token_count > 0
        assert isinstance(token_count, int)

    @pytest.mark.asyncio
    async def test_load_context(self, mock_storage):
        """Test loading context."""
        # Create a test project
        config = await mock_storage.load_config("test_project")
        project = TranslationProject(
            project_id="test_project",
            config=config,
            dst_language="es",
            storage=mock_storage,
        )

        # Load context
        context = await project._load_context()

        # Verify context was loaded correctly
        assert len(context) > 0
        assert "Context 1" in context
        assert "Context 2" in context

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
            phrases: list[str],
            indices: list[int],
            translations: list[dict[str, str]],
            progress: dict[str, str],
            model: str,
            base_language: str,
            dst_language: str,
            prompt: str,
            context: Optional[str] = None,
            delay_seconds: float = 1.0,
            max_retries: int = 3,
        ) -> int:
            # Simulate translation
            for i, phrase in enumerate(phrases):
                translations[indices[i]][dst_language] = f"{phrase} (translated)"
                progress[phrase] = f"{phrase} (translated)"
            return len(phrases)

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
            dst_language="es",
            storage=mock_storage,
            prompt="Translate from {base_language} to {dst_language}",
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
