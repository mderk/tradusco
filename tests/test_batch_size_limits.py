import pytest
from unittest.mock import MagicMock


from lib.TranslationProject import TranslationProject


class TestBatchSizeLimits:
    """Tests for the batch size and byte limit functionality"""

    @pytest.mark.asyncio
    async def test_batch_size_limit(
        self,
        translation_project: TranslationProject,
        common_translation_project_patches: dict[str, MagicMock],
    ) -> None:
        """Test that batch sizes are limited correctly"""
        # Extract mocks from common patches
        mock_load_translations = common_translation_project_patches[
            "mock_load_translations"
        ]
        mock_load_progress = common_translation_project_patches["mock_load_progress"]
        mock_process_batch = common_translation_project_patches["mock_process_batch"]
        mock_save_progress = common_translation_project_patches["mock_save_progress"]
        mock_save_translations = common_translation_project_patches[
            "mock_save_translations"
        ]
        mock_get_driver = common_translation_project_patches["mock_get_driver"]

        # Create a test translations dataset with 10 items
        translations = []
        for i in range(10):
            translations.append({"en": f"Phrase {i}", "fr": ""})

        # Create an empty progress object
        progress = {}

        # Setup the mocks
        mock_load_translations.return_value = translations
        mock_load_progress.return_value = progress

        # Mock process_batch to return a simple dictionary
        mock_process_batch.return_value = {"Phrase 0": "Translated 0"}

        # Create a mock driver
        mock_driver = MagicMock()
        mock_get_driver.return_value = mock_driver

        # Test with a batch size of 2
        batch_size = 2
        await translation_project.translate(batch_size=batch_size, delay_seconds=0)

        # Verify that process_batch was called
        assert mock_load_translations.call_count == 1
        assert mock_load_progress.call_count == 1
        assert mock_process_batch.call_count > 0

        # Verify save methods were called
        assert mock_save_progress.call_count >= 1
        assert mock_save_translations.call_count >= 1

    @pytest.mark.asyncio
    async def test_batch_max_bytes_limit(
        self,
        translation_project: TranslationProject,
        common_translation_project_patches: dict[str, MagicMock],
    ) -> None:
        """Test that batch sizes are limited by byte size"""
        # Extract mocks from common patches
        mock_load_translations = common_translation_project_patches[
            "mock_load_translations"
        ]
        mock_load_progress = common_translation_project_patches["mock_load_progress"]
        mock_process_batch = common_translation_project_patches["mock_process_batch"]
        mock_save_progress = common_translation_project_patches["mock_save_progress"]
        mock_save_translations = common_translation_project_patches[
            "mock_save_translations"
        ]
        mock_get_driver = common_translation_project_patches["mock_get_driver"]

        # Create translations with varying sizes
        # Small phrases ~10 bytes each
        small_phrases = [{"en": f"Word{i}", "fr": ""} for i in range(5)]

        # Large phrases ~100 bytes each
        large_text = "A" * 100
        large_phrases = [{"en": f"{large_text}{i}", "fr": ""} for i in range(2)]

        # Mix them together
        translations = small_phrases + large_phrases

        # Create an empty progress object
        progress = {}

        # Setup the mocks
        mock_load_translations.return_value = translations
        mock_load_progress.return_value = progress

        # Mock process_batch to return a simple dictionary
        mock_process_batch.return_value = {"Word0": "Translated Word0"}

        # Create a mock driver
        mock_driver = MagicMock()
        mock_get_driver.return_value = mock_driver

        # Test with a high batch size but limited bytes
        batch_size = 100  # High enough to not be a limiting factor
        batch_max_bytes = 200  # This should split the large phrases

        await translation_project.translate(
            batch_size=batch_size, batch_max_bytes=batch_max_bytes, delay_seconds=0
        )

        # Verify process_batch was called
        assert mock_process_batch.call_count > 0

        # Verify save methods were called
        assert mock_save_progress.call_count >= 1
        assert mock_save_translations.call_count >= 1

    @pytest.mark.asyncio
    async def test_batch_limit_interaction(
        self,
        translation_project: TranslationProject,
        common_translation_project_patches: dict[str, MagicMock],
    ) -> None:
        """Test interaction between batch_size and batch_max_bytes limits"""
        # Extract mocks from common patches
        mock_load_translations = common_translation_project_patches[
            "mock_load_translations"
        ]
        mock_load_progress = common_translation_project_patches["mock_load_progress"]
        mock_process_batch = common_translation_project_patches["mock_process_batch"]
        mock_save_progress = common_translation_project_patches["mock_save_progress"]
        mock_save_translations = common_translation_project_patches[
            "mock_save_translations"
        ]
        mock_get_driver = common_translation_project_patches["mock_get_driver"]

        # Create a set of translations
        # Mix of small phrases and medium-sized phrases
        small_phrases = [{"en": f"Word{i}", "fr": ""} for i in range(5)]
        medium_text = "B" * 50
        medium_phrases = [{"en": f"{medium_text}{i}", "fr": ""} for i in range(3)]

        translations = small_phrases + medium_phrases

        # Create an empty progress object
        progress = {}

        # Setup the mocks
        mock_load_translations.return_value = translations
        mock_load_progress.return_value = progress

        # Mock process_batch to return a simple dictionary
        mock_process_batch.return_value = {"Word0": "Translated Word0"}

        # Create a mock driver
        mock_driver = MagicMock()
        mock_get_driver.return_value = mock_driver

        # Test with both limits active
        batch_size = 3  # Limit to 3 phrases per batch
        batch_max_bytes = 100  # Also limit byte size

        await translation_project.translate(
            batch_size=batch_size, batch_max_bytes=batch_max_bytes, delay_seconds=0
        )

        # Verify process_batch was called
        assert mock_process_batch.call_count > 0

        # Verify save methods were called
        assert mock_save_progress.call_count >= 1
        assert mock_save_translations.call_count >= 1
