import pytest
from unittest.mock import patch, MagicMock
import tiktoken
from lib.TranslationProject import TranslationProject


class TestTokenCounting:
    """Tests for the token counting functionality"""

    def test_tiktoken_counting(self):
        """Test tiktoken-based token counting"""
        # Test text samples of varying lengths
        samples = [
            "",  # Empty string
            "Hello",  # Single word
            "Hello world!",  # Two words with punctuation
            "This is a longer sentence with multiple words.",  # Full sentence
            # A paragraph with various punctuation
            "The translator project handles various languages. It can process English, French, German, etc. Each has its own quirks!",
        ]

        # Get expected counts directly from tiktoken for comparison
        encoder = tiktoken.get_encoding("cl100k_base")

        # Create a mock driver that uses tiktoken for token counting
        mock_driver = MagicMock()
        mock_driver.count_tokens.side_effect = lambda text: len(encoder.encode(text))

        # Mock the get_driver function to return our mock driver
        with patch("lib.TranslationProject.get_driver", return_value=mock_driver):
            for sample in samples:
                # Count directly with tiktoken
                expected_count = len(encoder.encode(sample))
                # Count with our method
                actual_count = TranslationProject.count_tokens(sample)

                # Both should match exactly since we're using tiktoken directly
                assert (
                    actual_count == expected_count
                ), f"Token count mismatch for '{sample}'"

                # The model parameter should be ignored since we're mocking get_driver
                with_model_param = TranslationProject.count_tokens(sample, "some-model")
                assert (
                    with_model_param == expected_count
                ), f"Model parameter affected token count for '{sample}'"

    def test_token_counting_realistic(self):
        """Test token counting with realistic translation content"""
        # Simulate a realistic paragraph that might be translated
        paragraph = """
        The artificial intelligence translation system uses advanced language models to convert text
        between different languages. It can handle complex grammar, idiomatic expressions, and specialized
        terminology with reasonable accuracy. The system adapts to different content types and domains by
        using appropriate context information and specialized prompts. Each phrase is processed as part of a
        batch to efficiently utilize API calls and maintain consistent translations throughout a document.
        """

        # Get expected count directly from tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        expected_count = len(encoder.encode(paragraph))

        # Create a mock driver that uses tiktoken for token counting
        mock_driver = MagicMock()
        mock_driver.count_tokens.side_effect = lambda text: len(encoder.encode(text))

        # Mock the get_driver function to return our mock driver
        with patch("lib.TranslationProject.get_driver", return_value=mock_driver):
            # Count tokens with our method
            token_count = TranslationProject.count_tokens(paragraph)

            # They should match exactly
            assert (
                token_count == expected_count
            ), f"Token count {token_count} doesn't match expected {expected_count}"

    def test_fallback_behavior(self):
        """Test that the fallback mechanism works when tiktoken fails"""
        # Create a mock driver that raises an exception when counting tokens
        mock_driver = MagicMock()
        mock_driver.count_tokens.side_effect = Exception(
            "Simulated token counting failure"
        )

        # Mock the get_driver function to return our mock driver
        with patch("lib.TranslationProject.get_driver", return_value=mock_driver):
            text = "This is a sample text to test fallback behavior."

            # Should use character-based approximation
            token_count = TranslationProject.count_tokens(text)

            # Expected: length / 4 (character-based approximation)
            expected_tokens = max(1, len(text) // 4)

            # They should match exactly
            assert (
                token_count == expected_tokens
            ), f"Fallback token count {token_count} doesn't match expected {expected_tokens}"

            # Test empty string
            empty_count = TranslationProject.count_tokens("")
            assert (
                empty_count == 0
            ), f"Empty string should return 0 tokens, got {empty_count}"
