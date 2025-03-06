import unittest
from unittest.mock import patch, mock_open
from pathlib import Path

from lib.TranslationProject import TranslationProject


class TestPromptHandling(unittest.TestCase):
    """Tests for prompt loading and handling functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Create patches to avoid actual file operations
        self.config_patcher = patch.object(TranslationProject, "_load_config")
        self.mock_load_config = self.config_patcher.start()
        self.mock_load_config.return_value = {
            "name": "test_project",
            "sourceFile": "translations.csv",
            "languages": ["en", "fr"],
            "baseLanguage": "en",
            "keyColumn": "en",
        }

        self.path_exists_patcher = patch("pathlib.Path.exists")
        self.mock_path_exists = self.path_exists_patcher.start()
        self.mock_path_exists.return_value = True

        self.makedirs_patcher = patch("os.makedirs")
        self.mock_makedirs = self.makedirs_patcher.start()

    def tearDown(self):
        """Tear down test fixtures"""
        self.config_patcher.stop()
        self.path_exists_patcher.stop()
        self.makedirs_patcher.stop()

    @patch("builtins.open", new_callable=mock_open, read_data="Default prompt content")
    @patch.object(Path, "exists", return_value=True)
    def test_load_default_prompt(self, mock_exists, mock_file):
        """Test loading the default prompt"""
        with patch.object(Path, "parent", return_value=Path("/fake/path")):
            project = TranslationProject("test_project", "fr")
            result = project._load_default_prompt("test_prompt.txt")
            self.assertEqual(result, "Default prompt content")

    @patch("builtins.open", new_callable=mock_open, read_data="Custom prompt content")
    @patch("os.path.exists", return_value=True)
    def test_load_custom_prompt_with_valid_file(self, mock_exists, mock_file):
        """Test loading a custom prompt with a valid file path"""
        project = TranslationProject("test_project", "fr")
        result = project._load_custom_prompt("custom_prompt.txt")
        self.assertEqual(result, "Custom prompt content")

    @patch.object(Path, "exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data="Default prompt content")
    def test_load_custom_prompt_with_none(self, mock_file, mock_path_exists):
        """Test loading a custom prompt with None as the file path"""
        with patch.object(
            TranslationProject, "_load_default_prompt", return_value="Default prompt"
        ):
            project = TranslationProject("test_project", "fr")
            result = project._load_custom_prompt(None)
            self.assertEqual(result, "")

    @patch.object(Path, "exists", return_value=True)
    @patch("os.path.exists", return_value=False)
    @patch("builtins.open", new_callable=mock_open, read_data="Default prompt content")
    def test_load_custom_prompt_with_nonexistent_file(
        self, mock_file, mock_os_exists, mock_path_exists
    ):
        """Test loading a custom prompt with a nonexistent file path"""
        with patch.object(
            TranslationProject, "_load_default_prompt", return_value="Default prompt"
        ):
            project = TranslationProject("test_project", "fr")
            result = project._load_custom_prompt("nonexistent.txt")
            self.assertEqual(result, "")
            mock_os_exists.assert_called_once_with("nonexistent.txt")

    @patch.object(TranslationProject, "_load_default_prompt")
    @patch.object(TranslationProject, "_load_custom_prompt")
    def test_custom_prompt_precedence(self, mock_load_custom, mock_load_default):
        """Test that custom prompts take precedence over default prompts"""
        # Setup mocks
        mock_load_default.return_value = "Default prompt"
        mock_load_custom.return_value = "Custom prompt"

        # Initialize with a custom prompt file
        project = TranslationProject("test_project", "fr", prompt_file="custom.txt")

        # Create a batch prompt
        phrases = ["Hello", "World"]
        result = project._create_batch_prompt(phrases)

        # Verify custom prompt was used
        mock_load_custom.assert_called_with("custom.txt")
        self.assertIn("Custom prompt", result)

    @patch.object(TranslationProject, "_load_default_prompt")
    @patch.object(TranslationProject, "_load_custom_prompt")
    def test_default_prompt_fallback(self, mock_load_custom, mock_load_default):
        """Test fallback to default prompt when custom prompt is not available"""
        # Setup mocks
        mock_load_default.return_value = "Default prompt"
        mock_load_custom.return_value = ""  # Empty string indicates no custom prompt

        # Initialize with a nonexistent custom prompt file
        project = TranslationProject(
            "test_project", "fr", prompt_file="nonexistent.txt"
        )

        # Create a batch prompt
        phrases = ["Hello", "World"]
        result = project._create_batch_prompt(phrases)

        # Verify default prompt was used
        mock_load_custom.assert_called_with("nonexistent.txt")
        self.assertIn("Default prompt", result)


if __name__ == "__main__":
    unittest.main()
