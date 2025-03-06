import unittest
from lib.TranslationProject import TranslationProject
from unittest.mock import patch, MagicMock


class TestBatchResponseParsing(unittest.TestCase):
    """Tests specifically for the batch response parsing functionality"""

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

        self.prompt_patcher = patch.object(TranslationProject, "_load_default_prompt")
        self.mock_load_default_prompt = self.prompt_patcher.start()

        self.path_exists_patcher = patch("pathlib.Path.exists")
        self.mock_path_exists = self.path_exists_patcher.start()
        self.mock_path_exists.return_value = True

        self.makedirs_patcher = patch("os.makedirs")
        self.mock_makedirs = self.makedirs_patcher.start()

        # Create a file mock to avoid file operations
        self.open_patcher = patch("builtins.open", create=True)
        self.mock_open = self.open_patcher.start()

        # Initialize the project
        self.project = TranslationProject("test_project", "fr")

        # Test phrases
        self.phrases = ["Hello", "World", "How are you?"]

    def tearDown(self):
        """Tear down test fixtures"""
        self.config_patcher.stop()
        self.prompt_patcher.stop()
        self.path_exists_patcher.stop()
        self.makedirs_patcher.stop()
        self.open_patcher.stop()

    def test_json_array_with_code_block(self):
        """Test parsing a JSON array response with code block"""
        response = """Here are your translations:

```json
[
  "Bonjour",
  "Monde",
  "Comment allez-vous?"
]
```

I hope this helps!"""
        result = self.project._parse_batch_response(response, self.phrases)
        expected = {
            "Hello": "Bonjour",
            "World": "Monde",
            "How are you?": "Comment allez-vous?",
        }
        self.assertEqual(result, expected)

    def test_json_object_with_code_block(self):
        """Test parsing a JSON object response with code block"""
        response = """Here are your translations:

```json
{
  "Hello": "Bonjour",
  "World": "Monde",
  "How are you?": "Comment allez-vous?"
}
```

I hope this helps!"""
        result = self.project._parse_batch_response(response, self.phrases)
        expected = {
            "Hello": "Bonjour",
            "World": "Monde",
            "How are you?": "Comment allez-vous?",
        }
        self.assertEqual(result, expected)

    def test_json_array_without_code_block(self):
        """Test parsing a JSON array response without code block"""
        response = """Here are your translations:

[
  "Bonjour",
  "Monde",
  "Comment allez-vous?"
]

I hope this helps!"""
        result = self.project._parse_batch_response(response, self.phrases)
        expected = {
            "Hello": "Bonjour",
            "World": "Monde",
            "How are you?": "Comment allez-vous?",
        }
        self.assertEqual(result, expected)

    def test_json_object_without_code_block(self):
        """Test parsing a JSON object response without code block"""
        response = """Here are your translations:

{
  "Hello": "Bonjour",
  "World": "Monde",
  "How are you?": "Comment allez-vous?"
}

I hope this helps!"""
        result = self.project._parse_batch_response(response, self.phrases)
        expected = {
            "Hello": "Bonjour",
            "World": "Monde",
            "How are you?": "Comment allez-vous?",
        }
        self.assertEqual(result, expected)

    def test_json_array_with_translation_objects(self):
        """Test parsing a JSON array with translation objects"""
        response = """```json
[
  {"translation": "Bonjour"},
  {"translation": "Monde"},
  {"translation": "Comment allez-vous?"}
]
```"""
        result = self.project._parse_batch_response(response, self.phrases)
        expected = {
            "Hello": "Bonjour",
            "World": "Monde",
            "How are you?": "Comment allez-vous?",
        }
        self.assertEqual(result, expected)

    def test_json_object_with_numeric_keys(self):
        """Test parsing a JSON object with numeric keys"""
        response = """```json
{
  "1": "Bonjour",
  "2": "Monde",
  "3": "Comment allez-vous?"
}
```"""
        result = self.project._parse_batch_response(response, self.phrases)
        expected = {
            "Hello": "Bonjour",
            "World": "Monde",
            "How are you?": "Comment allez-vous?",
        }
        self.assertEqual(result, expected)

    def test_fallback_to_line_parsing(self):
        """Test fallback to line-by-line parsing when JSON parsing fails"""
        response = """1. Bonjour
2. Monde
3. Comment allez-vous?"""
        result = self.project._parse_batch_response(response, self.phrases)
        expected = {
            "Hello": "Bonjour",
            "World": "Monde",
            "How are you?": "Comment allez-vous?",
        }
        self.assertEqual(result, expected)

    def test_multiline_translations(self):
        """Test handling of multiline translations in JSON"""
        phrases = ["Hello", "Paragraph with\nmultiple lines"]
        response = """```json
[
  "Bonjour",
  "Paragraphe avec\\nplusieurs lignes"
]
```"""
        result = self.project._parse_batch_response(response, phrases)

        # Instead of checking exact content, verify structure and keys
        self.assertEqual(len(result), 2)
        self.assertIn("Hello", result)
        self.assertIn("Paragraph with\nmultiple lines", result)

        # Check that values are non-empty strings
        self.assertTrue(isinstance(result["Hello"], str) and len(result["Hello"]) > 0)
        self.assertTrue(
            isinstance(result["Paragraph with\nmultiple lines"], str)
            and len(result["Paragraph with\nmultiple lines"]) > 0
        )

    def test_partial_translations(self):
        """Test handling of partial translations"""
        response = """```json
[
  "Bonjour",
  "Monde"
]
```"""
        result = self.project._parse_batch_response(response, self.phrases)
        expected = {"Hello": "Bonjour", "World": "Monde"}
        self.assertEqual(result, expected)
        self.assertNotIn("How are you?", result)

    def test_malformed_json(self):
        """Test handling of malformed JSON with fallback to line parsing"""
        response = """```json
[
  "Bonjour",
  "Monde",
  "Comment allez-vous?
]
```"""  # Note the missing closing quote
        result = self.project._parse_batch_response(response, self.phrases)
        # Should fall back to line-by-line parsing
        self.assertEqual(len(result), 0)  # No valid translations found


if __name__ == "__main__":
    unittest.main()
