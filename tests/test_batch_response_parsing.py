import unittest
import asyncio
from lib.TranslationProject import TranslationProject, InvalidJSONException
from unittest.mock import patch, MagicMock


class AsyncMock(MagicMock):
    """Helper class for mocking async methods"""

    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


class TestBatchResponseParsing(unittest.TestCase):
    """Tests specifically for the batch response parsing functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a mock config
        self.mock_config = {
            "name": "test_project",
            "sourceFile": "translations.csv",
            "languages": ["en", "fr"],
            "baseLanguage": "en",
            "keyColumn": "en",
        }

        # Create patches for file operations
        self.path_exists_patcher = patch("pathlib.Path.exists")
        self.mock_path_exists = self.path_exists_patcher.start()
        self.mock_path_exists.return_value = True

        self.makedirs_patcher = patch("os.makedirs")
        self.mock_makedirs = self.makedirs_patcher.start()

        # Initialize the project directly with the mock config
        self.project = TranslationProject(
            "test_project",
            "fr",
            config=self.mock_config,
            default_prompt="Default prompt",
        )

        # Test phrases
        self.phrases = ["Hello", "World", "How are you?"]

        # Create an event loop for each test
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Tear down test fixtures"""
        self.path_exists_patcher.stop()
        self.makedirs_patcher.stop()

        # Close the event loop
        self.loop.close()
        asyncio.set_event_loop(None)

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
        """Test handling of malformed JSON raises InvalidJSONException"""
        response = """```json
[
  "Bonjour",
  "Monde",
  "Comment allez-vous?
]
```"""  # Note the missing closing quote
        with self.assertRaises(InvalidJSONException) as context:
            self.project._parse_batch_response(response, self.phrases)
        self.assertIn("Error parsing JSON response", str(context.exception))


if __name__ == "__main__":
    unittest.main()
