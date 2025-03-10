import pytest
from lib.TranslationProject import TranslationProject
from lib.TranslationTool import InvalidJSONException


class TestBatchResponseParsing:
    """Tests specifically for the batch response parsing functionality"""

    def test_json_object_with_code_block(
        self, translation_project: TranslationProject
    ) -> None:
        """Test parsing a JSON object response inside code blocks"""
        response = """```json
{
    "Hello": "Bonjour",
    "World": "Monde"
}
```"""
        phrases = ["Hello", "World"]
        result = translation_project._parse_batch_response(response, phrases)

        assert result == {"Hello": "Bonjour", "World": "Monde"}

    def test_json_object_without_code_block(
        self, translation_project: TranslationProject
    ) -> None:
        """Test parsing a JSON object response without code blocks"""
        response = """{
    "Hello": "Bonjour",
    "World": "Monde"
}"""
        phrases = ["Hello", "World"]
        result = translation_project._parse_batch_response(response, phrases)

        assert result == {"Hello": "Bonjour", "World": "Monde"}

    def test_json_array_with_code_block(
        self, translation_project: TranslationProject
    ) -> None:
        """Test parsing a JSON array response inside code blocks"""
        response = """```json
[
    "Bonjour",
    "Monde"
]
```"""
        phrases = ["Hello", "World"]
        result = translation_project._parse_batch_response(response, phrases)

        assert result == {"Hello": "Bonjour", "World": "Monde"}

    def test_json_array_without_code_block(
        self, translation_project: TranslationProject
    ) -> None:
        """Test parsing a JSON array response without code blocks"""
        response = """[
    "Bonjour",
    "Monde"
]"""
        phrases = ["Hello", "World"]
        result = translation_project._parse_batch_response(response, phrases)

        assert result == {"Hello": "Bonjour", "World": "Monde"}

    def test_malformed_json(self, translation_project: TranslationProject) -> None:
        """Test handling malformed JSON"""
        response = """Invalid JSON that cannot be parsed"""
        phrases = ["Hello", "World"]

        with pytest.raises(InvalidJSONException):
            translation_project._parse_batch_response(response, phrases)

    def test_json_array_with_translation_objects(
        self, translation_project: TranslationProject
    ) -> None:
        """Test parsing a JSON array with translation objects"""
        response = """```json
[
    {"translation": "Bonjour"},
    {"translation": "Monde"}
]
```"""
        phrases = ["Hello", "World"]
        result = translation_project._parse_batch_response(response, phrases)

        assert result == {"Hello": "Bonjour", "World": "Monde"}

    def test_json_object_with_numeric_keys(
        self, translation_project: TranslationProject
    ) -> None:
        """Test parsing a JSON object with numeric keys"""
        response = """```json
{
    "1": "Bonjour",
    "2": "Monde"
}
```"""
        phrases = ["Hello", "World"]
        result = translation_project._parse_batch_response(response, phrases)

        assert result == {"Hello": "Bonjour", "World": "Monde"}

    def test_multiline_translations(
        self, translation_project: TranslationProject
    ) -> None:
        """Test handling multiline text in translations"""
        response = """```json
{
    "Hello\\nWorld": "Bonjour\\nMonde"
}
```"""
        phrases = ["Hello\nWorld"]
        result = translation_project._parse_batch_response(response, phrases)

        assert result == {"Hello\nWorld": "Bonjour\nMonde"}

    def test_partial_translations(
        self, translation_project: TranslationProject
    ) -> None:
        """Test handling partial translations"""
        # Only "Hello" is translated, "World" is not included in response
        response = """```json
{
    "Hello": "Bonjour"
}
```"""
        phrases = ["Hello", "World"]

        # The method should return what it can parse, even if incomplete
        result = translation_project._parse_batch_response(response, phrases)

        # It should contain only the translated phrase
        assert "Hello" in result
        assert result["Hello"] == "Bonjour"
        assert "World" not in result
