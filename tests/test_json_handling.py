import pytest
import json
from unittest.mock import Mock, AsyncMock
from lib.TranslationProject import TranslationProject, InvalidJSONException
from lib.llm import BaseDriver


class MockDriver(BaseDriver):
    def __init__(self, model: str = "test"):
        super().__init__(model)
        self.translate_async = AsyncMock()


@pytest.fixture
def translation_project():
    return TranslationProject(
        project_name="test_project",
        dst_language="es",
        config={
            "languages": ["en", "es"],
            "baseLanguage": "en",
            "sourceFile": "test.csv",
        },
    )


class TestJsonHandling:
    def test_parse_batch_response_with_code_block(self, translation_project):
        """Test parsing JSON response within code blocks"""
        response = """Here's the translation:
```json
{
    "Hello": "Hola",
    "World": "Mundo"
}
```"""
        original_phrases = ["Hello", "World"]
        result = translation_project._parse_batch_response(response, original_phrases)
        assert result == {"Hello": "Hola", "World": "Mundo"}

    def test_parse_batch_response_with_direct_json(self, translation_project):
        """Test parsing direct JSON response without code blocks"""
        response = '{"Hello": "Hola", "World": "Mundo"}'
        original_phrases = ["Hello", "World"]
        result = translation_project._parse_batch_response(response, original_phrases)
        assert result == {"Hello": "Hola", "World": "Mundo"}

    def test_parse_batch_response_with_list_format(self, translation_project):
        """Test parsing JSON response in list format"""
        response = """```json
["Hola", "Mundo"]
```"""
        original_phrases = ["Hello", "World"]
        result = translation_project._parse_batch_response(response, original_phrases)
        assert result == {"Hello": "Hola", "World": "Mundo"}

    def test_parse_batch_response_with_dict_format(self, translation_project):
        """Test parsing JSON response with dictionary format containing translation field"""
        response = """```json
[
    {"translation": "Hola"},
    {"translation": "Mundo"}
]
```"""
        original_phrases = ["Hello", "World"]
        result = translation_project._parse_batch_response(response, original_phrases)
        assert result == {"Hello": "Hola", "World": "Mundo"}

    def test_parse_batch_response_with_text_format(self, translation_project):
        """Test parsing JSON response with dictionary format containing text field"""
        response = """```json
[
    {"text": "Hello", "translation": "Hola"},
    {"text": "World", "translation": "Mundo"}
]
```"""
        original_phrases = ["Hello", "World"]
        result = translation_project._parse_batch_response(response, original_phrases)
        assert result == {"Hello": "Hola", "World": "Mundo"}

    def test_parse_batch_response_with_numeric_keys(self, translation_project):
        """Test parsing JSON response with numeric keys"""
        response = """```json
{
    "1": "Hola",
    "2": "Mundo"
}
```"""
        original_phrases = ["Hello", "World"]
        result = translation_project._parse_batch_response(response, original_phrases)
        assert result == {"Hello": "Hola", "World": "Mundo"}

    def test_parse_batch_response_invalid_json(self, translation_project):
        """Test handling of invalid JSON response"""
        response = """```json
{
    "Hello": "Hola",
    "World": "Mundo",
    invalid json here
}
```"""
        original_phrases = ["Hello", "World"]
        with pytest.raises(InvalidJSONException) as exc_info:
            translation_project._parse_batch_response(response, original_phrases)
        assert "Error parsing JSON response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fix_invalid_json_success(self, translation_project):
        """Test successful fixing of invalid JSON"""
        invalid_json = """{
            "Hello": "Hola",
            "World": "Mundo"
            invalid json here
        }"""

        fixed_json = """{
            "Hello": "Hola",
            "World": "Mundo"
        }"""

        mock_driver = MockDriver()
        mock_driver.translate_async.return_value = f"```json\n{fixed_json}\n```"

        result = await translation_project._fix_invalid_json(invalid_json, mock_driver)
        assert json.loads(result) == {"Hello": "Hola", "World": "Mundo"}

    @pytest.mark.asyncio
    async def test_fix_invalid_json_direct_object(self, translation_project):
        """Test fixing invalid JSON when response is a direct object"""
        invalid_json = """{
            "Hello": "Hola"
            "World": "Mundo"
        }"""

        fixed_json = '{"Hello": "Hola", "World": "Mundo"}'

        mock_driver = MockDriver()
        mock_driver.translate_async.return_value = fixed_json

        result = await translation_project._fix_invalid_json(invalid_json, mock_driver)
        assert json.loads(result) == {"Hello": "Hola", "World": "Mundo"}

    @pytest.mark.asyncio
    async def test_fix_invalid_json_failure(self, translation_project):
        """Test handling of failed JSON fix attempt"""
        invalid_json = """{
            completely invalid json
            that cannot be fixed
        }"""

        mock_driver = MockDriver()
        mock_driver.translate_async.side_effect = Exception("Failed to fix JSON")

        result = await translation_project._fix_invalid_json(invalid_json, mock_driver)
        assert result == invalid_json  # Should return original on failure

    @pytest.mark.asyncio
    async def test_fix_invalid_json_with_array(self, translation_project):
        """Test fixing invalid JSON array"""
        invalid_json = """[
            {"text": "Hello", "translation": "Hola"}
            {"text": "World", "translation": "Mundo"}
        ]"""

        fixed_json = """[
            {"text": "Hello", "translation": "Hola"},
            {"text": "World", "translation": "Mundo"}
        ]"""

        mock_driver = MockDriver()
        mock_driver.translate_async.return_value = f"```json\n{fixed_json}\n```"

        result = await translation_project._fix_invalid_json(invalid_json, mock_driver)
        assert isinstance(json.loads(result), list)
        assert len(json.loads(result)) == 2
