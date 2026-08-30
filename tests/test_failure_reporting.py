import os
import sys
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.failure_reporting import batch_error_kind_to_category, make_failure_record
from lib.llm.BaseDriver import BaseDriver
from lib.TranslationTool import classify_llm_error


class _TestDriver(BaseDriver):
    def __init__(self):
        super().__init__("test-model")
        self.llm = None


def test_batch_error_kind_to_category():
    assert batch_error_kind_to_category("blocked") == "refusal"
    assert batch_error_kind_to_category("rate_limit") == "network_error"
    assert batch_error_kind_to_category("model_error") == "parse_error"
    assert batch_error_kind_to_category("unknown") == "other"


def test_make_failure_record_truncates_message():
    record = make_failure_record(
        model="google/gemini-2.5-flash",
        phrase="Hello",
        category="parse_error",
        message="x" * 600,
        method="structured",
    )
    assert record["model"] == "google/gemini-2.5-flash"
    assert record["phrase"] == "Hello"
    assert record["category"] == "parse_error"
    assert record["method"] == "structured"
    assert len(str(record["message"])) == 500


def test_openrouter_request_kwargs():
    driver = _TestDriver()
    assert driver._openrouter_request_kwargs() == {}

    driver.openrouter_require_parameters = True
    assert driver._openrouter_request_kwargs() == {
        "extra_body": {"provider": {"require_parameters": True}}
    }


class _FakeStatusError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def test_classify_llm_error_maps_common_cases():
    blocked = classify_llm_error(Exception("content policy blocked"))
    assert blocked.kind == "blocked"

    rate_limit = classify_llm_error(_FakeStatusError(429, "too many requests"))
    assert rate_limit.kind == "rate_limit"
    assert rate_limit.status_code == 429

    auth = classify_llm_error(_FakeStatusError(401, "invalid api key"))
    assert auth.kind == "auth_error"

    other = classify_llm_error(Exception("something unexpected"))
    assert other.kind == "model_error"


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
@patch("lib.llm._fetch_openrouter_supported_parameters_by_model", return_value={})
def test_openrouter_driver_enables_require_parameters(_mock_caps):
    from lib.llm import get_driver

    driver = get_driver("google/gemini-2.5-flash")
    assert driver.openrouter_require_parameters is True
