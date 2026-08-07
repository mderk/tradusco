import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.failure_reporting import batch_error_kind_to_category, make_failure_record
from lib.llm.BaseDriver import BaseDriver


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
