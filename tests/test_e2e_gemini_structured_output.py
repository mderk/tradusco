import csv
import json
import os
from pathlib import Path

import pytest

from lib.TranslationProject import TranslationProject
from lib.storage.filesystem import FileSystemStorageAdapter
from lib.utils import is_valid_translation


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_temp_project_gemini25_structured_output(
    tmp_path: Path, capsys, caplog
):
    """
    E2E-ish test:
    - creates a temporary Tradusco project on disk
    - runs translation with the real Gemini 2.5 driver (auto -> structured)
    - verifies persisted outputs (CSV + progress) contain valid translations
      and preserve placeholders / Lingui tags for tricky phrases.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    caplog.set_level("WARNING")

    project_dir = tmp_path / "tmp_tradusco_project"
    project_dir.mkdir(parents=True, exist_ok=True)

    (project_dir / "config.json").write_text(
        json.dumps(
            {
                "name": "tmp_tradusco_project",
                "sourceFile": "translations.csv",
                "languages": ["en", "es"],
                "baseLanguage": "en",
                "keyColumn": "en",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    phrases: list[dict[str, str]] = [
        {
            "id": "HELLO",
            "en": "Hello world",
            "es": "",
            "context": "Simple greeting.",
        },
        {
            "id": "WELCOME_USER",
            "en": "Welcome, {name}!",
            "es": "",
            "context": "Keep the `{name}` placeholder exactly.",
        },
        {
            "id": "MESSAGES_COUNT",
            "en": "You have {count} new messages.",
            "es": "",
            "context": "Keep the `{count}` placeholder exactly.",
        },
        {
            "id": "RICH_TEXT",
            "en": "<0>Learn more</0>",
            "es": "",
            "context": "Keep Lingui numeric tags exactly (e.g. <0>...</0>).",
        },
        {
            "id": "BROKEN_HTML",
            "en": "Broken markup (keep as-is): <b>bold</i>",
            "es": "",
            "context": "Do not fix or alter the markup; preserve it verbatim.",
        },
        {
            "id": "MULTILINE",
            "en": "Line 1\\nLine 2",
            "es": "",
            "context": "Preserve the newline as a newline character.",
        },
    ]

    csv_path = project_dir / "translations.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "en", "es", "context"])
        writer.writeheader()
        for row in phrases:
            writer.writerow(row)

    storage = FileSystemStorageAdapter(project_dir)
    storage.set_active_language("es")
    storage.set_overwrite_active_language(False)

    translator = await TranslationProject.create(
        project_name="tmp_tradusco_project",
        dst_language="es",
        storage=storage,
        context=None,
    )

    await translator.translate(
        model="gemini",
        translation_method="auto",
        batch_size=25,
        batch_max_tokens=1500,
        max_retries=2,
        delay_seconds=1.0,
        regenerate=False,
    )

    out = capsys.readouterr().out
    assert "Using translation method: structured" in out
    # Gemini free-tier quotas are easy to hit in CI / repeated local runs.
    # If we were rate-limited, skip instead of failing this integration test.
    combined = f"{out}\n{caplog.text}"
    if "429" in combined or "Quota exceeded" in combined or "ResourceExhausted" in combined:
        pytest.skip(f"Gemini API quota exceeded:\n{combined}")

    # Verify progress cache exists and is non-empty.
    progress_path = project_dir / "es" / "progress.json"
    assert progress_path.exists()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert isinstance(progress, dict)

    # Verify CSV now has Spanish translations and they pass validation.
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(phrases)

    for row in rows:
        src = row["en"]
        dst = row["es"]

        assert dst, f"Missing translation for: {src}"
        assert is_valid_translation(dst), f"Invalid translation value: {dst!r}"

        ok, reason = translator.translation_tool.validate_translation_text(dst)
        assert ok, reason

        # Placeholder + Lingui-tag preservation checks (only apply when present).
        ok, reason = translator.translation_tool.validate_placeholders(src, dst)
        assert ok, reason

        # Ensure the translation was also written to progress.
        assert src in progress
        assert progress[src] == dst

        if "<b>" in src or "</i>" in src:
            assert "<b>" in dst and "</i>" in dst


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_structured_output_can_return_failure_contract():
    """
    Contract test: prove the model can return a failure record in structured output.

    We avoid relying on real "policy refusals" (which are variable) by using a
    sentinel string and instructions to treat it as untranslatable.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    from lib.llm import get_driver
    from lib.contracts.translation_contracts import TranslationsResponse

    driver = get_driver("gemini")

    prompt = (
        "You are a professional translator.\n"
        "Translate the following phrases from EN to ES.\n"
        "If a phrase contains the token [[REFUSE]], do not translate it. Instead:\n"
        "- Set its translation to an empty string\n"
        "- Add an entry to `failures` with:\n"
        "  - index (0-based)\n"
        "  - phrase (original)\n"
        "  - category = \"refusal\"\n"
        "  - message = \"forced refusal for testing\"\n"
        "\n"
        "Phrases (in order):\n"
        "0) Hello\n"
        "1) [[REFUSE]] do not translate\n"
    )

    try:
        raw = await driver.translate_structured_async(
            prompt, delay_seconds=1.0, max_retries=2
        )
    except Exception as e:
        msg = str(e)
        if "429" in msg or "Quota exceeded" in msg or "ResourceExhausted" in msg:
            pytest.skip(f"Gemini API quota exceeded: {msg}")
        raise
    parsed = TranslationsResponse.model_validate(raw)

    assert len(parsed.translations) == 2
    assert parsed.translations[0].strip()
    assert parsed.translations[1] == ""

    assert parsed.failures, "Expected at least one failure record"
    refusal = parsed.failures[0]
    assert refusal.index == 1
    assert "[[REFUSE]]" in refusal.phrase
    assert refusal.category == "refusal"

