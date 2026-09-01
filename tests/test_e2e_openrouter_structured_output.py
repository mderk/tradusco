import csv
import json
import os
from pathlib import Path

import pytest

from lib.TranslationProject import TranslationProject
from lib.storage.filesystem import FileSystemStorageAdapter
from lib.utils import is_valid_translation


def _openrouter_model_for_tests() -> str:
    # Prefer the same env override used by other OpenRouter integration tests.
    return (
        os.environ.get("OPENROUTER_E2E_MODEL")
        or os.environ.get("OPENROUTER_STRUCTURED_MODEL")
        or "google/gemini-2.5-flash"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_temp_project_openrouter_gemini25_structured_output(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
):
    """
    E2E-ish test (OpenRouter-first):
    - creates a temporary Tradusco project on disk
    - runs translation via OpenRouter (Gemini 2.5 model ID)
    - verifies `auto` selects structured output
    - validates persisted outputs (CSV + progress) contain usable translations
      and preserve placeholders / Lingui tags for tricky phrases.
    """
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    # Ensure we exercise the structured-output path even if model capability probing fails.
    monkeypatch.setenv("TRADUSCO_OPENROUTER_SO", "force")

    model = _openrouter_model_for_tests()

    project_dir = tmp_path / "tmp_tradusco_openrouter_project"
    project_dir.mkdir(parents=True, exist_ok=True)

    (project_dir / "config.json").write_text(
        json.dumps(
            {
                "name": project_dir.name,
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

    # Include tricky cases:
    # - curly placeholders
    # - Lingui numeric tags
    # - "markup" including broken tags (we don't validate HTML tags today, but include it)
    phrases: list[dict[str, str]] = [
        {"id": "HELLO", "en": "Hello world", "es": "", "context": "Simple greeting."},
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
            "id": "BROKEN_MARKUP",
            "en": "Broken markup (keep as-is): <b>bold</i>",
            "es": "",
            "context": "Do not 'fix' tags; preserve them as-is.",
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
        project_name=project_dir.name,
        dst_languages=["es"],
        storage=storage,
        context=None,
    )

    await translator.translate(
        model=model,
        translation_method="auto",
        batch_size=25,
        batch_max_input_tokens=65536,
        max_retries=2,
        delay_seconds=1.0,
        regenerate=False,
    )

    out = capsys.readouterr().out
    assert "Using translation method: structured" in out

    progress_path = project_dir / "es" / "progress.json"
    assert progress_path.exists()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert isinstance(progress, dict)

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

        ok, reason = translator.translation_tool.validate_placeholders(src, dst)
        assert ok, reason

        assert src in progress
        assert progress[src] == dst
