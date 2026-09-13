import csv
import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from lib.TranslationProject import TranslationProject
from lib.storage.filesystem import FileSystemStorageAdapter
from lib.utils import Config
from tests.mock_llm_driver import MockLLMDriver


@pytest.mark.asyncio
async def test_offline_round_trip_preserves_retries_and_resumes(tmp_path):
    project_path = tmp_path / "small-shop"
    project_path.mkdir()
    config = Config(
        name=project_path.name,
        sourceFile="translations.csv",
        baseLanguage="en",
        languages=["en", "fr", "de", "ja"],
        keyColumn="en",
    )
    (project_path / "config.json").write_text(
        json.dumps(config.model_dump()), encoding="utf-8"
    )
    with (project_path / "translations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["id", "en", "fr", "de", "ja"])
        writer.writeheader()
        writer.writerows(
            [
                {
                    "id": "checkout.title",
                    "en": "Checkout",
                    "fr": "Paiement",
                    "de": "Kasse",
                    "ja": "チェックアウト",
                },
                {"id": "checkout.pay", "en": "Pay {amount}"},
                {"id": "checkout.receipt", "en": "Email receipt"},
            ]
        )

    storage = FileSystemStorageAdapter(project_path)
    storage.set_active_languages(["fr", "de", "ja"])
    project = TranslationProject(
        project_id=project_path.name,
        config=config,
        dst_languages=["fr", "de", "ja"],
        storage=storage,
        prompt="Translate {phrases_json} from {base_language} to {dst_languages}",
    )
    responses = [
        {
            "fr": {
                "Pay {amount}": "Payer {amount}",
                "Email receipt": "Reçu par e-mail",
            },
            "de": {
                "Pay {amount}": "{amount} bezahlen",
                "Email receipt": "E-Mail-Beleg",
            },
            "ja": {"Pay {amount}": "支払う", "Email receipt": "メール領収書"},
        },
        {"ja": {"Pay {amount}": "{amount}を支払う"}},
    ]
    translate = AsyncMock(side_effect=responses)

    with (
        patch("lib.TranslationProject.get_driver", return_value=MockLLMDriver()),
        patch.object(project.translation_tool, "translate_standard", translate),
    ):
        await project.translate(model="test-model", delay_seconds=0)
        assert [
            phrase for phrase, _context in translate.await_args_list[0].args[0]
        ] == [
            "Pay {amount}",
            "Email receipt",
        ]
        rows = await storage.load_translations(project_path.name)
        assert rows[0] == {
            "id": "checkout.title",
            "en": "Checkout",
            "fr": "Paiement",
            "de": "Kasse",
            "ja": "チェックアウト",
        }
        assert rows[1]["fr"] == "Payer {amount}"
        assert rows[1]["de"] == "{amount} bezahlen"
        assert rows[1]["ja"] == ""
        assert rows[2]["ja"] == "メール領収書"

        await project.translate(model="test-model", delay_seconds=0)
        assert [
            phrase for phrase, _context in translate.await_args_list[1].args[0]
        ] == ["Pay {amount}"]
        await project.translate(model="test-model", delay_seconds=0)
        assert translate.await_count == 2

    rows = await storage.load_translations(project_path.name)
    assert rows[1]["ja"] == "{amount}を支払う"
    failures = [
        json.loads(line)
        for line in (project_path / "ja" / "failures.jsonl").read_text().splitlines()
    ]
    assert failures[-1]["phrase"] == "Pay {amount}"
    assert failures[-1]["category"] == "placeholder_mismatch"


@pytest.mark.asyncio
async def test_restart_repairs_csv_after_progress_was_saved(tmp_path):
    project_path = tmp_path / "interrupted-shop"
    project_path.mkdir()
    config = Config(
        name=project_path.name,
        sourceFile="translations.csv",
        baseLanguage="en",
        languages=["en", "es"],
        keyColumn="en",
    )
    (project_path / "config.json").write_text(
        json.dumps(config.model_dump()), encoding="utf-8"
    )
    with (project_path / "translations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["id", "en", "es"])
        writer.writeheader()
        writer.writerow({"id": "checkout.pay", "en": "Pay {amount}", "es": ""})

    storage = FileSystemStorageAdapter(project_path)
    storage.set_active_language("es")
    project = TranslationProject(
        project_id=project_path.name,
        config=config,
        dst_languages=["es"],
        storage=storage,
        prompt="Translate {phrases_json} from {base_language} to {dst_languages}",
    )
    translate = AsyncMock(return_value={"es": {"Pay {amount}": "Pagar {amount}"}})

    with (
        patch("lib.TranslationProject.get_driver", return_value=MockLLMDriver()),
        patch.object(project.translation_tool, "translate_standard", translate),
        patch.object(
            storage,
            "save_translations",
            AsyncMock(side_effect=RuntimeError("interrupted before CSV save")),
        ),
        pytest.raises(RuntimeError, match="interrupted before CSV save"),
    ):
        await project.translate(model="test-model", delay_seconds=0)

    progress = json.loads((project_path / "es" / "progress.json").read_text())
    assert progress == {"Pay {amount}": "Pagar {amount}"}
    assert (await storage.load_translations(project_path.name))[0]["es"] == ""

    restarted = TranslationProject(
        project_id=project_path.name,
        config=config,
        dst_languages=["es"],
        storage=storage,
        prompt="Translate {phrases_json} from {base_language} to {dst_languages}",
    )
    after_restart = AsyncMock()
    with (
        patch("lib.TranslationProject.get_driver", return_value=MockLLMDriver()),
        patch.object(restarted.translation_tool, "translate_standard", after_restart),
    ):
        await restarted.translate(model="test-model", delay_seconds=0)

    after_restart.assert_not_awaited()
    assert (await storage.load_translations(project_path.name))[0][
        "es"
    ] == "Pagar {amount}"


@pytest.mark.asyncio
async def test_regeneration_preserves_explicit_editorial_value(tmp_path):
    project_path = tmp_path / "reviewed-shop"
    project_path.mkdir()
    config = Config(name=project_path.name, sourceFile="translations.csv", baseLanguage="en", languages=["en", "fr"], keyColumn="en")
    (project_path / "config.json").write_text(json.dumps(config.model_dump()), encoding="utf-8")
    with (project_path / "translations.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["en", "fr"])
        writer.writeheader()
        writer.writerows([{"en": "Reviewed", "fr": "Édité"}, {"en": "Machine", "fr": "Ancien"}])
    (project_path / "editorial.json").write_text(json.dumps({"fr": {"Reviewed": "Édité"}}), encoding="utf-8")
    (project_path / "fr").mkdir()
    (project_path / "fr/progress.json").write_text(json.dumps({"Reviewed": "Édité", "Machine": "Ancien"}), encoding="utf-8")

    storage = FileSystemStorageAdapter(project_path)
    storage.set_active_language("fr")
    storage.set_overwrite_active_language(True)
    project = TranslationProject(project_id=project_path.name, config=config, dst_languages=["fr"], storage=storage, prompt="Translate {phrases_json} from {base_language} to {dst_languages}")
    translate = AsyncMock(return_value={"fr": {"Machine": "Nouveau"}})
    with patch("lib.TranslationProject.get_driver", return_value=MockLLMDriver()), patch.object(project.translation_tool, "translate_standard", translate):
        await project.translate(model="test-model", regenerate=True, delay_seconds=0)
    assert [phrase for phrase, _ in translate.await_args.args[0]] == ["Machine"]
    rows = await storage.load_translations(project_path.name)
    assert rows == [{"en": "Reviewed", "fr": "Édité"}, {"en": "Machine", "fr": "Nouveau"}]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_multilanguage_round_trip(tmp_path):
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    project_path = tmp_path / "live-small-shop"
    project_path.mkdir()
    config = Config(
        name=project_path.name,
        sourceFile="translations.csv",
        baseLanguage="en",
        languages=["en", "fr", "de", "ja"],
        keyColumn="en",
    )
    (project_path / "config.json").write_text(
        json.dumps(config.model_dump()), encoding="utf-8"
    )
    with (project_path / "translations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["id", "en", "fr", "de", "ja"])
        writer.writeheader()
        writer.writerows(
            [
                {
                    "id": "checkout.title",
                    "en": "Checkout",
                    "fr": "Paiement",
                    "de": "Kasse",
                    "ja": "チェックアウト",
                },
                {"id": "checkout.pay", "en": "Pay {amount}"},
                {"id": "checkout.receipt", "en": "Email receipt"},
            ]
        )

    storage = FileSystemStorageAdapter(project_path)
    storage.set_active_languages(["fr", "de", "ja"])
    project = TranslationProject(
        project_id=project_path.name,
        config=config,
        dst_languages=["fr", "de", "ja"],
        storage=storage,
        prompt="Translate {phrases_json} from {base_language} to {dst_languages}",
    )
    await project.translate(
        model="gemini",
        translation_method="auto",
        batch_size=10,
        max_retries=1,
        delay_seconds=0,
    )

    rows = await storage.load_translations(project_path.name)
    assert rows[0]["fr"] == "Paiement"
    assert rows[0]["de"] == "Kasse"
    assert rows[0]["ja"] == "チェックアウト"
    for row in rows[1:]:
        for language in ("fr", "de", "ja"):
            assert row[language]
            ok, reason = project.translation_tool.validate_placeholders(
                row["en"], row[language]
            )
            assert ok, reason

    model_call = AsyncMock(side_effect=AssertionError("completed cells were resent"))
    with patch.object(project.translation_tool, "translate_structured", model_call):
        await project.translate(
            model="gemini",
            translation_method="auto",
            batch_size=10,
            max_retries=1,
            delay_seconds=0,
        )
    model_call.assert_not_awaited()
