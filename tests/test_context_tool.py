import asyncio
import csv
import json
import subprocess
from pathlib import Path

from lib.PromptManager import PromptManager
from lib.TranslationTool import TranslationTool, language_ref
from lib.envelope import EnvelopeBuilder


ROOT = Path(__file__).parents[1]
TOOL = ROOT / "tools" / "context.js"


def run_tool(project: Path, *args: str, check: bool = True):
    return subprocess.run(
        ["node", str(TOOL), *args, "--config", str(project / "tradusco.config.json")],
        check=check,
        capture_output=True,
        text=True,
    )


def test_context_resolution_preview_apply_and_decisions(tmp_path):
    engine = tmp_path / ".tradusco" / "shop"
    engine.mkdir(parents=True)
    (engine / "config.json").write_text(
        json.dumps(
            {
                "name": "shop",
                "sourceFile": "translations.csv",
                "baseLanguage": "en",
                "languages": ["en", "fr"],
                "keyColumn": "en",
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {"en": "Pay", "context": "", "fr": ""},
        {"en": "Receipt", "context": "", "fr": ""},
        {"en": "Travel Pack", "context": "", "fr": ""},
        {"en": "Override", "context": "old", "fr": ""},
        {"en": "Keep", "context": "Existing context.", "fr": ""},
        {"en": "Mystery", "context": "", "fr": ""},
        {"en": "Weapon", "context": "", "fr": ""},
        {"en": "Chest", "context": "", "fr": ""},
    ]
    with (engine / "translations.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["en", "context", "fr"])
        writer.writeheader()
        writer.writerows(rows)
    (engine / "contexts.json").write_text(
        json.dumps({"Override": "Manual context."}), encoding="utf-8"
    )
    (engine / "glossary.json").write_text(
        json.dumps({"terms": {"Chest": {"mode": "exact", "t": {"fr": "Coffre"}}}}),
        encoding="utf-8",
    )
    (tmp_path / "context-provider.js").write_text(
        """
module.exports = {
  createApi() {
    return {
      revision: "fixture-v1",
      table: () => [{label: "Pay"}],
      classOf: () => "shop",
      domainOf: (text) => text === "Receipt" ? "shop" : null,
      hit: (text) => ({refs: [`src/${text === "Mystery" || text === "Weapon" ? "ui" : "shop"}.js:1`]})
    };
  },
  csv: {catalog: {label: () => "Checkout action button."}},
  po: {shop: () => "Receipt shown after checkout."},
  rules: [
    {name: "pack", match: / Pack$/, context: () => "Name of a shop pack."},
    {name: "override", match: /^Override$/, context: () => "Generated context."},
    {name: "keep", match: /^Keep$/, context: () => "Replacement that must not be written."}
  ]
};
""",
        encoding="utf-8",
    )
    (tmp_path / "tradusco.config.json").write_text(
        json.dumps(
            {
                "projectDir": ".tradusco/shop",
                "contextProviderFile": "context-provider.js",
            }
        ),
        encoding="utf-8",
    )

    preview = json.loads(run_tool(tmp_path, "apply").stdout)
    assert preview["changes"] == 4
    assert len(preview["conflicts"]) == 1
    assert len(preview["masked_by_manual"]) == 1
    assert preview["retained"] == [
        {
            "text": "Keep",
            "current": "Existing context.",
            "proposed": "Replacement that must not be written.",
            "via": "rule:keep",
        }
    ]
    assert {item["via"] for item in preview["examples"]} == {
        "column",
        "domain:shop",
        "rule:pack",
        "manual",
    }
    rejected = run_tool(tmp_path, "apply", "--write", "--expect", "wrong", check=False)
    assert rejected.returncode == 1
    assert "input revision changed" in rejected.stderr
    run_tool(tmp_path, "apply", "--write", "--expect", preview["revision"])

    with (engine / "translations.csv").open(encoding="utf-8", newline="") as file:
        applied = {row["en"]: row["context"] for row in csv.DictReader(file)}
    assert applied == {
        "Pay": "Checkout action button.",
        "Receipt": "Receipt shown after checkout.",
        "Travel Pack": "Name of a shop pack.",
        "Override": "Manual context.",
        "Keep": "Existing context.",
        "Mystery": "",
        "Weapon": "",
        "Chest": "",
    }

    pending = json.loads(run_tool(tmp_path, "next").stdout)
    assert pending["group"] == "src/ui.js"
    assert [item["text"] for item in pending["strings"]] == ["Mystery", "Weapon"]
    context_answer = (
        "submit",
        "--group",
        "src/ui.js",
        "--source",
        "Mystery",
        "--context",
        "Label for an unknown reward.",
    )
    run_tool(tmp_path, *context_answer)
    run_tool(tmp_path, *context_answer)
    run_tool(
        tmp_path,
        "submit",
        "--group",
        "src/ui.js",
        "--source",
        "Weapon",
        "--needs-glossary",
        "Name of an equipment category.",
    )
    assert json.loads((engine / "contexts.json").read_text())[
        "Mystery"
    ] == ("Label for an unknown reward.")
    assert json.loads((engine / "terms_queue.json").read_text()) == {
        "Weapon": "Name of an equipment category."
    }
    assert json.loads(run_tool(tmp_path, "next").stdout) == {"done": True}

    second = json.loads(run_tool(tmp_path, "apply").stdout)
    assert second["changes"] == 1
    run_tool(tmp_path, "apply", "--write", "--expect", second["revision"])
    with (engine / "translations.csv").open(encoding="utf-8", newline="") as file:
        final_rows = list(csv.DictReader(file))
    final = {row["en"]: row["context"] for row in final_rows}
    assert final["Mystery"] == "Label for an unknown reward."
    assert final["Weapon"] == ""

    index = next(i for i, row in enumerate(final_rows) if row["en"] == "Mystery")
    batch = EnvelopeBuilder({}, final_rows, "en", ["fr"], []).build(
        [("Mystery", final["Mystery"])], {"Mystery": index}
    )
    prompt = asyncio.run(
        TranslationTool(PromptManager(None, "shop")).create_prompt(  # type: ignore[arg-type]
            [("Mystery", final["Mystery"])],
            "en",
            [language_ref("fr")],
            "Translate {phrases_json} from {base_language} to {dst_languages}",
            batch_input=batch,
        )
    )
    assert "Label for an unknown reward." in prompt
