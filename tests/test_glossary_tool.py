import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
TOOL = ROOT / "tools" / "glossary.js"


def run_tool(project: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", str(TOOL), *args, "--config", str(project / "tradusco.config.json")],
        check=True,
        capture_output=True,
        text=True,
    )


def test_glossary_prepare_decisions_and_lint(tmp_path):
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
    with (engine / "translations.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["en", "fr"])
        writer.writeheader()
        writer.writerows(
            [
                {"en": "enter Oblivion now", "fr": "entrer dans l'oubli"},
                {"en": "return to Oblivion", "fr": "retourner dans l'oubli"},
                {"en": "leave Oblivion", "fr": "quitter l'oubli"},
            ]
        )
    glossary_file = engine / "glossary.json"
    glossary_file.write_text(
        json.dumps({"terms": {}, "manual": {"Hero": {"mode": "keep"}}}),
        encoding="utf-8",
    )
    provider = tmp_path / "provider.js"
    provider.write_text(
        """
const fs = require("node:fs");
const output = process.argv[process.argv.indexOf("--output") + 1];
fs.writeFileSync(output, JSON.stringify({Chest: {group: "shop", mode: "stem", t: {fr: "Coffre"}}}));
""",
        encoding="utf-8",
    )
    (tmp_path / "tradusco.config.json").write_text(
        json.dumps(
            {
                "projectDir": ".tradusco/shop",
                "glossarySourceCommand": ["node", "provider.js"],
                "locales": ["fr"],
            }
        ),
        encoding="utf-8",
    )

    preview = run_tool(tmp_path, "prepare")
    assert "preview only" in preview.stdout
    assert json.loads(glossary_file.read_text())["terms"] == {}
    run_tool(tmp_path, "prepare", "--write")
    prepared = json.loads(glossary_file.read_text())
    assert prepared["terms"]["Chest"]["t"] == {"fr": "Coffre"}
    assert prepared["manual"] == {"Hero": {"mode": "keep"}}
    report = run_tool(tmp_path, "report")
    assert "coverage: unmatched 2; whole-phrase only 0" in report.stdout

    (engine / "terms_queue.json").write_text(
        json.dumps({"Weapon": "A category label."}), encoding="utf-8"
    )
    queued = json.loads(run_tool(tmp_path, "next").stdout)
    assert queued["term"] == "Weapon"
    reject = tmp_path / "reject.json"
    reject.write_text(
        json.dumps({"term": "Weapon", "not_a_term": "Generic word here."}),
        encoding="utf-8",
    )
    run_tool(tmp_path, "submit", "--json", str(reject))
    assert json.loads((engine / "not_terms.json").read_text()) == {
        "Weapon": "Generic word here."
    }
    assert json.loads((engine / "contexts.json").read_text()) == {
        "Weapon": "A category label."
    }

    candidate = json.loads(run_tool(tmp_path, "next", "--min", "3").stdout)
    assert candidate["term"] == "Oblivion"
    accept = tmp_path / "accept.json"
    accept.write_text(
        json.dumps(
            {
                "term": "Oblivion",
                "entry": {
                    "mode": "stem",
                    "note": "Name of a place.",
                    "t": {"fr": "Oubli"},
                },
            }
        ),
        encoding="utf-8",
    )
    run_tool(tmp_path, "submit", "--json", str(accept))
    run_tool(tmp_path, "submit", "--json", str(accept))
    assert json.loads(glossary_file.read_text())["manual"]["Oblivion"]["t"] == {
        "fr": "Oubli"
    }
    assert json.loads(run_tool(tmp_path, "next").stdout) == {"done": True}

    lint = subprocess.run(
        ["node", str(TOOL), "lint", "--config", str(tmp_path / "tradusco.config.json")],
        capture_output=True,
        text=True,
    )
    assert lint.returncode == 0
    assert "glossary findings: 0" in lint.stdout
    assert not (tmp_path / "translation_glossary.json").exists()
