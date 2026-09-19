import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
TOOL = ROOT / "review_translations.py"


def run_tool(root: Path, *args: str, check: bool = True):
    return subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(TOOL), *args, "--config", str(root / "tradusco.config.json")],
        check=check,
        capture_output=True,
        text=True,
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["en", "fr"])
        writer.writeheader()
        writer.writerows(rows)


def test_guarded_review_back_sync_and_partial_export(tmp_path):
    project = tmp_path / ".tradusco/shop"
    (project / "fr").mkdir(parents=True)
    (project / "config.json").write_text(
        json.dumps({"sourceFile": "translations.csv", "baseLanguage": "en", "languages": ["en", "fr"]}),
        encoding="utf-8",
    )
    write_csv(project / "translations.csv", [{"en": "Pay {n}", "fr": "Payer {n}"}, {"en": "Blank", "fr": ""}])
    write_csv(tmp_path / "catalog.csv", [{"en": "Pay {n}", "fr": "Ancien {n}"}, {"en": "Blank", "fr": "Legacy"}, {"en": "Host only", "fr": "Garder"}])
    (project / "fr/progress.json").write_text(json.dumps({"Pay {n}": "Payer {n}"}), encoding="utf-8")
    (tmp_path / "tradusco.config.json").write_text(
        json.dumps({"projectDir": ".tradusco/shop", "sourceCsv": "catalog.csv"}), encoding="utf-8"
    )

    edits = tmp_path / "edits.json"
    edits.write_text(json.dumps([{"source": "Pay {n}", "language": "fr", "from": "Payer {n}", "to": "Régler {n}"}]), encoding="utf-8")
    assert json.loads(run_tool(tmp_path, "apply", "--edits", str(edits)).stdout)["applied"] == 1
    run_tool(tmp_path, "apply", "--edits", str(edits), "--write")
    assert json.loads((project / "editorial.json").read_text()) == {"fr": {"Pay {n}": "Régler {n}"}}

    conflict = tmp_path / "conflict.json"
    conflict.write_text(json.dumps([{"source": "Pay {n}", "language": "fr", "from": "Payer {n}", "to": "Autre {n}"}]), encoding="utf-8")
    assert run_tool(tmp_path, "apply", "--edits", str(conflict), "--write", check=False).returncode == 1

    # An idempotent rerun repairs a later file when an earlier write survived.
    (project / "fr/progress.json").write_text("{}", encoding="utf-8")
    run_tool(tmp_path, "apply", "--edits", str(edits), "--write")
    assert json.loads((project / "fr/progress.json").read_text())["Pay {n}"] == "Régler {n}"

    stale = run_tool(tmp_path, "export", "--write", "--expect", "stale", check=False)
    assert stale.returncode == 1 and "input revision changed" in stale.stderr
    preview = json.loads(run_tool(tmp_path, "export").stdout)
    run_tool(tmp_path, "export", "--write", "--expect", preview["revision"])
    with (tmp_path / "catalog.csv").open(encoding="utf-8", newline="") as file:
        rows = {row["en"]: row["fr"] for row in csv.DictReader(file)}
    assert rows == {"Pay {n}": "Régler {n}", "Blank": "Legacy", "Host only": "Garder"}

    # Explicit back-sync records the external value as editorial and rejects stale previews.
    write_csv(tmp_path / "catalog.csv", [{"en": "Pay {n}", "fr": "Payer maintenant {n}"}, {"en": "Blank", "fr": "Legacy"}, {"en": "Host only", "fr": "Garder"}])
    preview = json.loads(run_tool(tmp_path, "back-sync").stdout)
    run_tool(tmp_path, "back-sync", "--write", "--expect", preview["revision"])
    assert json.loads((project / "editorial.json").read_text())["fr"]["Pay {n}"] == "Payer maintenant {n}"


def test_export_selected_locale_leaves_other_locale_alone(tmp_path):
    project = tmp_path / ".tradusco/shop"
    project.mkdir(parents=True)
    (project / "config.json").write_text(json.dumps({"sourceFile": "translations.csv", "baseLanguage": "en", "languages": ["en", "fr", "de"]}))
    fields = ["en", "fr", "de"]
    for file, values in [(project / "translations.csv", ["Pay", "Payer", "Zahlen"]), (tmp_path / "catalog.csv", ["Pay", "Old fr", "Old de"])]:
        with file.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerow(dict(zip(fields, values)))
    (tmp_path / "tradusco.config.json").write_text(json.dumps({"projectDir": ".tradusco/shop", "sourceCsv": "catalog.csv"}))

    preview = json.loads(run_tool(tmp_path, "export", "--langs", "fr").stdout)
    assert preview["changes"] == 1
    run_tool(tmp_path, "export", "--langs", "fr", "--write", "--expect", preview["revision"])
    with (tmp_path / "catalog.csv").open(encoding="utf-8", newline="") as source:
        row = next(csv.DictReader(source))
    assert row == {"en": "Pay", "fr": "Payer", "de": "Old de"}


def test_back_sync_flags_whole_rows_tradusco_never_translated(tmp_path):
    # A reviewer changes single cells. A source that Tradusco has no translation
    # for while the host offers it in several languages is almost always an old
    # translation re-keyed onto new source text, not editorial work.
    project = tmp_path / ".tradusco/shop"
    project.mkdir(parents=True)
    (project / "config.json").write_text(json.dumps({"sourceFile": "translations.csv", "baseLanguage": "en", "languages": ["en", "fr", "de"]}))
    fields = ["en", "fr", "de"]
    with (project / "translations.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{"en": "Pay {n}", "fr": "Payer {n}", "de": "Zahlen {n}"}, {"en": "Pay now {n}", "fr": "", "de": ""}])
    with (tmp_path / "catalog.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{"en": "Pay {n}", "fr": "Régler {n}", "de": "Zahlen {n}"}, {"en": "Pay now {n}", "fr": "Payer {n}", "de": "Zahlen {n}"}])
    (tmp_path / "tradusco.config.json").write_text(json.dumps({"projectDir": ".tradusco/shop", "sourceCsv": "catalog.csv"}))

    preview = json.loads(run_tool(tmp_path, "back-sync").stdout)
    assert preview["changes"] == 3
    assert preview["untranslated_sources"] == ["Pay now {n}"]


def test_export_hands_resolved_context_back_to_the_host_table(tmp_path):
    project = tmp_path / ".tradusco/shop"
    project.mkdir(parents=True)
    (project / "config.json").write_text(json.dumps({"sourceFile": "translations.csv", "baseLanguage": "en", "languages": ["en", "fr"]}))
    fields = ["en", "context", "fr"]
    with (project / "translations.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"en": "Pay {n}", "context": "Button in the shop.", "fr": "Payer {n}"})
    with (tmp_path / "catalog.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"en": "Pay {n}", "context": "", "fr": ""})
    (tmp_path / "tradusco.config.json").write_text(json.dumps({"projectDir": ".tradusco/shop", "sourceCsv": "catalog.csv"}))

    preview = json.loads(run_tool(tmp_path, "export").stdout)
    assert {(e["language"], e["to"]) for e in preview["edits"]} == {("fr", "Payer {n}"), ("context", "Button in the shop.")}
    run_tool(tmp_path, "export", "--write", "--expect", preview["revision"])
    with (tmp_path / "catalog.csv").open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["context"] == "Button in the shop." and row["fr"] == "Payer {n}"
