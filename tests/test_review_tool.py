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
