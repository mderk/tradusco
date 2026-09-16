import csv
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples/reference-host"


def test_reference_host_prepares_and_builds(tmp_path):
    host = tmp_path / "reference-host"
    shutil.copytree(EXAMPLE, host, ignore=shutil.ignore_patterns("app"))
    config_file = host / ".tradusco/config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config["traduscoRoot"] = str(ROOT)
    config_file.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    prepared = subprocess.run(
        [
            "node",
            str(ROOT / "tools/run.js"),
            "--config",
            str(config_file),
            "--skip-translate",
            "--skip-audit",
            "--skip-delivery",
        ],
        cwd=host,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "glossary: done" in prepared.stdout
    assert "context: done" in prepared.stdout
    glossary = json.loads((host / ".tradusco/app/glossary.json").read_text())
    assert glossary["terms"]["Travel Pack"]["t"]["fr"] == "Pack Voyage"
    with (host / ".tradusco/app/translations.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == ["en", "context", "fr", "de"]
    assert [row["context"] for row in rows] == [
        "Title of the checkout screen.",
        "Button that confirms a payment for the displayed amount.",
        "Name of a shop bundle for a short trip.",
    ]

    review = [str(ROOT / ".venv/bin/python"), str(ROOT / "review_translations.py")]
    preview = subprocess.run(
        [*review, "back-sync", "--config", str(config_file)],
        cwd=host,
        text=True,
        capture_output=True,
        check=True,
    )
    revision = json.loads(preview.stdout)["revision"]
    subprocess.run(
        [*review, "back-sync", "--config", str(config_file), "--write", "--expect", revision],
        cwd=host,
        check=True,
    )
    assert json.loads((host / ".tradusco/app/fr/progress.json").read_text()) == {
        "Checkout": "Paiement",
        "Pay {amount}": "Payer {amount}",
        "Travel Pack": "Pack Voyage",
    }

    internal_csv = host / ".tradusco/app/translations.csv"
    with internal_csv.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["de"] = {"Checkout": "Kasse", "Pay {amount}": "{amount} bezahlen", "Travel Pack": "Reisepaket"}[row["en"]]
    with internal_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    delivered = subprocess.run(
        [
            "node",
            str(ROOT / "tools/run.js"),
            "--config",
            str(config_file),
            "--langs",
            "de",
            "--skip-extract",
            "--skip-sync",
            "--skip-glossary",
            "--skip-context",
            "--skip-translate",
            "--skip-audit",
        ],
        cwd=host,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "delivery: done" in delivered.stdout
    with (host / "catalog.csv").open(encoding="utf-8", newline="") as file:
        catalog = list(csv.DictReader(file))
    assert [row["de"] for row in catalog] == ["Kasse", "{amount} bezahlen", "Reisepaket"]
    assert json.loads((host / "dist/de.json").read_text()) == {
        "Checkout": "Kasse",
        "Pay {amount}": "{amount} bezahlen",
        "Travel Pack": "Reisepaket",
    }
