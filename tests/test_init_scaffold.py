import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "skills/tradusco-init/scripts/scaffold.py"
RUNNER = Path(__file__).parents[1] / "tools/run.js"


def test_scaffold_creates_valid_config_without_overwriting(tmp_path):
    source = tmp_path / "catalog.csv"
    with source.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["en", "context", "fr"])
        writer.writeheader()
        writer.writerow({"en": "Open", "context": "Button label", "fr": "Ouvrir"})
    command = [sys.executable, str(SCRIPT), "--host", str(tmp_path), "--csv", "catalog.csv", "--base", "en", "--locales", "fr"]
    first = subprocess.run(command, text=True, capture_output=True, check=True)
    config_file = tmp_path / ".tradusco/config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    assert first.stdout.strip() == str(config_file)
    assert (tmp_path / ".tradusco/app").is_dir()
    assert config["sourceCsv"] == "../catalog.csv"
    assert config["locales"] == ["fr"]
    assert config["translate"] == {"protectLangs": []}
    assert Path(config["traduscoRoot"]).is_dir()

    prepared = subprocess.run(
        ["node", str(RUNNER), "--config", str(config_file), "--skip-translate", "--skip-delivery"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "sync: done" in prepared.stdout
    assert (tmp_path / ".tradusco/app/translations.csv").is_file()
    assert source.read_text(encoding="utf-8").endswith("Open,Button label,Ouvrir\n")

    second = subprocess.run(command, text=True, capture_output=True)
    assert second.returncode != 0
    assert "will not be overwritten" in second.stderr
    assert json.loads(config_file.read_text(encoding="utf-8")) == config


def test_scaffold_rejects_duplicate_source_keys(tmp_path):
    source = tmp_path / "catalog.csv"
    source.write_text("en,context,fr\nOpen,First,\nOpen,Second,\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--host", str(tmp_path), "--csv", "catalog.csv", "--base", "en", "--locales", "fr"],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "unique base-language text keys" in result.stderr
    assert not (tmp_path / ".tradusco/config.json").exists()
