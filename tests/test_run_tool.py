import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
TOOL = ROOT / "tools" / "run.js"


def run_tool(project: Path, *args: str, check: bool = True):
    return subprocess.run(
        ["node", str(TOOL), *args, "--config", str(project / "tradusco.config.json")],
        check=check,
        capture_output=True,
        text=True,
    )


def test_run_syncs_prepares_context_reports_and_locks(tmp_path):
    with (tmp_path / "source.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["en", "context", "fr"])
        writer.writeheader()
        writer.writerow({"en": "Pay", "context": "", "fr": ""})
    (tmp_path / "context-provider.js").write_text(
        """
module.exports = {
  createApi: () => ({revision: "v1", hit: () => ({refs: ["src/shop.js:1"]})}),
  rules: [{name: "pay", match: /^Pay$/, context: () => "Checkout action button."}]
};
""",
        encoding="utf-8",
    )
    (tmp_path / "artifact-keys.js").write_text('console.log(JSON.stringify(["Pay"]));\n', encoding="utf-8")
    (tmp_path / "build.js").write_text("", encoding="utf-8")
    config = {
        "traduscoRoot": str(ROOT),
        "projectDir": ".tradusco/shop",
        "sourceCsv": "source.csv",
        "baseCol": "en",
        "locales": ["fr"],
        "contextProviderFile": "context-provider.js",
        "translate": {"regenerateLangs": []},
        "artifactKeysCommand": ["node", "artifact-keys.js"],
        "deliveryCommands": [["node", "build.js"]],
    }
    (tmp_path / "tradusco.config.json").write_text(json.dumps(config), encoding="utf-8")

    result = run_tool(tmp_path, "--skip-extract", "--skip-glossary", "--skip-translate")
    assert "sync: done" in result.stdout
    assert "context: done" in result.stdout
    assert "status: rows 1; fr 0/1" in result.stdout
    assert 'selected: 1; "Pay"' in result.stdout
    with (tmp_path / ".tradusco/shop/translations.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        assert list(csv.DictReader(file))[0]["context"] == "Checkout action button."
    assert not (tmp_path / ".tradusco/shop/.run.lock").exists()

    (tmp_path / "artifact-keys.js").write_text('console.log("[]");\n', encoding="utf-8")
    incomplete = run_tool(tmp_path, "--skip-extract", "--skip-sync", "--skip-glossary", "--skip-context", "--skip-translate", "--skip-audit", check=False)
    assert incomplete.returncode == 1
    assert 'delivery artifact missing keys: "Pay"' in incomplete.stderr
    (tmp_path / "artifact-keys.js").write_text('console.log(JSON.stringify(["Pay"]));\n', encoding="utf-8")

    (tmp_path / "build.js").write_text("process.exit(7);\n", encoding="utf-8")
    failed_build = run_tool(tmp_path, "--skip-extract", "--skip-sync", "--skip-glossary", "--skip-context", "--skip-translate", "--skip-audit", check=False)
    assert failed_build.returncode == 1
    (tmp_path / "build.js").write_text("", encoding="utf-8")
    resumed_delivery = run_tool(tmp_path, "--skip-extract", "--skip-sync", "--skip-glossary", "--skip-context", "--skip-translate", "--skip-audit")
    assert "translate: skipped" in resumed_delivery.stdout
    assert "delivery: done" in resumed_delivery.stdout

    lock = tmp_path / ".tradusco/shop/.run.lock"
    lock.write_text("another writer\n", encoding="utf-8")
    blocked = run_tool(tmp_path, "--dry-run", check=True)
    assert "sync_project_from_csv.py" in blocked.stdout
    (tmp_path / "only.json").write_text(json.dumps(["Pay"]), encoding="utf-8")
    selected = run_tool(tmp_path, "--dry-run", "--only-keys-file", "only.json")
    assert 'selected: 1; "Pay"' in selected.stdout
    assert "--only-keys-file" in selected.stdout
    refused = run_tool(
        tmp_path,
        "--skip-extract",
        "--skip-sync",
        "--skip-context",
        "--skip-glossary",
        "--skip-translate",
        check=False,
    )
    assert refused.returncode == 1
    assert "project is locked" in refused.stderr
    assert lock.read_text(encoding="utf-8") == "another writer\n"
    lock.unlink()

    protected = run_tool(
        tmp_path,
        "--skip-extract",
        "--skip-sync",
        "--skip-context",
        "--skip-glossary",
        "--regenerate",
        check=False,
    )
    assert protected.returncode == 1
    assert "regeneration is not allowed for: fr" in protected.stderr
    assert not (tmp_path / ".tradusco/shop/.run.lock").exists()
