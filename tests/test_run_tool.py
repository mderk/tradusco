import csv
import json
import os
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
        "translate": {"protectLangs": ["fr"]},
        "artifactKeysCommand": ["node", "artifact-keys.js"],
        "deliveryCommands": [["node", "build.js"]],
    }
    (tmp_path / "tradusco.config.json").write_text(json.dumps(config), encoding="utf-8")

    default_config = {
        **config,
        "projectDir": "shop",
        "sourceCsv": "../source.csv",
        "contextProviderFile": "../context-provider.js",
        "artifactKeysCommand": ["node", "../artifact-keys.js"],
        "deliveryCommands": [["node", "../build.js"]],
    }
    (tmp_path / ".tradusco").mkdir()
    (tmp_path / ".tradusco/config.json").write_text(
        json.dumps(default_config), encoding="utf-8"
    )
    default_run = subprocess.run(
        ["node", str(TOOL), "--dry-run"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "status: project source not synced" in default_run.stdout

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
    assert "refusing to regenerate protected locales: fr" in protected.stderr
    assert not (tmp_path / ".tradusco/shop/.run.lock").exists()

    config["translate"] = {"regenerateLangs": ["fr"]}
    (tmp_path / "tradusco.config.json").write_text(json.dumps(config), encoding="utf-8")
    legacy = run_tool(tmp_path, "--dry-run", check=False)
    assert legacy.returncode == 1
    assert "regenerateLangs was replaced by translate.protectLangs" in legacy.stderr


def test_explicit_env_file_overrides_inherited_credentials(tmp_path):
    (tmp_path / "auth.env").write_text("OPENROUTER_API_KEY=file-key\n")
    (tmp_path / "check-env.js").write_text('if (process.env.OPENROUTER_API_KEY !== "file-key") process.exit(1);\n')
    (tmp_path / "tradusco.config.json").write_text(json.dumps({
        "projectDir": "shop",
        "envFile": "auth.env",
        "extractCommands": [["node", "check-env.js"]],
    }))
    inherited = {**os.environ, "OPENROUTER_API_KEY": "ambient-key"}
    result = subprocess.run(
        ["node", str(TOOL), "--config", str(tmp_path / "tradusco.config.json"),
         "--skip-sync", "--skip-glossary", "--skip-context", "--skip-translate",
         "--skip-audit", "--skip-delivery"],
        env=inherited, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "extract: done" in result.stdout


def test_target_reference_language_is_excluded_for_that_run(tmp_path):
    (tmp_path / "tradusco.config.json").write_text(json.dumps({
        "traduscoRoot": str(ROOT),
        "projectDir": "shop",
        "locales": ["ru", "ja"],
        "translate": {"referenceLangs": ["ru", "ja"]},
    }), encoding="utf-8")

    ru = run_tool(tmp_path, "--dry-run", "--langs", "ru")
    assert "-l ru" in ru.stdout
    assert "--reference-langs ja" in ru.stdout

    both = run_tool(tmp_path, "--dry-run", "--langs", "ru,ja")
    assert "-l ru,ja" in both.stdout
    assert "--reference-langs" not in both.stdout
