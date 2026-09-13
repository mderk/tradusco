import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
PYTHON = ROOT / ".venv/bin/python"


def test_po_partial_export_preserves_existing_and_sorts(tmp_path):
    project = tmp_path / "project"
    po_dir = tmp_path / "locale_src/fr"
    (project / "fr").mkdir(parents=True)
    po_dir.mkdir(parents=True)
    (project / "fr/progress.json").write_text(json.dumps({"Alpha": "Alpha traduit"}), encoding="utf-8")
    po = po_dir / "messages.po"
    po.write_text(
        'msgid ""\nmsgstr ""\n\nmsgid "Zulu"\nmsgstr "Déjà traduit"\n\nmsgid "Alpha"\nmsgstr ""\n',
        encoding="utf-8",
    )

    subprocess.run(
        [str(PYTHON), str(ROOT / "apply_progress_to_po.py"), "--lang", "fr", "--project-dir", str(project), "--po-dir", str(po_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(PYTHON), str(ROOT / "sort_po.py"), str(po_dir)], check=True, capture_output=True, text=True)
    content = po.read_text(encoding="utf-8")
    assert content.index('msgid "Alpha"') < content.index('msgid "Zulu"')
    assert 'msgstr "Alpha traduit"' in content
    assert 'msgstr "Déjà traduit"' in content
