"""
Characterization snapshots of the prompt that actually reaches the model.

The prompt is assembled from four pieces — the template, the global context,
the per-phrase context and the output-format block — and all four are about to
change. Prompt drift does not fail anything: the run succeeds and the text is
merely translated worse, which only surfaces weeks later during proofreading.
These tests pin the assembled string so every change to it has to be looked at.

Refresh after an intended change, then read the diff before committing it:

    UPDATE_PROMPT_SNAPSHOTS=1 uv run pytest tests/test_prompt_snapshot.py
"""

import difflib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.PromptManager import PromptManager
from lib.envelope import BatchEnvelope, Example, PhraseEnvelope
from lib.storage.base import StorageAdapter
from lib.TranslationTool import language_ref, TranslationTool
from lib.utils import Config
from tests.mock_llm_driver import MockLLMDriver

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
UPDATE = os.environ.get("UPDATE_PROMPT_SNAPSHOTS") == "1"


class _DefaultPromptStorage(StorageAdapter):
    """
    A project that ships no prompts of its own, so PromptManager falls back to
    the defaults in ``prompts/``. That is the configuration these snapshots are
    about: the shipped prompt, not a project override.
    """

    async def load_config(self, project_id: str) -> Config:
        return MagicMock()

    async def load_progress(self, project_id: str, language: str) -> Dict[str, str]:
        return {}

    async def save_progress(
        self,
        project_id: str,
        language: str,
        progress: Dict[str, str],
        overwrite_keys: Optional[set] = None,
    ) -> None:
        pass

    async def load_translations(self, project_id: str) -> List[Dict[str, str]]:
        return []

    async def save_translations(
        self, project_id: str, translations: List[Dict[str, str]]
    ) -> None:
        pass

    async def load_context(self, project_id: str, language: str) -> List[str]:
        return []

    async def load_prompt(self, project_id: str, prompt_type: str) -> str:
        return ""

    async def append_failure(
        self,
        project_id: str,
        language: str,
        record: dict[str, str | None],
    ) -> None:
        pass


# One phrase per property the prompt is supposed to protect, so a template edit
# that drops a rule shows up as a changed snapshot rather than as silent damage.
PHRASES = [
    # Placeholder that must survive verbatim.
    ("You gained {count} gems.", "Reward popup after a battle."),
    # Lingui tag pair.
    ("Read the <0>rules</0> first.", None),
    # Multi-line source: the template forbids breaking it into pieces.
    ("Line one.\nLine two.", "Two lines of a tooltip."),
    # Short label whose meaning depends entirely on the context column.
    ("Chest", "A container the player opens, not a body part."),
]

GLOBAL_CONTEXT = (
    "A fantasy RPG with adult content; the tone is relaxed.\n"
    "Every playable hero is a woman; the player character is a man.\n\n"
    "[ru]\nUse established Russian fantasy terminology.\n\n"
    "[uk]\nUse Ukrainian, never interpret uk as the United Kingdom."
)

BATCH_INPUT = BatchEnvelope(
    glossary=[
        {
            "term": "gems",
            "mode": "stem",
            "t": {"ru": "самоцветы", "uk": "самоцвіти"},
        }
    ],
    phrases=[
        PhraseEnvelope(
            phrase=PHRASES[0][0],
            context=PHRASES[0][1],
            reference={"ja": "ジェムを{count}個獲得した。"},
            examples=[
                Example(
                    phrase="You gained {count} coins.",
                    t={"ru": "Вы получили {count} монет."},
                )
            ],
        ),
        *[
            PhraseEnvelope(phrase=phrase, context=context)
            for phrase, context in PHRASES[1:]
        ],
    ],
)


@pytest.fixture
def translation_tool() -> TranslationTool:
    return TranslationTool(PromptManager(_DefaultPromptStorage(), "snapshot_project"))


def assert_matches_snapshot(name: str, actual: str) -> None:
    """Compare against the stored snapshot, reporting a readable diff."""
    path = SNAPSHOT_DIR / name
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        return

    assert path.exists(), (
        f"Missing snapshot {path.name}. "
        "Create it with UPDATE_PROMPT_SNAPSHOTS=1 and review the result."
    )

    expected = path.read_text(encoding="utf-8")
    if actual == expected:
        return

    diff = "\n".join(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile=f"{name} (stored)",
            tofile=f"{name} (produced)",
            lineterm="",
        )
    )
    pytest.fail(f"Assembled prompt changed:\n{diff}", pytrace=False)


async def _assemble(tool: TranslationTool, method_name: str) -> str:
    """Run the real assembly path in TranslationTool.setup."""
    prompt = await tool.prompt_manager.load_prompt("translation")
    with patch("lib.TranslationTool.get_driver", return_value=MockLLMDriver()):
        _driver, batch_prompt = await tool.setup(
            phrases=PHRASES,
            model="mock-model",
            base_language="en",
            dst_languages=[language_ref("ru"), language_ref("uk")],
            prompt=prompt,
            context=GLOBAL_CONTEXT,
            method_name=method_name,
            batch_input=BATCH_INPUT,
        )
    return batch_prompt


@pytest.mark.asyncio
async def test_standard_prompt_snapshot(translation_tool):
    """The standard method appends the output-format block; structured does not."""
    assert_matches_snapshot(
        "prompt_standard.txt", await _assemble(translation_tool, "standard")
    )


@pytest.mark.asyncio
async def test_structured_prompt_snapshot(translation_tool):
    assert_matches_snapshot(
        "prompt_structured.txt", await _assemble(translation_tool, "structured")
    )
