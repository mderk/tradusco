from pathlib import Path


_PROMPTS_DIR = Path(__file__).resolve().parent

translation = (_PROMPTS_DIR / "translation.txt").read_text(encoding="utf-8")

output_format = (_PROMPTS_DIR / "output_format.txt").read_text(encoding="utf-8")


__all__ = [
    "translation",
    "output_format",
]
