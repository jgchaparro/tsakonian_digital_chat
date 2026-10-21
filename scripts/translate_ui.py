"""Fill the Tsakonian interface texts and starter phrases in assets/i18n.json with Leonida.

Only empty "tsd" values are translated, so reviewed corrections are never overwritten.
Clear a value to translate it again. Run from the repository directory:

    .venv/Scripts/python.exe -X utf8 scripts/translate_ui.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import translate  # noqa: E402

I18N = ROOT / "assets" / "i18n.json"


def main():
    load_dotenv(ROOT / ".env")
    model = os.environ.get("LEONIDA_MODEL", "leonida")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    data = json.loads(I18N.read_text(encoding="utf-8"))
    items = [(key, entry, "el") for key, entry in data["strings"].items()]
    items += [(f"starter {i + 1}", entry, "ell") for i, entry in enumerate(data["starters"])]
    for name, entry, source in items:
        if entry.get("tsd"):
            continue
        entry["tsd"] = translate(entry[source], "Greek", "Tsakonian", model=model, base_url=base_url)
        print(f"{name}: {entry[source]} → {entry['tsd']}")
        # Save after each item so an interrupted run keeps its progress.
        I18N.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
