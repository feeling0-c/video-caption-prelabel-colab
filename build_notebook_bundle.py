"""Embed the current local source snapshot in the self-contained Colab notebook."""
from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks" / "Video_Caption_Prelabel_All_in_Colab.ipynb"


def source_files() -> list[Path]:
    files = [ROOT / ".gitignore", ROOT / "README.md", ROOT / "requirements-colab.txt"]
    files.extend(sorted(ROOT.glob("*.py")))
    files.extend(sorted((ROOT / "prompts").glob("*.txt")))
    return [path for path in files if path.is_file()]


def build_bundle() -> str:
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_files():
            archive.write(path, path.relative_to(ROOT).as_posix())
    return base64.b64encode(raw.getvalue()).decode("ascii")


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    bundle_line = "SOURCE_BUNDLE_B64 = '" + build_bundle() + "'\n"
    replaced = False
    for cell in notebook.get("cells", []):
        source = cell.get("source")
        if not isinstance(source, list):
            continue
        for index, line in enumerate(source):
            if isinstance(line, str) and line.startswith("SOURCE_BUNDLE_B64 = "):
                source[index] = bundle_line
                replaced = True
    if not replaced:
        raise RuntimeError("SOURCE_BUNDLE_B64 cell was not found")
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Embedded {len(source_files())} source files in {NOTEBOOK}")


if __name__ == "__main__":
    main()
