"""Normalize previously delivered caption JSONL files to the tool's strict schema.

This is a local-only repair utility: it neither reads media nor calls a model.
It keeps the four required sections and their contents, removes the unsupported
Scene Summary field, and canonicalizes the Overview field labels and order.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SECTIONS_EN = ("Overview", "Storyline", "Speech Transcript", "Visible Text")
SECTIONS_ZH = ("概览", "故事线", "语音转录", "可见文字")
FIELD_MAP_EN = {
    "Overall Visual Style:": "Overall Visual Style:",
    "Overall Audio Style:": "Overall Audio Style:",
    "Character Profiles:": "Character Profiles:",
    "Narrative Theme:": "Narrative Theme:",
    "Scene Summary:": None,
}
FIELD_MAP_ZH = {
    "整体视觉风格：": "整体视觉风格：",
    "整体音频风格：": "整体音频风格：",
    "人物档案：": "人物档案：",
    "人物概况：": "人物档案：",
    "叙事主题：": "叙事主题：",
    "场景摘要：": None,
}
ORDER_EN = tuple(value for value in FIELD_MAP_EN.values() if value is not None)
ORDER_ZH = ("整体视觉风格：", "整体音频风格：", "人物档案：", "叙事主题：")


def split_sections(text: str, sections: tuple[str, ...]) -> dict[str, str]:
    matches = list(re.finditer(r"(?m)^##[ \t]+(.+?)[ \t]*$", text))
    headings = tuple(match.group(1).strip() for match in matches)
    if headings != sections:
        raise ValueError(f"section headings must be {sections}; got {headings}")
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result[headings[index]] = text[match.end():end].strip()
    return result


def normalize_overview(body: str, field_map: dict[str, str | None], order: tuple[str, ...]) -> str:
    label_re = re.compile(r"(?m)^(.+?[：:])[ \t]*$")
    matches = list(label_re.finditer(body))
    if not matches:
        raise ValueError("Overview has no field labels")
    values: dict[str, str] = {}
    for index, match in enumerate(matches):
        raw_label = match.group(1).strip()
        if raw_label not in field_map:
            raise ValueError(f"unsupported Overview field: {raw_label}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        canonical_label = field_map[raw_label]
        if canonical_label is None:
            continue
        if canonical_label in values:
            raise ValueError(f"duplicate Overview field: {canonical_label}")
        values[canonical_label] = body[match.end():end].strip()
    missing = [label for label in order if label not in values]
    if missing:
        raise ValueError(f"missing Overview fields: {missing}")
    return "\n\n".join(f"{label}\n{values[label]}" for label in order)


def normalize_caption(text: Any, sections: tuple[str, ...], field_map: dict[str, str | None], order: tuple[str, ...]) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("caption is empty or not a string")
    blocks = split_sections(text.replace("\r\n", "\n"), sections)
    blocks[sections[0]] = normalize_overview(blocks[sections[0]], field_map, order)
    return "\n\n".join(f"## {section}\n\n{blocks[section]}".rstrip() for section in sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("--output must be a new file; the input is never overwritten")
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open(encoding="utf-8-sig") as source, args.output.open("w", encoding="utf-8", newline="\n") as target:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: record is not an object")
            try:
                row["caption_en"] = normalize_caption(row.get("caption_en"), SECTIONS_EN, FIELD_MAP_EN, ORDER_EN)
                row["caption_zh"] = normalize_caption(row.get("caption_zh"), SECTIONS_ZH, FIELD_MAP_ZH, ORDER_ZH)
            except ValueError as exc:
                raise ValueError(f"line {line_no}: {exc}") from exc
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    print(f"NORMALIZED {count} records -> {args.output}")


if __name__ == "__main__":
    main()
