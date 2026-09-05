"""Build a tool-importable bilingual package from fused records.

This is intentionally strict. It never fills a missing Chinese caption or
video with a placeholder. Incomplete records are written to pending_review.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

SECTIONS_EN = ("Overview", "Storyline", "Speech Transcript", "Visible Text")
SECTIONS_ZH = ("概览", "故事线", "语音转录", "可见文字")
OVERVIEW_FIELDS_EN = ("Overall Visual Style:", "Overall Audio Style:",
                      "Character Profiles:", "Narrative Theme:")
OVERVIEW_FIELDS_ZH = ("整体视觉风格：", "整体音频风格：", "人物档案：", "叙事主题：")
ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{n} is not a JSON object")
                yield n, value


def safe_rel(raw: str) -> PurePosixPath:
    normalized = str(raw).replace("\\", "/")
    p = PurePosixPath(normalized)
    if not normalized or p.is_absolute() or ".." in p.parts or ":" in p.parts[0]:
        raise ValueError("unsafe video_path")
    return p


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def has_sections(text: Any, sections: tuple[str, ...]) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    return all(re.search(rf"(?im)^##[ \t]+{re.escape(s)}[ \t]*$", text) for s in sections)


def strict_caption_schema(text: Any, sections: tuple[str, ...], overview_fields: tuple[str, ...]) -> bool:
    """Validate the exact Markdown contract required by the annotation tool."""
    if not has_sections(text, sections):
        return False
    headings = tuple(re.findall(r"(?m)^##[ \t]+(.+?)[ \t]*$", text))
    if headings != sections:
        return False
    if any(f"\n\n## {section}" not in text for section in sections[1:]):
        return False
    overview_match = re.search(
        rf"(?ms)^##[ \t]+{re.escape(sections[0])}[ \t]*\n(.*?)^##[ \t]+{re.escape(sections[1])}[ \t]*$",
        text,
    )
    if not overview_match:
        return False
    labels = tuple(
        line.strip() for line in overview_match.group(1).splitlines()
        if line.strip().endswith((":", "："))
    )
    return labels == overview_fields


def valid_interval(start: Any, end: Any, duration: int) -> bool:
    try:
        return 0 <= int(start) < int(end) <= duration
    except (TypeError, ValueError):
        return False


def validate(item: dict[str, Any], translation: dict[str, Any] | None,
             manifest: dict[str, Any], allow_unresolved: bool) -> list[str]:
    issues: list[str] = []
    item_id = item.get("_id")
    if not isinstance(item_id, str) or not ID_RE.fullmatch(item_id):
        issues.append("invalid_id")
    duration = int(item.get("duration_ms") or manifest.get("duration_ms") or 0)
    if duration <= 0:
        issues.append("missing_duration")
    video_path = item.get("video_path") or manifest.get("video_path")
    try:
        rel = safe_rel(video_path)
    except (TypeError, ValueError):
        issues.append("unsafe_video_path")
        rel = None
    source = manifest.get("local_video_path")
    if not source or not Path(source).is_file():
        issues.append("video_missing")
    if rel is None:
        issues.append("video_path_missing")
    if not strict_caption_schema(item.get("caption_en"), SECTIONS_EN, OVERVIEW_FIELDS_EN):
        issues.append("english_caption_schema_invalid")
    if not translation:
        issues.append("translation_missing")
    cn = (translation or {}).get("caption_zh") or (translation or {}).get("cn")
    if not strict_caption_schema(cn, SECTIONS_ZH, OVERVIEW_FIELDS_ZH):
        issues.append("chinese_caption_schema_invalid")
    if item.get("validation", {}).get("schema_valid") is False:
        issues.append("fused_schema_invalid")
    if not allow_unresolved:
        unresolved = set(item.get("validation", {}).get("warnings", []))
        unresolved.update({"audio_style_not_assessed", "speaker_unresolved"})
        provenance = item.get("provenance", {})
        if provenance.get("audio_style_source") == "not_assessed":
            unresolved.add("audio_style_not_assessed")
        if any(s.get("speaker_link_status", "unknown") == "unknown" for s in item.get("speech") or []):
            unresolved.add("speaker_unresolved")
        if any(x in unresolved for x in ("audio_style_not_assessed", "speaker_unresolved")):
            issues.append("required_audio_or_speaker_field_unresolved")
    for i, event in enumerate(item.get("storyline") or [], 1):
        if not valid_interval(event.get("start_ms"), event.get("end_ms"), duration):
            issues.append(f"storyline_{i}_invalid_time")
    for i, speech in enumerate(item.get("speech") or [], 1):
        if not valid_interval(speech.get("start_ms"), speech.get("end_ms"), duration):
            issues.append(f"speech_{i}_invalid_time")
    return issues


def ranges(total: int, parts: int) -> list[tuple[int, int]]:
    if total < 1 or parts < 1 or total < parts:
        raise ValueError("records must be >= parts >= 1")
    base, rem = divmod(total, parts)
    result, start = [], 0
    for i in range(parts):
        end = start + base + (1 if i < rem else 0)
        result.append((start, end))
        start = end
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True,
                        help="JSONL with _id and caption_zh (or cn)")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="prepared manifest with local_video_path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--prefix", default="caption_pilot")
    parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--allow-unresolved", action="store_true",
                        help="allow explicitly unresolved audio/speaker fields")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    manifest_rows = {x["_id"]: x for x in json.loads(args.manifest.read_text(encoding="utf-8"))}
    translations = {}
    for line_no, row in read_jsonl(args.translations):
        if row.get("_id") in translations:
            raise ValueError(f"duplicate translation ID at line {line_no}")
        translations[row.get("_id")] = row

    ready: list[tuple[dict[str, Any], Path, PurePosixPath]] = []
    pending: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, item in read_jsonl(args.fused):
        item_id = item.get("_id")
        if item_id in seen:
            pending.append({"line": line_no, "_id": item_id, "reasons": ["duplicate_fused_id"]})
            continue
        seen.add(item_id)
        manifest = manifest_rows.get(item_id)
        trans = translations.get(item_id)
        if not manifest:
            pending.append({"line": line_no, "_id": item_id, "reasons": ["manifest_missing"]})
            continue
        issues = validate(item, trans, manifest, args.allow_unresolved)
        if issues:
            pending.append({"line": line_no, "_id": item_id, "reasons": issues})
            continue
        rel = safe_rel(item.get("video_path") or manifest["video_path"])
        source = Path(manifest["local_video_path"])
        output_json = {
            "_id": item_id,
            "video_path": rel.as_posix(),
            "caption_en": item["caption_en"],
            "caption_zh": trans.get("caption_zh") or trans.get("cn"),
        }
        ready.append((output_json, source, rel))

    package_summaries = []
    if ready:
        width = max(4, len(str(len(ready))))
        for part, (start, end) in enumerate(ranges(len(ready), args.parts), 1):
            folder = f"{start+1:0{width}d}-{end:0{width}d}"
            package = args.output / folder
            package.mkdir()
            json_name = f"{args.prefix}_{start+1:0{width}d}_{end:0{width}d}_final_caption_zh.jsonl"
            with (package / json_name).open("w", encoding="utf-8", newline="\n") as out:
                for output_json, source, rel in ready[start:end]:
                    out.write(json.dumps(output_json, ensure_ascii=False, separators=(",", ":")) + "\n")
                    destination = package / Path(*rel.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if args.copy_mode == "copy":
                        shutil.copy2(source, destination)
                    else:
                        destination.hardlink_to(source)
            package_summaries.append({"part": part, "folder": folder,
                                     "jsonl": json_name, "records": end-start})

    with (args.output / "delivery_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for output_json, source, rel in ready:
            out.write(json.dumps({"_id":output_json["_id"],"video_path":output_json["video_path"],
                                  "source":str(source),"sha256":digest(source),
                                  "caption_en":True,"caption_zh":True}, ensure_ascii=False) + "\n")
    with (args.output / "pending_review.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for row in pending:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"stage":"bilingual_ready_for_annotation" if ready and not pending else "partial_pending_review",
               "ready_records":len(ready),"pending_records":len(pending),"parts":package_summaries,
               "allow_unresolved":args.allow_unresolved,
               "required_sections":{"en":SECTIONS_EN,"zh":SECTIONS_ZH}}
    (args.output / "delivery_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
