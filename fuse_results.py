"""Deterministically fuse ASR and Qwen visual results.

The model supplies semantic fields only. This program owns all timestamps,
IDs, ordering, and the final Markdown rendering. It never treats a missing
audio-capable result as evidence that audio is absent.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_ms(value: int) -> str:
    value = max(0, int(value))
    minutes, rem = divmod(value, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def clean_text(value: Any, fallback: str = "Unknown.") -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def normalize_visual(raw: Any, shots: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    """Accept the declared schema plus common Qwen string fallbacks.

    The model is asked for objects/lists, but a production pipeline must not
    crash when a provider returns a concise string. String fallbacks are kept
    as data and marked for review rather than silently discarded.
    """
    if not isinstance(raw, dict):
        warnings.append("visual_result_not_object")
        raw = {}

    overview_raw = raw.get("overview")
    if isinstance(overview_raw, dict):
        overview = dict(overview_raw)
    elif isinstance(overview_raw, str) and overview_raw.strip():
        overview = {"scene_summary_en": overview_raw.strip()}
        warnings.append("overview_string_normalized")
    else:
        overview = {}

    characters: list[dict[str, Any]] = []
    raw_characters = raw.get("characters")
    if isinstance(raw_characters, list):
        for i, person in enumerate(raw_characters, 1):
            if isinstance(person, dict):
                characters.append({
                    "person_id": person.get("person_id") or f"p{i:02d}",
                    "description_en": person.get("description_en") or person.get("description") or "Unknown.",
                    "evidence_frame_ids": person.get("evidence_frame_ids") or [],
                })
            elif isinstance(person, str) and person.strip():
                characters.append({"person_id": f"p{i:02d}", "description_en": person.strip(),
                                   "evidence_frame_ids": []})
                warnings.append(f"character_{i}_string_normalized")
    elif isinstance(raw_characters, str) and raw_characters.strip():
        characters.append({"person_id": "p01", "description_en": raw_characters.strip(),
                           "evidence_frame_ids": []})
        warnings.append("characters_string_normalized")

    storyline: list[dict[str, Any]] = []
    raw_storyline = raw.get("storyline")
    if isinstance(raw_storyline, list):
        for i, event in enumerate(raw_storyline, 1):
            if isinstance(event, dict):
                shot_id = event.get("shot_id")
                if not shot_id and shots:
                    shot_id = shots[min(i - 1, len(shots) - 1)].get("shot_id")
                    warnings.append(f"storyline_{i}_shot_id_inferred")
                storyline.append({
                    "event_id": event.get("event_id") or f"e{i:03d}",
                    "shot_id": shot_id,
                    "text_en": event.get("text_en") or event.get("text") or "Unknown.",
                    "evidence_frame_ids": event.get("evidence_frame_ids") or [],
                })
            elif isinstance(event, str) and event.strip() and shots:
                storyline.append({"event_id": f"e{i:03d}", "shot_id": shots[min(i - 1, len(shots) - 1)].get("shot_id"),
                                  "text_en": event.strip(), "evidence_frame_ids": []})
                warnings.append(f"storyline_{i}_string_normalized")
    elif isinstance(raw_storyline, str) and raw_storyline.strip() and shots:
        storyline.append({"event_id": "e001", "shot_id": shots[0].get("shot_id"),
                          "text_en": raw_storyline.strip(), "evidence_frame_ids": []})
        warnings.append("storyline_string_assigned_to_first_shot")

    return {"overview": overview, "characters": characters, "storyline": storyline,
            "speech_context": raw.get("speech_context") or []}


def render_markdown(item: dict[str, Any], visual: dict[str, Any], speech: list[dict[str, Any]]) -> str:
    overview = visual.get("overview") or {}
    characters = visual.get("characters") or []
    storyline = visual.get("storyline") or []
    shot_by_id = {s["shot_id"]: s for s in item.get("shots", [])}
    lines = [
        "## Overview", "", "Overall Visual Style:",
        clean_text(overview.get("visual_style_en")), "",
        "Scene Summary:", clean_text(overview.get("scene_summary_en"), "Not provided."), "",
        "Overall Audio Style:",
        "Audio was not assessed by the visual model.", "",
        "Narrative Theme:", clean_text(overview.get("narrative_theme_en")), "",
        "Character Profiles:",
    ]
    if characters:
        for person in characters:
            person_id = clean_text(person.get("person_id"), "unknown_person")
            desc = clean_text(person.get("description_en"))
            lines.append(f"- {person_id}: {desc}")
    else:
        lines.append("- No character profile was generated.")

    lines.extend(["", "## Storyline", ""])
    if storyline:
        for event in storyline:
            shot = shot_by_id.get(event.get("shot_id"))
            if not shot:
                continue
            lines.extend([
                f"{fmt_ms(shot['start_ms'])} - {fmt_ms(shot['end_ms'])}",
                clean_text(event.get("text_en")), "",
            ])
    else:
        lines.append("No storyline description is available.")

    lines.extend(["## Speech Transcript", ""])
    if speech:
        for segment in speech:
            lines.extend([
                f"{fmt_ms(segment['start_ms'])} - {fmt_ms(segment['end_ms'])}",
                "Speaker: unknown (not linked to a visible person)",
                "State: not assessed by an audio-capable model",
                f"Content: {json.dumps(segment.get('text', ''), ensure_ascii=False)}", "",
            ])
    else:
        lines.append("No ASR speech segment is available.")

    lines.extend(["## Visible Text", "", "Visible text was not assessed in this run."])
    return "\n".join(lines)


def fuse(item: dict[str, Any], visual_file: Path, asr_file: Path) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    visual_doc = load_json(visual_file)
    asr_doc = load_json(asr_file)
    if visual_doc is None:
        warnings.append("visual_missing")
    if asr_doc is None:
        warnings.append("asr_missing")

    visual = normalize_visual((visual_doc or {}).get("result"), item.get("shots", []), warnings)
    raw_speech = (asr_doc or {}).get("segments") or []
    duration = int(item["duration_ms"])
    speech: list[dict[str, Any]] = []

    for i, segment in enumerate(raw_speech, 1):
        start = int(round(float(segment.get("start_ms", 0))))
        end = int(round(float(segment.get("end_ms", 0))))
        if not (0 <= start < end <= duration):
            warnings.append(f"speech_{i}_time_out_of_range")
            continue
        speech.append({
            "speech_id": segment.get("speech_id") or f"u{i:03d}",
            "start_ms": start,
            "end_ms": end,
            "text": clean_text(segment.get("text"), ""),
            "timing_source": (asr_doc or {}).get("timing_source", "unknown"),
            "speaker_link_status": "unknown",
            "speaker_person_id": None,
        })

    shots = item.get("shots", [])
    shot_by_id = {s["shot_id"]: s for s in shots}
    storyline: list[dict[str, Any]] = []
    for i, event in enumerate(visual.get("storyline") or [], 1):
        shot_id = event.get("shot_id")
        shot = shot_by_id.get(shot_id)
        if not shot:
            warnings.append(f"storyline_{i}_unknown_shot_id")
            continue
        evidence = event.get("evidence_frame_ids") or []
        storyline.append({
            "event_id": event.get("event_id") or f"e{i:03d}",
            "shot_id": shot_id,
            # These times come from manifest, never from the model response.
            "start_ms": int(shot["start_ms"]),
            "end_ms": int(shot["end_ms"]),
            "text_en": clean_text(event.get("text_en")),
            "evidence_frame_ids": evidence,
        })

    result = {
        "_id": item["_id"],
        "video_path": item["video_path"],
        "duration_ms": duration,
        "overview": {
            "visual_style_en": clean_text((visual.get("overview") or {}).get("visual_style_en")),
            "scene_summary_en": clean_text((visual.get("overview") or {}).get("scene_summary_en"), "Not provided."),
            "audio_style_en": None,
            "narrative_theme_en": clean_text((visual.get("overview") or {}).get("narrative_theme_en")),
        },
        "characters": visual.get("characters") or [],
        "shots": shots,
        "storyline": storyline,
        "speech": speech,
        "visible_text": {"status": "not_assessed", "items": []},
        "caption_en": render_markdown(item, visual, speech),
        "caption_zh": None,
        "translation_status": "pending",
        "validation": {
            "schema_valid": not any(x.endswith("_out_of_range") or x.endswith("_unknown_shot_id") for x in warnings),
            "needs_human_review": True,
            "warnings": warnings,
        },
        "provenance": {
            "visual_file": str(visual_file),
            "asr_file": str(asr_file),
            "timestamp_owner": "manifest_and_asr_program",
            "audio_style_source": "not_assessed",
        },
    }
    return result, warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--asr-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "fused_preannotations.jsonl"
    errors = args.out / "fuse_warnings.jsonl"
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as out, errors.open("w", encoding="utf-8", newline="\n") as err:
        for item in manifest:
            result, warnings = fuse(
                item,
                args.visual_root / item["_id"] / "visual.json",
                args.asr_root / item["_id"] / "asr.json",
            )
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            if warnings:
                err.write(json.dumps({"_id": item["_id"], "warnings": warnings}, ensure_ascii=False) + "\n")
            count += 1
    print(f"FUSED {count} records -> {output}")
    print(f"Warnings -> {errors}")


if __name__ == "__main__":
    main()
