"""Translate fused semantic fields and render deterministic Chinese captions.

The model returns field_id -> Chinese text. IDs, time ranges, and original
speech are rendered by code and never translated by the model.
"""
from __future__ import annotations
import argparse, json, os, urllib.error, urllib.request
from pathlib import Path
from typing import Any

MODEL = "qwen3.7-flash-2026-07-15"
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def fmt_ms(value: int) -> str:
    m, r = divmod(max(0, int(value)), 60_000)
    s, ms = divmod(r, 1_000)
    return f"{m:02d}:{s:02d}.{ms:03d}"


def fields_for(item: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    overview = item.get("overview") or {}
    for key in ("visual_style_en", "scene_summary_en", "narrative_theme_en"):
        if isinstance(overview.get(key), str) and overview[key].strip():
            fields["overview." + key.removesuffix("_en")] = overview[key]
    for person in item.get("characters") or []:
        pid = person.get("person_id")
        if pid and person.get("description_en"):
            fields[f"character.{pid}.description"] = person["description_en"]
    for event in item.get("storyline") or []:
        eid = event.get("event_id")
        if eid and event.get("text_en"):
            fields[f"storyline.{eid}.text"] = event["text_en"]
    return fields


def call_translate(fields: dict[str, str], timeout: int = 120) -> tuple[dict[str, str], dict[str, Any]]:
    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not key:
        raise RuntimeError("Set DASHSCOPE_API_KEY in the runtime secret store")
    prompt = {
        "instruction": "Translate each English value into concise Simplified Chinese. Return JSON only with exactly the same field IDs. Do not add facts.",
        "fields": fields,
    }
    body = json.dumps({"model": MODEL, "temperature": 0,
        "messages":[{"role":"user","content":json.dumps(prompt, ensure_ascii=False)}],
        "response_format":{"type":"json_object"}}).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
        headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status, raw, headers = response.status, response.read(), response.headers
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise RuntimeError("403: free-only quota stopped the request") from exc
        raise RuntimeError(f"API HTTP {exc.code}") from exc
    data = json.loads(raw)
    text = data["choices"][0]["message"]["content"]
    if isinstance(text, list):
        text = "".join(x.get("text", "") for x in text if isinstance(x, dict))
    translated = json.loads(text)
    if set(translated) != set(fields):
        raise ValueError("translation field IDs do not exactly match input field IDs")
    return translated, {"status":status,"usage":data.get("usage",{}),"request_id":headers.get("x-request-id")}


def render_zh(item: dict[str, Any], translated: dict[str, str]) -> str:
    overview = item.get("overview") or {}
    lines = ["## 概览", "", "整体视觉风格：", translated.get("overview.visual_style", "待翻译"), "",
             "场景摘要：", translated.get("overview.scene_summary", "待翻译"), "",
             "整体音频风格：", "本次未使用音频理解模型，待人工试听。", "",
             "叙事主题：", translated.get("overview.narrative_theme", "待翻译"), "", "人物概况："]
    for person in item.get("characters") or []:
        pid = person.get("person_id", "unknown_person")
        lines.append(f"- {pid}：{translated.get(f'character.{pid}.description', '待翻译')}")
    lines.extend(["", "## 故事线", ""])
    for event in item.get("storyline") or []:
        eid = event.get("event_id")
        lines.extend([f"{fmt_ms(event['start_ms'])} - {fmt_ms(event['end_ms'])}",
                      translated.get(f"storyline.{eid}.text", "待翻译"), ""])
    lines.extend(["## 语音转录", ""])
    for speech in item.get("speech") or []:
        lines.extend([f"{fmt_ms(speech['start_ms'])} - {fmt_ms(speech['end_ms'])}",
                      "说话人：待人工确认", "状态：本次未使用音频理解模型，待人工试听。",
                      f"内容：{json.dumps(speech.get('text',''), ensure_ascii=False)}", ""])
    if not item.get("speech"):
        lines.append("语音是否存在及其内容待人工确认。")
    lines.extend(["## 可见文字", "", "本次未评估可见文字。"])
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fused", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-items", type=int, default=2)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    records = [row for _, row in read_jsonl(args.fused)][:args.max_items]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.execute:
        print(f"DRY RUN: {len(records)} records; no network call")
        return
    with args.out.open("w", encoding="utf-8", newline="\n") as out:
        for n, item in enumerate(records, 1):
            fields = fields_for(item)
            translated, meta = call_translate(fields)
            row = {"_id":item["_id"], "caption_zh":render_zh(item, translated),
                   "translated_fields":translated, "usage":meta}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("TRANSLATED", n, item["_id"], meta.get("usage",{}), flush=True)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip(): yield None, json.loads(line)


if __name__ == "__main__": main()
