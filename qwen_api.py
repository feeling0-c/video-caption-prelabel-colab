"""Quota-safe Qwen3.7-Flash visual preannotation runner.

The script is deliberately opt-in: without --execute it makes no network call.
It reads DASHSCOPE_API_KEY from the environment and never prints the key.
"""
from __future__ import annotations
import argparse, base64, hashlib, json, os, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "qwen3.7-flash-2026-07-15"
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def image_part(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"type":"image_url", "image_url":{"url":"data:image/jpeg;base64," + base64.b64encode(raw).decode()}}

def compact_frames(item: dict[str, Any], input_root: Path, max_frames: int) -> list[dict[str, Any]]:
    frames = item["frames"]
    if len(frames) <= max_frames:
        chosen = frames
    else:
        positions = [round(i * (len(frames)-1)/(max_frames-1)) for i in range(max_frames)]
        chosen = [frames[i] for i in positions]
    result = []
    for frame in chosen:
        result.append({"frame_id": frame["frame_id"], "time_ms": frame["time_ms"],
                       "part": image_part(input_root / frame["file"])})
    return result

def api_call(payload: dict[str, Any], model: str, timeout: int) -> tuple[dict[str, Any], dict[str, Any]]:
    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not key:
        raise RuntimeError("Set DASHSCOPE_API_KEY (or QWEN_API_KEY) in the runtime secret store")
    import urllib.request, urllib.error
    body = json.dumps({"model":model,"messages":payload["messages"],"temperature":0,
                       "response_format":{"type":"json_object"}}).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
        headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            raw = response.read()
            response_headers = response.headers
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise RuntimeError("API returned 403; free-only stop may have halted quota use") from exc
        detail = exc.read(500).decode(errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {detail}") from exc
    body = json.loads(raw)
    text = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    if isinstance(text, list):
        text = "".join(x.get("text", "") for x in text if isinstance(x, dict))
    return json.loads(text), {"http_status":status,"usage":usage,"request_id":response_headers.get("x-request-id")}

def visual_payload(item: dict[str, Any], input_root: Path, prompt: str, max_frames: int) -> dict[str, Any]:
    frames = compact_frames(item, input_root, max_frames)
    frame_lines = "\n".join(f"{x['frame_id']} at {x['time_ms']} ms" for x in frames)
    shots = json.dumps(item["shots"], ensure_ascii=False)
    text = (prompt + "\n\nFrame index (timestamps are authoritative):\n" + frame_lines +
            "\n\nShot candidates (boundaries are authoritative):\n" + shots +
            "\n\nASR will be added separately; do not claim you heard audio.")
    return {"messages":[{"role":"user","content":[{"type":"text","text":text}]+[x["part"] for x in frames]}]}

def render_en(item: dict[str, Any], visual: dict[str, Any], asr: dict[str, Any] | None) -> str:
    ov = visual.get("overview", {})
    out = ["## Overview", "", "Overall Visual Style:", str(ov.get("visual_style_en") or "Unknown."), "",
           "Overall Audio Style:", "Audio not assessed by the visual model.", "", "Narrative Theme:",
           str(ov.get("narrative_theme_en") or "Unknown."), "", "Character Profiles:"]
    for p in visual.get("characters", []):
        out.append(f"- {p.get('person_id')}: {p.get('description_en') or 'Unknown.'}")
    out += ["", "## Storyline", ""]
    byid = {s["shot_id"]: s for s in item["shots"]}
    for event in visual.get("storyline", []):
        shot = byid.get(event.get("shot_id"))
        if not shot: continue
        out += [f"{shot['start_ms']//60000:02d}:{(shot['start_ms']//1000)%60:02d}.{shot['start_ms']%1000:03d} - "
                f"{shot['end_ms']//60000:02d}:{(shot['end_ms']//1000)%60:02d}.{shot['end_ms']%1000:03d}",
                event.get("text_en") or "Unknown.", ""]
    out += ["## Speech Transcript", ""]
    speech = (asr or {}).get("segments", [])
    if speech:
        for u in speech:
            out += [f"{u['start_ms']//60000:02d}:{(u['start_ms']//1000)%60:02d}.{u['start_ms']%1000:03d} - "
                    f"{u['end_ms']//60000:02d}:{(u['end_ms']//1000)%60:02d}.{u['end_ms']%1000:03d}",
                    f"Content: {json.dumps(u.get('text',''), ensure_ascii=False)}", ""]
    else:
        out.append("No ASR result is available yet.")
    out += ["## Visible Text", "", "Visible text was not assessed in this run."]
    return "\n".join(out)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prompt", type=Path, default=ROOT/"prompts"/"visual_fusion_schema.txt")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--max-items", type=int, default=20)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))[:args.max_items]
    prompt = args.prompt.read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.execute:
        print(f"DRY RUN: {len(manifest)} items, model={args.model}, max_frames={args.max_frames}; no network call")
        return
    for i, item in enumerate(manifest, 1):
        dest = args.out / item["_id"] / "visual.json"
        if dest.exists():
            print("CACHED", i, item["_id"], flush=True); continue
        started = time.time()
        result, meta = api_call(visual_payload(item, args.input_root, prompt, args.max_frames), args.model, 180)
        atomic_json(dest, {"_id":item["_id"],"model":args.model,"result":result,"api":meta,
                           "elapsed_sec":round(time.time()-started,2),"input_sha256":sha(json.dumps(item,sort_keys=True).encode())})
        print("DONE", i, item["_id"], meta.get("usage",{}), flush=True)

if __name__ == "__main__": main()
