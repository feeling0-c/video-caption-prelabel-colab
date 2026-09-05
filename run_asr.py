"""Generate timestamped ASR JSON from a prepared manifest in Colab."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--max-items", type=int, default=20)
    args = parser.parse_args()

    from faster_whisper import WhisperModel
    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False
    device, compute_type = ("cuda", "int8_float16") if use_cuda else ("cpu", "int8")
    print(f"ASR model={args.model} device={device} compute_type={compute_type}", flush=True)
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))[:args.max_items]
    args.out.mkdir(parents=True, exist_ok=True)
    for number, item in enumerate(manifest, 1):
        destination = args.out / item["_id"] / "asr.json"
        if destination.exists():
            print("CACHED", number, item["_id"], flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not item.get("audio_path"):
            destination.write_text(json.dumps({"_id": item["_id"], "audio_status": "no_audio", "segments": []},
                                              ensure_ascii=False, indent=2), encoding="utf-8")
            continue
        segments, info = model.transcribe(
            str(args.input_root / item["audio_path"]), beam_size=5, word_timestamps=True,
            vad_filter=True, condition_on_previous_text=False,
        )
        rows = []
        for segment in segments:
            start_ms, end_ms = round(segment.start * 1000), round(segment.end * 1000)
            if not 0 <= start_ms < end_ms <= int(item["duration_ms"]):
                continue
            words = [
                {"start_ms": round(word.start * 1000), "end_ms": round(word.end * 1000), "text": word.word}
                for word in (segment.words or [])
                if word.start is not None and word.end is not None
            ]
            rows.append({"speech_id": f"u{len(rows) + 1:03d}", "start_ms": start_ms,
                         "end_ms": end_ms, "text": segment.text.strip(), "words": words})
        output = {
            "_id": item["_id"], "language": info.language,
            "language_probability": info.language_probability,
            "timing_source": "faster_whisper_estimated_word_timestamps", "segments": rows,
        }
        destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print("ASR", number, item["_id"], len(rows), info.language, flush=True)


if __name__ == "__main__":
    main()
