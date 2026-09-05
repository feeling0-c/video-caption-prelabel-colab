"""Prepare raw MP4 clips entirely inside a Colab runtime.

The input is an extracted private ZIP containing MP4 clips.  An optional JSONL
source index can preserve pre-existing _id and video_path values, but captions
are never read or used.  Output paths and time ranges are owned by this code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import av
import cv2
from scenedetect import AdaptiveDetector, detect


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def norm_path(value: str) -> str:
    return str(PurePosixPath(str(value).replace("\\", "/"))).lstrip("./")


def read_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"source index line {line_no} is not an object")
            source_file = row.get("source_file") or row.get("source_path") or row.get("video_path")
            if not isinstance(source_file, str) or not source_file.strip():
                raise ValueError(f"source index line {line_no} lacks source_file/video_path")
            key = norm_path(source_file)
            if key in result:
                raise ValueError(f"duplicate source path in index: {key}")
            result[key] = row
    return result


def find_index_row(rel: str, index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if rel in index:
        return index[rel]
    candidates = [row for key, row in index.items() if rel.endswith("/" + key) or key.endswith("/" + rel)]
    if len(candidates) > 1:
        raise ValueError(f"ambiguous source index mapping for {rel}")
    return candidates[0] if candidates else None


def probe(path: Path) -> dict[str, Any]:
    with av.open(str(path)) as container:
        video = next((stream for stream in container.streams if stream.type == "video"), None)
        audio = next((stream for stream in container.streams if stream.type == "audio"), None)
        if video is None:
            raise ValueError("no video stream")
        if video.duration is not None:
            duration_sec = float(video.duration * video.time_base)
        elif container.duration is not None:
            duration_sec = float(container.duration / av.time_base)
        else:
            raise ValueError("duration unavailable")
        fps = float(video.average_rate) if video.average_rate else None
        if not fps or fps <= 0:
            raise ValueError("fps unavailable")
        return {
            "duration_ms": round(duration_sec * 1000),
            "fps": fps,
            "width": int(video.width),
            "height": int(video.height),
            "has_audio": audio is not None,
        }


def shot_candidates(path: Path, duration_ms: int, max_shots: int) -> list[dict[str, int | str]]:
    scenes = detect(str(path), AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=5), start_in_scene=True)
    shots = [
        {"shot_id": f"s{i + 1:03d}", "start_ms": round(start.get_seconds() * 1000),
         "end_ms": min(round(end.get_seconds() * 1000), duration_ms)}
        for i, (start, end) in enumerate(scenes)
    ]
    if not shots:
        shots = [{"shot_id": "s001", "start_ms": 0, "end_ms": duration_ms}]
    shots[0]["start_ms"] = 0
    shots[-1]["end_ms"] = duration_ms
    if len(shots) > max_shots:
        raise ValueError(f"{len(shots)} shot candidates exceeds max_shots={max_shots}")
    return shots


def sample_frames(video: Path, target_dir: Path, item_id: str,
                  shots: list[dict[str, int | str]], duration_ms: int,
                  max_frames: int) -> list[dict[str, Any]]:
    # Sample twice per detected shot first, then distribute remaining frames.
    targets = set()
    for shot in shots:
        start, end = int(shot["start_ms"]), int(shot["end_ms"])
        targets.update((start + (end - start) * 0.25, start + (end - start) * 0.75))
    for i in range(max_frames * 2):
        if len(targets) >= max_frames:
            break
        candidate = (i + 0.5) * duration_ms / (max_frames * 2)
        if all(abs(candidate - previous) > 150 for previous in targets):
            targets.add(candidate)
    targets_ms = sorted(targets)[:max_frames]

    frames: list[dict[str, Any]] = []
    target_dir.mkdir(parents=True, exist_ok=True)
    next_target = 0
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            time_ms = round(float(frame.pts * frame.time_base) * 1000)
            if next_target >= len(targets_ms) or time_ms < targets_ms[next_target]:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            height, width = rgb.shape[:2]
            scale = min(1.0, 512 / max(width, height))
            if scale < 1.0:
                rgb = cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            name = f"frame_{len(frames):03d}.jpg"
            okay, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not okay:
                raise ValueError("frame JPEG encoding failed")
            (target_dir / name).write_bytes(encoded.tobytes())
            frames.append({"frame_id": f"f{len(frames):03d}", "time_ms": time_ms,
                           "file": f"{item_id}/{name}"})
            while next_target < len(targets_ms) and targets_ms[next_target] <= time_ms:
                next_target += 1
    if len(frames) < min(4, len(targets_ms)):
        raise ValueError("too few decodable frames")
    for shot in shots:
        if not any(int(shot["start_ms"]) <= frame["time_ms"] < int(shot["end_ms"]) for frame in frames):
            raise ValueError(f"no sampled frame in {shot['shot_id']}")
    return frames


def prepare_one(video: Path, media_root: Path, index: dict[str, dict[str, Any]],
                inputs: Path, max_duration_sec: int, max_shots: int,
                max_frames: int) -> dict[str, Any]:
    local_rel = video.relative_to(media_root).as_posix()
    mapped = find_index_row(local_rel, index)
    item_id = str(mapped["_id"]) if mapped and mapped.get("_id") else hashlib.sha256(local_rel.encode()).hexdigest()[:24]
    if not item_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("unsafe _id")
    video_path = norm_path(mapped.get("video_path")) if mapped and mapped.get("video_path") else local_rel
    if video_path.startswith("/") or ".." in PurePosixPath(video_path).parts:
        raise ValueError("unsafe video_path")
    meta = probe(video)
    if not 1000 < meta["duration_ms"] <= max_duration_sec * 1000:
        raise ValueError(f"duration {meta['duration_ms']} ms outside (1 s, {max_duration_sec} s]")
    shots = shot_candidates(video, meta["duration_ms"], max_shots)
    target_dir = inputs / item_id
    frames = sample_frames(video, target_dir, item_id, shots, meta["duration_ms"], max_frames)
    audio_path = None
    if meta["has_audio"]:
        audio_path = f"{item_id}/audio.wav"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(video),
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(inputs / audio_path),
        ], check=True)
    row = {
        "_id": item_id,
        "video_path": video_path,
        "local_video_path": str(video.resolve()),
        "source_group_hint": Path(video_path).parent.name,
        **meta,
        "video_sha256": sha256_file(video),
        "shots": shots,
        "frames": frames,
        "audio_path": audio_path,
        "decode_ok": True,
        "shot_source": "PySceneDetect AdaptiveDetector",
    }
    dump(target_dir / "prepared.json", row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-duration-sec", type=int, default=60)
    parser.add_argument("--max-shots", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=32)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    media_root = args.media_root.resolve()
    videos = sorted([*media_root.rglob("*.mp4"), *media_root.rglob("*.MP4")])
    if not videos:
        raise FileNotFoundError("no MP4 files beneath media-root")
    index = read_index(args.source_index)
    inputs = args.out / "inputs"
    prepared: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for video in videos:
        try:
            row = prepare_one(video, media_root, index, inputs, args.max_duration_sec,
                              args.max_shots, args.max_frames)
            if row["_id"] in seen_ids:
                raise ValueError("duplicate _id")
            seen_ids.add(row["_id"])
            prepared.append(row)
            print("PREPARED", len(prepared), row["_id"], row["duration_ms"], flush=True)
            if len(prepared) >= args.limit:
                break
        except Exception as exc:
            failures.append({"source_file": str(video), "reason": str(exc)})
            print("PREPARE_FAILED", video.name, type(exc).__name__, flush=True)
    dump(inputs / "manifest.json", prepared)
    dump(args.out / "preparation_failures.json", failures)
    dump(args.out / "preparation_report.json", {
        "source_videos": len(videos), "prepared": len(prepared), "failed": len(failures),
        "total_seconds": sum(x["duration_ms"] for x in prepared) / 1000,
        "captions_read": False,
    })
    if not prepared:
        raise SystemExit("No video was prepared successfully")


if __name__ == "__main__":
    main()
