"""One-command, all-in-Colab preannotation pipeline.

Input: an extracted private directory of MP4 clips (plus an optional source
index).  Output: a ZIP containing the tool-importable bilingual JSONL and its
matching MP4 files.  No API key is accepted by CLI argument or saved to disk.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(label: str, args: list[str]) -> None:
    print("\n=====", label, "=====", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-root", type=Path, required=True,
                        help="Extracted private ZIP directory containing MP4 files")
    parser.add_argument("--run-dir", type=Path, default=Path("/content/video_caption_run"))
    parser.add_argument("--source-index", type=Path,
                        help="Optional JSONL with _id, video_path and source_file; captions are ignored")
    parser.add_argument("--max-items", type=int, default=2,
                        help="Use 2 for smoke test; then repeat with 20")
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--visual-model", default="qwen3.7-flash-2026-07-15")
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--allow-unresolved", action="store_true",
                        help="Include records whose audio style/speaker need human annotation")
    parser.add_argument("--clean-run-dir", action="store_true",
                        help="Delete only --run-dir before starting; safe for a fresh /content directory")
    args = parser.parse_args()
    if args.max_items < 1:
        raise ValueError("max-items must be positive")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("Set DASHSCOPE_API_KEY through Colab Secrets before running")
    run_dir = args.run_dir.resolve()
    if args.clean_run_dir and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs = run_dir / "inputs"
    manifest = inputs / "manifest.json"
    asr = run_dir / "asr"
    visual = run_dir / "visual"
    fused = run_dir / "fused"
    translations = run_dir / "translations.jsonl"
    delivery = run_dir / "delivery"

    prepare_args = ["prepare_raw_videos.py", "--media-root", str(args.media_root), "--out", str(run_dir),
                    "--limit", str(args.max_items)]
    if args.source_index:
        prepare_args += ["--source-index", str(args.source_index)]
    run("prepare raw video inputs", prepare_args)
    run("ASR", ["run_asr.py", "--manifest", str(manifest), "--input-root", str(inputs),
                "--out", str(asr), "--model", args.asr_model, "--max-items", str(args.max_items)])
    run("Qwen visual preannotation", [
        "qwen_api.py", "--manifest", str(manifest), "--input-root", str(inputs), "--out", str(visual),
        "--model", args.visual_model, "--max-items", str(args.max_items), "--execute",
    ])
    run("fuse deterministic timestamps", ["fuse_results.py", "--manifest", str(manifest),
        "--asr-root", str(asr), "--visual-root", str(visual), "--out", str(fused)])
    run("Chinese translation", ["translate_fields.py", "--fused", str(fused / "fused_preannotations.jsonl"),
        "--out", str(translations), "--max-items", str(args.max_items), "--execute"])
    final_args = ["finalize_delivery.py", "--fused", str(fused / "fused_preannotations.jsonl"),
                  "--translations", str(translations), "--manifest", str(manifest),
                  "--output", str(delivery), "--parts", str(args.parts)]
    if args.allow_unresolved:
        final_args.append("--allow-unresolved")
    run("validate and package", final_args)
    archive = shutil.make_archive(str(run_dir / "video_caption_delivery"), "zip", run_dir, "delivery")
    print("\nDELIVERY_ZIP", archive, flush=True)
    print("DELIVERY_FOLDER", delivery, flush=True)


if __name__ == "__main__":
    main()
