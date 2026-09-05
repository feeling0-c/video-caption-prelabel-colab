"""Prepare private pilot inputs. Never include legacy captions in model inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / '.deps'))
import av
import cv2
import imageio_ffmpeg
from scenedetect import detect, AdaptiveDetector


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def metadata(path):
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        duration = float(v.duration * v.time_base) if v.duration else c.duration / av.time_base
        audio = c.streams.audio[0] if c.streams.audio else None
        return dict(duration_ms=round(duration * 1000), fps=float(v.average_rate),
                    width=v.width, height=v.height, has_audio=bool(audio),
                    video_start_sec=float((v.start_time or 0) * v.time_base),
                    audio_start_sec=float((audio.start_time or 0) * audio.time_base) if audio else None)


def inventory(source, media_root):
    index = {}
    for path in sorted(media_root.rglob('*.mp4')):
        parts = path.parts
        if 'video_clips' not in parts:
            continue
        i = parts.index('video_clips')
        rel = '/'.join(parts[i-1:])
        index.setdefault(rel, path)
    rows = []
    seen = set()
    with source.open(encoding='utf-8-sig') as f:
        for line in f:
            old = json.loads(line)
            item_id, rel = old['_id'], old['video_path'].replace('\\', '/')
            if item_id in seen:
                raise ValueError('Duplicate input ID')
            seen.add(item_id)
            if rel in index:
                if not re.fullmatch(r'[a-zA-Z0-9_-]+', item_id):
                    raise ValueError('Unsafe ID')
                rows.append({'_id': item_id, 'video_path': rel,
                             'local_video_path': str(index[rel]),
                             'source_group_hint': Path(rel).parent.name})
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda x: hashlib.sha256(x['_id'].encode()).hexdigest()):
        groups[row['source_group_hint']].append(row)
    ordered = []
    while groups:
        for key in sorted(list(groups)):
            ordered.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
    return ordered


def prepare(row, out):
    path = Path(row['local_video_path'])
    d = out / row['_id']
    d.mkdir(parents=True, exist_ok=True)
    meta = metadata(path)
    if not (1000 < meta['duration_ms'] <= 60000):
        raise ValueError('Pilot requires a 1–60 second clip')
    # Preserve the input MP4 timeline; nonzero AV starts need explicit offset handling.
    if abs(meta['video_start_sec']) > .05 or (meta['has_audio'] and abs(meta['audio_start_sec']) > .1):
        raise ValueError('Nonzero media origin: requires AV offset review')
    scenes = detect(str(path), AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=5),
                    start_in_scene=True)
    shots = [{ 'shot_id': f's{i+1:03d}', 'start_ms': round(a.get_seconds()*1000),
              'end_ms': min(round(b.get_seconds()*1000), meta['duration_ms'])}
             for i,(a,b) in enumerate(scenes)]
    if not shots:
        shots = [{'shot_id': 's001', 'start_ms': 0, 'end_ms': meta['duration_ms']}]
    shots[0]['start_ms'] = 0
    shots[-1]['end_ms'] = meta['duration_ms']
    if len(shots) > 12:
        raise ValueError('More than 12 shots: windowed inference required')
    # Two anchor frames per shot, then uniform coverage, with actual decoded PTS saved.
    targets = set()
    for s in shots:
        lo, hi = s['start_ms']/1000, s['end_ms']/1000
        targets.update([lo+(hi-lo)*.25, lo+(hi-lo)*.75])
    for i in range(24):
        if len(targets) >= 32:
            break
        t = (i+.5) * meta['duration_ms']/24000
        if all(abs(t-x) > .15 for x in targets):
            targets.add(t)
    targets = sorted(targets)
    frames, pts = [], []
    j = 0
    with av.open(str(path)) as c:
        for n, frame in enumerate(c.decode(video=0)):
            t = float(frame.pts * frame.time_base)
            pts.append(t)
            if j < len(targets) and t >= targets[j]:
                rgb = frame.to_ndarray(format='rgb24')
                h, w = rgb.shape[:2]
                scale = min(1., 512/max(w,h))
                rgb = cv2.resize(rgb, (max(1,round(w*scale)), max(1,round(h*scale))))
                name = f'frame_{len(frames):03d}.jpg'
                ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise ValueError('Frame encoding failed')
                (d/name).write_bytes(encoded.tobytes())
                frames.append({'frame_id': f'f{len(frames):03d}', 'time_ms': round(t*1000),
                               'file': f"{row['_id']}/{name}"})
                while j < len(targets) and targets[j] <= t:
                    j += 1
    if len(pts) < 2 or len(frames) < 4:
        raise ValueError('Too few decoded frames')
    intervals = [b-a for a,b in zip(pts,pts[1:])]
    if max(abs(dt-1/meta['fps']) for dt in intervals) > .005:
        raise ValueError('VFR detected: shot frame-to-time mapping needs review')
    if abs((pts[-1]+1/meta['fps'])*1000-meta['duration_ms']) > 150:
        raise ValueError('Decoded duration differs from metadata')
    for s in shots:
        if not any(s['start_ms'] <= f['time_ms'] < s['end_ms'] for f in frames):
            raise ValueError('Shot has no sampled frame')
    audio_path = None
    if meta['has_audio']:
        audio_path = f"{row['_id']}/audio.wav"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-nostdin', '-y',
                        '-i', str(path), '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000',
                        '-c:a', 'pcm_s16le', str(out/audio_path)], check=True, capture_output=True)
    cloud = {k: row[k] for k in ('_id','video_path','source_group_hint')}
    cloud.update(meta, video_sha256=digest(path), shots=shots, frames=frames,
                 audio_path=audio_path, decode_ok=True, decoded_frame_count=len(pts),
                 shot_source='PySceneDetect 0.7.1 AdaptiveDetector; newly detected candidates',
                 selection_status='engineering_stratified_by_path_group; semantic_coverage_not_verified')
    dump(d/'prepared.json', cloud)
    return cloud


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--media-root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=PROJECT/'runs'/'pilot_20260905')
    p.add_argument('--limit', type=int, default=20)
    args = p.parse_args()
    candidates = inventory(args.source, args.media_root)
    dump(args.out/'local_inventory.json', candidates)
    inputs = args.out/'inputs'
    prepared, failures = [], []
    for row in candidates:
        try:
            item = prepare(row, inputs)
            prepared.append(item)
            print('PREPARED', len(prepared), row['_id'], item['duration_ms'],
                  'shots',len(item['shots']),'frames',len(item['frames']), flush=True)
            if len(prepared) == args.limit:
                break
        except Exception as e:
            failures.append({'_id':row['_id'],'reason':str(e)})
            print('PREPARE_FAILED', row['_id'], type(e).__name__, flush=True)
    dump(inputs/'manifest.json', prepared)
    dump(args.out/'preparation_failures.json', failures)
    archive = args.out/'pilot_inputs.zip'
    # Explicit allowlist: no legacy captions, no absolute local paths, no credentials.
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(inputs/'manifest.json', 'manifest.json')
        for item in prepared:
            names = [f['file'] for f in item['frames']]
            if item['audio_path']:
                names.append(item['audio_path'])
            for name in names:
                z.write(inputs/name, name)
    dump(args.out/'preparation_report.json', {'matched_video_records':len(candidates),
        'prepared':len(prepared),'failed_attempts':len(failures),
        'total_seconds':sum(x['duration_ms'] for x in prepared)/1000,
        'archive_bytes':archive.stat().st_size,'archive_sha256':digest(archive),
        'archive_contains':'new metadata, frames, audio only; NO legacy captions'})
    print('ARCHIVE', str(archive), archive.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
