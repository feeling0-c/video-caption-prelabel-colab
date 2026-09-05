"""Cells for Colab; copy each # %% block into a notebook or upload as .py."""
# %% [markdown]
# Caption prelabel pilot: private input zip -> ASR -> optional Qwen visual JSON.
# This notebook intentionally defaults to two items for a quota/quality smoke test.

# %%
# In a Colab code cell run: !pip -q install faster-whisper==1.2.1 requests

# %%
from google.colab import files
uploaded = files.upload()  # choose pilot_inputs.zip; keep the notebook private
import io, json, os, zipfile, shutil, time
from pathlib import Path
zip_name = next(iter(uploaded))
ROOT = Path('/content/pilot_inputs')
if ROOT.exists(): shutil.rmtree(ROOT)
ROOT.mkdir(parents=True)
with zipfile.ZipFile(io.BytesIO(uploaded[zip_name])) as z:
    z.extractall(ROOT)
manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
print('items:', len(manifest), 'total seconds:', round(sum(x['duration_ms'] for x in manifest)/1000, 1))

# %%
from faster_whisper import WhisperModel
import torch
ASR_LIMIT = 20
asr_out = Path('/content/asr'); asr_out.mkdir(exist_ok=True)
model = WhisperModel('large-v3', device='cuda', compute_type='int8_float16')
for n, item in enumerate(manifest[:ASR_LIMIT], 1):
    dest = asr_out/item['_id']/'asr.json'
    if dest.exists():
        print('CACHED', n, item['_id']); continue
    if not item.get('audio_path'):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({'_id':item['_id'],'segments':[],'audio_status':'no_audio'}, ensure_ascii=False), encoding='utf-8')
        continue
    segments, info = model.transcribe(str(ROOT/item['audio_path']), beam_size=5,
        word_timestamps=True, vad_filter=True, condition_on_previous_text=False)
    rows=[]
    for s in list(segments):
        words=[]
        for w in (s.words or []):
            words.append({'start_ms':round(w.start*1000),'end_ms':round(w.end*1000),'text':w.word})
        rows.append({'speech_id':f"u{len(rows)+1:03d}",'start_ms':round(s.start*1000),
                     'end_ms':round(s.end*1000),'text':s.text.strip(),'words':words})
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({'_id':item['_id'],'language':info.language,
        'language_probability':info.language_probability,'timing_source':'faster_whisper_estimated_word_timestamps',
        'segments':rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ASR', n, item['_id'], len(rows), info.language, flush=True)
del model
torch.cuda.empty_cache()

# %%
# Optional visual pass. Put DASHSCOPE_API_KEY in Colab Secrets under that exact name.
import base64, hashlib, requests
from google.colab import userdata
try:
    API_KEY = userdata.get('DASHSCOPE_API_KEY')
except Exception:
    API_KEY = None
MODEL = 'qwen3.7-flash-2026-07-15'
VISUAL_LIMIT = 2  # raise to 20 only after the two-item smoke test is inspected
visual_out = Path('/content/visual'); visual_out.mkdir(exist_ok=True)
prompt = '''Return JSON only and follow this exact schema:
{"overview":{"visual_style_en":"","audio_style_en":null,"narrative_theme_en":""},
"characters":[{"person_id":"p01","description_en":"","evidence_frame_ids":[]}],
"storyline":[{"event_id":"e01","shot_id":"s001","text_en":"","evidence_frame_ids":[]}],
"speech_context":[{"speech_id":"u001","state_en":null,"speaker_person_id":null,
"speaker_link_status":"unknown","review_reason":null}]}
Use only visible evidence in the supplied frames. Frame IDs and shot boundaries are authoritative.
Never invent or alter shot_id or speech_id. A visual-only model cannot hear audio, so audio_style_en
and speech state must be null. Use anonymous person_id values and cite evidence_frame_ids.
If a field is unknown, use an empty string, empty array, or null according to the schema.'''
def frame_part(path):
    b = path.read_bytes()
    return {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(b).decode()}}
def call_qwen(item):
    frames = item['frames']
    idx = sorted(set(round(i*(len(frames)-1)/7) for i in range(min(8,len(frames)))))
    chosen = [frames[i] for i in idx]
    index = '\n'.join(f"{f['frame_id']} at {f['time_ms']} ms" for f in chosen)
    content = [{'type':'text','text':prompt+'\nFrame index:\n'+index+'\nShots:\n'+json.dumps(item['shots'])}]
    content += [frame_part(ROOT/f['file']) for f in chosen]
    r = requests.post('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
        headers={'Authorization':'Bearer '+API_KEY,'Content-Type':'application/json'},
        json={'model':MODEL,'messages':[{'role':'user','content':content}], 'temperature':0,
              'response_format':{'type':'json_object'}}, timeout=180)
    if r.status_code == 403: raise RuntimeError('403: free-only quota stopped the request')
    r.raise_for_status()
    body = r.json(); text = body['choices'][0]['message']['content']
    if isinstance(text, list): text=''.join(x.get('text','') for x in text if isinstance(x,dict))
    return json.loads(text), {'usage':body.get('usage',{}),'request_id':r.headers.get('x-request-id')}
if not API_KEY:
    print('No DASHSCOPE_API_KEY secret: ASR is complete; visual API pass is paused.')
else:
    for n, item in enumerate(manifest[:VISUAL_LIMIT], 1):
        dest=visual_out/item['_id']/'visual.json'; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists(): print('CACHED', n, item['_id']); continue
        result, meta=call_qwen(item)
        dest.write_text(json.dumps({'_id':item['_id'],'model':MODEL,'result':result,'api':meta}, ensure_ascii=False, indent=2), encoding='utf-8')
        print('VISUAL', n, item['_id'], meta['usage'], flush=True)

# %%
# Translation is also run inside this Secret-enabled Colab runtime.
# Upload fused_preannotations.jsonl and translate_fields.py when this cell runs.
from google.colab import userdata
import os, subprocess
try:
    os.environ['DASHSCOPE_API_KEY'] = userdata.get('DASHSCOPE_API_KEY')
except Exception:
    os.environ.pop('DASHSCOPE_API_KEY', None)
from google.colab import files as colab_files
translation_upload = colab_files.upload()
fused_name = next((name for name in translation_upload if name.endswith('.jsonl')), None)
script_name = next((name for name in translation_upload if name == 'translate_fields.py'), None)
if not fused_name or not script_name:
    raise RuntimeError('Select both fused_preannotations.jsonl and translate_fields.py')
TRANSLATION_LIMIT = 2  # inspect two records before changing to 20
subprocess.run([
    'python', script_name, '--fused', fused_name,
    '--out', '/content/translations.jsonl',
    '--max-items', str(TRANSLATION_LIMIT), '--execute'
], check=True)
colab_files.download('/content/translations.jsonl')

# %%
# Download durable outputs before the Colab runtime is reclaimed.
package_root = Path('/content/pilot_outputs')
if package_root.exists():
    shutil.rmtree(package_root)
package_root.mkdir(parents=True)
shutil.copytree(asr_out, package_root/'asr', dirs_exist_ok=True)
if visual_out.exists():
    shutil.copytree(visual_out, package_root/'visual', dirs_exist_ok=True)
shutil.make_archive('/content/pilot_outputs', 'zip', '/content', 'pilot_outputs')
files.download('/content/pilot_outputs.zip')
