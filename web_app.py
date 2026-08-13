"""
web_app.py

Unified web application for ASL Learning / Test / Translate modes.
- Single-page frontend (menu + mode views) with a warm, Duolingo-like UI.
- Browser captures webcam frames and uploads to /api/predict as base64.
- Learning/Test modes use gesture.py (unchanged) for per-frame prediction.
- Translate mode uses gesture_v3.py (unchanged) for per-frame prediction and semantic correction (correct_word).
- Sessions maintained in-memory (not persistent). Start a session via /api/session.

Run: python web_app.py
Open: http://localhost:5001

Dependencies: flask, opencv-python, mediapipe, numpy
"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory
import time
import uuid
import base64
import os
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

import gesture
try:
    import gesture_v3 as gesture3
except Exception:
    gesture3 = None

# Config
PORT = 5001
CONFIRM_TIME = 1.4
CONF_THRESHOLD = 0.55
NO_HAND_COMMIT_FRAMES = 20
PRACTICE_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

app = Flask(__name__)
mp_hands = mp.solutions.hands.Hands(model_complexity=1, max_num_hands=1,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# In-memory session storage
sessions = {}

# Serve letter images from repository's letter/ folder
LETTER_DIR = os.path.join(os.path.dirname(__file__), 'letter')

INDEX_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ASL Tutor</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg1:#FFF7E6;--card:#ffffff;--accent:#6DD3A2;--muted:#6b6b6b}
body{margin:0;font-family:Inter,Arial,Helvetica,sans-serif;background:linear-gradient(180deg,var(--bg1),#fff);color:#222}
.header{display:flex;align-items:center;justify-content:space-between;padding:18px 28px}
.logo{font-weight:700;color:#0b6b47}
.modes{display:flex;gap:10px}
.mode-btn{background:transparent;border:none;padding:10px 14px;border-radius:10px;font-weight:600;cursor:pointer}
.mode-btn.active{background:var(--accent);color:#fff}
.container{display:flex;gap:20px;padding:0 28px 28px}
.left{flex:1;display:flex;flex-direction:column;gap:14px}
.card{background:var(--card);border-radius:14px;padding:16px;box-shadow:0 6px 18px rgba(15,15,15,0.06)}
.center{width:520px;display:flex;flex-direction:column;align-items:center;gap:12px}
.video-wrap{width:480px;height:360px;border-radius:12px;overflow:hidden;background:#000;display:flex;align-items:center;justify-content:center}
video{width:100%;height:100%;object-fit:cover}
.hud{display:flex;gap:12px;align-items:center}
.letter-big{font-size:72px;color:#0b6b47;font-weight:800}
.progress{height:10px;background:#eee;border-radius:10px;width:240px;overflow:hidden}
.progress > i{display:block;height:100%;background:linear-gradient(90deg,#ffda79,#6dd3a2);width:0%}
.right{width:300px;display:flex;flex-direction:column;gap:12px}
.sample-img{width:100%;height:120px;object-fit:contain;background:#f6f6f6;border-radius:8px;padding:6px}
.small{font-size:13px;color:var(--muted)}
.footer{padding:12px 28px;color:#666;font-size:13px}
.msg{padding:10px;border-radius:8px;background:rgba(11,107,71,0.06);color:#0b6b47;font-weight:600}
.center-bottom{display:flex;gap:10px}
.button{background:var(--accent);border:none;color:#fff;padding:10px 14px;border-radius:10px;cursor:pointer;font-weight:700}
</style>
</head>
<body>
  <div class="header">
    <div class="logo">ASL Tutor</div>
    <div class="modes">
      <button class="mode-btn active" data-mode="learning">Learning</button>
      <button class="mode-btn" data-mode="test">Test</button>
      <button class="mode-btn" data-mode="translate">Translate</button>
    </div>
  </div>

  <div class="container">
    <div class="left">
      <div class="card">
        <div style="font-weight:700">Session</div>
        <div class="small">ID: <span id="sid">-</span></div>
        <div style="margin-top:8px">Mode: <b id="modeLbl">learning</b></div>
        <div style="margin-top:8px">Score: <b id="score">0</b></div>
        <div style="margin-top:8px" class="small">Instructions: Allow camera; hold the handshape until confirmed.</div>
      </div>

      <div class="card">
        <div style="font-weight:700">Practice Table</div>
        <div id="practice" style="margin-top:8px;display:flex;flex-wrap:wrap;gap:6px"></div>
      </div>

      <div class="card small" id="help">Camera: initializing...</div>
    </div>

    <div class="center">
      <div class="video-wrap card">
        <video id="video" autoplay playsinline></video>
      </div>
      <div class="hud">
        <div class="letter-big" id="det">-</div>
        <div class="progress"><i id="prog"></i></div>
      </div>
      <div class="center-bottom">
        <button class="button" id="startBtn">Start</button>
        <button class="button" id="stopBtn" style="background:#ddd;color:#333">Stop</button>
        <div id="msgBox" style="min-width:220px"></div>
      </div>
    </div>

    <div class="right">
      <div class="card">
        <div style="font-weight:700">Sample</div>
        <img id="sampleImg" class="sample-img" src="" alt="sample"/>
        <div class="small" id="sampleLbl">Target: -</div>
      </div>

      <div class="card">
        <div style="font-weight:700">Candidates</div>
        <div id="cands" style="margin-top:8px" class="small">-</div>
      </div>

      <div class="card">
        <div style="font-weight:700">Translate Output</div>
        <div id="raw" class="small">raw: -</div>
        <div id="final" class="small">text: -</div>
      </div>
    </div>
  </div>

  <div class="footer">If camera doesn't start, check browser permissions or try Chrome/Edge. For remote access use HTTPS.</div>

<script>
let sid=null; let mode='learning'; let running=false; const FPS=6; let captureInterval=null; let stream=null; let score=0; let practice=[];
const progEl=document.getElementById('prog'); const detEl=document.getElementById('det'); const candsEl=document.getElementById('cands'); const sampleImg=document.getElementById('sampleImg'); const sampleLbl=document.getElementById('sampleLbl');

async function createSession(){ const r=await fetch('/api/session',{method:'POST'}); const j=await r.json(); sid=j.session_id; document.getElementById('sid').innerText=sid; practice=j.practice; score=0; updatePractice(); }
function updatePractice(){ const el=document.getElementById('practice'); el.innerHTML=''; practice.forEach((p,i)=>{ const span=document.createElement('span'); span.style.padding='6px 8px'; span.style.borderRadius='8px'; span.style.background=(i===0?'#6dd3a2':'#f2f2f2'); span.style.fontWeight=700; span.innerText=p; el.appendChild(span); }); }

async function setMode(m){ mode=m; document.getElementById('modeLbl').innerText=m; document.querySelectorAll('.mode-btn').forEach(b=>b.classList.toggle('active', b.dataset.mode===m)); await fetch('/api/set_mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,mode:m})}); }

document.querySelectorAll('.mode-btn').forEach(b=>b.addEventListener('click', ()=>{ setMode(b.dataset.mode); }));

async function startCamera(){ if(running) return; if(!sid) await createSession(); const v=document.getElementById('video'); try{ stream = await navigator.mediaDevices.getUserMedia({video:{width:640}}); v.srcObject=stream; running=true; captureInterval = setInterval(captureFrame, 1000/FPS); document.getElementById('help').innerText='Camera active'; }catch(e){ document.getElementById('help').innerText='Camera error: '+e.message; console.error(e);}}

document.getElementById('startBtn').addEventListener('click', ()=>startCamera());
document.getElementById('stopBtn').addEventListener('click', stopCamera);

function stopCamera(){ if(!running) return; running=false; if(captureInterval) clearInterval(captureInterval); if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; } document.getElementById('help').innerText='Camera stopped'; }

async function captureFrame(){ const v=document.getElementById('video'); if(!v || v.readyState<2) return; const c=document.createElement('canvas'); c.width=480; c.height=360; const ctx=c.getContext('2d'); ctx.drawImage(v,0,0,c.width,c.height); c.toBlob(async function(blob){ const b64 = await toBase64(blob); await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,mode:mode,image:b64})}).then(r=>r.json()).then(updateFromServer).catch(e=>console.error(e)); }, 'image/jpeg', 0.7); }

function toBase64(blob){ return new Promise(res=>{ const reader=new FileReader(); reader.onloadend=()=>res(reader.result.split(',')[1]); reader.readAsDataURL(blob); }); }

function updateFromServer(j){ if(j.letter) detEl.innerText=j.letter; if(j.confidence) candsEl.innerText = (j.candidates||[]).map(x=>x[0]+':'+Math.round(x[1]*100)+'%').join('  '); if(j.progress) sampleLbl.innerText='Target: '+j.target; if(j.sample_url) sampleImg.src=j.sample_url; if(j.updated) score=j.score, document.getElementById('score').innerText=score; if(j.progress_text) document.getElementById('raw').innerText='raw: '+j.progress_text; if(j.final_text) document.getElementById('final').innerText='text: '+j.final_text; if(j.prog_pct!==undefined) progEl.style.width=j.prog_pct+'%'; if(j.message) document.getElementById('msgBox').innerText=j.message; }

window.onload = async ()=>{ await createSession(); await setMode('learning'); };
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/letter/<path:fname>')
def letter_file(fname):
    # serve images from letter dir
    safe = os.path.basename(fname)
    path = os.path.join(LETTER_DIR, safe)
    if os.path.exists(path):
        return send_from_directory(LETTER_DIR, safe)
    return ('', 404)

@app.route('/api/session', methods=['POST'])
def api_session():
    sid = str(uuid.uuid4())
    practice = PRACTICE_LETTERS.copy()
    # keep first-practice element as target; shuffle
    import random
    random.shuffle(practice)
    sessions[sid] = {
        'practice': practice,
        'pos': 0,
        'score': 0,
        'target': practice[0] if practice else 'A',
        'stable_letter': None,
        'stable_start': None,
        'last_committed': None,
        'raw_seq': deque(),
        'final_words': [],
        'no_hand_frames': 0,
    }
    return jsonify({'session_id': sid, 'practice': sessions[sid]['practice']})

@app.route('/api/set_mode', methods=['POST'])
def api_set_mode():
    data = request.get_json()
    sid = data.get('session_id')
    mode = data.get('mode')
    s = sessions.get(sid)
    if not s:
        return jsonify({'error': 'invalid session'}), 400
    # reset mode-specific state
    s['pos'] = 0
    s['score'] = 0
    s['target'] = s['practice'][0] if s['practice'] else 'A'
    s['stable_letter'] = None
    s['stable_start'] = None
    s['last_committed'] = None
    s['raw_seq'].clear()
    s['final_words'].clear()
    s['no_hand_frames'] = 0
    s['mode'] = mode
    return jsonify({'ok': True})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    sid = data.get('session_id')
    mode = data.get('mode', 'learning')
    b64 = data.get('image')
    s = sessions.get(sid)
    if not s:
        return jsonify({'error': 'invalid session'}), 400

    try:
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return jsonify({'error': 'bad image', 'e': str(e)}), 400

    results = mp_hands.process(frame_rgb)
    letter = '?'
    confidence = 0.0
    candidates = []
    updated = False
    message = ''
    prog_pct = 0

    # choose engine
    use_v3 = (mode == 'translate') and (gesture3 is not None)

    if results.multi_hand_landmarks:
        s['no_hand_frames'] = 0
        lm = results.multi_hand_landmarks[0]
        if use_v3:
            out = gesture3.predict_letter(lm.landmark)
        else:
            out = gesture.predict_letter(lm.landmark)
        letter = out.get('letter', '?')
        confidence = out.get('confidence', 0.0)
        candidates = out.get('candidates', [])
        now = time.time()

        if letter == s['stable_letter']:
            if s['stable_start'] is None:
                s['stable_start'] = now
            elapsed = now - s['stable_start']
            prog_pct = min(100, int(elapsed / CONFIRM_TIME * 100))
            if (elapsed >= CONFIRM_TIME and confidence >= CONF_THRESHOLD and letter != '?' and letter != s['last_committed']):
                s['last_committed'] = letter
                if mode in ('learning', 'test'):
                    target = s['target']
                    if letter.upper() == target.upper():
                        s['score'] += 1
                        s['pos'] += 1
                        if s['pos'] < len(s['practice']):
                            s['target'] = s['practice'][s['pos']]
                        else:
                            s['target'] = 'DONE'
                        message = 'Correct!'
                        updated = True
                    else:
                        message = f"Detected {letter}, expect {target}"
                else:  # translate mode
                    alts = [c for c, _ in candidates[1:]]
                    s['raw_seq'].append({'letter': letter, 'alts': alts})
                    s['last_committed'] = letter
                    updated = True
                s['stable_start'] = None
        else:
            s['stable_letter'] = letter
            s['stable_start'] = now
    else:
        s['stable_letter'] = None
        s['stable_start'] = None
        s['last_committed'] = None
        s['no_hand_frames'] += 1
        # auto-commit for translate mode when no hand seen for some frames
        if mode == 'translate' and s['no_hand_frames'] >= NO_HAND_COMMIT_FRAMES and len(s['raw_seq'])>0:
            if gesture3 is not None and hasattr(gesture3, 'correct_word'):
                try:
                    word = gesture3.correct_word(list(s['raw_seq']))
                except Exception:
                    word = ''.join(item['letter'] for item in s['raw_seq'])
            else:
                word = ''.join(item['letter'] for item in s['raw_seq'])
            s['final_words'].append(word)
            s['raw_seq'].clear()
            message = f"Committed: {word}"

    # prepare response
    resp = {
        'letter': letter,
        'confidence': confidence,
        'candidates': candidates,
        'updated': updated,
        'message': message,
        'score': s.get('score', 0),
        'target': s.get('target'),
        'prog_pct': prog_pct,
        'progress_text': ''.join(item['letter'] for item in s['raw_seq']),
        'final_text': ' '.join(s['final_words'])
    }
    # include sample image URL for learning mode
    if mode == 'learning':
        target = s.get('target', 'A')
        img_path = f"/letter/{target}.png"
        # fallback check by existence
        if os.path.exists(os.path.join(LETTER_DIR, f"{target}.png")):
            resp['sample_url'] = img_path
        else:
            resp['sample_url'] = ''
    return jsonify(resp)

@app.route('/api/next', methods=['POST'])
def api_next():
    data = request.get_json()
    sid = data.get('session_id')
    s = sessions.get(sid)
    if not s:
        return jsonify({'error': 'invalid session'}), 400
    # advance practice
    s['pos'] = min(len(s['practice']), s['pos']+1)
    if s['pos'] < len(s['practice']):
        s['target'] = s['practice'][s['pos']]
    else:
        s['target'] = 'DONE'
    return jsonify({'target': s['target'], 'score': s['score']})

@app.route('/api/back', methods=['POST'])
def api_back():
    data = request.get_json()
    sid = data.get('session_id')
    s = sessions.get(sid)
    if not s:
        return jsonify({'error': 'invalid session'}), 400
    s['pos'] = max(0, s['pos']-1)
    s['target'] = s['practice'][s['pos']] if s['pos'] < len(s['practice']) else 'DONE'
    return jsonify({'target': s['target'], 'score': s['score']})


def main():
    if gesture3 is None:
        print('[Warning] gesture_v3 not imported; translate mode will fall back to gesture.predict_letter and no semantic correction.')
    print(f"Starting web app at http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)


if __name__ == '__main__':
    main()
