"""
web_app.py (updated)

Adjustments per user request:
- Use gesture_v3.detect_thumb_gesture(lm.landmark) when available to detect 'up'/'down' thumbs and treat them as accept/reject (do not append thumbs to raw_seq).
- Test mode: use a shuffled sentence_order that cycles without repetition until exhausted, then reshuffles. When a sentence is completed, server advances and returns sentence_removed & next_sentence so frontend can animate.
- Learning mode: Practice table only visible in learning; sample image lookup supports .png/.jpg/.jpeg and returns sample_url.
- Restart endpoint clears raw_seq/final_words/pending_suggestion as requested.
- Keep gesture.py and gesture_v3.py unchanged; treat them as libraries.

"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory
import time
import uuid
import base64
import os
import random
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
SENTENCES = ["THANKS", "HAVE A NICE DAY", "GOOD JOB", "HELLO", "PLEASE", "SEE YOU", "GOOD LUCK"]

app = Flask(__name__)
mp_hands = mp.solutions.hands.Hands(model_complexity=1, max_num_hands=1,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# In-memory session storage
sessions = {}

# Serve letter images from repository's letter/ folder
LETTER_DIR = os.path.join(os.path.dirname(__file__), 'letter')

# Simple encouragement messages
ENCOURAGE = ["Great!", "Nice job!", "Well done!", "Keep it up!", "Excellent!"]

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
.center{width:640px;display:flex;flex-direction:column;align-items:center;gap:12px}
.video-wrap{width:640px;height:480px;border-radius:12px;overflow:hidden;background:#000;display:flex;align-items:center;justify-content:center}
video{width:100%;height:100%;object-fit:cover}
.hud{display:flex;gap:12px;align-items:center}
.letter-big{font-size:72px;color:#0b6b47;font-weight:800}
.progress{height:10px;background:#eee;border-radius:10px;width:240px;overflow:hidden}
.progress > i{display:block;height:100%;background:linear-gradient(90deg,#ffda79,#6dd3a2);width:0%}
.right{width:320px;display:flex;flex-direction:column;gap:12px}
.sample-img{width:100%;height:160px;object-fit:contain;background:#f6f6f6;border-radius:8px;padding:6px}
.small{font-size:13px;color:var(--muted)}
.footer{padding:12px 28px;color:#666;font-size:13px}
.msg{padding:10px;border-radius:8px;background:rgba(11,107,71,0.06);color:#0b6b47;font-weight:600}
.center-bottom{display:flex;gap:10px}
.button{background:var(--accent);border:none;color:#fff;padding:10px 14px;border-radius:10px;cursor:pointer;font-weight:700}
.hidden{display:none}
.sentence{font-size:28px;font-weight:700}
.sentence-wrap{height:48px;overflow:hidden;position:relative}
.sentence-item{position:absolute;left:0;top:0;transition:transform 0.45s ease, opacity 0.45s ease}
.translate-output{font-size:36px;font-weight:800;color:#0b6b47}
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
      <div class="card" id="sessionCard">
        <div style="font-weight:700">Session</div>
        <div class="small">ID: <span id="sid">-</span></div>
        <div style="margin-top:8px">Mode: <b id="modeLbl">learning</b></div>
        <div style="margin-top:8px">Score: <b id="score">0</b></div>
        <div style="margin-top:8px" class="small">Instructions: Allow camera; hold the handshape until confirmed.</div>
      </div>

      <div class="card" id="practiceCard">
        <div style="font-weight:700">Practice Table</div>
        <div id="practice" style="margin-top:8px;display:flex;flex-wrap:wrap;gap:6px"></div>
      </div>

      <div class="card small" id="help">Camera: initializing...</div>
    </div>

    <div class="center">
      <div class="video-wrap card">
        <video id="video" autoplay playsinline></video>
      </div>

      <!-- translate output sits under camera when in translate mode -->
      <div id="translateOutput" class="card hidden" style="width:640px;text-align:center">
        <div style="font-weight:700">Translate Output</div>
        <div id="finalLarge" class="translate-output">-</div>
        <div id="rawLarge" class="small">raw: -</div>
        <div style="margin-top:8px;display:flex;gap:8px;justify-content:center">
          <button class="button" id="acceptBtn">👍 Accept</button>
          <button class="button" id="rejectBtn" style="background:#e76f51">👎 Reject</button>
          <button class="button" id="restartBtn" style="background:#f2c94c;color:#000">Restart</button>
        </div>
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

      <div id="sentenceCard" class="card hidden" style="margin-top:12px;width:640px;text-align:center">
        <div class="sentence-wrap"><div id="sentenceItem" class="sentence-item sentence">-</div></div>
      </div>

    </div>

    <div class="right">
      <div class="card" id="sampleCard">
        <div style="font-weight:700">Sample</div>
        <img id="sampleImg" class="sample-img" src="" alt="sample"/>
        <div class="small" id="sampleLbl">Target: -</div>
      </div>

      <div class="card" id="candsCard">
        <div style="font-weight:700">Candidates</div>
        <div id="cands" style="margin-top:8px" class="small">-</div>
      </div>

    </div>
  </div>

  <div class="footer">If camera doesn't start, check browser permissions or try Chrome/Edge. For remote access use HTTPS.</div>

<script>
let sid=null; let mode='learning'; let running=false; const FPS=6; let captureInterval=null; let stream=null; let score=0; let practice=[]; let sentenceOrder=[]; let sentenceIdx=0; let pendingSuggestion=null;
const progEl=document.getElementById('prog'); const detEl=document.getElementById('det'); const candsEl=document.getElementById('cands'); const sampleImg=document.getElementById('sampleImg');

async function createSession(){ const r=await fetch('/api/session',{method:'POST'}); const j=await r.json(); sid=j.session_id; document.getElementById('sid').innerText=sid; practice=j.practice; score=0; updatePractice(); sentenceOrder=j.sentence_order; sentenceIdx=0; updateSentenceUI(); }
function updatePractice(){ const el=document.getElementById('practice'); el.innerHTML=''; practice.forEach((p,i)=>{ const span=document.createElement('span'); span.style.padding='6px 8px'; span.style.borderRadius='8px'; span.style.background=(i===0?'#6dd3a2':'#f2f2f2'); span.style.fontWeight=700; span.innerText=p; el.appendChild(span); }); }

async function setMode(m){ mode=m; document.getElementById('modeLbl').innerText=m; document.querySelectorAll('.mode-btn').forEach(b=>b.classList.toggle('active', b.dataset.mode===m)); await fetch('/api/set_mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,mode:m})});
  // UI toggles
  document.getElementById('practiceCard').style.display = (m==='learning')? 'block' : 'none';
  document.getElementById('sampleCard').style.display = (m==='learning')? 'block' : 'none';
  document.getElementById('sentenceCard').style.display = (m==='test')? 'block' : 'none';
  document.getElementById('translateOutput').classList.toggle('hidden', m!=='translate');
}

document.querySelectorAll('.mode-btn').forEach(b=>b.addEventListener('click', ()=>{ setMode(b.dataset.mode); }));

async function startCamera(){ if(running) return; if(!sid) await createSession(); const v=document.getElementById('video'); try{ stream = await navigator.mediaDevices.getUserMedia({video:{width:640}}); v.srcObject=stream; running=true; captureInterval = setInterval(captureFrame, 1000/FPS); document.getElementById('help').innerText='Camera active'; }catch(e){ document.getElementById('help').innerText='Camera error: '+e.message; console.error(e);}}

document.getElementById('startBtn').addEventListener('click', ()=>startCamera());
document.getElementById('stopBtn').addEventListener('click', stopCamera);

document.getElementById('acceptBtn').addEventListener('click', ()=>handleAccept());
document.getElementById('rejectBtn').addEventListener('click', ()=>handleReject());
document.getElementById('restartBtn').addEventListener('click', ()=>handleRestart());

function stopCamera(){ if(!running) return; running=false; if(captureInterval) clearInterval(captureInterval); if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; } document.getElementById('help').innerText='Camera stopped'; }

async function captureFrame(){ const v=document.getElementById('video'); if(!v || v.readyState<2) return; const c=document.createElement('canvas'); c.width=640; c.height=480; const ctx=c.getContext('2d'); ctx.drawImage(v,0,0,c.width,c.height); c.toBlob(async function(blob){ const b64 = await toBase64(blob); await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,mode:mode,image:b64})}).then(r=>r.json()).then(updateFromServer).catch(e=>console.error(e)); }, 'image/jpeg', 0.7); }

function toBase64(blob){ return new Promise(res=>{ const reader=new FileReader(); reader.onloadend=()=>res(reader.result.split(',')[1]); reader.readAsDataURL(blob); }); }

function updateFromServer(j){ if(j.letter) detEl.innerText=j.letter; if(j.candidates) candsEl.innerText = (j.candidates||[]).map(x=>x[0]+':'+Math.round(x[1]*100)+'%').join('  '); if(j.sample_url) { sampleImg.src=j.sample_url; document.getElementById('sampleLbl').innerText='Target: '+j.target; } else { sampleImg.src=''; document.getElementById('sampleLbl').innerText='Target: '+j.target; }
  if(j.updated) { score=j.score; document.getElementById('score').innerText=score; // show encouragement
    showMessage(j.message || 'Good!'); }
  if(j.prog_pct!==undefined) progEl.style.width=j.prog_pct+'%';
  if(mode==='test' && j.sentence){ if(j.sentence_removed){ displaySentence(j.next_sentence,true); } else { displaySentence(j.sentence,false); } }
  if(mode==='translate'){
    document.getElementById('rawLarge').innerText = 'raw: ' + (j.progress_text || '');
    document.getElementById('finalLarge').innerText = (j.final_text || '-') ;
    if(j.pending_suggestion){ pendingSuggestion = j.pending_suggestion; document.getElementById('msgBox').innerText = 'Suggestion: '+pendingSuggestion; }
  }
}

function showMessage(txt){ const el=document.getElementById('msgBox'); el.innerText = txt; setTimeout(()=>{ if(el.innerText===txt) el.innerText=''; },1500); }

// sentence animation / display
function displaySentence(txt, removed){ const item=document.getElementById('sentenceItem'); if(removed){ item.style.transform='translateX(-120%)'; item.style.opacity=0; setTimeout(()=>{ item.style.transition='none'; item.style.transform='translateX(100%)'; item.innerText=txt; setTimeout(()=>{ item.style.transition='transform 0.45s ease, opacity 0.45s ease'; item.style.transform='translateX(0%)'; item.style.opacity=1; },50); },480); } else { item.innerText=txt; item.style.transform='translateX(0%)'; item.style.opacity=1; } }

async function handleAccept(){ if(!sid) return; await fetch('/api/accept',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid})}).then(r=>r.json()).then(j=>{ if(j.ok) showMessage('Accepted'); }); }
async function handleReject(){ if(!sid) return; await fetch('/api/reject',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid})}).then(r=>r.json()).then(j=>{ if(j.ok) showMessage('Rejected'); }); }
async function handleRestart(){ if(!sid) return; await fetch('/api/restart',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid})}).then(r=>r.json()).then(j=>{ if(j.ok) showMessage('Restarted'); document.getElementById('finalLarge').innerText='-'; document.getElementById('rawLarge').innerText='raw: -'; }); }

function updateSentenceUI(){ if(!sentenceOrder || sentenceOrder.length===0) return; document.getElementById('sentenceItem').innerText = sentenceOrder[sentenceIdx]; }

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
    random.shuffle(practice)
    # prepare shuffled sentence order (non-repeating until exhausted)
    sentence_order = SENTENCES.copy()
    random.shuffle(sentence_order)
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
        'mode': 'learning',
        'pending_suggestion': None,
        'pending_raw': None,
        'sentence_order': sentence_order,
        'sentence_idx': 0,
    }
    return jsonify({'session_id': sid, 'practice': sessions[sid]['practice'], 'sentence_order': sessions[sid]['sentence_order']})

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
    s['pending_suggestion'] = None
    s['pending_raw'] = None
    # when entering test mode, ensure sentence_idx valid
    if 'sentence_order' not in s or not s['sentence_order']:
        s['sentence_order'] = SENTENCES.copy()
        random.shuffle(s['sentence_order'])
        s['sentence_idx'] = 0
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

    use_v3 = (mode == 'translate') and (gesture3 is not None)

    if results.multi_hand_landmarks:
        s['no_hand_frames'] = 0
        lm = results.multi_hand_landmarks[0]
        thumb = None
        if use_v3:
            out = gesture3.predict_letter(lm.landmark)
            # try detect_thumb_gesture if available
            if hasattr(gesture3, 'detect_thumb_gesture'):
                try:
                    thumb = gesture3.detect_thumb_gesture(lm.landmark)
                except Exception:
                    thumb = None
        else:
            out = gesture.predict_letter(lm.landmark)
        letter = out.get('letter', '?')
        confidence = out.get('confidence', 0.0)
        candidates = out.get('candidates', [])
        now = time.time()

        # If translate mode and thumb detected by gesture3, handle accept/reject and DO NOT append thumb to raw_seq
        if mode == 'translate' and thumb in ('up', 'down'):
            if thumb == 'up':
                if s.get('pending_suggestion'):
                    s['final_words'].append(s['pending_suggestion'])
                    s['pending_suggestion'] = None
                    s['pending_raw'] = None
                    message = 'Suggestion accepted'
            elif thumb == 'down':
                if s.get('pending_suggestion'):
                    s['pending_suggestion'] = None
                    s['pending_raw'] = None
                    message = 'Suggestion rejected'
            # do not run stable-letter confirmation for thumbs
        else:
            # normal stable-letter confirmation
            if letter == s['stable_letter']:
                if s['stable_start'] is None:
                    s['stable_start'] = now
                elapsed = now - s['stable_start']
                prog_pct = min(100, int(elapsed / CONFIRM_TIME * 100))
                if (elapsed >= CONFIRM_TIME and confidence >= CONF_THRESHOLD and letter != '?' and letter != s['last_committed']):
                    s['last_committed'] = letter
                    if mode == 'learning':
                        target = s['target']
                        if letter.upper() == target.upper():
                            s['score'] += 1
                            s['pos'] += 1
                            if s['pos'] < len(s['practice']):
                                s['target'] = s['practice'][s['pos']]
                            else:
                                s['target'] = 'DONE'
                            message = random.choice(ENCOURAGE)
                            updated = True
                        else:
                            message = f"Detected {letter}, expect {target}"
                    elif mode == 'test':
                        # compare against current sentence
                        idx = s.get('sentence_idx', 0)
                        if idx < len(s['sentence_order']):
                            sentence = s['sentence_order'][idx]
                            chars = ''.join(c for c in sentence if c.isalpha())
                            expected = chars[s['pos']] if s['pos'] < len(chars) else None
                            if expected and letter.upper() == expected.upper():
                                s['pos'] += 1
                                if s['pos'] >= len(chars):
                                    # completed sentence: advance index and indicate removal
                                    s['pos'] = 0
                                    s['sentence_idx'] += 1
                                    if s['sentence_idx'] >= len(s['sentence_order']):
                                        # reshuffle when exhausted
                                        s['sentence_idx'] = 0
                                        random.shuffle(s['sentence_order'])
                                    message = 'Sentence completed'
                                    updated = True
                                    # we'll include next sentence in response below
                                else:
                                    message = 'Correct letter'
                            else:
                                message = f"Detected {letter}, expect {expected}"
                    else:  # translate mode (non-thumb)
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
            s['pending_suggestion'] = word
            s['pending_raw'] = list(s['raw_seq'])
            s['raw_seq'].clear()
            message = f"Suggestion: {word} (thumb up to accept, down to reject)"

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

    # include sample image URL for learning mode (try png/jpg/jpeg)
    if mode == 'learning':
        target = s.get('target', 'A')
        for ext in ('.png', '.jpg', '.jpeg'):
            p = os.path.join(LETTER_DIR, f"{target}{ext}")
            if os.path.exists(p):
                resp['sample_url'] = f"/letter/{target}{ext}"
                break
        else:
            resp['sample_url'] = ''

    # include sentence for test mode and whether a sentence was completed
    if mode == 'test':
        idx = s.get('sentence_idx', 0)
        current = s['sentence_order'][idx] if idx < len(s['sentence_order']) else ''
        resp['sentence'] = current
        # if updated and message indicates completion, provide next sentence
        if updated and message == 'Sentence completed':
            # next index already advanced above; provide next sentence to frontend
            next_idx = s.get('sentence_idx', 0)
            next_sentence = s['sentence_order'][next_idx] if next_idx < len(s['sentence_order']) else ''
            resp['sentence_removed'] = True
            resp['next_sentence'] = next_sentence
        else:
            resp['sentence_removed'] = False

    # include pending suggestion if any (translate)
    if s.get('pending_suggestion'):
        resp['pending_suggestion'] = s['pending_suggestion']

    return jsonify(resp)

@app.route('/api/accept', methods=['POST'])
def api_accept():
    data = request.get_json()
    sid = data.get('session_id')
    s = sessions.get(sid)
    if not s:
        return jsonify({'error': 'invalid session'}), 400
    if s.get('pending_suggestion'):
        s['final_words'].append(s['pending_suggestion'])
        s['pending_suggestion'] = None
        s['pending_raw'] = None
        return jsonify({'ok': True})
    return jsonify({'ok': False, 'reason': 'no pending suggestion'})

@app.route('/api/reject', methods=['POST'])
def api_reject():
    data = request.get_json()
    sid = data.get('session_id')
    s = sessions.get(sid)
    if not s:
        return jsonify({'error': 'invalid session'}), 400
    s['pending_suggestion'] = None
    s['pending_raw'] = None
    return jsonify({'ok': True})

@app.route('/api/restart', methods=['POST'])
def api_restart():
    data = request.get_json()
    sid = data.get('session_id')
    s = sessions.get(sid)
    if not s:
        return jsonify({'error': 'invalid session'}), 400
    s['raw_seq'].clear()
    s['final_words'].clear()
    s['pending_suggestion'] = None
    s['pending_raw'] = None
    return jsonify({'ok': True})

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
