"""
text.py

Web-based Test Mode for ASL fingerspelling.
- Browser captures webcam and POSTs frames to /predict
- Server provides a sentence and checks sequential letter correctness
- Minimal UI similar to learning.py (no sample images)

Run: python text.py (opens web UI at http://localhost:5001/text)
"""

from flask import Flask, request, jsonify, render_template_string
import time
import uuid
import base64
import random
import os
import cv2
import numpy as np
import mediapipe as mp
import gesture

PORT = 5001
CONFIRM_TIME = 1.4
CONF_THRESHOLD = 0.55

app = Flask(__name__)
mp_hands = mp.solutions.hands.Hands(model_complexity=1, max_num_hands=1,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

SENTENCES = [
    "HELLO",
    "GOOD MORNING",
    "HOW ARE YOU",
    "I LOVE CODING",
    "THANK YOU",
]

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ASL Test Mode</title>
  <style>body{background:#111;color:#eee;font-family:Arial} #wrap{display:flex;gap:16px;padding:16px} .panel{background:#222;padding:12px;border-radius:8px}</style>
</head>
<body>
  <h2>ASL 测试模式 (Web)</h2>
  <div id="wrap">
    <div class="panel"><video id="video" autoplay playsinline width=480 height=360></video><canvas id="canvas" width=480 height=360 style="display:none"></canvas><div><button id="start">开始</button><button id="stop">停止</button></div></div>
    <div class="panel"><div>句子: <b id="sentence"></b></div><div>进度: <b id="progress"></b></div><div>最近识别: <span id="det">-</span></div><div id="msg"></div></div>
  </div>
<script>
let sid=null; let running=false; const FPS=6;
async function newSession(){ const r=await fetch('/session',{method:'POST'}); const j=await r.json(); sid=j.session_id; document.getElementById('sentence').innerText=j.sentence; document.getElementById('progress').innerText=j.progress; }
async function sendFrame(blob){ if(!sid) return; const b64=await toBase64(blob); await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,image:b64})}).then(r=>r.json()).then(j=>{ document.getElementById('det').innerText=j.letter+' ('+Math.round(j.confidence*100)+'%)'; document.getElementById('progress').innerText=j.progress; if(j.msg) document.getElementById('msg').innerText=j.msg; if(j.completed) document.getElementById('sentence').innerText='Completed - press space to next'; }).catch(e=>console.log(e)); }
function toBase64(blob){return new Promise(res=>{const reader=new FileReader();reader.onloadend=()=>res(reader.result.split(',')[1]);reader.readAsDataURL(blob);});}
async function startCamera(){ if(running) return; if(!sid) await newSession(); const v=document.getElementById('video'); const c=document.getElementById('canvas'); const ctx=c.getContext('2d'); const stream=await navigator.mediaDevices.getUserMedia({video:{width:480}}); v.srcObject=stream; running=true; const interval=1000/FPS; async function loop(){ if(!running) return; ctx.drawImage(v,0,0,c.width,c.height); c.toBlob(async function(blob){ await sendFrame(blob); },'image/jpeg',0.7); setTimeout(loop,interval); } loop(); }
function stopCamera(){ running=false; const v=document.getElementById('video'); if(v.srcObject){ v.srcObject.getTracks().forEach(t=>t.stop()); v.srcObject=null; }}
document.getElementById('start').addEventListener('click',()=>startCamera()); document.getElementById('stop').addEventListener('click',()=>stopCamera()); window.onload=()=>newSession(); document.addEventListener('keydown', async (e)=>{ if(e.code==='Space'){ await fetch('/next',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid})}).then(()=>newSession()); } if(e.key==='b'){ await fetch('/back',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid})}).then(r=>r.json()).then(j=>{ document.getElementById('progress').innerText=j.progress; }); }});
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


sessions = {}

@app.route('/session', methods=['POST'])
def new_session():
    sid = str(uuid.uuid4())
    sentence = random.choice(SENTENCES)
    sessions[sid] = {
        'sentence': sentence,
        'chars': sentence.replace(' ', ''),
        'pos': 0,
        'stable_letter': None,
        'stable_start': None,
        'last_committed': None,
    }
    return jsonify({'session_id': sid, 'sentence': sentence, 'progress': sessions[sid]['chars'][:0] + '|' + sessions[sid]['chars'][0:]})


@app.route('/state')
def state():
    sid = request.args.get('session_id')
    s = sessions.get(sid)
    if not s:
        return jsonify({'error': 'no session'}), 400
    return jsonify({'sentence': s['sentence'], 'progress': s['chars'][:s['pos']] + '|' + s['chars'][s['pos']:]})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sid = data.get('session_id')
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
    now = time.time()
    completed = False
    msg = ''

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        out = gesture.predict_letter(lm.landmark)
        letter = out['letter']
        confidence = out['confidence']
        if letter == s['stable_letter']:
            if s['stable_start'] is None:
                s['stable_start'] = now
            elapsed = now - s['stable_start']
            if elapsed >= CONFIRM_TIME and confidence >= CONF_THRESHOLD and letter != '?' and letter != s['last_committed']:
                s['last_committed'] = letter
                expected = s['chars'][s['pos']] if s['pos'] < len(s['chars']) else None
                if expected and letter.upper() == expected.upper():
                    s['pos'] += 1
                    msg = '正确'
                else:
                    msg = f'识别 {letter}，期望 {expected}'
                s['stable_start'] = None
        else:
            s['stable_letter'] = letter
            s['stable_start'] = now
    else:
        s['stable_letter'] = None
        s['stable_start'] = None
        s['last_committed'] = None

    if s['pos'] >= len(s['chars']):
        completed = True

    return jsonify({'letter': letter, 'confidence': confidence, 'progress': s['chars'][:s['pos']] + '|' + s['chars'][s['pos']:], 'msg': msg, 'completed': completed})


@app.route('/next', methods=['POST'])
def next_sentence():
    data = request.get_json()
    sid = data.get('session_id')
    s = sessions.get(sid)
    if not s:
        return jsonify({'error': 'invalid session'}), 400
    sentence = random.choice(SENTENCES)
    s['sentence'] = sentence
    s['chars'] = sentence.replace(' ', '')
    s['pos'] = 0
    return jsonify({'ok': True})


@app.route('/back', methods=['POST'])
def back():
    data = request.get_json()
    sid = data.get('session_id')
    s = sessions.get(sid)
    if not s:
        return jsonify({'error': 'invalid session'}), 400
    s['pos'] = max(0, s['pos'] - 1)
    return jsonify({'progress': s['chars'][:s['pos']] + '|' + s['chars'][s['pos']:']})


def main():
    print(f"Running Test Mode web server on http://localhost:{PORT}/")
    app.run(host='0.0.0.0', port=PORT, debug=False)


if __name__ == '__main__':
    main()
