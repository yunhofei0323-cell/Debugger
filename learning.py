"""
learning.py

Web-based Learning Mode for ASL fingerspelling.
- Uses Flask to serve a simple web frontend that captures webcam frames in the browser
  and POSTs them to /predict as base64 JPEG data.
- Server uses MediaPipe Hands + gesture.predict_letter to classify the handshape.
- Maintains simple per-session state (target letter, score, practice table, stable confirmation)
- When learner holds the correct letter for CONFIRM_TIME and above CONF_THRESHOLD, score increments
  and we move to the next letter in the practice table.

Run: python learning.py  (opens web UI at http://localhost:5001)
Dependencies: flask, opencv-python, mediapipe, numpy
"""

from flask import Flask, request, jsonify, render_template_string
import time
import uuid
import base64
import io
import os
import random
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import gesture

# Config
PORT = 5001
CONFIRM_TIME = 1.4
CONF_THRESHOLD = 0.55
PRACTICE_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
COMMON_WORDS = ["HELLO", "THANK YOU", "PLEASE", "YES", "NO", "GOOD", "MORNING"]

app = Flask(__name__)

# In-memory sessions: not persistent. Keyed by session_id
sessions = {}

mp_hands = mp.solutions.hands.Hands(model_complexity=1, max_num_hands=1,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ASL Learning Mode</title>
  <style>
    body { background:#111; color:#eee; font-family: Arial, Helvetica, sans-serif; }
    #container { display:flex; gap:20px; padding:20px }
    #videoWrap, #info { background:#222; padding:12px; border-radius:8px }
    canvas { background: #000 }
    button { padding:8px 12px; margin:6px }
    .letter { font-size:72px; color:#5EEAD4 }
  </style>
</head>
<body>
  <h2>ASL 学习模式 (Web)</h2>
  <div id="container">
    <div id="videoWrap">
      <video id="video" autoplay playsinline width=480 height=360></video>
      <canvas id="canvas" width=480 height=360 style="display:none"></canvas>
      <div>
        <button id="start">开始</button>
        <button id="stop">停止</button>
      </div>
    </div>
    <div id="info">
      <div>会话: <span id="sid"></span></div>
      <div>目标字母: <span class="letter" id="target">A</span></div>
      <div>得分: <span id="score">0</span></div>
      <div>练习表:</div>
      <div id="table"></div>
      <div style="margin-top:12px">最近识别: <span id="det">-</span></div>
      <div id="enc"> </div>
    </div>
  </div>

<script>
let sid = null;
let running = false;
const FPS = 6; // frames per second to upload

async function startSession(){
  const r = await fetch('/session', {method:'POST'});
  const j = await r.json(); sid = j.session_id; document.getElementById('sid').innerText = sid;
  updateInfo();
}

async function updateInfo(){
  if(!sid) return;
  const r = await fetch('/state?session_id='+sid);
  const j = await r.json();
  document.getElementById('target').innerText = j.target;
  document.getElementById('score').innerText = j.score;
  document.getElementById('table').innerText = j.practice.join(' ');
}

async function sendFrame(blob){
  if(!sid) return;
  const b64 = await toBase64(blob);
  await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({session_id: sid, image: b64})})
    .then(r=>r.json()).then(j=>{
      document.getElementById('det').innerText = j.letter + ' (' + Math.round(j.confidence*100) + '%)';
      if(j.message) document.getElementById('enc').innerText = j.message;
      if(j.updated) updateInfo();
    }).catch(e=>console.log(e));
}

function toBase64(blob){
  return new Promise(res=>{
    const reader = new FileReader(); reader.onloadend = ()=>res(reader.result.split(',')[1]); reader.readAsDataURL(blob);
  });
}

async function startCamera(){
  if(running) return;
  if(!sid) await startSession();
  const v = document.getElementById('video');
  const c = document.getElementById('canvas');
  const ctx = c.getContext('2d');
  const stream = await navigator.mediaDevices.getUserMedia({video:{width:480}});
  v.srcObject = stream;
  running = true;
  const interval = 1000 / FPS;
  async function loop(){
    if(!running) return;
    ctx.drawImage(v, 0, 0, c.width, c.height);
    c.toBlob(async function(blob){ await sendFrame(blob); }, 'image/jpeg', 0.7);
    setTimeout(loop, interval);
  }
  loop();
}

function stopCamera(){ running=false; const v=document.getElementById('video'); if(v.srcObject){ v.srcObject.getTracks().forEach(t=>t.stop()); v.srcObject=null; }}

document.getElementById('start').addEventListener('click', ()=>startCamera());
document.getElementById('stop').addEventListener('click', ()=>stopCamera());

// init
window.onload = ()=>{ startSession(); setInterval(()=>updateInfo(), 2000); };
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


@app.route('/session', methods=['POST'])
def new_session():
    sid = str(uuid.uuid4())
    # practice order: full alphabet shuffled for this session
    practice = PRACTICE_LETTERS.copy()
    random.shuffle(practice)
    sessions[sid] = {
        'practice': practice,
        'pos': 0,
        'score': 0,
        'target': practice[0] if practice else 'A',
        'stable_letter': None,
        'stable_start': None,
        'last_committed': None,
        'msg': '',
        'msg_until': 0,
    }
    return jsonify({'session_id': sid})


@app.route('/state')
def state():
    sid = request.args.get('session_id')
    s = sessions.get(sid)
    if not s:
        return jsonify({'error': 'no session'}), 400
    return jsonify({'practice': s['practice'], 'pos': s['pos'], 'score': s['score'], 'target': s['target']})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sid = data.get('session_id')
    b64 = data.get('image')
    s = sessions.get(sid)
    if not s:
        return jsonify({'error': 'invalid session'}), 400

    # decode image
    try:
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return jsonify({'error': 'bad image', 'e': str(e)}), 400

    # mediapipe
    results = mp_hands.process(frame_rgb)
    letter = '?'
    confidence = 0.0
    candidates = []
    updated = False
    message = ''

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        out = gesture.predict_letter(lm.landmark)
        letter = out['letter']
        confidence = out['confidence']
        candidates = out['candidates']
        now = time.time()
        # stable confirmation
        if letter == s['stable_letter']:
            if s['stable_start'] is None:
                s['stable_start'] = now
            elapsed = now - s['stable_start']
            if (elapsed >= CONFIRM_TIME and confidence >= CONF_THRESHOLD and letter != '?' and letter != s['last_committed']):
                s['last_committed'] = letter
                # check correctness
                target = s['target']
                if letter.upper() == target.upper():
                    s['score'] += 1
                    s['pos'] += 1
                    if s['pos'] < len(s['practice']):
                        s['target'] = s['practice'][s['pos']]
                    else:
                        s['target'] = 'DONE'
                    s['msg'] = '干得好！'
                    s['msg_until'] = time.time() + 1.5
                    updated = True
                else:
                    s['msg'] = f'识别为 {letter}，目标是 {target}。再试一次。'
                    s['msg_until'] = time.time() + 1.2
                s['stable_start'] = None
        else:
            s['stable_letter'] = letter
            s['stable_start'] = now
    else:
        s['stable_letter'] = None
        s['stable_start'] = None
        s['last_committed'] = None

    if s.get('msg') and time.time() < s.get('msg_until', 0):
        message = s['msg']
    else:
        message = ''

    return jsonify({'letter': letter, 'confidence': confidence, 'candidates': candidates, 'updated': updated, 'message': message})


def main():
    print(f"Running Learning Mode web server on http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)


if __name__ == '__main__':
    main()
