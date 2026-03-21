"""
SynapxeAI Dashboard — Flask + SocketIO real-time dashboard
Reads from agent_log.jsonl and activity_log.jsonl
Run alongside main.py and activity_monitor.py
"""
from flask import Flask, render_template_string
from flask_socketio import SocketIO
import json
import os
import time
import threading
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'synapxe_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

AGENT_LOG    = "agent_log.jsonl"
ACTIVITY_LOG = "activity_log.jsonl"
BASELINE     = "activity_baseline.json"
PATIENT      = "patient_profile.json"

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SynapxeAI — Patient Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  :root {
    --bg:       #080c0f;
    --surface:  #0d1419;
    --border:   #1a2530;
    --text:     #c8d8e0;
    --muted:    #4a6070;
    --accent:   #00e5cc;
    --danger:   #ff3a3a;
    --warn:     #ff9500;
    --safe:     #00e5a0;
    --font-display: 'Syne', sans-serif;
    --font-mono:    'DM Mono', monospace;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Scanline overlay */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,229,204,0.015) 2px,
      rgba(0,229,204,0.015) 4px
    );
    pointer-events: none;
    z-index: 1000;
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 32px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .logo {
    font-family: var(--font-display);
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #fff;
  }
  .logo span { color: var(--accent); }
  .status-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--muted);
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .pulse {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--safe);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(1.3); }
  }

  .grid {
    display: grid;
    grid-template-columns: 280px 1fr 320px;
    grid-template-rows: auto 1fr auto;
    gap: 1px;
    background: var(--border);
    min-height: calc(100vh - 65px);
  }

  .panel {
    background: var(--bg);
    padding: 24px;
  }

  /* Patient card */
  .patient-card {
    grid-row: 1 / 2;
    border-right: 1px solid var(--border);
  }
  .panel-label {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
  }
  .patient-name {
    font-family: var(--font-display);
    font-size: 20px;
    font-weight: 600;
    color: #fff;
    margin-bottom: 4px;
  }
  .patient-meta {
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 16px;
  }
  .tag {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 3px;
    font-size: 11px;
    background: rgba(0,229,204,0.08);
    color: var(--accent);
    border: 1px solid rgba(0,229,204,0.2);
    margin: 2px;
  }

  /* Risk meter */
  .risk-panel {
    grid-column: 2;
    grid-row: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    background: var(--surface);
  }
  .risk-label {
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
  }
  .risk-value {
    font-family: var(--font-display);
    font-size: 72px;
    font-weight: 800;
    line-height: 1;
    transition: color 0.5s;
  }
  .risk-bar-wrap {
    width: 300px;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }
  .risk-bar {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s, background 0.5s;
  }
  .risk-status {
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 500;
  }

  /* Alert feed */
  .alert-panel {
    grid-column: 3;
    grid-row: 1 / 4;
    border-left: 1px solid var(--border);
    overflow-y: auto;
    max-height: calc(100vh - 65px);
  }
  .alert-item {
    padding: 14px 0;
    border-bottom: 1px solid var(--border);
    animation: slideIn 0.3s ease;
  }
  @keyframes slideIn {
    from { opacity:0; transform:translateX(10px); }
    to   { opacity:1; transform:translateX(0); }
  }
  .alert-time {
    font-size: 10px;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .alert-action {
    font-size: 13px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .alert-conf {
    font-size: 11px;
    color: var(--muted);
    margin-top: 2px;
  }
  .action-notify   { color: var(--danger); }
  .action-escalate { color: #ff6060; }
  .action-recovery { color: var(--safe); }
  .action-wait     { color: var(--muted); }
  .action-ping     { color: var(--warn); }
  .action-getting_up { color: var(--accent); }

  /* Activity chart */
  .chart-panel {
    grid-column: 1 / 3;
    grid-row: 2;
    border-top: 1px solid var(--border);
  }
  .chart-wrap {
    height: 180px;
    position: relative;
  }

  /* Stats row */
  .stats-panel {
    grid-column: 1 / 3;
    grid-row: 3;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 0;
  }
  .stat {
    flex: 1;
    padding: 16px 24px;
    border-right: 1px solid var(--border);
  }
  .stat:last-child { border-right: none; }
  .stat-value {
    font-family: var(--font-display);
    font-size: 28px;
    font-weight: 700;
    color: #fff;
  }
  .stat-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 2px;
  }

  /* Critical flash */
  body.critical-state {
    animation: critFlash 0.5s ease 3;
  }
  @keyframes critFlash {
    0%,100% { background: var(--bg); }
    50%      { background: #1a0000; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">Synapxe<span>AI</span></div>
  <div style="font-family:var(--font-display);font-size:13px;color:var(--muted)" id="patient-header">Loading...</div>
  <div class="status-pill">
    <div class="pulse" id="status-dot"></div>
    <span id="status-text">MONITORING</span>
  </div>
</header>

<div class="grid">

  <!-- Patient card -->
  <div class="panel patient-card">
    <div class="panel-label">Patient Profile</div>
    <div class="patient-name" id="p-name">—</div>
    <div class="patient-meta" id="p-meta">—</div>
    <div id="p-conditions"></div>
    <div style="margin-top:16px">
      <div class="panel-label">Fall History</div>
      <div style="font-size:12px;color:var(--text);line-height:1.6" id="p-history">—</div>
    </div>
    <div style="margin-top:16px">
      <div class="panel-label">Mobility</div>
      <div style="font-size:12px;color:var(--text)" id="p-mobility">—</div>
    </div>
    <div style="margin-top:16px">
      <div class="panel-label">Last Updated</div>
      <div style="font-size:12px;color:var(--muted)" id="last-update">—</div>
    </div>
  </div>

  <!-- Risk meter -->
  <div class="panel risk-panel">
    <div class="risk-label">Current Risk Level</div>
    <div class="risk-value" id="risk-pct" style="color:var(--safe)">0%</div>
    <div class="risk-bar-wrap">
      <div class="risk-bar" id="risk-bar" style="width:0%;background:var(--safe)"></div>
    </div>
    <div class="risk-status" id="risk-status" style="color:var(--safe)">LOW</div>
    <div style="font-size:11px;color:var(--muted);margin-top:4px" id="last-action">Waiting for data...</div>
  </div>

  <!-- Alert feed -->
  <div class="panel alert-panel">
    <div class="panel-label">Alert Feed</div>
    <div id="alert-list"></div>
  </div>

  <!-- Activity chart -->
  <div class="panel chart-panel">
    <div class="panel-label" style="margin-bottom:12px">Activity Timeline (last 30 mins)</div>
    <div class="chart-wrap">
      <canvas id="activityChart"></canvas>
    </div>
  </div>

  <!-- Stats -->
  <div class="stats-panel">
    <div class="stat">
      <div class="stat-value" id="stat-alerts">0</div>
      <div class="stat-label">Total Alerts</div>
    </div>
    <div class="stat">
      <div class="stat-value" id="stat-falls">0</div>
      <div class="stat-label">Falls Detected</div>
    </div>
    <div class="stat">
      <div class="stat-value" id="stat-recoveries">0</div>
      <div class="stat-label">Recoveries</div>
    </div>
    <div class="stat">
      <div class="stat-value" id="stat-uptime">0m</div>
      <div class="stat-label">Monitor Uptime</div>
    </div>
    <div class="stat">
      <div class="stat-value" id="stat-days">0</div>
      <div class="stat-label">Days Learned</div>
    </div>
  </div>

</div>

<script>
const socket = io();
const startTime = Date.now();

// Activity chart
const ctx = document.getElementById('activityChart').getContext('2d');
const actChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Activity',
      data: [],
      borderColor: '#00e5cc',
      backgroundColor: 'rgba(0,229,204,0.05)',
      borderWidth: 1.5,
      pointRadius: 2,
      fill: true,
      tension: 0.4
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color:'#4a6070', font:{size:10} }, grid:{color:'#1a2530'} },
      y: { min:0, max:1, ticks:{color:'#4a6070',font:{size:10}}, grid:{color:'#1a2530'} }
    }
  }
});

// Update patient info
socket.on('patient', data => {
  document.getElementById('p-name').textContent = data.name;
  document.getElementById('p-meta').textContent = `${data.age} • ${data.gender}`;
  document.getElementById('patient-header').textContent = data.name;
  document.getElementById('p-history').textContent = data.fall_history;
  document.getElementById('p-mobility').textContent = data.mobility;
  const cDiv = document.getElementById('p-conditions');
  cDiv.innerHTML = (data.conditions||[]).map(c => `<span class="tag">${c}</span>`).join('');
});

// Update risk
socket.on('risk_update', data => {
  const pct   = Math.round(data.confidence * 100);
  const level = data.level;
  const color = level === 'CRITICAL' ? 'var(--danger)' :
                level === 'HIGH'     ? 'var(--warn)'   : 'var(--safe)';

  document.getElementById('risk-pct').textContent  = pct + '%';
  document.getElementById('risk-pct').style.color  = color;
  document.getElementById('risk-bar').style.width  = pct + '%';
  document.getElementById('risk-bar').style.background = color;
  document.getElementById('risk-status').textContent   = level;
  document.getElementById('risk-status').style.color   = color;
  document.getElementById('last-update').textContent   = 'Updated ' + new Date().toLocaleTimeString();
  document.getElementById('last-action').textContent   = 'Action: ' + data.action.toUpperCase();

  if (level === 'CRITICAL') document.body.classList.add('critical-state');
  else document.body.classList.remove('critical-state');

  // Update dot
  document.getElementById('status-dot').style.background = color;
});

// New alert
socket.on('new_alert', data => {
  const list = document.getElementById('alert-list');
  const cls  = `action-${data.action.replace('_','-')}`;
  const div  = document.createElement('div');
  div.className = 'alert-item';
  div.innerHTML = `
    <div class="alert-time">${data.time}</div>
    <div class="alert-action ${cls}">${data.action.replace(/_/g,' ')}</div>
    <div class="alert-conf">conf: ${(data.confidence*100).toFixed(0)}% · ${data.level}</div>
  `;
  list.insertBefore(div, list.firstChild);
  if (list.children.length > 50) list.removeChild(list.lastChild);
});

// Activity update
socket.on('activity_update', data => {
  const labels = actChart.data.labels;
  const values = actChart.data.datasets[0].data;
  labels.push(data.time);
  values.push(data.activity);
  if (labels.length > 60) { labels.shift(); values.shift(); }
  actChart.update();
});

// Stats
socket.on('stats', data => {
  document.getElementById('stat-alerts').textContent     = data.total_alerts;
  document.getElementById('stat-falls').textContent      = data.falls;
  document.getElementById('stat-recoveries').textContent = data.recoveries;
  document.getElementById('stat-days').textContent       = data.days_learned;
  const mins = Math.floor((Date.now() - startTime) / 60000);
  document.getElementById('stat-uptime').textContent     = mins + 'm';
});
</script>
</body>
</html>
"""

def load_patient():
    try:
        with open(PATIENT) as f:
            return json.load(f)
    except:
        return {}

def get_stats():
    total = falls = recoveries = 0
    if os.path.exists(AGENT_LOG):
        with open(AGENT_LOG) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    total += 1
                    if e.get('action') == 'notify':    falls += 1
                    if e.get('action') == 'recovery_notify': recoveries += 1
                except: pass
    days = 0
    if os.path.exists(ACTIVITY_LOG):
        dates = set()
        with open(ACTIVITY_LOG) as f:
            for line in f:
                try: dates.add(json.loads(line).get('date',''))
                except: pass
        days = len(dates)
    return {'total_alerts': total, 'falls': falls,
            'recoveries': recoveries, 'days_learned': days}

def tail_agent_log(last_pos):
    entries = []
    if not os.path.exists(AGENT_LOG):
        return entries, last_pos
    with open(AGENT_LOG) as f:
        f.seek(last_pos)
        for line in f:
            try:
                e = json.loads(line)
                entries.append({
                    'time':       datetime.fromtimestamp(e['timestamp']).strftime('%H:%M:%S'),
                    'action':     e.get('action','wait'),
                    'confidence': e.get('confidence', 0),
                    'level':      e.get('level','LOW')
                })
            except: pass
        last_pos = f.tell()
    return entries, last_pos

def tail_activity_log(last_pos):
    entries = []
    if not os.path.exists(ACTIVITY_LOG):
        return entries, last_pos
    with open(ACTIVITY_LOG) as f:
        f.seek(last_pos)
        for line in f:
            try:
                e = json.loads(line)
                entries.append({
                    'time':     f"{e['hour']:02d}:{e['minute']:02d}",
                    'activity': e.get('activity', 0)
                })
            except: pass
        last_pos = f.tell()
    return entries, last_pos

def background_thread():
    agent_pos    = 0
    activity_pos = 0
    while True:
        # Patient info
        socketio.emit('patient', load_patient())

        # Agent log
        entries, agent_pos = tail_agent_log(agent_pos)
        for e in entries:
            socketio.emit('new_alert', e)
            level = 'CRITICAL' if e['action'] in ['notify','escalate'] else 'LOW'
            socketio.emit('risk_update', {
                'confidence': e['confidence'],
                'level':      level,
                'action':     e['action']
            })

        # Activity log
        act_entries, activity_pos = tail_activity_log(activity_pos)
        for e in act_entries:
            socketio.emit('activity_update', e)

        # Live risk from main.py
        try:
            with open("live_risk.json") as f:
                live = json.load(f)
            if time.time() - live.get("timestamp", 0) < 5:
                socketio.emit("risk_update", {
                    "confidence": live["confidence"],
                    "level": live["level"],
                    "action": live.get("action", "wait")
                })
        except: pass

        # Stats
        socketio.emit('stats', get_stats())
        time.sleep(2)


@app.route('/')
def index():
    return render_template_string(HTML)

@socketio.on('connect')
def on_connect():
    # Send existing alert history on connect
    entries, _ = tail_agent_log(0)
    for e in entries[-30:]:
        socketio.emit('new_alert', e)
    socketio.emit('patient', load_patient())
    socketio.emit('stats', get_stats())

if __name__ == '__main__':
    thread = threading.Thread(target=background_thread, daemon=True)
    thread.start()
    print("\n[DASHBOARD] Running at http://localhost:5050\n")
    socketio.run(app, host='0.0.0.0', port=5050, debug=False, allow_unsafe_werkzeug=True)
