"""
Web 远程访问服务（Flask）。

提供给家属端（手机/电脑浏览器）的接口：
    GET  /                     单页应用首页（HTML + JS）
    GET  /stream               MJPEG 视频流（直接给 <img> 标签用）
    GET  /api/snapshot.jpg     单张 JPEG 当前帧
    GET  /api/events           Server-Sent Events 推送告警 / 日报等
    GET  /api/recent           最近事件历史（JSON）
    GET  /api/status           系统状态（在线 / 设备 / 订阅数等）

主程序通过 services.web_state 推送数据，本服务端只读不写。
"""
from __future__ import annotations

import json
import socket
import threading
import time
from typing import Optional

from services.web_state import get_state


# ============================================================
# 工具
# ============================================================
def get_local_ip() -> str:
    """尝试获取本机内网 IP（用于显示给家属端）。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 用一个不会真发包的"探测"找到出口网卡 IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        try:
            s.close()
        except Exception:
            pass
    return ip


# ============================================================
# 前端 HTML（嵌入式）
# ============================================================
INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>家属端 · 远程陪护视图</title>
<style>
  :root {
    --bg:#0d1117; --panel:#161b22; --panel2:#1c2128;
    --border:#30363d; --text:#e6edf3; --dim:#7d8590;
    --accent:#58a6ff; --danger:#f85149; --warn:#d29922;
    --success:#3fb950; --gold:#d2a8ff;
  }
  * { box-sizing:border-box; }
  html,body { margin:0; padding:0; background:var(--bg); color:var(--text);
              font-family:"Microsoft YaHei UI","Segoe UI",sans-serif; }
  header { background:var(--panel); border-bottom:1px solid var(--border);
           padding:12px 16px; display:flex; align-items:center; gap:12px; }
  header .dot { width:10px; height:10px; border-radius:50%;
                background:var(--success); box-shadow:0 0 8px var(--success);}
  header h1 { font-size:16px; margin:0; font-weight:600;}
  header .meta { color:var(--dim); font-size:12px; margin-left:auto;}
  main { padding:14px; display:grid; gap:14px;
         grid-template-columns: 1.7fr 1fr; }
  @media (max-width: 760px) { main { grid-template-columns: 1fr; } }
  .card { background:var(--panel); border:1px solid var(--border);
          border-radius:10px; padding:12px; }
  .card h2 { font-size:13px; margin:0 0 10px; color:var(--accent);
             font-weight:600; letter-spacing:0.04em;}
  #stream-wrap { position:relative; background:#000; border-radius:8px;
                 overflow:hidden; aspect-ratio:16/9; min-height:200px;}
  #stream { width:100%; height:100%; object-fit:contain; display:block;
            background:#000; }
  #stream-tag { position:absolute; top:8px; left:8px; padding:3px 8px;
                background:rgba(248,81,73,0.85); color:#fff; font-size:11px;
                border-radius:4px; font-weight:600; letter-spacing:0.05em;}
  .events { max-height:62vh; overflow:auto; padding-right:4px; }
  .event { padding:8px 10px; margin:6px 0; border-radius:6px;
           background:var(--panel2); border-left:3px solid var(--accent);
           font-size:13px; line-height:1.45;}
  .event.fall { border-left-color:var(--danger);
                background:rgba(248,81,73,0.08); }
  .event.report { border-left-color:var(--gold);
                  background:rgba(210,168,255,0.06);}
  .event.describe { border-left-color:var(--accent);}
  .event.status, .event.ping { display:none; }
  .event .time { color:var(--dim); font-size:11px; margin-bottom:2px;
                 letter-spacing:0.04em;}
  .event .title { font-weight:600; margin-bottom:4px; font-size:13px;
                  display:flex; align-items:center; gap:6px;}
  .event.fall .title { color:var(--danger);}
  .event.report .title { color:var(--gold);}
  .event .text { color:var(--text); white-space:pre-wrap; }
  .event .meta { color:var(--dim); font-size:11px; margin-top:4px;}
  footer { color:var(--dim); font-size:11px; padding:8px 16px;
           text-align:center; border-top:1px solid var(--border);
           background:var(--panel);}
</style>
</head>
<body>
<header>
  <span class="dot" id="dot"></span>
  <h1>家属端 · 远程陪护视图</h1>
  <span class="meta" id="meta">连接中...</span>
</header>
<main>
  <section>
    <div class="card">
      <h2>实时画面</h2>
      <div id="stream-wrap">
        <img id="stream" src="/stream"
             onerror="this.src='/api/snapshot.jpg?'+Date.now()">
        <span id="stream-tag">LIVE</span>
      </div>
    </div>
  </section>
  <section>
    <div class="card">
      <h2>事件与告警</h2>
      <div id="events" class="events">
        <div class="event status"><div class="time">--:--:--</div>
        <div class="title">连接中...</div></div>
      </div>
    </div>
  </section>
</main>
<footer>
  Azure Kinect Pose &amp; Gesture Suite · Web 端
</footer>
<script>
const eventsEl = document.getElementById('events');
const metaEl = document.getElementById('meta');
const dotEl = document.getElementById('dot');
let recentLoaded = false;

function fmt(ev) {
  const div = document.createElement('div');
  const cls = (ev.type || 'status').toLowerCase();
  div.className = 'event ' + cls;
  let title = ev.title || ev.type || '事件';
  let text = ev.text || '';
  let meta = ev.meta_text || '';
  let icon = '';
  if (cls === 'fall') icon = '⚠️';
  else if (cls === 'report') icon = '📋';
  else if (cls === 'describe') icon = '🤖';
  div.innerHTML =
    '<div class="time">' + (ev.time_str || '--') + '</div>' +
    '<div class="title">' + icon + ' ' + title + '</div>' +
    (text ? '<div class="text">' + text + '</div>' : '') +
    (meta ? '<div class="meta">' + meta + '</div>' : '');
  return div;
}

function appendEvent(ev) {
  if (!ev || !ev.type || ev.type === 'ping') return;
  const div = fmt(ev);
  eventsEl.insertBefore(div, eventsEl.firstChild);
  // 限制最多 50 条
  while (eventsEl.children.length > 50) {
    eventsEl.removeChild(eventsEl.lastChild);
  }
}

async function loadRecent() {
  try {
    const r = await fetch('/api/recent');
    const data = await r.json();
    eventsEl.innerHTML = '';
    (data.events || []).forEach(appendEvent);
    recentLoaded = true;
  } catch (e) { /* ignore */ }
}

function startSSE() {
  const es = new EventSource('/api/events');
  es.onmessage = (m) => {
    try {
      const ev = JSON.parse(m.data);
      appendEvent(ev);
    } catch (e) {}
    metaEl.textContent = '已连接 · ' + new Date().toLocaleTimeString();
    dotEl.style.background = '#3fb950';
  };
  es.onerror = () => {
    metaEl.textContent = '连接断开，正在重连...';
    dotEl.style.background = '#d29922';
  };
}

loadRecent().then(startSSE);

// 定期检查图像存活
setInterval(() => {
  const img = document.getElementById('stream');
  if (!img.naturalWidth) {
    // 尝试 fallback 到 snapshot
    img.src = '/api/snapshot.jpg?t=' + Date.now();
  }
}, 4000);
</script>
</body>
</html>
"""


# ============================================================
# Flask App
# ============================================================
def create_app():
    """构造 Flask app。延迟导入 flask 避免没装时主程序起不来。"""
    from flask import Flask, Response, jsonify

    app = Flask(__name__)
    state = get_state()

    @app.route("/")
    def index():
        return INDEX_HTML

    @app.route("/stream")
    def stream():
        def gen():
            last_seq = 0
            while True:
                jpeg, last_seq = state.wait_new_frame(last_seq, timeout=2.0)
                if jpeg is None:
                    # 服务还没收到第一帧，先发个占位
                    time.sleep(0.2)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )

        return Response(
            gen(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/snapshot.jpg")
    def snapshot():
        jpeg = state.get_latest_frame()
        if jpeg is None:
            return Response(status=503)
        return Response(jpeg, mimetype="image/jpeg")

    @app.route("/api/events")
    def events_sse():
        def gen():
            for ev in state.subscribe():
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

        resp = Response(gen(), mimetype="text/event-stream")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    @app.route("/api/recent")
    def recent():
        return jsonify({"events": state.recent_events(since_seconds=3600)})

    @app.route("/api/status")
    def status():
        meta = state.get_meta()
        return jsonify({
            "online": True,
            "subscribers": state.subscriber_count,
            **meta,
        })

    return app


# ============================================================
# 服务线程
# ============================================================
class WebServerThread(threading.Thread):
    """守护线程：运行 Flask 服务器。停止只能整个进程退出。"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        super().__init__(daemon=True, name="WebServerThread")
        self.host = host
        self.port = int(port)
        self._app = None
        self._url: Optional[str] = None
        self._error: Optional[str] = None

    @property
    def url(self) -> str:
        return self._url or ""

    @property
    def error(self) -> Optional[str]:
        return self._error

    def run(self):
        try:
            self._app = create_app()
            ip = get_local_ip()
            self._url = f"http://{ip}:{self.port}"
            # 用 werkzeug 默认 server，足够课设演示用
            # threaded=True 让多个客户端可以并发访问
            self._app.run(
                host=self.host, port=self.port,
                threaded=True, use_reloader=False, debug=False,
            )
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
