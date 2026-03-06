use std::sync::mpsc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use axum::Router;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PortalPassTiming {
    pub name: String,
    pub gpu_ms: f32,
    pub cpu_ms: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PortalFrameSnapshot {
    pub frame: u64,
    pub frame_time_ms: f32,
    pub frame_to_frame_ms: f32,
    pub total_gpu_ms: f32,
    pub total_cpu_ms: f32,
    pub pass_timings: Vec<PortalPassTiming>,
    pub pipeline_order: Vec<String>,
    pub timestamp_ms: u128,
}

pub struct LivePortalHandle {
    tx: mpsc::Sender<PortalFrameSnapshot>,
    pub url: String,
    _server_thread: std::thread::JoinHandle<()>,
}

impl LivePortalHandle {
    pub fn publish(&self, snapshot: PortalFrameSnapshot) {
        let _ = self.tx.send(snapshot);
    }
}

pub fn start_live_portal(bind_addr: &str) -> std::io::Result<LivePortalHandle> {
    let bind_addr = bind_addr.to_string();
    let url = format!("http://{}", bind_addr);

    let (tx, rx) = mpsc::channel::<PortalFrameSnapshot>();
    let (ready_tx, ready_rx) = mpsc::channel::<std::io::Result<()>>();

    let server_bind = bind_addr.clone();
    let server_thread = std::thread::Builder::new()
        .name("helio-live-portal-server".to_string())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = ready_tx.send(Err(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())));
                    return;
                }
            };

            runtime.block_on(async move {
                let (broadcast_tx, _) = broadcast::channel::<String>(256);

                let bridge_tx = broadcast_tx.clone();
                let bridge_thread = std::thread::Builder::new()
                    .name("helio-live-portal-bridge".to_string())
                    .spawn(move || {
                        while let Ok(snapshot) = rx.recv() {
                            if let Ok(json) = serde_json::to_string(&snapshot) {
                                let _ = bridge_tx.send(json);
                            }
                        }
                    });

                let app = Router::new()
                    .route("/", get(index))
                  .route("/favicon.ico", get(favicon))
                    .route("/ws", get(ws_upgrade))
                    .with_state(broadcast_tx);

                match tokio::net::TcpListener::bind(&server_bind).await {
                    Ok(listener) => {
                        let _ = ready_tx.send(Ok(()));
                        let _ = axum::serve(listener, app).await;
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                    }
                }

                if let Ok(handle) = bridge_thread {
                    let _ = handle.join();
                }
            });
        })?;

    match ready_rx.recv() {
        Ok(Ok(())) => {
            Ok(LivePortalHandle {
                tx,
                url,
                _server_thread: server_thread,
            })
        }
        Ok(Err(e)) => {
            let _ = server_thread.join();
            Err(e)
        }
        Err(e) => {
            let _ = server_thread.join();
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Portal startup channel failed: {e}"),
            ))
        }
    }
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn favicon() -> impl IntoResponse {
  axum::http::StatusCode::NO_CONTENT
}

async fn ws_upgrade(
    ws: WebSocketUpgrade,
    State(tx): State<broadcast::Sender<String>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| ws_client(socket, tx.subscribe()))
}

async fn ws_client(mut socket: WebSocket, mut rx: broadcast::Receiver<String>) {
    while let Ok(msg) = rx.recv().await {
        if socket.send(Message::Text(msg.into())).await.is_err() {
            break;
        }
    }
}

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Helio Live Portal</title>
  <style>
    :root {
      --bg: #0e141b;
      --panel: #121b24;
      --ink: #e6eef7;
      --muted: #8ea4b8;
      --line: #233545;
      --accent: #63d1ff;
      --hot: #ff8f5a;
      --ok: #6de69b;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: radial-gradient(1000px 600px at 10% -10%, #1a2a38 0%, var(--bg) 45%); color: var(--ink); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 20px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { background: linear-gradient(180deg, #152230, var(--panel)); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
    h1 { font-size: 18px; margin: 0 0 12px; letter-spacing: 0.08em; text-transform: uppercase; }
    h2 { font-size: 13px; margin: 0 0 10px; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; }
    .metric { display: flex; gap: 12px; flex-wrap: wrap; }
    .pill { padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; background: #0f1720; }
    .val { color: var(--accent); font-weight: 700; }
    .list { max-height: 320px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; }
    td, th { padding: 7px 9px; border-bottom: 1px solid #1d2e3d; font-size: 12px; }
    th { text-align: left; color: var(--muted); position: sticky; top: 0; background: #11202e; }
    .name { color: #d1e2f1; }
    .gpu { color: var(--hot); }
    .cpu { color: var(--ok); }
    .pipeline { display: grid; gap: 6px; font-size: 12px; }
    .step { padding: 6px 8px; border: 1px solid var(--line); border-radius: 7px; background: #0f1720; }
    .status { color: var(--muted); font-size: 12px; margin-top: 8px; }
    @media (max-width: 900px) { .row { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Helio Live Pipeline Portal</h1>
    <div class="row">
      <section class="card">
        <h2>Frame Metrics</h2>
        <div class="metric">
          <div class="pill">Frame <span class="val" id="frame">-</span></div>
          <div class="pill">Frame ms <span class="val" id="frameMs">-</span></div>
          <div class="pill">Frame-to-frame ms <span class="val" id="ftfMs">-</span></div>
          <div class="pill">GPU total ms <span class="val" id="gpuTotal">-</span></div>
          <div class="pill">CPU total ms <span class="val" id="cpuTotal">-</span></div>
        </div>
        <div class="status" id="status">Connecting...</div>
      </section>
      <section class="card">
        <h2>Pipeline Layout</h2>
        <div class="pipeline" id="pipeline"></div>
      </section>
    </div>

    <section class="card" style="margin-top:16px">
      <h2>Per-Pass Timings</h2>
      <div class="list">
        <table>
          <thead><tr><th>Pass</th><th>GPU ms</th><th>CPU ms</th></tr></thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
    </section>
  </div>

  <script>
    const statusEl = document.getElementById('status');
    const frameEl = document.getElementById('frame');
    const frameMsEl = document.getElementById('frameMs');
    const ftfMsEl = document.getElementById('ftfMs');
    const gpuTotalEl = document.getElementById('gpuTotal');
    const cpuTotalEl = document.getElementById('cpuTotal');
    const rowsEl = document.getElementById('rows');
    const pipelineEl = document.getElementById('pipeline');

    function render(snapshot) {
      frameEl.textContent = snapshot.frame;
      frameMsEl.textContent = snapshot.frame_time_ms.toFixed(2);
      ftfMsEl.textContent = snapshot.frame_to_frame_ms.toFixed(2);
      gpuTotalEl.textContent = snapshot.total_gpu_ms.toFixed(2);
      cpuTotalEl.textContent = snapshot.total_cpu_ms.toFixed(2);

      rowsEl.innerHTML = '';
      for (const t of snapshot.pass_timings) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td class="name">${t.name}</td><td class="gpu">${t.gpu_ms.toFixed(3)}</td><td class="cpu">${t.cpu_ms.toFixed(3)}</td>`;
        rowsEl.appendChild(tr);
      }

      pipelineEl.innerHTML = '';
      for (let i = 0; i < snapshot.pipeline_order.length; i++) {
        const d = document.createElement('div');
        d.className = 'step';
        d.textContent = `${i}. ${snapshot.pipeline_order[i]}`;
        pipelineEl.appendChild(d);
      }
    }

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => { statusEl.textContent = 'Connected'; };
    ws.onclose = () => { statusEl.textContent = 'Disconnected'; };
    ws.onerror = () => { statusEl.textContent = 'Socket error'; };
    ws.onmessage = (ev) => {
      try {
        const snapshot = JSON.parse(ev.data);
        render(snapshot);
      } catch (_) {}
    };
  </script>
</body>
</html>
"#;
