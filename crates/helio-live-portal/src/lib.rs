use std::sync::mpsc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use axum::Router;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

// helper to serve javascript modules from assets/js
async fn serve_js(axum::extract::Path(file): axum::extract::Path<String>) -> impl IntoResponse {
    // compute absolute path based on manifest dir so it works from any cwd
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/js");
    let path = base.join(&file);
    match tokio::fs::read(&path).await {
        Ok(data) => (
            [
                (axum::http::header::CONTENT_TYPE, "application/javascript"),
                (axum::http::header::CACHE_CONTROL, "no-store"),
            ],
            data,
        )
            .into_response(),
        Err(_) => {
            eprintln!("serve_js missing {}", path.display());
            axum::http::StatusCode::NOT_FOUND.into_response()
        }
    }
}

async fn serve_vendor(axum::extract::Path(file): axum::extract::Path<String>) -> impl IntoResponse {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/vendor");
    let path = base.join(&file);
    let mime = match path.extension().and_then(|e| e.to_str()) {
        Some("js") => "application/javascript",
        Some("css") => "text/css",
        _ => "application/octet-stream",
    };
    match tokio::fs::read(&path).await {
        Ok(data) => (
            [
                (axum::http::header::CONTENT_TYPE, mime),
                (axum::http::header::CACHE_CONTROL, "no-store"),
            ],
            data,
        )
            .into_response(),
        Err(_) => {
            eprintln!("serve_vendor missing {}", path.display());
            axum::http::StatusCode::NOT_FOUND.into_response()
        }
    }
}

async fn serve_static(axum::extract::Path(file): axum::extract::Path<String>) -> impl IntoResponse {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
    let path = base.join(&file);
    match tokio::fs::read(&path).await {
        Ok(data) => (
            [(axum::http::header::CONTENT_TYPE, "application/octet-stream")],
            data,
        )
            .into_response(),
        Err(_) => {
            eprintln!("serve_static missing {}", path.display());
            axum::http::StatusCode::NOT_FOUND.into_response()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PortalPassTiming {
    pub name: String,
    pub gpu_ms: f32,
    pub cpu_ms: f32,
}

/// A single top-level CPU timing stage sent to the portal.
/// The bridge auto-populates this from the individual *_ms fields so the JS
/// A node in the frame-timing tree sent to the portal frontend.
/// `children` are rendered as a second row below this node.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PortalStageTiming {
    pub id: String,
    pub name: String,
    pub ms: f32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<PortalStageTiming>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PortalSceneObject {
    pub id: u32,
    pub bounds_center: [f32; 3],
    pub bounds_radius: f32,
    pub has_material: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PortalSceneLight {
    pub id: u32,
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PortalSceneBillboard {
    pub id: u32,
    pub position: [f32; 3],
    pub scale: [f32; 2],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PortalSceneCamera {
    pub position: [f32; 3],
    pub forward: [f32; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PortalSceneLayout {
    pub objects: Vec<PortalSceneObject>,
    pub lights: Vec<PortalSceneLight>,
    pub billboards: Vec<PortalSceneBillboard>,
    pub camera: Option<PortalSceneCamera>,
}

/// Delta update: only includes changed elements to reduce bandwidth
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PortalSceneLayoutDelta {
    /// Objects to add or update: send full data on add, sparse data on update
    pub object_changes: Vec<PortalSceneObject>,
    /// Object IDs that moved (position changed)
    pub moved_object_ids: Vec<u32>,
    /// Object IDs to remove (not present in object_changes)
    pub removed_object_ids: Vec<u32>,

    /// Lights to add or update
    pub light_changes: Vec<PortalSceneLight>,
    /// Light IDs to remove
    pub removed_light_ids: Vec<u32>,

    /// Billboards to add or update
    pub billboard_changes: Vec<PortalSceneBillboard>,
    /// Billboard IDs to remove
    pub removed_billboard_ids: Vec<u32>,

    /// Only present if camera changed
    pub camera: Option<Option<PortalSceneCamera>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrawCallMetrics {
    pub total: usize,
    pub opaque: usize,
    pub transparent: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PortalFrameSnapshot {
    pub frame: u64,
    pub timestamp_ms: u128,
    pub frame_time_ms: f32,
    pub frame_to_frame_ms: f32,
    pub total_gpu_ms: f32,
    pub total_cpu_ms: f32,

    /// Per-pass GPU/CPU timing from hardware timestamp queries.
    pub pass_timings: Vec<PortalPassTiming>,
    /// Render-graph pass execution order (names only).
    pub pipeline_order: Vec<String>,
    /// ID of the `stage_timings` node that owns the pass sub-graph.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline_stage_id: Option<String>,

    /// Delta updates for scene data (only changes sent; first frame is full).
    pub scene_delta: Option<PortalSceneLayoutDelta>,

    // Scene object/light/billboard counts.
    pub object_count: usize,
    pub light_count: usize,
    pub billboard_count: usize,

    pub draw_calls: DrawCallMetrics,

    /// CPU timing tree built from `profile_scope!` macros — always populated.
    pub stage_timings: Vec<PortalStageTiming>,
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

    /// Return a clone of the internal sender so callers can dispatch from other
    /// threads without holding on to the full handle.
    pub fn sender(&self) -> std::sync::mpsc::Sender<PortalFrameSnapshot> {
        self.tx.clone()
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
                    let _ = ready_tx.send(Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    )));
                    return;
                }
            };

            runtime.block_on(async move {
                let (broadcast_tx, _) = broadcast::channel::<String>(256);

                let bridge_tx = broadcast_tx.clone();
                let bridge_thread = std::thread::Builder::new()
                    .name("helio-live-portal-bridge".to_string())
                    .spawn(move || {
                        use std::sync::mpsc::RecvTimeoutError;
                        use std::time::Duration;

                        let mut buffer = Vec::new();
                        loop {
                            match rx.recv_timeout(Duration::from_millis(100)) {
                                Ok(snapshot) => {
                                    buffer.push(snapshot);
                                }
                                Err(RecvTimeoutError::Timeout) => {
                                    // time to flush whatever we collected
                                }
                                Err(RecvTimeoutError::Disconnected) => break,
                            }

                            if !buffer.is_empty() {
                                if let Ok(json) = serde_json::to_string(&buffer) {
                                    let _ = bridge_tx.send(json);
                                }
                                buffer.clear();
                            }
                        }
                    });

                let app = Router::new()
                    .route("/", get(index))
                    .route("/favicon.ico", get(favicon))
                    // serve JS modules and other static assets
                    .route("/js/{*file}", get(serve_js))
                    .route("/vendor/{*file}", get(serve_vendor))
                    .route("/assets/{*file}", get(serve_static))
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
        Ok(Ok(())) => Ok(LivePortalHandle {
            tx,
            url,
            _server_thread: server_thread,
        }),
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

const INDEX_HTML: &str = include_str!("../assets/index.html");

