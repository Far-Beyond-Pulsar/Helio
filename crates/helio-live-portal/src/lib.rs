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
pub struct PortalSceneObject {
    pub bounds_center: [f32; 3],
    pub bounds_radius: f32,
    pub has_material: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PortalSceneLight {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PortalSceneBillboard {
    pub position: [f32; 3],
    pub scale: [f32; 2],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PortalSceneCamera {
    pub position: [f32; 3],
    pub forward: [f32; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PortalSceneLayout {
    pub objects: Vec<PortalSceneObject>,
    pub lights: Vec<PortalSceneLight>,
    pub billboards: Vec<PortalSceneBillboard>,
    pub camera: Option<PortalSceneCamera>,
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
    pub scene_layout: Option<PortalSceneLayout>,
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
