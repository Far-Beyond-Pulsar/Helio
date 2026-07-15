//! `cargo run --bin web` — fullscreen TUI that builds all WASM demos
//! and serves them locally on http://127.0.0.1:8000.
//!
//! Arrow keys to navigate, Enter to view live logs, Q to quit.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::{Frame, Terminal};

const DEMOS: &[&str] = &[
    "render_v2_basic",
    "render_v2_sky",
    "debug_shapes",
    "indoor_room",
    "indoor_corridor",
    "outdoor_night",
    "outdoor_canyon",
    "indoor_cathedral",
    "indoor_server_room",
    "outdoor_city",
    "outdoor_volcano",
    "space_station",
    "light_benchmark",
    "hlfs_benchmark",
    "sdf_demo",
    "rc_benchmark",
    "load_fbx",
    "load_fbx_embedded",
    "ship_flight",
    "simple_graph",
    "outdoor_rocks",
    "editor_demo",
];

#[derive(Clone)]
enum Status {
    Pending,
    Building,
    Success(u64), // size in KiB
    Failed,
}

struct BuildState {
    status: Status,
    log: Arc<Mutex<Vec<String>>>,
}

struct App {
    builds: Vec<BuildState>,
    list_state: ListState,
    log_scroll: usize,
    show_log: bool,
    done: usize,
    ok_count: usize,
    fail_count: usize,
    start: Instant,
    total: usize,
    cc: String,
    manifest_dir: PathBuf,
}

impl App {
    fn new(manifest_dir: PathBuf) -> Self {
        let total = DEMOS.len();
        let builds = (0..total)
            .map(|_| BuildState {
                status: Status::Pending,
                log: Arc::new(Mutex::new(Vec::new())),
            })
            .collect();
        let mut list_state = ListState::default();
        list_state.select(Some(0));
        Self {
            builds,
            list_state,
            log_scroll: 0,
            show_log: false,
            done: 0,
            ok_count: 0,
            fail_count: 0,
            start: Instant::now(),
            total,
            cc: std::env::var("CC").unwrap_or_default(),
            manifest_dir,
        }
    }

    fn selected(&self) -> Option<usize> {
        self.list_state.selected()
    }

    fn status_icon(frame: u64, s: &Status) -> &'static str {
        let spinners = &["◴", "◷", "◶", "◵", "◐", "◑", "◒", "◓"][..];
        match s {
            Status::Pending => "  ",
            Status::Building => spinners[(frame as usize) % spinners.len()],
            Status::Success(_) => "✓",
            Status::Failed => "✗",
        }
    }

    fn status_color(s: &Status) -> Color {
        match s {
            Status::Pending => Color::DarkGray,
            Status::Building => Color::Cyan,
            Status::Success(_) => Color::Green,
            Status::Failed => Color::Red,
        }
    }

    fn status_text(s: &Status) -> String {
        match s {
            Status::Pending => String::new(),
            Status::Building => String::new(),
            Status::Success(kb) => format!("  {} KiB", kb),
            Status::Failed => "  FAILED".into(),
        }
    }
}

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_base = manifest_dir.join("target/wasm-prebuilt");
    let cc = std::env::var("CC").unwrap_or_default();

    enable_raw_mode().unwrap();
    std::io::stdout().execute(EnterAlternateScreen).unwrap();
    let mut terminal = Terminal::new(ratatui::backend::CrosstermBackend::new(std::io::stdout())).unwrap();
    terminal.clear().unwrap();

    let mut app = App::new(manifest_dir.clone());
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    // ── Spawn build threads ─────────────────────────────────────────────────
    for (i, name) in DEMOS.iter().enumerate() {
        let out_dir = out_base.join(name);
        let manifest_dir = manifest_dir.clone();
        let cc = cc.clone();
        let state = &app.builds[i];
        let log = Arc::clone(&state.log);
        let running = Arc::clone(&running);

        std::thread::spawn(move || {
            // Poke the log so the UI sees activity
            let mut log_guard = log.lock().unwrap();
            log_guard.push(format!("[{}] Starting wasm-pack build...", name));
            drop(log_guard);

            let mut child = match Command::new("wasm-pack")
                .args([
                    "build",
                    "--release",
                    "--target",
                    "web",
                    "--no-default-features",
                    "--features",
                    name,
                ])
                .current_dir(manifest_dir.join("crates/helio-web-demos"))
                .env("CC", &cc)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
            {
                Ok(c) => c,
                Err(e) => {
                    let mut log_guard = log.lock().unwrap();
                    log_guard.push(format!("Failed to spawn wasm-pack: {e}"));
                    return;
                }
            };

            let stdout = child.stdout.take().unwrap();
            let stderr = child.stderr.take().unwrap();
            let out_reader = BufReader::new(stdout);
            let err_reader = BufReader::new(stderr);

            for line in out_reader.lines().flatten() {
                if !running.load(Ordering::Relaxed) { break; }
                let mut log_guard = log.lock().unwrap();
                log_guard.push(line);
            }
            for line in err_reader.lines().flatten() {
                if !running.load(Ordering::Relaxed) { break; }
                let mut log_guard = log.lock().unwrap();
                log_guard.push(line);
            }

            let status = child.wait().ok();
            let success = status.map(|s| s.success()).unwrap_or(false);

            // Try to move pkg/ to the output directory, then write index.html
            let moved = success && {
                let pkg_dir = manifest_dir.join("crates/helio-web-demos/pkg");
                if pkg_dir.exists() {
                    let _ = std::fs::remove_dir_all(&out_dir);
                    if std::fs::rename(&pkg_dir, &out_dir).is_ok() {
                        let html = format!(
                            r#"<!DOCTYPE html><html><head>
<meta charset="utf-8"><title>{name}</title>
<style>body{{margin:0;overflow:hidden;background:#000}}
#info{{position:absolute;bottom:8px;left:8px;color:#888;font:14px monospace}}
</style></head><body>
<script type="module">
import init from "./helio_web_demos.js";
init().catch(e=>document.body.innerHTML=`<pre style=color:red>${{e}}</pre>`);
</script>
<div id=info>{name}</div>
</body></html>"#
                        );
                        std::fs::write(out_dir.join("index.html"), &html).is_ok()
                    } else {
                        false
                    }
                } else {
                    false
                }
            };

            let mut log_guard = log.lock().unwrap();
            if moved {
                let size_kb = out_dir
                    .join("helio_web_demos_bg.wasm")
                    .metadata()
                    .ok()
                    .map(|m| m.len() / 1024)
                    .unwrap_or(0);
                log_guard.push(format!("OK ({size_kb} KiB)"));
            } else {
                log_guard.push("FAILED".into());
            }
        });
    }

    // ── TUI event loop ────────────────────────────────────────────────────
    let tick = std::time::Duration::from_millis(100);
    let all_done = Arc::new(AtomicBool::new(false));
    let serve_started = Arc::new(AtomicBool::new(false));

    'main: loop {
        let all_done_val = all_done.load(Ordering::Relaxed);

        terminal.draw(|f| ui(f, &mut app, all_done_val)).unwrap();

        if event::poll(tick).unwrap() {
            if let Event::Key(key) = event::read().unwrap() {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break 'main,
                        KeyCode::Down => {
                            if app.show_log {
                                app.log_scroll = app.log_scroll.saturating_add(1);
                            } else {
                                let i = app.list_state.selected().unwrap_or(0);
                                if i + 1 < app.total {
                                    app.list_state.select(Some(i + 1));
                                }
                            }
                        }
                        KeyCode::Up => {
                            if app.show_log {
                                app.log_scroll = app.log_scroll.saturating_sub(1);
                            } else {
                                let i = app.list_state.selected().unwrap_or(0);
                                if i > 0 {
                                    app.list_state.select(Some(i - 1));
                                }
                            }
                        }
                        KeyCode::Enter if !all_done_val || app.show_log => {
                            app.show_log = !app.show_log;
                            app.log_scroll = 0;
                        }
                        KeyCode::Enter => {
                            // All done, Enter starts the server
                            serve_started.store(true, Ordering::Relaxed);
                            break 'main;
                        }
                        KeyCode::Backspace | KeyCode::Esc if app.show_log => {
                            app.show_log = false;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Check if all builds are done
        if !all_done_val {
            let done = app.builds.iter().filter(|b| {
                let last = b.log.lock().unwrap().last().cloned().unwrap_or_default();
                last.starts_with("OK") || last == "FAILED"
            }).count();
            if done == app.total {
                all_done.store(true, Ordering::Relaxed);
            }
        }
    }

    disable_raw_mode().unwrap();
    std::io::stdout().execute(LeaveAlternateScreen).unwrap();

    if serve_started.load(Ordering::Relaxed) || all_done.load(Ordering::Relaxed) {
        let out_base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/wasm-prebuilt");
        println!("\nServing at http://127.0.0.1:8000/");
        serve(&out_base, "127.0.0.1:8000");
    }
}

// ── UI rendering ───────────────────────────────────────────────────────────

fn ui(f: &mut Frame, app: &App, all_done: bool) {
    if app.show_log {
        log_ui(f, app);
    } else {
        list_ui(f, app, all_done);
    }
}

fn list_ui(f: &mut Frame, app: &App, all_done: bool) {
    let total = app.total;
    let done = app.builds.iter().filter(|b| {
        let last = b.log.lock().unwrap().last().cloned().unwrap_or_default();
        last.starts_with("OK") || last == "FAILED"
    }).count();
    let ok = app.builds.iter().filter(|b| {
        b.log.lock().unwrap().last().cloned().unwrap_or_default().starts_with("OK")
    }).count();
    let fail = done.saturating_sub(ok);
    let pct = if total > 0 { done as f64 / total as f64 } else { 0.0 };
    let frame = app.builds.iter().map(|b| b.log.lock().unwrap().len() as u64).sum::<u64>();

    let bar_width = f.area().width.saturating_sub(4) as usize;
    let filled = (pct * bar_width as f64).round() as usize;
    let empty = bar_width.saturating_sub(filled);
    let bar = format!(
        "{}│{}",
        "█".repeat(filled),
        "░".repeat(empty),
    );

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(1)])
        .split(f.area());

    // Header bar
    let mut header_spans = vec![
        Span::styled(" Helio ", Style::new().fg(Color::Magenta).bold()),
        Span::raw("│ "),
        Span::raw(format!("{total} demos  ")),
        Span::styled(format!("✔ {ok}"), Style::new().fg(Color::Green).bold()),
        Span::raw("  "),
        Span::styled(format!("✘ {fail}"), Style::new().fg(Color::Red).bold()),
        Span::raw(format!("  {done}/{total}  ")),
        Span::styled(bar, Style::default().fg(Color::Cyan)),
    ];
    if all_done {
        header_spans.push(Span::raw("  │  "));
        header_spans.push(Span::styled(
            "http://127.0.0.1:8000/",
            Style::new().fg(Color::Green).bold(),
        ));
    }
    let header = Paragraph::new(Line::from(header_spans))
        .block(Block::default());
    f.render_widget(header, layout[0]);

    // Demo list
    let items: Vec<ListItem> = app.builds.iter().enumerate().map(|(i, b)| {
        let last = b.log.lock().unwrap().last().cloned().unwrap_or_default();
        let status = if last.starts_with("OK") {
            Status::Success(last.trim_start_matches("OK (")
                .trim_end_matches(" KiB)").parse().unwrap_or(0))
        } else if last == "FAILED" {
            Status::Failed
        } else if b.log.lock().unwrap().len() > 0 {
            Status::Building
        } else {
            Status::Pending
        };
        let icon = App::status_icon(frame, &status);
        let color = App::status_color(&status);
        let text = App::status_text(&status);
        let name = DEMOS[i];
        let is_selected = app.list_state.selected() == Some(i);
        let bg = if is_selected { Color::DarkGray } else { Color::Reset };
        let fg = if is_selected { Color::White } else { Color::Reset };
        let icon_style = Style::default().fg(color).bg(bg).add_modifier(Modifier::BOLD);
        let name_style = Style::default().fg(fg).bg(bg);
        let text_style = Style::default().fg(color).bg(bg);
        ListItem::new(Line::from(vec![
            Span::styled(format!(" {} ", icon), icon_style),
            Span::styled(name, name_style),
            Span::styled(text, text_style),
        ])).style(Style::default().bg(bg))
    }).collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Demos "))
        .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    f.render_stateful_widget(list, layout[1], &mut app.list_state.clone());
}

fn log_ui(f: &mut Frame, app: &App) {
    let idx = app.list_state.selected().unwrap_or(0);
    let name = DEMOS[idx];
    let log = app.builds[idx].log.lock().unwrap();
    let lines: Vec<Line> = log.iter().map(|l| Line::from(Span::raw(l))).collect();

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(1)])
        .split(f.area());

    let title = format!(" Build Log: {name}  ");
    let log_widget = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(title))
        .scroll((app.log_scroll as u16, 0))
        .wrap(Wrap { trim: false });
    f.render_widget(log_widget, layout[1]);

    let footer = Paragraph::new(Line::from(vec![
        Span::raw(" ↑↓ Scroll  "),
        Span::styled("Enter/BS", Style::new().bold()),
        Span::raw(" Close  "),
        Span::styled("Q", Style::new().bold()),
        Span::raw(" Quit"),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(footer, layout[0]);
}

fn final_ui(f: &mut Frame, app: &App) {
    let ok = app.builds.iter().filter(|b| {
        b.log.lock().unwrap().last().cloned().unwrap_or_default().starts_with("OK")
    }).count();
    let failed_names: Vec<String> = app.builds.iter().enumerate().filter_map(|(i, b)| {
        let last = b.log.lock().unwrap().last().cloned().unwrap_or_default();
        if last == "FAILED" { Some(DEMOS[i].to_string()) } else { None }
    }).collect();
    let total = app.total;

    let mut text = vec![
        Line::from(Span::styled(" Build Complete ", Style::new().fg(Color::Magenta).bold())),
        Line::from(Span::raw("")),
        Line::from(Span::styled(format!("  ✔  {ok:>3} / {total}"), Style::new().fg(Color::Green).bold())),
    ];

    if !failed_names.is_empty() {
        text.push(Line::from(Span::raw("")));
        text.push(Line::from(Span::styled(
            format!("  ✘  {} failed:", failed_names.len()),
            Style::new().fg(Color::Red).bold(),
        )));
        for n in &failed_names {
            text.push(Line::from(Span::styled(format!("     {n}"), Style::new().fg(Color::Red))));
        }
    }

    text.push(Line::from(Span::raw("")));
    text.push(Line::from(vec![
        Span::raw("  "),
        Span::styled("http://127.0.0.1:8000/", Style::new().fg(Color::Cyan).underlined()),
    ]));
    text.push(Line::from(Span::raw("")));
    text.push(Line::from(vec![
        Span::raw("  "),
        Span::styled("Enter", Style::new().bold()),
        Span::raw("  Serve  "),
        Span::styled("Q", Style::new().bold()),
        Span::raw("  Quit"),
    ]));

    let p = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title(" Summary "))
        .wrap(Wrap { trim: false });
    f.render_widget(p, f.area());
}

// ── HTTP server ─────────────────────────────────────────────────────────────

fn serve(root: &PathBuf, addr: &str) {
    let server = tiny_http::Server::http(addr).unwrap();
    let root = root.clone();
    println!("\nServing at http://{addr}/");
    for request in server.incoming_requests() {
        let url = request.url().to_string();
        let path = {
            let stripped = url.trim_start_matches('/');
            let candidate = root.join(stripped);
            if candidate.is_dir() {
                candidate.join("index.html")
            } else {
                candidate
            }
        };

        let (status, contents) = match std::fs::read(&path) {
            Ok(data) => (tiny_http::StatusCode(200), data),
            Err(_) => (tiny_http::StatusCode(404), b"404 Not Found\n".to_vec()),
        };

        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        let cts = mime_for(ext);
        let response = tiny_http::Response::from_data(contents)
            .with_status_code(status)
            .with_header(
                tiny_http::Header::from_bytes(&b"Content-Type"[..], cts.as_bytes()).unwrap(),
            );
        let _ = request.respond(response);
    }
}

fn mime_for(ext: &str) -> &'static str {
    match ext {
        "html" => "text/html; charset=utf-8",
        "js" => "application/javascript",
        "wasm" => "application/wasm",
        "css" => "text/css; charset=utf-8",
        "png" => "image/png",
        "svg" => "image/svg+xml",
        "json" => "application/json",
        _ => "application/octet-stream",
    }
}
