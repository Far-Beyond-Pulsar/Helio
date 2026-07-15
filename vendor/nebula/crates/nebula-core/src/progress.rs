/// Receives progress updates from a bake pass.
///
/// The trait is object-safe so you can store `Box<dyn ProgressReporter>` and
/// pass one implementation through the whole pipeline.
pub trait ProgressReporter: Send + Sync {
    /// Called at the start of the pass.
    fn begin(&self, pass_name: &str, total_steps: u32);
    /// Called after each logical step (`step` is 0-based).
    fn step(&self, pass_name: &str, step: u32, message: &str);
    /// Called when the pass finishes (or on error).
    fn finish(&self, pass_name: &str, success: bool, message: &str);
}

/// A no-op reporter — used when progress tracking is not needed.
pub struct NullReporter;

impl ProgressReporter for NullReporter {
    fn begin(&self, _: &str, _: u32)               {}
    fn step(&self, _: &str, _: u32, _: &str)        {}
    fn finish(&self, _: &str, _: bool, _: &str)     {}
}

/// A reporter that logs to the `log` crate at INFO level.
pub struct LogReporter;

impl ProgressReporter for LogReporter {
    fn begin(&self, pass: &str, steps: u32) {
        log::info!("[nebula:{pass}] starting ({steps} steps)");
    }
    fn step(&self, pass: &str, step: u32, msg: &str) {
        log::info!("[nebula:{pass}] step {step}: {msg}");
    }
    fn finish(&self, pass: &str, success: bool, msg: &str) {
        if success {
            log::info!("[nebula:{pass}] done — {msg}");
        } else {
            log::error!("[nebula:{pass}] FAILED — {msg}");
        }
    }
}

/// A reporter backed by an `std::sync::mpsc` channel so the caller can poll
/// progress from another thread.
pub struct ChannelReporter {
    tx: std::sync::Mutex<std::sync::mpsc::Sender<ProgressEvent>>,
}

#[derive(Clone, Debug)]
pub struct ProgressEvent {
    pub pass:    String,
    pub kind:    ProgressEventKind,
}

#[derive(Clone, Debug)]
pub enum ProgressEventKind {
    Begin   { total_steps: u32 },
    Step    { step: u32, message: String },
    Finish  { success: bool, message: String },
}

impl ChannelReporter {
    pub fn new() -> (Self, std::sync::mpsc::Receiver<ProgressEvent>) {
        let (tx, rx) = std::sync::mpsc::channel();
        (Self { tx: std::sync::Mutex::new(tx) }, rx)
    }

    fn send(&self, pass: &str, kind: ProgressEventKind) {
        if let Ok(guard) = self.tx.lock() {
            let _ = guard.send(ProgressEvent { pass: pass.to_owned(), kind });
        }
    }
}

impl ProgressReporter for ChannelReporter {
    fn begin(&self, pass: &str, total_steps: u32) {
        self.send(pass, ProgressEventKind::Begin { total_steps });
    }
    fn step(&self, pass: &str, step: u32, message: &str) {
        self.send(pass, ProgressEventKind::Step { step, message: message.to_owned() });
    }
    fn finish(&self, pass: &str, success: bool, message: &str) {
        self.send(pass, ProgressEventKind::Finish { success, message: message.to_owned() });
    }
}
