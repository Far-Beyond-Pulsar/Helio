// cross-platform instant type used by renderer and graph.
//
// For native targets we simply alias `std::time::Instant`.  WebAssembly
// doesn't implement `Instant`, so for wasm32 we provide a tiny wrapper around
// `js_sys::Date::now()` that exposes the same API we need (now(),
// duration_since, elapsed).

#[cfg(not(target_arch = "wasm32"))]
pub type Instant = std::time::Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Copy, Clone)]
pub struct Instant(pub f64); // milliseconds since epoch

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub fn now() -> Self {
        Instant(js_sys::Date::now())
    }

    pub fn duration_since(&self, earlier: Instant) -> std::time::Duration {
        let ms = self.0 - earlier.0;
        if ms <= 0.0 {
            std::time::Duration::ZERO
        } else {
            std::time::Duration::from_secs_f64(ms / 1000.0)
        }
    }

    pub fn elapsed(&self) -> std::time::Duration {
        Instant::now().duration_since(*self)
    }
}
