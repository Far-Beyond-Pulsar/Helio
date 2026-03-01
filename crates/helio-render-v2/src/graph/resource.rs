//! Resource management for graph

/// Pass identifier
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct PassId(pub usize);

/// Resource handle for graph resources
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct ResourceHandle(pub u64);

impl ResourceHandle {
    /// Create a new resource handle
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Create a named resource handle (deterministic)
    pub fn named(name: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        Self(hasher.finish())
    }
}

impl Default for ResourceHandle {
    fn default() -> Self {
        Self::new()
    }
}
