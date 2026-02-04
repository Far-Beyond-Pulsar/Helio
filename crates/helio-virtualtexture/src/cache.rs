pub struct PageCache {
    pub capacity: u32,
    pub page_size: u32,
}

impl Default for PageCache {
    fn default() -> Self {
        Self {
            capacity: 8192,
            page_size: 256,
        }
    }
}
