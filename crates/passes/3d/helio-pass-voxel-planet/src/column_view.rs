//! Resident columns read back from the GPU (records and pool words),
//! decoded as the shaders read them (`common.wgsl`, docs/span-columns.md):
//! for tests and diagnostics.

/// Column record info bits (`Column.info` in `common.wgsl`).
pub mod info {
    pub const VALID: u32 = 0x8000_0000;
    pub const OVERFLOW: u32 = 0x4000_0000;
    pub const RELIEF: u32 = 0x1000_0000;
    pub const TOPOLOGY: u32 = 0x0800_0000;
    pub const RELIEF_INLINE: u32 = 0x0400_0000;
    pub const HEIGHTFIELD: u32 = 0x0200_0000;
    pub const CLIP_BELOW: u32 = 0x200;
    pub const CLIP_ABOVE: u32 = 0x400;
    pub const EDIT_MATERIALS: u32 = 0x800;
    pub const GENERATED: u32 = 0x1000;
    pub const TOPS_WIDE: u32 = 0x20;
}

/// Span kinds (`SPAN_*` in `common.wgsl`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanKind {
    Air,
    Solid,
    Lanes,
    Tops,
    Natural,
    Bricks,
}

impl SpanKind {
    fn of(code: u32) -> Self {
        match code {
            0 => Self::Air,
            1 => Self::Solid,
            2 => Self::Lanes,
            3 => Self::Tops,
            4 => Self::Natural,
            _ => Self::Bricks,
        }
    }
}

/// A span `[start, end)` and its payload's first pool word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub kind: SpanKind,
    pub start: i32,
    pub end: i32,
    payload: usize,
}

const UNIT_WORDS: usize = 16;
const NO_LAYER: i32 = -0x7fff_ffff;

/// One resident column: its record (8 words) over the pool's words.
pub struct ColumnView<'a> {
    record: &'a [u32],
    pool: &'a [u32],
    /// The terrain program stores a surface word per cell.
    surface_words: bool,
}

impl<'a> ColumnView<'a> {
    /// The column of record `record` (8 words); `surface_words` when the
    /// terrain program has `terrain_surface`.
    pub fn new(record: &'a [u32], pool: &'a [u32], surface_words: bool) -> Self {
        Self { record, pool, surface_words }
    }
    /// Every record of a records buffer (8 words each).
    pub fn all(records: &'a [u32], pool: &'a [u32], surface_words: bool) -> impl Iterator<Item = ColumnView<'a>> + 'a {
        records.chunks_exact(8).map(move |record| ColumnView::new(record, pool, surface_words))
    }
    pub fn info(&self) -> u32 {
        self.record[3]
    }
    pub fn valid(&self) -> bool {
        self.info() & (info::VALID | info::OVERFLOW) == info::VALID
    }
    /// `(face, level, ci, cj)`.
    pub fn key(&self) -> (u8, u32, i32, i32) {
        let k0 = self.record[0];
        (((k0 >> 24) & 7) as u8, k0 >> 27, (k0 & 0xff_ffff) as i32, self.record[1] as i32)
    }
    /// The natural tops' base (level cells).
    pub fn base(&self) -> i32 {
        self.record[2] as i32
    }
    fn run(&self) -> usize {
        self.record[4] as usize
    }
    /// First layer above every solid cell (level cells); clipped above, the
    /// window's top.
    pub fn top(&self) -> i32 {
        self.record[5] as i32
    }
    /// The lowest layer described, when clipped below.
    pub fn lo(&self) -> i32 {
        self.record[6] as i32
    }
    /// Whether the column describes layer `k`.
    pub fn knows(&self, k: i32) -> bool {
        !(self.info() & info::CLIP_BELOW != 0 && k < self.lo() || self.info() & info::CLIP_ABOVE != 0 && k >= self.top())
    }
    fn tops_units(&self) -> usize {
        if self.info() & info::TOPS_WIDE != 0 { 2 } else { 1 }
    }
    fn relief_wide(&self) -> bool {
        self.info() & info::RELIEF != 0 && self.info() & info::RELIEF_INLINE == 0
    }
    /// Header units (tops, relief fractions, surface words, offsets).
    pub fn header_units(&self) -> usize {
        self.tops_units() + if self.relief_wide() { 2 } else { 0 } + usize::from(self.surface_words) + 1
    }
    fn word(&self, unit: usize, word: usize) -> u32 {
        self.pool[(self.run() + unit) * UNIT_WORDS + word]
    }
    /// The stored top of cell (x, y) over the base: level cells, or base
    /// cells with inline relief.
    pub fn stored_top(&self, x: u32, y: u32) -> u32 {
        let cell = (x + y * 8) as usize;
        if self.info() & info::TOPS_WIDE != 0 {
            return (self.word(0, cell >> 1) >> ((cell & 1) * 16)) & 0xffff;
        }
        (self.word(0, cell >> 2) >> ((cell & 3) * 8)) & 255
    }
    /// The surface offset byte of cell (x, y) (`column_surface_offset`).
    pub fn surface_offset(&self, x: u32, y: u32) -> u32 {
        let cell = (x + y * 8) as usize;
        let unit = self.header_units() - 1;
        (self.word(unit, cell >> 2) >> ((cell & 3) * 8)) & 255
    }
    /// Natural surface top of cell (x, y): first air above the generated
    /// ground (level cells).
    pub fn natural_top(&self, x: u32, y: u32) -> i32 {
        let cell = (x + y * 8) as usize;
        if self.info() & info::TOPS_WIDE != 0 {
            return self.base() + ((self.word(0, cell >> 1) >> ((cell & 1) * 16)) & 0xffff) as i32;
        }
        let offset = (self.word(0, cell >> 2) >> ((cell & 3) * 8)) & 255;
        if self.info() & info::RELIEF_INLINE != 0 {
            let level = self.key().1;
            return self.base() + ((offset + (1 << level) - 1) >> level) as i32;
        }
        self.base() + offset as i32
    }
    /// The column's spans, the implicit solid below the first one excluded.
    pub fn spans(&self) -> Vec<Span> {
        let top = self.top();
        if self.info() & info::HEIGHTFIELD != 0 {
            return vec![Span { kind: SpanKind::Natural, start: NO_LAYER, end: top, payload: 0 }];
        }
        let table = (self.run() + self.header_units()) * UNIT_WORDS;
        let n = (self.info() & 31) as usize;
        let mut spans: Vec<Span> = Vec::with_capacity(n);
        for e in 0..n {
            let start = self.pool[table + e * 2] as i32;
            let entry = self.pool[table + e * 2 + 1];
            if let Some(last) = spans.last_mut() {
                last.end = start;
            }
            spans.push(Span { kind: SpanKind::of(entry & 7), start, end: top, payload: table + (entry >> 3) as usize });
        }
        spans
    }
    /// The span holding layer `k` below the top (solid below the first).
    pub fn span_at(&self, k: i32) -> Span {
        let spans = self.spans();
        let first = spans.first().map_or(self.top(), |s| s.start);
        if k < first {
            return Span { kind: SpanKind::Solid, start: NO_LAYER, end: first, payload: 0 };
        }
        *spans.iter().rev().find(|s| s.start <= k).expect("k is at or above the first span")
    }
    /// Layer below which lane (x, y) is solid inside lane span `s`.
    pub fn lane_top(&self, s: &Span, x: u32, y: u32) -> i32 {
        let cell = (x + y * 8) as usize;
        match s.kind {
            SpanKind::Air | SpanKind::Bricks => s.start,
            SpanKind::Solid => s.end,
            SpanKind::Lanes => {
                if (self.pool[s.payload + (cell >> 5)] >> (cell & 31)) & 1 != 0 { s.end } else { s.start }
            }
            SpanKind::Tops => s.start + ((self.pool[s.payload + (cell >> 2)] >> ((cell & 3) * 8)) & 255) as i32,
            SpanKind::Natural => self.natural_top(x, y).clamp(s.start, s.end),
        }
    }
    /// Brick `b` of a BRICKS span: `(0 air | 1 solid | 2 mixed, pool unit)`.
    pub fn brick(&self, s: &Span, b: u32) -> (u32, usize) {
        let words = (((s.end - s.start) as u32 >> 3) as usize).div_ceil(32);
        let (w, bit) = ((b >> 5) as usize, b & 31);
        let mixed = self.pool[s.payload + 1 + w];
        if (mixed >> bit) & 1 != 0 {
            let rank = (0..w).map(|q| self.pool[s.payload + 1 + q].count_ones()).sum::<u32>() + (mixed & ((1u32 << bit) - 1)).count_ones();
            return (2, self.run() + self.pool[s.payload] as usize + rank as usize);
        }
        let solid = (self.pool[s.payload + 1 + words + w] >> bit) & 1 != 0;
        (u32::from(solid), 0)
    }
    /// Occupancy of cell (x, y, k): `None` beyond a clipped window.
    pub fn cell(&self, x: u32, y: u32, k: i32) -> Option<bool> {
        if !self.knows(k) {
            return None;
        }
        if k >= self.top() {
            return Some(false);
        }
        let s = self.span_at(k);
        if s.kind == SpanKind::Bricks {
            let (state, unit) = self.brick(&s, ((k - s.start) >> 3) as u32);
            if state == 2 {
                let bit = x + y * 8 + (k & 7) as u32 * 64;
                return Some((self.pool[unit * UNIT_WORDS + (bit >> 5) as usize] >> (bit & 31)) & 1 != 0);
            }
            return Some(state == 1);
        }
        Some(k < self.lane_top(&s, x, y))
    }
}
