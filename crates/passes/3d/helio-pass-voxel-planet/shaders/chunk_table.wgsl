// Chunk table updates (docs/chunk-octree.md, `chunks/drawn.rs`): the
// entries that changed, uploaded at once and scattered into the table.
// Change n is changes[3n] (x: the entry's index), its key, its value.

@group(0) @binding(0) var<storage, read> changes: array<vec4<i32>>;
@group(0) @binding(1) var<storage, read_write> chunk_table: array<vec4<i32>>;

@compute @workgroup_size(64)
fn table_update(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&changes) / 3u { return; }
    let entry = u32(changes[id.x * 3u].x);
    chunk_table[entry * 2u] = changes[id.x * 3u + 1u];
    chunk_table[entry * 2u + 1u] = changes[id.x * 3u + 2u];
}
