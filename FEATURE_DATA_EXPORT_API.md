# Feature Data Export API

## Overview

The Feature Data Export API provides a standardized, **zero-copy** mechanism for features to share data with each other. This enables complex feature interactions like material properties flowing to lighting calculations for PBR rendering.

## Key Concepts

### Zero-Copy Guarantee

All data sharing uses zero-copy semantics:
- **CPU data**: Shared via `Arc<T>` - only reference counts are incremented, never the actual data
- **GPU resources**: Shared via handles/views - lightweight references to GPU memory

### Core Types

#### `FeatureData` Trait
Marker trait for CPU-side data that can be exported:

```rust
pub trait FeatureData: Any + Send + Sync {
    fn description(&self) -> &str {
        "Feature data"
    }
}
```

Any struct implementing this trait can be wrapped in an `Arc` and exported.

#### `ExportedData` Enum
Represents exported data in one of three forms:

```rust
pub enum ExportedData {
    CpuData(Arc<dyn FeatureData>),      // CPU metadata (zero-copy via Arc)
    GpuBuffer(gpu::BufferPiece),        // GPU buffer handle
    GpuTexture(gpu::TextureView),       // GPU texture view
}
```

#### Methods on `ExportedData`
- `downcast_arc<T>()` - Downcast CPU data to `Arc<T>` (zero-copy Arc clone)
- `as_gpu_buffer()` - Get GPU buffer handle
- `as_gpu_texture()` - Get GPU texture view

## Usage Guide

### 1. Define Exportable Data

Create a struct representing the data you want to share:

```rust
#[derive(Debug, Clone)]
pub struct MaterialProperties {
    pub metallic: f32,
    pub roughness: f32,
    pub base_color: [f32; 4],
}

impl FeatureData for MaterialProperties {
    fn description(&self) -> &str {
        "PBR Material Properties"
    }
}
```

### 2. Export Data from a Feature

Override the `export_data()` method in your Feature implementation:

```rust
impl Feature for MaterialFeature {
    // ... other Feature methods ...
    
    fn export_data(&self) -> HashMap<String, ExportedData> {
        let mut exports = HashMap::new();
        
        // Export CPU metadata (zero-copy via Arc)
        let props = Arc::new(MaterialProperties {
            metallic: self.metallic,
            roughness: self.roughness,
            base_color: self.base_color,
        });
        exports.insert("properties".to_string(), ExportedData::CpuData(props));
        
        // Export GPU buffer handle (cheap copy)
        if let Some(buffer) = &self.material_buffer {
            exports.insert("buffer".to_string(), ExportedData::GpuBuffer(*buffer));
        }
        
        // Export GPU texture view
        if let Some(texture) = &self.material_texture {
            exports.insert("texture".to_string(), ExportedData::GpuTexture(texture.clone()));
        }
        
        exports
    }
}
```

### 3. Query Data from Another Feature

Use the `FeatureRegistry` methods to query exported data:

```rust
// Query specific export by feature name and export name
if let Some(data) = registry.get_exported_data("materials", "properties") {
    // Downcast to expected type (zero-copy Arc clone)
    if let Some(props) = data.downcast_arc::<MaterialProperties>() {
        // Use the properties
        println!("Metallic: {}", props.metallic);
        println!("Roughness: {}", props.roughness);
        
        // Apply PBR lighting using these properties
        self.apply_pbr_lighting(&props);
    }
}

// Query GPU buffer
if let Some(data) = registry.get_exported_data("materials", "buffer") {
    if let Some(buffer) = data.as_gpu_buffer() {
        // Bind buffer to shader
        encoder.bind_buffer(buffer);
    }
}

// Get all exports from a feature
if let Some(exports) = registry.get_all_exported_data("materials") {
    for (name, data) in exports {
        println!("Export: {}", name);
    }
}
```

## API Reference

### `FeatureRegistry` Methods

#### `get_exported_data(feature_name: &str, export_name: &str) -> Option<ExportedData>`
Query a specific data export from a feature.

**Parameters:**
- `feature_name` - Name of the feature to query (e.g., "materials")
- `export_name` - Name of the specific export (e.g., "properties", "buffer")

**Returns:**
- `Some(ExportedData)` if found
- `None` if feature doesn't exist or doesn't have that export

#### `get_all_exported_data(feature_name: &str) -> Option<HashMap<String, ExportedData>>`
Get all exports from a feature.

**Returns:**
- `Some(HashMap)` with all exports from the feature
- `None` if feature doesn't exist

### `Feature` Trait Method

#### `export_data(&self) -> HashMap<String, ExportedData>`
Export data that other features can consume.

**Default implementation:** Returns empty HashMap

**Override this method** to export data. Export names should be:
- Descriptive (e.g., "properties", "buffer", "shadow_map")
- Unique within the feature
- Lowercase with underscores

## Common Patterns

### Material → Lighting Data Flow

**Materials Feature:**
```rust
fn export_data(&self) -> HashMap<String, ExportedData> {
    let mut exports = HashMap::new();
    let props = Arc::new(MaterialProperties {
        metallic: 0.8,
        roughness: 0.3,
        base_color: [1.0, 1.0, 1.0, 1.0],
    });
    exports.insert("properties".to_string(), ExportedData::CpuData(props));
    exports
}
```

**Lighting Feature:**
```rust
fn prepare_frame(&mut self, context: &FeatureContext, registry: &FeatureRegistry) {
    if let Some(data) = registry.get_exported_data("materials", "properties") {
        if let Some(props) = data.downcast_arc::<MaterialProperties>() {
            // Update PBR uniforms based on material properties
            self.update_pbr_uniforms(props.metallic, props.roughness);
        }
    }
}
```

### Shadow Rendering → Post-Processing

**Shadow Feature:**
```rust
fn export_data(&self) -> HashMap<String, ExportedData> {
    let mut exports = HashMap::new();
    if let Some(shadow_map) = &self.shadow_map_view {
        exports.insert("shadow_map".to_string(), ExportedData::GpuTexture(shadow_map.clone()));
    }
    exports
}
```

**Post-Processing Feature:**
```rust
fn pre_render_pass(&mut self, encoder: &mut gpu::CommandEncoder, context: &FeatureContext, registry: &FeatureRegistry) {
    if let Some(data) = registry.get_exported_data("shadows", "shadow_map") {
        if let Some(texture_view) = data.as_gpu_texture() {
            // Bind shadow map for soft shadow post-processing
            encoder.bind_texture(texture_view);
        }
    }
}
```

## Performance Characteristics

### Zero-Copy Verification

The Arc reference counting ensures no data is copied:

```rust
// First access
let data = registry.get_exported_data("materials", "properties").unwrap();
let props1 = data.downcast_arc::<MaterialProperties>().unwrap();
let count1 = Arc::strong_count(&props1); // 2

// Second access
let props2 = data.downcast_arc::<MaterialProperties>().unwrap();
let count2 = Arc::strong_count(&props2); // 3

// Only the Arc reference count incremented, no data copying!
```

### Memory Overhead

- **CPU data**: `size_of::<Arc<T>>()` = 8 bytes per reference (just a pointer)
- **GPU handles**: Typically 8-16 bytes per handle
- **Original data**: Stored once, shared via references

## Migration Guide

### For Existing Features

No changes required! The `export_data()` method has a default empty implementation. Features only need to override it if they want to export data.

### Adding Data Export to a Feature

1. Define your exportable data struct
2. Implement `FeatureData` for it
3. Override `export_data()` in your Feature implementation
4. Wrap CPU data in `Arc::new()` before exporting

### Consuming Exported Data

1. Call `registry.get_exported_data(feature_name, export_name)`
2. Use `downcast_arc()`, `as_gpu_buffer()`, or `as_gpu_texture()` to access
3. Use the data (it's zero-copy, no performance concerns!)

## Example

See `crates/examples/data_export_demo.rs` for a complete working example demonstrating:
- Exporting material properties
- Querying data from another feature
- Zero-copy Arc reference counting
- PBR material → lighting data flow

Run with:
```bash
cargo run -p examples --bin data_export_demo
```

## Benefits

✓ **Zero-copy** - Data never duplicated, only Arc refcounts incremented  
✓ **Type-safe** - Compile-time type checking with runtime downcasting  
✓ **Flexible** - Supports CPU metadata and GPU resources  
✓ **Optional** - No breaking changes, default implementation provided  
✓ **Performant** - Minimal overhead (just Arc pointer copying)  
✓ **Enables PBR** - Materials can expose properties to lighting features  

## Limitations

- CPU data requires `Clone` to wrap in Arc (one-time cost)
- Downcasting requires `'static` types (no lifetimes in FeatureData)
- Registry must be passed to features that need to query data
- Export names are string-based (not compile-time checked)

## Future Enhancements

Potential future improvements:
- Compile-time checked export names via macros
- Automatic change tracking/invalidation
- Batch queries for multiple exports
- Export metadata (type information, size, etc.)
