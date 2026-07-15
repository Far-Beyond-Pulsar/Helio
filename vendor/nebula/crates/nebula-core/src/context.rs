use crate::error::NebulaError;
use std::sync::Arc;

/// Adapter info forwarded from wgpu so the baker can make quality decisions
/// based on the GPU vendor / limits without holding a reference to the adapter.
#[derive(Clone, Debug)]
pub struct AdapterInfo {
    pub name:   String,
    pub vendor: u32,
    pub limits: wgpu::Limits,
}

impl AdapterInfo {
    fn from_wgpu(info: &wgpu::AdapterInfo, limits: wgpu::Limits) -> Self {
        Self {
            name:   info.name.clone(),
            vendor: info.vendor,
            limits,
        }
    }
}

/// The GPU context shared across every bake pass.
///
/// Creating a `BakeContext` obtains a wgpu device and queue.  You can either
/// let Nebula create its own headless device (ideal for offline CLI tools), or
/// hand in an existing device from the Helio renderer so no extra device is
/// created.
///
/// ```rust,no_run
/// # async fn example() -> nebula_core::Result<()> {
/// let ctx = nebula_core::BakeContext::new().await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct BakeContext {
    pub device:       Arc<wgpu::Device>,
    pub queue:        Arc<wgpu::Queue>,
    pub adapter_info: AdapterInfo,
}

impl BakeContext {
    /// Create a new headless bake context.  Uses the highest-performance
    /// adapter available on the system.
    pub async fn new() -> Result<Self, NebulaError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| NebulaError::Gpu(e.to_string()))?;

        let adapter_info       = adapter.get_info();
        let adapter_limits     = adapter.limits();

        let (device, queue): (wgpu::Device, wgpu::Queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:                Some("nebula-bake-device"),
                    required_features:    wgpu::Features::TEXTURE_BINDING_ARRAY
                        | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
                        | wgpu::Features::BUFFER_BINDING_ARRAY,
                    required_limits:      adapter_limits.clone(),
                    memory_hints:         wgpu::MemoryHints::Performance,
                    experimental_features: Default::default(),
                    trace:                wgpu::Trace::Off,
                },
            )
            .await
            .map_err(|e: wgpu::RequestDeviceError| NebulaError::Gpu(e.to_string()))?;

        Ok(Self {
            device:       Arc::new(device),
            queue:        Arc::new(queue),
            adapter_info: AdapterInfo::from_wgpu(&adapter_info, adapter_limits),
        })
    }

    /// Borrow an existing renderer device instead of creating a new one.
    /// This is the preferred path when running inside the Helio editor.
    pub fn from_wgpu(
        device:       Arc<wgpu::Device>,
        queue:        Arc<wgpu::Queue>,
        name:   String,
        vendor: u32,
        limits: wgpu::Limits,
    ) -> Self {
        Self {
            device,
            queue,
            adapter_info: AdapterInfo { name, vendor, limits },
        }
    }
}
