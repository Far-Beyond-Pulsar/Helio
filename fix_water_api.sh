# Fix water surface pass
sed -i 's/bind_group_layouts: &\[&bgl_0, &bgl_1, &bgl_2\]/bind_group_layouts: \&[Some(\&bgl_0), Some(\&bgl_1), Some(\&bgl_2)]/g' crates/helio-pass-water-surface/src/lib.rs
sed -i 's/push_constant_ranges: &\[\]/immediate_size: 0/g' crates/helio-pass-water-surface/src/lib.rs
sed -i 's/multiview: None/multiview_mask: None,\n            cache: None/g' crates/helio-pass-water-surface/src/lib.rs
sed -i 's/depth_write_enabled: false/depth_write_enabled: Some(false)/g' crates/helio-pass-water-surface/src/lib.rs
sed -i 's/depth_compare: wgpu::CompareFunction::LessEqual/depth_compare: Some(wgpu::CompareFunction::LessEqual)/g' crates/helio-pass-water-surface/src/lib.rs
sed -i 's/fn prepare(&mut self, ctx: &PrepareContext) -> Result<()>/fn prepare(\&mut self, ctx: \&PrepareContext) -> HelioResult<()>/g' crates/helio-pass-water-surface/src/lib.rs
sed -i 's/fn execute(&mut self, ctx: &mut PassContext) -> Result<()>/fn execute(\&mut self, ctx: \&mut PassContext) -> HelioResult<()>/g' crates/helio-pass-water-surface/src/lib.rs

# Fix underwater pass
sed -i 's/bind_group_layouts: &\[&bgl\]/bind_group_layouts: \&[Some(\&bgl)]/g' crates/helio-pass-underwater/src/lib.rs
sed -i 's/push_constant_ranges: &\[\]/immediate_size: 0/g' crates/helio-pass-underwater/src/lib.rs
sed -i 's/multiview: None/multiview_mask: None,\n            cache: None/g' crates/helio-pass-underwater/src/lib.rs
sed -i 's/fn prepare(&mut self, ctx: &PrepareContext) -> Result<()>/fn prepare(\&mut self, ctx: \&PrepareContext) -> HelioResult<()>/g' crates/helio-pass-underwater/src/lib.rs
sed -i 's/fn execute(&mut self, ctx: &mut PassContext) -> Result<()>/fn execute(\&mut self, ctx: \&mut PassContext) -> HelioResult<()>/g' crates/helio-pass-underwater/src/lib.rs

# Replace anyhow Result with HelioResult
sed -i 's/use anyhow::Result;$/use helio_v3::Result as HelioResult;/g' crates/helio-pass-underwater/src/lib.rs

# Remove anyhow dependencies
sed -i '/anyhow = "1.0"/d' crates/helio-pass-water-surface/Cargo.toml
sed -i '/anyhow = "1.0"/d' crates/helio-pass-underwater/Cargo.toml

