// Water volumes/hitboxes are authored exclusively as SceneDB rows now
// (`helio_pass_water_sim::WaterVolumeComponent`/`WaterHitboxComponent` --
// see those components' docs). There is no Renderer-owned CPU arena for
// them any more; `WaterSimPass`/`DeferredLightPass` resolve
// `"water_volumes"`/`"water_hitboxes"` straight from `ctx.scene_buffers`
// every frame.
