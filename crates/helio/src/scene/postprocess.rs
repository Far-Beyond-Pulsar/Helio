// Post-process volumes are authored exclusively as
// `helio_pass_postprocess::PostProcessVolumeComponent` SceneDB rows now --
// see that component's doc. There is no Renderer-owned CPU arena for them
// any more; `PostProcessVolumeBlendPass` resolves `"post_process_volumes"`
// straight from `ctx.scene_buffers` every frame.
