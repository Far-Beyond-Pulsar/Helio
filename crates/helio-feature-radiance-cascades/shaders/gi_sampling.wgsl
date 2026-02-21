// Radiance cascades GI injection snippet
// This single line calls the apply_radiance_cascades_gi() function
// which is defined in gi_functions.wgsl (injected at FragmentPreamble)
    final_color = apply_radiance_cascades_gi(final_color, input.world_position, input.world_normal);
