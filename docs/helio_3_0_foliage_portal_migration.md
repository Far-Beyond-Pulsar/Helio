# Helio 3.0 foliage and portal migration

Foliage and portal authoring now terminates in typed SceneDB columns owned by the
consuming passes:

| domain | SceneDB keys | owner |
| --- | --- | --- |
| foliage | `foliage_types`, `foliage_layers`, `foliage_interactors`, `foliage_wind` | `helio-pass-foliage-place` |
| portals | `portal_views`, `portal_chains` | `helio-pass-portal-cull` |

The foliage placement and G-buffer passes, portal cull, portal mask, portal instances,
and editor overlay resolve these keys from `ctx.scene_buffers` and bind the returned
GPU buffers. The renderer no longer uploads or publishes foliage/portal data through
`generic transient resource registry`; the old renderer façade methods and foliage cache were removed.

The packed records intentionally retain the established 96/32/32/48-byte foliage and
144/16-byte portal layouts. The existing frame slots in `libhelio::generic transient resource registry`
remain only because this migration is not allowed to edit `libhelio` or `helio-core`.
They are now unused by these domains and should be deleted in the next core-interface
change, along with the corresponding `FoliageFrameData`/`PortalsFrameData` types.

## Integration follow-up

The parent engine backend should register the new pass-owned records alongside its
other explicit GPU-column registrations:

```text
helio_pass_foliage_place::components::{FoliageTypeComponent,
  FoliageLayerComponent, FoliageInteractorComponent, FoliageWindComponent}
helio_pass_portal_cull::components::{PortalViewComponent, PortalChainComponent}
```

Portal pair-map/coordinate-space generation remains a separate transient scene service;
the direct SceneDB view record uses coordinate-space zero until that service exposes a
typed SceneDB-owned portal-space projection.
