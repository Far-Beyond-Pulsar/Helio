# Planet maps: drainage, rivers, lakes and climate

The terrain's octaves are local: each column is noise at its own point.
Nothing in them knows where water goes, where the coast is or how far a
place lies from the sea, so a planet of them has no drainage basins,
rivers, lakes or climate. A stack's **planet maps** give it that context:
baked once per world from its seed and layers, deterministic, integer
data the CPU and the GPU sample with the same integer arithmetic
(`src/planet_maps.rs`, `shaders/planet_maps.wgsl`).

## What a stack bakes

- A `Climate` (`layers::Climate`) with `maps` bakes temperature and
  moisture over the planet; without, the climate of its `latitude_deg`
  holds everywhere (heights still cool it).
- A `Rivers` layer (`LayerKind::Rivers`) bakes the drainage and carves its
  rivers and lakes into the layers before it in the stack.

Earth's preset has both. The Moon, the desert and flat worlds are
unmapped (no bake, no table); voxel_craft's 16 km world keeps one climate.

## Layout

Maps cover the planet as six **gnomonic** cube faces (one square over a
plane), `n` texels a side at about `texel_km` at face centres (Earth: 25
km, 510 a side). A lookup takes a domain point (any level's column, a
cell's volume point): its face is its largest component's, and its face
coordinate `u = (a . p) / (n . p)` comes from one Q30 reciprocal
(`landform::recip_q30`): no transcendental functions, the same bits on both
sides, the same map at every level and voxel size.

The program's table (`TerrainProgram::tables`, `terrain_tables`, binding
40, `array<vec4<u32>>`) holds:

1. **River index**: per face `(n + 4)^2` words, a ring of two texels from
   the neighbouring faces around each face (a plane's ring repeats its
   edge). A word is a record id + 1 (0: none) and the record's reach in
   texels (Q4, top byte).
2. **Climate**: per face `(n/2 + 2)^2` words at half the resolution with a
   ring of one: sea-level annual mean temperature (1/128 °C, low half) and
   moisture (Q16, high half). Bilinear between texel centres; the rings
   make the faces' seams continuous.
3. **River segments**, eight words each: the node (a domain point on a
   16-unit lattice, jittered within the middle half of its texel), the
   valley's geometry (wall slope Q6, floodplain half-width in 8 m, channel
   half-width in 0.5 m, a lake flag), the unit direction to the downstream
   node (Q15) and the segment's length (64 units), the bed at the node
   (mm), its drop to the downstream node's bed and the channel's depth.

`LandformConstants::maps` (`TerrainConstants::maps`) is the header: the
flags, sizes and offsets, the plane's half width, an unmapped world's
uniform climate, the lapse rate and the snow temperature, the valleys'
shoulder rounding.

## Bake

`planet_maps::bake`, once per process and world (`planet_maps::maps`
caches by the stack's constants, map settings and grid):

1. **Coarse heights**: the mean (and highest) of 2x2 samples a texel of the
   field without its rivers (`height_parts` without maps), at the level
   that resolves wavelengths of half a texel.
2. **Climate** at half resolution:
   - temperature: Earth's zonal means between `equator_c` and `pole_c` by
     latitude (around `pole`; a plane's from `latitude_deg` at its origin,
     north along -Z), with a little noise;
   - moisture: the rain belts by latitude (wet tropics, dry subtropics,
     storm tracks, dry poles), times the way to the sea: marching against
     the prevailing wind (trades and polar easterlies, westerlies) to the
     first sea texel, and across the coast any way (Dijkstra); rain
     falls off with distance (`continentality_km`) and behind ground
     higher than the place (`rain_shadow_m`); noise; then the rivers add
     moisture to the ground along them.
3. **Drainage**: priority flood (Barnes et al. 2014, with an epsilon of a
   millimetre) from the sea, a plane's edges and dry basins (local minima
   where moisture is under 0.22: endorheic sinks; wetter basins fill and
   spill: lakes). Every texel drains to its steepest lower neighbour (8
   neighbours, across cube edges by 3D position); discharge is the wet
   area (area times moisture) upstream, accumulated from the highest
   filled texel down.
4. **Rivers**: every land texel whose discharge reaches the layer's source
   (`region_km`² of wet land) and every lake (filled 2 m or more) gets a
   segment to its downstream texel's node. Beds: the filled height less
   the incision (a lake's is its level), made non-increasing downstream.
   Floodplain and channel grow with the square root of the discharge,
   incision and depth with its fourth root. Every quantity is in metres
   and powers of a half, with only IEEE basic operations and square roots
   (sines by their series), in a fixed order: every machine bakes the same
   maps.

Earth (seed 7, texels of 25 km): see the measurements below.

## Carving

The `Rivers` layer cuts what lies before it in the stack
(`planet_maps::carve`, `map_carve` in WGSL). A column's lookup visits the
5x5 index window around its texel; a record whose reach (in texels, a
box test) cannot meet the column is skipped. A segment's surface at the
column is

    bed(t) + wall * max(d - floodplain, 0) - channel(d)

with `d` the 3D distance (domain units, `unit_q30`) to the segment, `t`
the position along it (beds interpolate from node to node), `channel(d)`
a parabola of the channel's depth over its half-width; the walls' foot
curves over half the floodplain (at most 409 m), so slopes never jump
there. The column's height
is the smooth minimum (`landform::smooth_min`, the layer's rounding) of
its terrain and the lowest surface of the segments within reach. A union
of continuous surfaces is continuous: confluences merge, segments meet at
their nodes, faces' rings hold the same 3D segments.

A record's reach is where its walls clear the highest ground the terrain
may have near it (the texels' highest samples plus every octave the maps
do not resolve), at most 0.8 of a texel: the walls steepen where a valley
would otherwise reach past it, so that no segment of a texel outside the
window can touch a column (a segment's texels lie two texels or more from
the column's).

**Levels.** Every level carves the same valleys (point samples of the same
surface). The channel shows where it spans four of the level's cells and
fades out by two (coarser levels keep its floodplain), so coarse levels
are band-limited samples and stay at or above finer ones.

**Bounds.** The layer's height range drops by the deepest incision and
channel; its slope bound (walls, beds and channels together,
`PlanetMaps::valley_bounds`) adds to the bound margins (`check_field`
samples them).

**Surface word.** A column in a channel whose bed is its ground gets the
surface word `RIVER_BED` (0x80; erosion bytes stop at -127): Earthlike
materials draw gravel and sand over silt there, rules count it as the
deepest gully and may require or refuse it (`MaterialRule::river_bed`).

## Climate in materials and features

`terrain_climate(p, height)` (`TerrainField::climate`): the sea-level
temperature less the lapse rate at `height`, and the moisture.

- **Earthlike**: dry lands where moisture is under 0.3 (sand over sandstone
  and clay strata); the snowline is where the lapse rate brings the
  sea-level temperature to `snow_c` (lower toward the poles, under the
  tropics' 4 km), moved by moisture and noise as before; its alpine rock
  band under it.
- **Rules**: moisture and temperature ranges at the cell.
- **Features**: a spawn's temperature and moisture ranges at its ground
  (with its height wander); Earth's woods need rain, broadleaves give way
  to conifers below 7.5 °C, and the tree line is at 0.5 °C, so forests
  follow the climate and tree lines fall toward the poles.

## Costs

TODO measurements.
