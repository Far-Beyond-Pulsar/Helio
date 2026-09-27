struct CanonicalPosition { cell:vec3<i32>, fraction:vec3<f32> }

// Keep whole cells and the sub-cell displacement separate. Multiplying a
// distant ray into one f32 position discards authored voxels before traversal.
fn canonical_position(origin:vec3<i32>,fraction:vec3<f32>,ro:vec3<f32>,rd:vec3<f32>,t:f32)->CanonicalPosition {
    let travelled=rd*t;
    let travelled_error=fma(rd,vec3<f32>(t),-travelled);
    let units=travelled*10.0;
    let units_error=fma(travelled,vec3<f32>(10.0),-units)+travelled_error*10.0;
    let offset=ro*10.0;
    let offset_error=fma(ro,vec3<f32>(10.0),-offset);
    let offset_integral=floor(offset);
    let integral=floor(units);
    // Keep camera fraction before the local-origin terms. Reordering these
    // terms passed the isolated test but regressed integrated orbital rays.
    let remainder=(units-integral)+units_error+fraction+(offset-offset_integral)+offset_error;
    return CanonicalPosition(origin+vec3<i32>(integral)+vec3<i32>(offset_integral)+vec3<i32>(floor(remainder)),fract(remainder));
}
