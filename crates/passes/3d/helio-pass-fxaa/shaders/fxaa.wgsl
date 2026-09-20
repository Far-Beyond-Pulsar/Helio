// FXAA (Fast Approximate Anti-Aliasing) shader
// Edge-directed spatial filtering with bounded span search.

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

const EDGE_THRESHOLD_MIN: f32 = 0.0312;
const EDGE_THRESHOLD_MAX: f32 = 0.125;
const SUBPIXEL_QUALITY: f32 = 0.75;
const ITERATIONS: i32 = 12;

fn load_luma(pixel: vec2<i32>) -> f32 {
    let hi=vec2<i32>(textureDimensions(input_tex))-vec2<i32>(1);
    return rgb2luma(textureLoad(input_tex,clamp(pixel,vec2<i32>(0),hi),0).rgb);
}

fn rgb2luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = textureDimensions(input_tex);
    let texel_size = 1.0 / vec2<f32>(dimensions);
    let texel = vec2<i32>(in.uv * vec2<f32>(dimensions));
    
    // Sample center and neighbors
    let rgb_center = textureSampleLevel(input_tex, input_sampler, in.uv, 0.0).rgb;
    let luma_center = rgb2luma(rgb_center);
    
    let luma_down = load_luma(texel + vec2<i32>(0, -1));
    let luma_up = load_luma(texel + vec2<i32>(0, 1));
    let luma_left = load_luma(texel + vec2<i32>(-1, 0));
    let luma_right = load_luma(texel + vec2<i32>(1, 0));
    
    // Find min/max luma
    let luma_min = min(luma_center, min(min(luma_down, luma_up), min(luma_left, luma_right)));
    let luma_max = max(luma_center, max(max(luma_down, luma_up), max(luma_left, luma_right)));
    
    let luma_range = luma_max - luma_min;
    
    // Early exit if no edge
    if luma_range < max(EDGE_THRESHOLD_MIN, luma_max * EDGE_THRESHOLD_MAX) {
        return vec4<f32>(rgb_center, 1.0);
    }
    
    // Sample corners
    let luma_down_left = load_luma(texel + vec2<i32>(-1, -1));
    let luma_up_right = load_luma(texel + vec2<i32>(1, 1));
    let luma_up_left = load_luma(texel + vec2<i32>(-1, 1));
    let luma_down_right = load_luma(texel + vec2<i32>(1, -1));
    
    // Compute gradient
    let luma_down_up = luma_down + luma_up;
    let luma_left_right = luma_left + luma_right;
    let luma_left_corners = luma_down_left + luma_up_left;
    let luma_down_corners = luma_down_left + luma_down_right;
    let luma_right_corners = luma_down_right + luma_up_right;
    let luma_up_corners = luma_up_right + luma_up_left;
    
    let edge_horizontal = abs(-2.0 * luma_left + luma_left_corners) +
                         abs(-2.0 * luma_center + luma_down_up) * 2.0 +
                         abs(-2.0 * luma_right + luma_right_corners);
    let edge_vertical = abs(-2.0 * luma_up + luma_up_corners) +
                       abs(-2.0 * luma_center + luma_left_right) * 2.0 +
                       abs(-2.0 * luma_down + luma_down_corners);
    
    let is_horizontal = edge_horizontal >= edge_vertical;
    
    // Choose the perpendicular side from its contrast with the center,
    // not from the horizontal/vertical second-derivative edge metric.
    let luma1=select(luma_left,luma_down,is_horizontal);
    let luma2=select(luma_right,luma_up,is_horizontal);
    let gradient1=luma1-luma_center;
    let gradient2=luma2-luma_center;
    let negative_side=abs(gradient1)>=abs(gradient2);
    let gradient=max(abs(gradient1),abs(gradient2));
    let edge_luma=0.5*(luma_center+select(luma2,luma1,negative_side));
    let perpendicular=select(vec2<f32>(texel_size.x,0.0),vec2<f32>(0.0,texel_size.y),is_horizontal);
    let direction=select(1.0,-1.0,negative_side);
    let along=select(vec2<f32>(0.0,texel_size.y),vec2<f32>(texel_size.x,0.0),is_horizontal);
    let edge_uv=in.uv+perpendicular*(0.5*direction);
    var uv1=edge_uv-along;
    var uv2=edge_uv+along;
    var delta1=rgb2luma(textureSampleLevel(input_tex,input_sampler,uv1,0.0).rgb)-edge_luma;
    var delta2=rgb2luma(textureSampleLevel(input_tex,input_sampler,uv2,0.0).rgb)-edge_luma;
    var done1=abs(delta1)>=gradient*0.25;
    var done2=abs(delta2)>=gradient*0.25;
    for (var i=0;i<ITERATIONS;i++) {
        if done1 && done2 { break; }
        if !done1 {
            uv1-=along;
            delta1=rgb2luma(textureSampleLevel(input_tex,input_sampler,uv1,0.0).rgb)-edge_luma;
            done1=abs(delta1)>=gradient*0.25;
        }
        if !done2 {
            uv2+=along;
            delta2=rgb2luma(textureSampleLevel(input_tex,input_sampler,uv2,0.0).rgb)-edge_luma;
            done2=abs(delta2)>=gradient*0.25;
        }
    }
    let distance1=select(in.uv.y-uv1.y,in.uv.x-uv1.x,is_horizontal);
    let distance2=select(uv2.y-in.uv.y,uv2.x-in.uv.x,is_horizontal);
    let near1=distance1<distance2;
    let end_delta=select(delta2,delta1,near1);
    let found_end=select(done2,done1,near1);
    var edge_offset=0.0;
    if found_end && ((end_delta<0.0)!=(luma_center-edge_luma<0.0)) {
        edge_offset=0.5-min(distance1,distance2)/(distance1+distance2);
    }
    // Subpixel coverage from the complete 3x3 neighborhood, bounded to
    // half a pixel so filtering never crosses the adjacent pixel center.
    let average=(2.0*(luma_down_up+luma_left_right)+luma_left_corners+luma_right_corners)/12.0;
    let coverage=clamp(abs(average-luma_center)/luma_range,0.0,1.0);
    let smooth_coverage=coverage*coverage*(3.0-2.0*coverage);
    let subpixel_offset=min(0.5,smooth_coverage*smooth_coverage*SUBPIXEL_QUALITY);
    let offset=max(edge_offset,subpixel_offset)*direction;
    return vec4<f32>(textureSampleLevel(input_tex,input_sampler,in.uv+perpendicular*offset,0.0).rgb,1.0);
}
