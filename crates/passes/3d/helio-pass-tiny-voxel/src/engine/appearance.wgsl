// Nine visible first-hit samples, shaded independently by the engine. Depth
// proximity, source material and the final receiver mask reject incompatible
// neighbors. This is a biased spatial reconstruction, not a coverage oracle.
@group(0) @binding(32) var appearance_lighting:texture_2d<f32>;
@group(0) @binding(33) var appearance_receiver:texture_2d<f32>;
@group(0) @binding(34) var appearance_output:texture_storage_2d<rgba16float,write>;
fn appearance_owned(q:vec2<i32>)->bool {
    return all(textureLoad(appearance_receiver,q,0).xy==vec2<f32>(-1.0,-2.0));
}
@compute @workgroup_size(8,8)
fn appearance(@builtin(global_invocation_id) id:vec3<u32>) {
    let size=textureDimensions(appearance_lighting);
    if any(id.xy>=size) {return;}
    let pixel=vec2<i32>(id.xy);
    let color=textureLoad(appearance_lighting,pixel,0);
    if p.settings.z<=0.0 || !appearance_owned(pixel) {
        textureStore(appearance_output,pixel,color);return;
    }
    let hit=primary_hits[id.x+id.y*size.x];
    let cell=max(0.1,p.settings.w*0.1);
    let width=hit.distance*p.up.w*2.0/f32(size.y);
    if (hit.status&3u)!=1u || width<=cell {
        textureStore(appearance_output,pixel,color);return;
    }
    let blend=smoothstep(1.0,3.0,width/cell);
    let position=hit.normal*hit.distance;
    let material=(hit.status>>8u)&3u;
    let limit=max(width*4.0,cell*2.0);
    var sum=vec3<f32>(0.0);var total=0.0;
    for(var y=-1;y<=1;y++) {for(var x=-1;x<=1;x++) {
        let q=clamp(pixel+vec2<i32>(x,y),vec2<i32>(0),vec2<i32>(size)-vec2<i32>(1));
        let other=primary_hits[u32(q.x)+u32(q.y)*size.x];
        if (other.status&3u)!=1u || ((other.status>>8u)&3u)!=material || !appearance_owned(q) {continue;}
        if distance(other.normal*other.distance,position)>limit {continue;}
        let weight=exp(-f32(x*x+y*y)/(2.0*0.65*0.65));
        sum+=textureLoad(appearance_lighting,q,0).xyz*weight;total+=weight;
    }}
    textureStore(appearance_output,pixel,vec4<f32>(mix(color.xyz,sum/max(total,1e-20),blend),color.w));
}
