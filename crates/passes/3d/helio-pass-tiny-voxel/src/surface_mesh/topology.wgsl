struct Quad { origin_face_material:u32, extent:u32 }
fn quad_vertex(q:Quad,vertex_id:u32)->vec4<f32> {
    let low=vec3<f32>(f32(q.origin_face_material&63u),f32((q.origin_face_material>>6u)&63u),f32((q.origin_face_material>>12u)&63u));
    let face=(q.origin_face_material>>18u)&7u;let axis=face/2u;
    let u=(axis+1u)%3u;let v=(axis+2u)%3u;
    let width=q.extent&63u;let height=(q.extent>>6u)&63u;
    let triangle=vertex_id/3u;
    let component=select(vertex_id%3u,(3u-vertex_id%3u)%3u,(face&1u)!=0u);
    var point=255u;
    if width==1u && height==1u {
        if triangle>=2u {return vec4<f32>(0.0);}
        point=array<u32,6>(0u,1u,2u,0u,2u,3u)[triangle*3u+component];
    } else {
        let perimeter=2u*(width+height);
        if triangle>=perimeter {return vec4<f32>(0.0);}
        if component>0u {point=(triangle+component-1u)%perimeter;}
    }
    var uv=vec2<f32>(f32(width),f32(height))*0.5;
    if point<width {uv=vec2<f32>(f32(point),0.0);}
    else if point<width+height {uv=vec2<f32>(f32(width),f32(point-width));}
    else if point<2u*width+height {uv=vec2<f32>(f32(2u*width+height-point),f32(height));}
    else if point<2u*(width+height) {uv=vec2<f32>(0.0,f32(2u*(width+height)-point));}
    var position=low;
    position[u]+=uv.x;position[v]+=uv.y;
    return vec4<f32>(position,1.0);
}
