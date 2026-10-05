//! Production face UV arithmetic must preserve fractions at planetary origins.
mod common;
use common::*;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Probe { eye: [i32; 4], hit: [i32; 4], relative: [f32; 4] }

#[test]
fn planetary_face_uv_preserves_cell_fractions_and_shared_boundaries() {
    let Some(gpu) = gpu() else { return };
    let surface = include_str!("../shaders/surface.wgsl");
    let start = surface.find("fn face_local_cell(").unwrap();
    let helper = &surface[start..start + surface[start..].find("\n}").unwrap() + 2];
    let source = format!(r#"
{helper}
struct Probe {{ eye:vec4<i32>, hit:vec4<i32>, relative:vec4<f32> }}
@group(0) @binding(0) var<storage,read> probes:array<Probe>;
@group(0) @binding(1) var<storage,read_write> answers:array<vec4<f32>>;
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{
    if id.x>=arrayLength(&probes) {{return;}}
    let p=probes[id.x];
    let cell=face_local_cell(p.eye.xyz,p.hit.xyz,p.relative.xyz,u32(p.hit.w));
    let axis=u32(p.eye.w)>>1u;
    var u_axis=select(0u,1u,axis==0u);
    var v_axis=select(2u,1u,axis==2u);
    if axis==2u {{u_axis=0u;v_axis=1u;}}
    let uv=clamp(vec2<f32>(cell[u_axis],cell[v_axis]),vec2<f32>(0.0),vec2<f32>(1.0));
    // Retain the rejected arithmetic only to certify that these probes expose
    // the old loss of fractions rather than testing small convenient indices.
    let old=(vec3<f32>(p.eye.xyz)+p.relative.xyz)/f32(1u<<u32(p.hit.w))-vec3<f32>(p.hit.xyz);
    answers[id.x*2u]=vec4<f32>(cell,0.0);
    answers[id.x*2u+1u]=vec4<f32>(uv,old[u_axis],old[v_axis]);
}}
"#);
    let mut probes=Vec::new();
    let mut expected=Vec::new();
    // Dyadic fractions give an exact independent f64 oracle. Include upper
    // boundary1 and the adjacent cell's0 at the identical ray coordinate.
    for level in [0,1,4,8,16] {
        let scale=1i32<<level;
        for sign in [-1,1] {
            for face in 0..6 {
                for code in 0..6 {
                    let base: [i32; 3]=[sign*(49_807_363+face*8),sign*(49_807_371-face*4),sign*18_003_379];
                    let hit=base.map(|x|x.div_euclid(scale));
                    let eye=[hit[0]*scale+13,hit[1]*scale-7,hit[2]*scale+5];
                    for fractions in [[0.0,0.25,0.75],[0.5,0.75,0.25],[1.0,1.0,1.0]] {
                        let relative=std::array::from_fn::<_,3,_>(|a|
                            (f64::from(hit[a])*f64::from(scale)-f64::from(eye[a])+fractions[a]*f64::from(scale)) as f32);
                        probes.push(Probe {eye:[eye[0],eye[1],eye[2],code],hit:[hit[0],hit[1],hit[2],level],
                            relative:[relative[0],relative[1],relative[2],0.0]});
                        expected.push(std::array::from_fn::<_,3,_>(|a|
                            (f64::from(eye[a])+f64::from(relative[a]))/f64::from(scale)-f64::from(hit[a])));
                        if fractions==[1.0;3] {
                            probes.push(Probe {eye:[eye[0],eye[1],eye[2],code],hit:[hit[0]+1,hit[1]+1,hit[2]+1,level],
                                relative:[relative[0],relative[1],relative[2],0.0]});
                            expected.push([0.0;3]);
                        }
                    }
                }
            }
        }
    }
    let module=gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:Some("production planetary face UV"),source:wgpu::ShaderSource::Wgsl(source.into()),
    });
    let input=gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:None,contents:bytemuck::cast_slice(&probes),usage:wgpu::BufferUsages::STORAGE,
    });
    let bytes=(probes.len()*32) as u64;
    let output=gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label:None,size:bytes,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false,
    });
    let pipeline=gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:None,layout:None,module:&module,entry_point:Some("probe"),compilation_options:Default::default(),cache:None,
    });
    let group=gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[
            wgpu::BindGroupEntry {binding:0,resource:input.as_entire_binding()},
            wgpu::BindGroupEntry {binding:1,resource:output.as_entire_binding()},
        ],
    });
    let mut encoder=gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass=encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);pass.set_bind_group(0,&group,&[]);
        pass.dispatch_workgroups((probes.len() as u32+63)/64,1,1);
    }
    gpu.queue.submit([encoder.finish()]);
    let data=read_buffer(&gpu,&output,bytes);
    let actual:&[[f32;4]]=bytemuck::cast_slice(&data);
    let mut old_wrong=0;
    for (n,(p,want)) in probes.iter().zip(expected).enumerate() {
        for a in 0..3 { assert_eq!(f64::from(actual[n*2][a]),want[a],"cell fraction probe{n} axis{a}"); }
        let (u,v)=match p.eye[3]>>1 {0=>(1,2),1=>(0,2),_=>(0,1)};
        assert_eq!([f64::from(actual[n*2+1][0]),f64::from(actual[n*2+1][1])],[want[u],want[v]],"face UV probe{n}");
        if actual[n*2+1][2] as f64!=want[u] || actual[n*2+1][3] as f64!=want[v] {old_wrong+=1;}
    }
    assert!(old_wrong>100,"fixture must expose the former planetary loss of UV fractions");
}
