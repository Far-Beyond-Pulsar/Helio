//! Resident shadow receivers stop at the surface, including fractional relief.
mod common;
use common::*;
use wgpu::util::DeviceExt;

fn function<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("fn {name}(")).unwrap();
    &source[start..start + source[start..].find("\n}").unwrap() + 2]
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Probe { integers: [i32; 4], values: [f32; 4] }

#[test]
fn filtered_riser_receiver_reaches_resident_top_without_overshooting() {
    let Some(gpu) = gpu() else { return };
    let surface = include_str!("../shaders/surface.wgsl");
    let trace = include_str!("../shaders/trace.wgsl");
    let helpers = [function(trace, "layer_height"), function(trace, "height_rel"),
        function(trace, "relief_height"), function(surface, "filtered_shadow_lift")].join("\n");
    // Stub only resident accessors; execute the actual production height and
    // receiver arithmetic. No convenient global-f32 height reconstruction.
    let source = format!(r#"
struct Frame {{ eye:vec4<f32>, layer:vec4<f32>, layer_i:vec4<i32> }}
struct Column {{ top:i32, fraction:u32, info:u32, fits:u32 }}
struct Hit {{ t:f32, i:i32, j:i32, info:u32 }}
struct Ray {{ eo:f32, ee:f32, ol:f32, el:f32 }}
struct Probe {{ integers:vec4<i32>, values:vec4<f32> }}
const INFO_RELIEF:u32=0x10000000u;
const INFO_TOPOLOGY:u32=0x08000000u;
var<private> frame:Frame;
override PLANE:bool=false;
fn is_plane()->bool {{ return PLANE; }}
fn column_tops_fit(c:Column)->bool {{ return c.fits!=0u; }}
fn column_top(c:Column,x:u32,y:u32)->i32 {{ return c.top; }}
fn column_relief_fraction(c:Column,x:u32,y:u32)->u32 {{ return c.fraction; }}
{helpers}
@group(0) @binding(0) var<storage,read> probes:array<Probe>;
@group(0) @binding(1) var<storage,read_write> answers:array<vec4<f32>>;
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{
    if id.x>=arrayLength(&probes) {{ return; }}
    let p=probes[id.x];
    frame.eye=vec4<f32>(0.0,1.0,0.0,6371000.0);
    frame.layer=vec4<f32>(0.375,p.values.x,0.0,0.0);
    frame.layer_i=vec4<i32>(p.integers.x,0,0,0);
    let flags=bitcast<u32>(p.integers.w);
    let c=Column(p.integers.z,u32(p.values.w),flags&0xfffffffeu,flags&1u);
    let h=Hit(0.0,0,0,u32(p.integers.y)<<5u);
    // A radial ray offset at t=0 tests both the plane and stable sphere formula.
    let r=Ray(p.values.y,p.values.y*p.values.y,0.0,0.0);
    let lift=filtered_shadow_lift(c,h,r);
    let hit_height=height_rel(r,h.t);
    answers[id.x]=vec4<f32>(lift,hit_height+lift,
        relief_height(c,0,0,u32(p.integers.y),c.fraction),hit_height);
}}
"#);
    let module=gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:Some("production filtered shadow receiver"),source:wgpu::ShaderSource::Wgsl(source.into()),
    });
    let mut probes=Vec::new();
    let mut expected=Vec::new();
    for level in [0,1,4,8,16] {
        for eye in [-49_807_363i32,49_807_363] {
            for voxel in [0.1f32,1.0] {
                let top=eye.div_euclid(1<<level)+1;
                for fraction in [0u32,8192,32768,65535] {
                    // L0 has no fractional relief metadata in production.
                    if level==0 && fraction!=0 { continue; }
                    let target=(f64::from((top<<level)-eye)-0.375)*f64::from(voxel)
                        - if fraction==0 {0.0} else {
                            (1.0-f64::from(fraction)/65536.0)*f64::from(1<<level)*f64::from(voxel)
                        };
                    for displacement in [-0.25f64,0.0,0.25] {
                        let hit=(target+displacement*f64::from(voxel)*f64::from(1<<level)) as f32;
                        let relief=if level==0 {0} else {0x10000000};
                        probes.push(Probe {integers:[eye,level,top,relief|1],
                            values:[voxel,hit,0.0,fraction as f32]});
                        expected.push((target,true));
                    }
                    // Topology and truncated top headers must never move a receiver.
                    for flags in [relief_flags(level)|0x08000000|1,relief_flags(level)] {
                        probes.push(Probe {integers:[eye,level,top,flags],
                            values:[voxel,(target-0.25) as f32,0.0,fraction as f32]});
                        expected.push((target,false));
                    }
                }
            }
        }
    }
    let input=gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:None,contents:bytemuck::cast_slice(&probes),usage:wgpu::BufferUsages::STORAGE,
    });
    let bytes=(probes.len()*16) as u64;
    let output=gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label:None,size:bytes,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false,
    });
    for plane in [true,false] {
        let pipeline=gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:None,layout:None,module:&module,entry_point:Some("probe"),
            compilation_options:wgpu::PipelineCompilationOptions {
                constants:&[("PLANE",if plane {1.0} else {0.0})],..Default::default()
            },cache:None,
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
        for (n,(target,allowed)) in expected.iter().copied().enumerate() {
            let [lift,receiver,stored,hit]=actual[n];
            // Relative operands can be kilometres while their sum is near
            // zero. Bound the actual f32 operations by their operand ULPs,
            // rather than granting precision that cancellation cannot retain.
            let cell=f64::from(probes[n].values[0])*f64::from(1<<probes[n].integers[1]);
            let scale=cell.max(f64::from(hit).abs()).max(target.abs()).max(1.0);
            let tolerance=4.0*f64::from(f32::EPSILON)*scale;
            assert!((f64::from(stored)-target).abs()<=tolerance,"resident target probe{n}");
            assert!(lift>=0.0,"receiver must never move down probe{n}");
            if !allowed { assert_eq!(lift,0.0,"guard probe{n}"); continue; }
            let wanted=f64::from(hit).max(target);
            assert!((f64::from(receiver)-wanted).abs()<=tolerance,
                "plane={plane}, probe{n}: receiver{receiver}, target{wanted}");
        }
    }
}

fn relief_flags(level:i32)->i32 { if level==0 {0} else {0x10000000} }
