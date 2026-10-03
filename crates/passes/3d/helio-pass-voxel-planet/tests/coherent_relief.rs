//! Exact shader helper checks; native appearance/performance remain separate gates.
mod common;
use common::*;
use helio_pass_voxel_planet::grid::face_axes;
use wgpu::util::DeviceExt;
use bytemuck::Zeroable;
use helio_pass_voxel_planet::{landform::{self, LandformConstants}, terrain};

#[test]
fn canonical_relief_uses_physical_chart_slope_and_continuous_support() {
    let Some(gpu) = gpu() else { return };
    let surface = include_str!("../shaders/surface.wgsl");
    let helpers = surface
        .split("fn canonical_relief_confidence")
        .nth(1)
        .unwrap()
        .split("// A natural riser")
        .next()
        .unwrap();
    let world = include_str!("../shaders/world.wgsl");
    let axes = world
        .split("fn face_axis")
        .nth(1)
        .unwrap()
        .split("fn mul_q24")
        .next()
        .unwrap();
    let noise=include_str!("../shaders/noise.wgsl");
    let landform=include_str!("../shaders/landform.wgsl");
    let materials=world.split("const M_AIR").nth(1).unwrap().split("// Face bases").next().unwrap();
    let mut terrain_constants=LandformConstants::zeroed();
    terrain_constants.header=[0,100,7,123];
    terrain_constants.levels=[0,0,2_000_000,-8000];
    terrain_constants.shape=[0,16,0,0];
    terrain_constants.octaves[6].shift=10;
    let mut terrain_bytes = bytemuck::bytes_of(&terrain_constants).to_vec();
    terrain_bytes.resize(terrain_bytes.len() + 66 * 16, 0);
    let terrain_uniform=gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:Some("canonical slope material constants"),contents:&terrain_bytes,usage:wgpu::BufferUsages::UNIFORM,
    });
    let source = format!(
        r#"
        {noise}
        const M_AIR{materials}
        {landform}
        @group(0) @binding(3) var<uniform> terrain:TerrainConstants;
        var<private> material_footprint:f32=0.0;
        var<private> material_snow_mix:vec4<f32>=vec4<f32>(-1.0,0.0,0.0,0.0);
        var<private> material_rock_id:u32=0u;
        var<private> material_rock_base_id:u32=0u;
        struct Face {{m_a:vec4<f32>,m_b:vec4<f32>}}
        struct Frame {{faces:array<Face,6>,layer:vec4<f32>}}
        struct Probe {{up:vec4<f32>,gradient:vec4<f32>,params:vec4<f32>}}
        @group(0) @binding(0) var<uniform> frame:Frame;
        @group(0) @binding(1) var<storage,read> probes:array<Probe>;
        @group(0) @binding(2) var<storage,read_write> answers:array<vec4<f32>>;
        override PLANE:bool=false;
        fn is_plane()->bool {{return PLANE;}}
        fn face_axis{axes}
        fn canonical_relief_confidence{helpers}
        @compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{
            if id.x>=arrayLength(&probes) {{return;}}
            let p=probes[id.x];
            answers[id.x*13u]=vec4<f32>(canonical_relief_slope(u32(p.params.x),p.up.xyz,p.gradient.xyz,p.params.y),
                canonical_relief_confidence(p.params.z),canonical_relief_face_weight(0u,p.params.w,false),canonical_relief_face_weight(4u,p.params.w,false));
            let base=detail_filter_weight(p.params.w);
            answers[id.x*13u+1u]=vec4<f32>(base,
                base*canonical_relief_face_weight(0u,p.params.w*2.0,true),
                base*canonical_relief_face_weight(0u,p.params.w*4.0,true),
                base*canonical_relief_face_weight(0u,p.params.w*32.0,false));
            let dithers=array<f32,4>(0.0,0.25,0.5,1.0);
            var stencil:vec4<f32>;
            for (var n=0u;n<4u;n++) {{
                let distance=(p.params.w*10.0+1.0)/(1.0-0.5*dithers[n]);
                stencil[n]=canonical_stencil_weight(distance,10.0,dithers[n]);
            }}
            answers[id.x*13u+2u]=stencil;
            answers[id.x*13u+3u]=column_relief_gradient(u32(p.params.x),p.up.xyz,
                vec2<f32>(p.up.w,p.gradient.w),p.params.y);
            answers[id.x*13u+4u]=vec4<f32>(detail_filter_weight(1.5*appearance_projection(1.0,4u).x),
                detail_filter_weight(1.5*appearance_projection(0.1,4u).x),detail_filter_weight(5.0*appearance_projection(0.1,0u).x),
                detail_filter_weight(1.5*appearance_projection(0.1,6u).x));
            answers[id.x*13u+5u]=vec4<f32>(detail_filter_weight(1.0*appearance_projection(1.0,4u).x),
                detail_filter_weight(2.0*appearance_projection(0.5,4u).x),detail_filter_weight(2.0*appearance_projection(-0.5,4u).x),
                detail_filter_weight(1.0*appearance_projection(0.1,6u).x));
            answers[id.x*13u+6u]=vec4<f32>(detail_filter_weight(1.5*appearance_projection(1.0,4u).y),
                detail_filter_weight(1.5*appearance_projection(0.1,4u).y),
                detail_filter_weight(5.0*appearance_projection(0.1,0u).y),
                detail_filter_weight(1.5*appearance_projection(0.1,6u).y));
            // One-pixel coplanar grazing separation: 6 world footprints,
            // but one perpendicular footprint. The original 4fp sphere
            // rejects it; bounded anisotropic support admits it.
            let grazing=vec3<f32>(sqrt(35.0)/6.0,0.0,1.0/6.0);
            answers[id.x*13u+7u]=vec4<f32>(shadow_reuse_distance_squared(
                vec3<f32>(1.0,0.0,-sqrt(35.0)),vec3<f32>(0.0,0.0,1.0),grazing,1.0,1.0),
                shadow_reuse_distance_squared(vec3<f32>(1.0,2.0,3.0),
                    vec3<f32>(0.0,0.0,1.0),vec3<f32>(0.0,0.0,1.0),1.0,1.0),
                shadow_reuse_distance_squared(vec3<f32>(0.0,0.0,17.0),
                    vec3<f32>(0.0,0.0,1.0),grazing,1.0,1.0),
                shadow_reuse_distance_squared(vec3<f32>(5.0,0.0,0.0),
                    vec3<f32>(0.0,0.0,1.0),grazing,1.0,1.0));
            let delta=vec3<f32>(1.0,0.0,-sqrt(35.0));
            let view=vec3<f32>(0.0,0.0,1.0);
            answers[id.x*13u+8u]=vec4<f32>(shadow_reuse_distance_squared(delta,view,grazing,0.0,1.0),
                shadow_reuse_distance_squared(delta,view,grazing,1.0,0.0),
                shadow_reuse_distance_squared(delta,view,grazing,0.0,0.0),
                shadow_reuse_distance_squared(delta,view,grazing,1.0,1.0));
            // Quantized 1:8 shallow stair: sample every phase, including the
            // block whose endpoints remain on the same authored terrace.
            let phase=i32(id.x&7u);
            let secant=column_secant_derivative(phase/8,(phase+7)/8,0,0);
            let local=f32((phase+1)/8-(phase-1+8)/8+1)/2.0;
            answers[id.x*13u+9u]=vec4<f32>(local,secant.x,secant.y,
                f32((phase+7)/8-phase/8));
            let up=vec3<f32>(0.0,0.0,1.0);
            let field_normal=normalize(up-vec3<f32>(secant.x,0.0,0.0));
            let face=vec3<f32>(1.0,0.0,0.0);
            answers[id.x*13u+10u]=vec4<f32>(normalize(mix(face,field_normal,
                detail_filter_weight(2.0))),dot(field_normal,up));
            let offset=i32((id.x>>3u)&7u);
            let cell=i32(id.x&7u);
            let left=max(cell-1,0);
            let right=min(cell+1,7);
            let block=(((offset+7)*7/4)-(offset*7/4))*8/7;
            let stair_local=f32(((offset+right)*7/4)-(offset+left)*7/4)*8.0/f32(right-left);
            let point=vec3<i32>(i32(hash3(i32(id.x),0,0,123u)&0xffffffu)*2+1,
                1<<27,i32(hash3(i32(id.x),2,0,123u)&0xffffffu)*2+1);
            let weights=vec3<f32>(0.0,0.5,1.0);
            var fine_ids:vec3<f32>;
            var coarse_ids:vec3<f32>;
            for (var w=0u;w<3u;w++) {{
                fine_ids[w]=f32(ground_material(point,1000000,0,
                    filtered_material_slope(block,stair_local,weights[w],0u),9999)&M_ID);
                coarse_ids[w]=f32(ground_material(point,1000000,0,
                    filtered_material_slope(block,stair_local,weights[w],1u),9999)&M_ID);
            }}
            answers[id.x*13u+11u]=vec4<f32>(fine_ids,stair_local);
            answers[id.x*13u+12u]=vec4<f32>(coarse_ids,f32(block));
        }}
    "#
    );
    let smooth = |lo: f64, hi: f64, x: f64| {
        let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("coherent relief helpers"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    for plane in [true, false] {
        for voxel in [0.1f64, 0.3, 1.0] {
            let datum = 6_371_000.0;
            let delta = if plane { voxel } else { voxel / datum };
            let mut frame = [0f32; 52];
            for face in 0u8..6 {
                let [_, a, b] = face_axes(face);
                frame[(face as usize) * 8..(face as usize) * 8 + 3]
                    .copy_from_slice(&a.as_vec3().to_array());
                frame[(face as usize) * 8 + 4..(face as usize) * 8 + 7]
                    .copy_from_slice(&b.as_vec3().to_array());
            }
            frame[49] = voxel as f32;
            frame[50] = delta as f32;
            let mut probes = Vec::<[[f32; 4]; 3]>::new();
            let mut expected = Vec::<[f64; 4]>::new();
            let mut expected_gradient=Vec::<[f64;4]>::new();
            for face in 0u8..6 {
                let [n, a, b] = face_axes(face);
                for (ta, tb) in [(0.0f64, 0.0f64), (0.55, -0.4), (-0.3, 0.61)] {
                    for radius in [datum, datum * 1.01] {
                        let point =
                            |x: f64, y: f64| (n + a * x.tan() + b * y.tan()).normalize() * radius;
                        let up = if plane { n } else { point(ta, tb).normalize() };
                        let g0 = a * 0.7 + b * 0.3;
                        let gradient = g0 - up * g0.dot(up);
                        let eps = 1e-6;
                        // Independent finite differences of the actual cube-sphere
                        // chart, expressed per one index step / radial voxel.
                        let di = if plane {
                            gradient.dot(a) * delta / voxel
                        } else {
                            gradient.dot(point(ta + eps, tb) - point(ta - eps, tb)) / (2.0 * eps)
                                * delta
                                / voxel
                        };
                        let dj = if plane {
                            gradient.dot(b) * delta / voxel
                        } else {
                            gradient.dot(point(ta, tb + eps) - point(ta, tb - eps)) / (2.0 * eps)
                                * delta
                                / voxel
                        };
                        let slope = 8.0 * di.abs().max(dj.abs());
                        for (gradient_squared, projected_cell) in [
                            (3.99, 0.5),
                            (4.0, 0.75),
                            (4.01, 0.7501),
                            (6.5, 1.25),
                            (9.0, 4.0),
                            (4.0, 0.7499),
                            (4.0, 0.9999),
                            (4.0, 1.0),
                            (4.0, 1.0001),
                            (4.0, 1.2499),
                            (4.0, 1.2501),
                        ] {
                            probes.push([
                                up.as_vec3().extend(di as f32).to_array(),
                                gradient.as_vec3().extend(dj as f32).to_array(),
                                [
                                    face as f32,
                                    radius as f32,
                                    gradient_squared as f32,
                                    projected_cell as f32,
                                ],
                            ]);
                            expected_gradient.push([gradient.x,gradient.y,gradient.z,slope]);
                            expected.push([
                                slope,
                                1.0 - smooth(4.0, 9.0, gradient_squared),
                                1.0 - smooth(0.75, 1.25, projected_cell),
                                1.0,
                            ]);
                        }
                    }
                }
            }
            let buffer = |label, bytes: &[u8], usage| {
                gpu.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: bytes,
                        usage,
                    })
            };
            let frame = buffer(
                "chart frame",
                bytemuck::cast_slice(&frame),
                wgpu::BufferUsages::UNIFORM,
            );
            let input = buffer(
                "chart probes",
                bytemuck::cast_slice(&probes),
                wgpu::BufferUsages::STORAGE,
            );
            let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chart answers"),
                size: (probes.len() * 208) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let pipeline = gpu
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &shader,
                    entry_point: Some("probe"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[("PLANE", if plane { 1.0 } else { 0.0 })],
                        ..Default::default()
                    },
                    cache: None,
                });
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: frame.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding:3,resource:terrain_uniform.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups((probes.len() as u32 + 63) / 64, 1, 1);
            }
            gpu.queue.submit([encoder.finish()]);
            let bytes = read_buffer(&gpu, &output, (probes.len() * 208) as u64);
            let pairs: &[[[f32; 4]; 13]] = bytemuck::cast_slice(&bytes);
            let actual: Vec<[f32; 4]> = pairs.iter().map(|p| p[0]).collect();
            let mut material_disagreements=0usize;
            for (index, pair) in pairs.iter().enumerate() {
                material_disagreements+=usize::from(pair[11][2]!=pair[12][2]);
                let offset=((index>>3)&7) as i32;
                let cell=(index&7) as i32;
                let height=|i:i32| ((offset+i)*7).div_euclid(4);
                let canonical_slope=terrain::block_slope(|x,_|height(x),cell,0);
                let left=(cell-1).max(0);
                let right=(cell+1).min(7);
                let local=8.0*f64::from(height(right)-height(left))/f64::from(right-left);
                assert!((13..=14).contains(&canonical_slope) && canonical_slope<terrain_constants.shape[1]);
                assert_eq!(pair[11][3],local as f32,"stair fixture did not expose canonical-neighbour disagreement");
                assert_eq!(pair[12][3],canonical_slope as f32,"GPU block support disagrees with CPU query");
                let point=glam::IVec3::new((helio_pass_voxel_planet::noise::hash3(index as i32,0,0,123)&0xffffff) as i32*2+1,
                    1<<27,(helio_pass_voxel_planet::noise::hash3(index as i32,2,0,123)&0xffffff) as i32*2+1);
                let canonical_id=landform::ground_material(&terrain_constants,point,1000000,0,canonical_slope,9999)&terrain::material::ID;
                assert_eq!(&pair[11][..3],&[canonical_id as f32;3],
                    "filtered L0 material disagrees with canonical CPU query at phase{offset}/cell{cell}");
                for (w,weight) in [0.0,0.5,1.0].into_iter().enumerate() {
                    let coarse_slope=(f64::from(canonical_slope)*(1.0-weight)+local*weight) as i32;
                    let expected=landform::ground_material(&terrain_constants,point,1000000,0,coarse_slope,9999)&terrain::material::ID;
                    assert_eq!(pair[12][w],expected as f32,"existing coarse material blend changed");
                }
                let phase=index&7;
                let authored_height=|i:i32| i.div_euclid(8);
                let local=f64::from(authored_height(phase as i32+1)-authored_height(phase as i32-1))/2.0;
                let secant=f64::from(authored_height(phase as i32+7)-authored_height(phase as i32))/7.0;
                assert!((f64::from(pair[9][0])-local).abs()<1e-6);
                assert!((f64::from(pair[9][1])-secant).abs()<1e-6);
                assert_eq!(pair[9][2],0.0,"cross-slope appeared on a one-axis stair");
                assert!((secant-0.125).abs()<=0.125,
                    "block support failed to suppress the quantized local slope pulse");
                if local==0.5 {
                    assert!((secant-0.125).abs()<(local-0.125).abs(),
                        "filtered stair riser retained its half-cell derivative spike");
                }
                assert_eq!(&pair[10][..3],&[1.0,0.0,0.0],
                    "resolved face normal changed under secant shading support");
                assert!((f64::from(pair[10][3])-1.0/(1.0+secant*secant).sqrt()).abs()<1e-6,
                    "fully filtered stair normal disagrees with its endpoint support");
                for value in &pair[8][..3] {
                    assert!((*value-36.0).abs()<1e-5 && *value>16.0,
                        "topology/unfiltered query or representative lost its original distance cap");
                }
                assert!((pair[8][3]-3.1875).abs()<1e-5,
                    "both filtered samples must enable bounded anisotropic support");
                assert!((pair[7][0]-3.1875).abs()<1e-5 && pair[7][0]<16.0,
                    "coplanar grazing pixel failed bounded representative reuse");
                assert_eq!(pair[7][1],14.0,"head-on world cap changed");
                assert!(pair[7][2]>16.0,"depth extension exceeded its fourfold bound");
                assert_eq!(pair[7][3],25.0,"perpendicular screen separation changed");
                assert_eq!(pair[4],[0.0,1.0,0.0,0.0],
                    "grazing hash support must preserve resolved walls and unknown faces");
                assert_eq!(pair[5],[0.5,0.5,0.5,0.5],
                    "head-on support, incidence sign and unknown-face support changed");
                assert_eq!(pair[6],[0.0,1.0,0.0,0.0],
                    "area support must retain resolved long faces and head-on/unknown support");
                for component in 0..4 {
                    assert!((f64::from(pair[3][component])-expected_gradient[index][component]).abs()<2e-5,
                        "physical fallback gradient disagrees with independent chart finite differences: plane{plane} voxel{voxel} probe{index} component{component}: {:?} expected{:?}",
                        pair[3],expected_gradient[index]);
                }
                let weight=expected[index][2] as f32;
                for level in 0..3 {
                    assert!((pair[1][level]-weight).abs()<2e-5,
                        "authored filtering changed across selected L0/L1/L2 at probe{index}");
                }
                assert_eq!(pair[1][3],0.0,
                    "oversized streamed fallback wall lost its geometric normal at probe{index}");
                for (n, dither) in [0.0, 0.25, 0.5, 1.0].into_iter().enumerate() {
                    let distance = (f64::from(probes[index][2][3]) * 10.0 + 1.0) / (1.0 - 0.5 * dither);
                    // Independently enumerate both ends of compatible depth
                    // and primary-dither intervals, rather than copying the
                    // shader's selected-distance shortcut.
                    let tolerance = (distance * 0.02).max(1.0);
                    let mut nearest = f64::INFINITY;
                    for depth in [distance - tolerance, distance + tolerance] {
                        for noise in [0.0, 1.0] {
                            nearest = nearest.min(depth * (1.0 + dither * (noise - 0.5)));
                        }
                    }
                    let expected = smooth(10.0, 12.5, nearest);
                    assert!((f64::from(pair[2][n]) - expected).abs() < 2e-5,
                        "stencil support disagrees with primary bounds at probe{index} dither{dither}");
                    if nearest <= 10.0 {
                        assert_eq!(pair[2][n], 0.0, "L0-compatible stencil must not acquire canonical confidence");
                    }
                    if nearest >= 12.5 {
                        assert_eq!(pair[2][n], 1.0, "fully coarse stencil must retain canonical relief");
                    }
                }
            }
            assert!(material_disagreements>0,"stair fixture failed to expose the old local-derivative material stripes");
            for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                for component in 0..4 {
                    assert!((a[component] as f64-e[component]).abs()<2e-5,"plane{plane} voxel{voxel} probe{index} component{component}: {:?} expected{:?}",a,e);
                }
            }
            assert_eq!(actual[1][1], 1.0);
            assert!(
                actual[2][1] > 0.9999,
                "normal support jumped at old slope cutoff"
            );
            assert_eq!(
                actual[3][2], 0.0,
                "resolved angular wall received canonical height normal"
            );
            assert_eq!(actual[1][2], 1.0, "subpixel lower boundary must be fully averaged");
            assert_eq!(actual[7][2], 0.5, "one pixel must retain half the authored contrast");
            assert_eq!(actual[10][2], 0.0, "resolvable detail must retain full contrast");
            for (a, b) in [(1usize, 2usize), (6, 7), (7, 8), (9, 10)] {
                assert!((actual[a][2] - actual[b][2]).abs() < 0.001,
                    "filter discontinuity around one-pixel boundaries: {a}/{b}");
            }
            eprintln!(
                "coherent relief plane{plane} voxel{voxel}:{} slope/support cases",
                probes.len()
            );
        }
    }
}


#[test]
fn fractional_natural_surface_material_altitude_preserves_authored_strata() {
    let Some(gpu) = gpu() else { return };
    let surface=include_str!("../shaders/surface.wgsl");
    let helper=surface.split("fn surface_material_layer").nth(1).unwrap()
        .split("// A natural riser").next().unwrap();
    let source=format!(r#"
        struct Probe {{a:vec4<i32>,b:vec4<i32>,c:vec4<i32>}}
        @group(0) @binding(0) var<storage,read> probes:array<Probe>;
        @group(0) @binding(1) var<storage,read_write> answers:array<i32>;
        fn surface_material_layer{helper}
        @compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{
            if id.x>=arrayLength(&probes) {{return;}}
            let p=probes[id.x];
            answers[id.x]=surface_material_layer(p.a.x,u32(p.a.y),u32(p.a.z),p.a.w,
                p.b.x,u32(p.b.y),bitcast<f32>(p.b.z),p.b.w!=0,p.c.x!=0);
        }}
    "#);
    let mut probes=Vec::<[[i32;4];3]>::new();
    let mut expected=Vec::<i32>::new();
    for authored_top in [-531i32,0,1,24657,50004] {
        for level in 1u32..=16 {
            let size=1i32<<level;
            let remainder=authored_top.rem_euclid(size);
            let top=authored_top.div_euclid(size)+i32::from(remainder!=0);
            let fraction=(remainder as u32)<<(16-level);
            let coarse_hit=(top-1)*size;
            for (depth,code,filtered,relief,topology) in [
                (0,4,0.0f32,true,false),
                (0,0,1.0,true,false),
                (0,0,0.5,true,false),
                (0,0,0.49,true,false),
                (3,4,1.0,true,false),
                (0,4,1.0,true,true),
                (0,4,1.0,false,false),
                (0,5,1.0,true,false),
            ] {
                let corrected=relief&&!topology&&depth==0&&(code==4||(code<4&&filtered>0.5));
                probes.push([[top,fraction as i32,level as i32,coarse_hit],
                    [depth,code,filtered.to_bits() as i32,i32::from(relief)],
                    [i32::from(topology),0,0,0]]);
                expected.push(if corrected {authored_top-1} else {coarse_hit});
            }
        }
    }
    let shader=gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:Some("fractional material altitude"),source:wgpu::ShaderSource::Wgsl(source.into()),
    });
    let input=gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:None,contents:bytemuck::cast_slice(&probes),usage:wgpu::BufferUsages::STORAGE,
    });
    let output=gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label:None,size:(probes.len()*4) as u64,
        usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false,
    });
    let pipeline=gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:None,layout:None,module:&shader,entry_point:Some("probe"),
        compilation_options:Default::default(),cache:None,
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
    let bytes=read_buffer(&gpu,&output,(probes.len()*4) as u64);
    let actual:&[i32]=bytemuck::cast_slice(&bytes);
    assert_eq!(actual,expected.as_slice(),"fractional surface layers changed authored strata or exposed cut/wall depth");
    for mm in [100i32,300,1000] {
        for (index,(actual,expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!((actual*mm).div_euclid(4500),(expected*mm).div_euclid(4500),"rock band probe{index}");
            assert_eq!((actual*mm).div_euclid(2100),(expected*mm).div_euclid(2100),"sandstone/clay band probe{index}");
        }
    }
}
