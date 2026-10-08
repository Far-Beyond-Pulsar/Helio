//! Exact shader helper checks; native appearance/performance remain separate gates.
mod common;
use common::*;
use helio_pass_voxel_planet::grid::face_axes;
use wgpu::util::DeviceExt;

#[test]
fn grazing_soil_lip_filters_radial_coverage_and_preserves_protected_faces() {
    let surface = include_str!("../shaders/surface.wgsl");
    let radial = surface.split("fn radial_material_span").nth(1).unwrap()
        .split("// Relief changes geometry").next().unwrap();
    let detail = surface.split("fn detail_filter_weight").nth(1).unwrap()
        .split("// Independent hash detail").next().unwrap();
    let natural = surface.split("fn natural_material_filter_allowed").nth(1).unwrap()
        .split("// A material's filtered").next().unwrap();
    let call = surface.split("soil_coverage = soil_lip_coverage").nth(1).unwrap()
        .split(';').next().unwrap();
    assert!(surface.contains("if axis < 2u && material_lip(material) != material {"));
    assert!(surface.contains("soil_coverage * (1.0 - appearance_w)"));
    assert!(surface.contains("ground_material(p, column_surface(c, x, y), ground.height, material_depth, slope, material_layer)"));
    let Some(gpu) = gpu() else { return };
    let source = format!(r#"
        struct Column {{info:u32, fit:u32}}
        struct Hit {{t:f32}}
        struct Probe {{lip_width:vec4<f32>, ray_flags:vec4<f32>, appearance:vec4<f32>}}
        const INFO_TOPOLOGY:u32=1u;
        const INFO_GENERATED:u32=2u;
        @group(0) @binding(0) var<storage,read> probes:array<Probe>;
        @group(0) @binding(1) var<storage,read_write> answers:array<vec4<f32>>;
        fn column_tops_fit(c:Column)->bool {{return c.fit!=0u;}}
        fn hit_up(t:f32,d:vec3<f32>)->vec3<f32> {{return vec3<f32>(0.0,0.0,1.0);}}
        fn detail_filter_weight{detail}
        fn natural_material_filter_allowed{natural}
        fn radial_material_span{radial}
        @compute @workgroup_size(32) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{
            if id.x>=arrayLength(&probes) {{return;}}
            let p=probes[id.x];
            let uv=vec2<f32>(0.5,p.lip_width.x);
            let lip=p.lip_width.y;
            let pixel=1.0/p.lip_width.z;
            let code=u32(p.lip_width.w);
            let flags=u32(p.ray_flags.w);
            let edited=(flags&1u)!=0u;
            let c=Column(select(0u,INFO_TOPOLOGY,(flags&2u)!=0u),select(1u,0u,(flags&4u)!=0u));
            let natural_material=natural_material_filter_allowed(edited,c);
            let d=p.ray_flags.xyz;
            let actual_normal=vec3<f32>(1.0,0.0,0.0);
            let h=Hit(p.appearance.y);
            let size=1.0;
            let appearance_w=p.appearance.x;
            var soil_coverage=0.0;
            if (code>>1u)<2u {{soil_coverage=soil_lip_coverage{call};}}
            answers[id.x]=vec4<f32>(soil_coverage,select(0.0,1.0,uv.y<1.0-lip),
                soil_coverage*(1.0-appearance_w),select(0.0,1.0,soil_coverage<1.0||appearance_w>0.0));
        }}
    "#);
    let ray = |cosine:f32, radial:bool| {
        let tangent=(1.0-cosine*cosine).sqrt();
        if radial {[cosine,0.0,tangent]} else {[cosine,tangent,0.0]}
    };
    let mut probes = Vec::<[[f32;4];3]>::new();
    for (v,lip,width,code,cosine,radial,flags,appearance) in [
        (0.2,0.27,20.0,0.0,0.01,true,0.0,0.0),
        (0.8,0.27,20.0,0.0,0.01,true,0.0,0.0),
        (0.2,0.27,20.0,0.0,0.001,true,0.0,0.0),
        (0.2,0.27,20.0,0.0,0.001,false,0.0,0.0),
        (0.8,0.27,20.0,0.0,0.001,false,0.0,0.0),
        (0.2,0.27,20.0,0.0,1.0,true,0.0,0.0),
        (0.8,0.27,20.0,0.0,1.0,true,0.0,0.0),
        (0.2,0.27,20.0,0.0,0.001,true,1.0,0.0),
        (0.2,0.27,20.0,0.0,0.001,true,2.0,0.0),
        (0.2,0.27,20.0,0.0,0.001,true,4.0,0.0),
        (0.2,0.27,20.0,4.0,0.001,true,0.0,0.0),
        (0.2,0.27,20.0,6.0,0.001,true,0.0,0.0),
        (0.7,0.27,20.0,0.0,0.05,true,0.0,0.0),
        (0.7,0.27,20.0,0.0,0.04,true,0.0,0.0),
        (0.2,0.87,20.0,0.0,-0.01,true,0.0,0.0),
        (0.2,0.27,20.0,0.0,0.01,true,0.0,0.6),
    ] {
        let d=ray(cosine,radial);
        probes.push([[v,lip,width,code],[d[0],d[1],d[2],flags],[appearance,100.0,0.0,0.0]]);
    }
    let shader=gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:Some("production radial soil coverage"),source:wgpu::ShaderSource::Wgsl(source.into()),
    });
    let input=gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:None,contents:bytemuck::cast_slice(&probes),usage:wgpu::BufferUsages::STORAGE,
    });
    let output=gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label:None,size:(probes.len()*16) as u64,
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
        pass.dispatch_workgroups(1,1,1);
    }
    gpu.queue.submit([encoder.finish()]);
    let bytes=read_buffer(&gpu,&output,(probes.len()*16) as u64);
    let actual:&[[f32;4]]=bytemuck::cast_slice(&bytes);
    for (index,(p,a)) in probes.iter().zip(actual).enumerate() {
        let [v,lip,width,code]=p[0].map(f64::from);
        let [dx,dy,dz,flags]=p[1].map(f64::from);
        let point=if v<1.0-lip {1.0} else {0.0};
        let coverage=if code>=4.0 {0.0} else if flags!=0.0 {point} else {
            // Independent double ray/plane intersections along two screen
            // tangents. The derivative of radial hit position gives the
            // pixel interval without reproducing the production cross form.
            let d=glam::DVec3::new(dx,dy,dz).normalize();
            let e0=glam::DVec3::Y.cross(d).normalize();
            let e1=d.cross(e0);
            let epsilon=1e-7;
            let hit=|direction:glam::DVec3| direction*(100.0*d.x/direction.x);
            let derivative=|e:glam::DVec3| (hit(d+e*epsilon).z-hit(d-e*epsilon).z)/(2.0*epsilon*100.0);
            let span=derivative(e0).hypot(derivative(e1))/width;
            // The filter follows the soil band's own width (1 - lip), not
            // the whole face: a thin band aliases while the face resolves.
            let t=(((1.0-lip)/span-0.75)/0.5).clamp(0.0,1.0);
            let weight=1.0-t*t*(3.0-2.0*t);
            let lo=(v-span/2.0).max(0.0);
            let hi=(v+span/2.0).min(1.0);
            let n=100_000;
            let covered=(0..n).filter(|i| (lo+(hi-lo)*(*i as f64+0.5)/n as f64) < 1.0-lip).count();
            point*(1.0-weight)+(covered as f64/n as f64)*weight
        };
        assert!((f64::from(a[0])-coverage).abs()<2e-5,"soil probe{index}: {a:?}, expected{coverage}");
        assert_eq!(f64::from(a[1]),point);
        assert!((f64::from(a[2])-coverage*(1.0-f64::from(p[2][0]))).abs()<2e-5);
        assert_eq!(a[3],if coverage<1.0 || p[2][0]>0.0 {1.0} else {0.0},
            "grass branch must execute for fractional coverage and preserve full-soil protection");
        // Distinct custom palette endpoints obey the same coverage. IDs and
        // occupied geometry are not replaced by a thresholded material.
        for (grass,dirt) in [(0.2,0.8),(0.9,0.1)] {
            let colour=grass+(dirt-grass)*f64::from(a[2]);
            let expected=grass+(dirt-grass)*coverage*(1.0-f64::from(p[2][0]));
            assert!((colour-expected).abs()<2e-5);
        }
    }
    assert!((actual[0][0]-actual[1][0]).abs()<1e-6,"unresolved radial phase must converge");
    assert_eq!(actual[3][0],1.0,"lateral grazing must not erase a resolved radial lip");
    assert_eq!(actual[4][0],0.0);
    for index in 7..=9 {assert_eq!(actual[index][0],1.0,"protected side{index}");}
}

#[test]
fn grazing_projection_uses_actual_support_and_preserves_edit_guards() {
    let surface = include_str!("../shaders/surface.wgsl");
    let helpers = surface.split("fn detail_filter_weight").nth(1).unwrap()
        .split("// Coplanar grazing").next().unwrap();
    let gate = surface.split("var projection = vec2<f32>(1.0);").nth(1).unwrap()
        .split("let hash_filter_w").next().unwrap();
    // Projected appearance is not an input to canonical classification or
    // the existing material-depth/layer selection.
    assert!(surface.contains("let top_material = code < 4u && smooth_w > 0.5;"));
    assert!(surface.contains("ground_material(p, column_surface(c, x, y), ground.height, material_depth, slope, material_layer)"));
    let Some(gpu) = gpu() else { return };
    let source = format!(r#"
        struct Column {{info:u32}}
        const INFO_TOPOLOGY:u32=1u;
        @group(0) @binding(0) var<storage,read> probes:array<vec4<f32>>;
        @group(0) @binding(1) var<storage,read_write> answers:array<vec4<f32>>;
        fn detail_filter_weight{helpers}
        @compute @workgroup_size(16) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{
            if id.x>=arrayLength(&probes) {{return;}}
            let p=probes[id.x];
            let code=u32(p.z);
            let flags=u32(p.w);
            let edited=(flags&1u)!=0u;
            let c=Column(select(0u,INFO_TOPOLOGY,(flags&2u)!=0u));
            let actual_normal=vec3<f32>(1.0,0.0,0.0);
            let d=vec3<f32>(p.y,sqrt(max(1.0-p.y*p.y,0.0)),0.0);
            var projection=vec2<f32>(1.0);
            {gate}
            let support=p.x*projection;
            answers[id.x]=vec4<f32>(support,detail_filter_weight(support.x),detail_filter_weight(support.y));
        }}
    "#);
    // Width, signed incidence, face code, edited/topology bits.
    let probes = [
        [20.0f32,0.01,4.0,0.0], [20.0,0.001,4.0,0.0],
        [20.0,0.01,0.0,0.0], [2.0,1.0,4.0,0.0],
        [5.0,0.5,0.0,0.0], [100.0,0.1,0.0,0.0],
        [20.0,0.001,6.0,0.0], [20.0,0.001,4.0,1.0],
        [20.0,0.001,0.0,2.0], [20.0,-0.01,4.0,0.0],
        [20.0,0.0,4.0,0.0],
    ];
    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:Some("grazing production appearance support"),source:wgpu::ShaderSource::Wgsl(source.into()),
    });
    let input = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:None,contents:bytemuck::cast_slice(&probes),usage:wgpu::BufferUsages::STORAGE,
    });
    let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label:None,size:(probes.len()*16) as u64,
        usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false,
    });
    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:None,layout:None,module:&shader,entry_point:Some("probe"),
        compilation_options:Default::default(),cache:None,
    });
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[
            wgpu::BindGroupEntry {binding:0,resource:input.as_entire_binding()},
            wgpu::BindGroupEntry {binding:1,resource:output.as_entire_binding()},
        ],
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass=encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);pass.set_bind_group(0,&group,&[]);
        pass.dispatch_workgroups(1,1,1);
    }
    gpu.queue.submit([encoder.finish()]);
    let bytes=read_buffer(&gpu,&output,(probes.len()*16) as u64);
    let actual:&[[f32;4]]=bytemuck::cast_slice(&bytes);
    for (index,(p,a)) in probes.iter().zip(actual).enumerate() {
        // Orthographic projection of a unit square has compressed length
        // |N.D| and area |N.D|, hence area-equivalent length sqrt(|N.D|).
        let cosine=if p[2]>=6.0 || p[3]!=0.0 {1.0} else {f64::from(p[1]).abs()};
        let axis=f64::from(p[0])*cosine;
        let area=f64::from(p[0])*cosine.sqrt();
        let filter=|x:f64| {
            let t=((x-0.75)/0.5).clamp(0.0,1.0);
            1.0-t*t*(3.0-2.0*t)
        };
        for (component,expected) in [axis,area,filter(axis),filter(area)].into_iter().enumerate() {
            assert!((f64::from(a[component])-expected).abs()<2e-5,"probe{index} component{component}");
        }
    }
    assert_eq!(&actual[0][2..], &[1.0,0.0]);
    assert_eq!(&actual[1][2..], &[1.0,1.0]);
    for index in 3..=8 { assert_eq!(&actual[index][2..], &[0.0,0.0]); }
}

#[test]
fn smooth_ground_uses_physical_chart_slope_and_continuous_support() {
    let Some(gpu) = gpu() else { return };
    let surface = include_str!("../shaders/surface.wgsl");
    let helpers = surface
        .split("fn detail_filter_weight")
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
    let source = format!(
        r#"
        {noise}
        var<private> material_footprint:f32=0.0;
        var<private> material_radial_span:f32=0.0;
        var<private> material_coverage:f32=-1.0;
        var<private> material_coverage_ids:vec2<u32>=vec2<u32>(0u);
        var<private> material_mix:vec4<f32>=vec4<f32>(-1.0,0.0,0.0,0.0);
        var<private> material_mix_ids:vec4<u32>=vec4<u32>(0u);
        var<private> material_fleck_base:u32=0u;
        var<private> material_weathered_skin:bool=false;
        struct Face {{m_a:vec4<f32>,m_b:vec4<f32>}}
        struct Frame {{faces:array<Face,6>,layer:vec4<f32>}}
        struct Probe {{up:vec4<f32>,gradient:vec4<f32>,params:vec4<f32>}}
        @group(0) @binding(0) var<uniform> frame:Frame;
        @group(0) @binding(1) var<storage,read> probes:array<Probe>;
        @group(0) @binding(2) var<storage,read_write> answers:array<vec4<f32>>;
        override PLANE:bool=false;
        fn is_plane()->bool {{return PLANE;}}
        fn face_axis{axes}
        fn detail_filter_weight{helpers}
        @compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{
            if id.x>=arrayLength(&probes) {{return;}}
            let p=probes[id.x];
            answers[id.x*13u]=vec4<f32>(0.0,0.0,
                smooth_face_weight(0u,p.params.w,false),smooth_face_weight(4u,p.params.w,false));
            let base=detail_filter_weight(p.params.w);
            answers[id.x*13u+1u]=vec4<f32>(base,
                base*smooth_face_weight(0u,p.params.w*2.0,true),
                base*smooth_face_weight(0u,p.params.w*4.0,true),
                base*smooth_face_weight(0u,p.params.w*32.0,false));
            answers[id.x*13u+3u]=vec4<f32>(chart_gradient(u32(p.params.x),p.up.xyz,
                vec2<f32>(p.up.w,p.gradient.w),p.params.y),0.0);
            answers[id.x*13u+4u]=vec4<f32>(detail_filter_weight(1.5*appearance_projection(1.0,4u).x),
                detail_filter_weight(1.5*appearance_projection(0.1,4u).x),detail_filter_weight(20.0*appearance_projection(0.1,0u).x),
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
            for (index, pair) in pairs.iter().enumerate() {
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
                for component in 0..3 {
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
            }
            for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                // Materials no longer read a screen-space slope (one slope
                // field at every level: `material_slope`).
                for component in 2..4 {
                    assert!((a[component] as f64-e[component]).abs()<2e-5,"plane{plane} voxel{voxel} probe{index} component{component}: {:?} expected{:?}",a,e);
                }
            }
            assert_eq!(
                actual[3][2], 0.0,
                "resolved angular wall received the smooth ground normal"
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
