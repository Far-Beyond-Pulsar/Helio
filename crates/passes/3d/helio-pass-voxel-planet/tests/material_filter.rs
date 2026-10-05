//! Appearance-only procedural filtering; canonical material queries remain exact.
mod common;
use common::*;
use wgpu::util::DeviceExt;

#[test]
fn unresolved_outcrop_preserves_canonical_ids_and_mean_palette() {
    let Some(gpu) = gpu() else { return };
    let noise = include_str!("../shaders/noise.wgsl");
    let landform = include_str!("../shaders/landform.wgsl");
    let world = include_str!("../shaders/world.wgsl");
    let materials = world
        .split("const M_AIR")
        .nth(1)
        .unwrap()
        .split("// Face bases")
        .next()
        .unwrap();
    let surface = include_str!("../shaders/surface.wgsl");
    let rock_helper = surface
        .split("fn filtered_rock_flecks")
        .nth(1)
        .unwrap()
        .split("\nfn ")
        .next()
        .unwrap();
    let source = format!(
        r#"
        {noise}
        const M_AIR{materials}
        {landform}
        fn palette(id:u32)->vec3<f32> {{
            let p=array<vec3<f32>,16>(vec3<f32>(200.,0.,200.),vec3<f32>(91.,125.,65.),
                vec3<f32>(120.,87.,61.),vec3<f32>(133.,139.,142.),vec3<f32>(203.,188.,151.),
                vec3<f32>(217.,228.,236.),vec3<f32>(28.,72.,92.),vec3<f32>(116.,111.,102.),
                vec3<f32>(185.,142.,104.),vec3<f32>(82.,88.,95.),
                vec3<f32>(101.,75.,53.),vec3<f32>(59.,102.,52.),vec3<f32>(155.,113.,89.),
                vec3<f32>(148.,77.,63.),vec3<f32>(158.,119.,79.),vec3<f32>(121.,126.,130.));
            return pow(p[id]/255.0,vec3<f32>(2.2));
        }}
        fn filtered_rock_flecks{rock_helper}
        @group(0) @binding(0) var<uniform> terrain:TerrainConstants;
        @group(0) @binding(1) var<storage,read> points:array<vec4<i32>>;
        @group(0) @binding(2) var<storage,read_write> answers:array<vec4<u32>>;
        @group(0) @binding(3) var<storage,read_write> coverage:array<vec4<f32>>;
        var<private> material_footprint:f32=0.0;
        var<private> material_weathered_skin:bool=false;
        var<private> material_radial_span:f32=0.0;
        var<private> material_stone_coverage:f32=-1.0;
        var<private> material_snow_mix:vec4<f32>=vec4<f32>(-1.0,0.0,0.0,0.0);
        var<private> material_rock_id:u32=0u;
        var<private> material_rock_base_id:u32=0u;
        @compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{
            if id.x>=arrayLength(&points) {{return;}}
            let p=points[id.x].xyz;
            material_footprint=0.0;
            let canonical=ground_material(p,4000000,0,5,39999);
            material_footprint=0.1;
            let near=ground_material(p,4000000,0,5,39999);
            let near_disabled=u32(material_snow_mix.x<0.0);
            material_footprint=128.0;
            let far=ground_material(p,4000000,0,5,39999);
            coverage[id.x*4u]=material_snow_mix;
            let far_rock=material_rock_id;
            // The 6.4m octave is unresolved here; the 51.2m octave remains.
            material_footprint=6.4;
            let middle=ground_material(p,4000000,0,5,39999);
            coverage[id.x*4u+1u]=material_snow_mix;
            let middle_rock=material_rock_id;
            material_footprint=0.0;
            let reset=ground_material(p,4000000,0,5,39999);
            answers[id.x*4u]=vec4<u32>(canonical,near,far,middle);
            answers[id.x*4u+1u]=vec4<u32>(far_rock,middle_rock,near_disabled,u32(material_snow_mix.x<0.0 && reset==canonical));
            // Below every snowline, a steep natural rock face also has dirt
            // flecks. Metadata changes appearance, never its canonical ID.
            let below=ground_material(p,1500000,0,20,14999);
            let below_base=material_rock_base_id;
            let near_colour=palette(below);
            material_footprint=128.0;
            let below_far=ground_material(p,1500000,0,20,14999);
            let far_base=material_rock_base_id;
            var far_colour=palette(below_far);
            if far_base!=M_AIR {{far_colour=filtered_rock_flecks(far_colour,1.0,far_base,1.0);}}
            coverage[id.x*4u+2u]=vec4<f32>(filtered_rock_flecks(near_colour,1.0,below_base,0.0),0.0);
            coverage[id.x*4u+3u]=vec4<f32>(far_colour,0.0);
            let snow_disabled=u32(material_snow_mix.x<0.0);
            let stone_coverage=material_stone_coverage;
            let basin=ground_material(p,-1000,0,0,0);
            answers[id.x*4u+2u]=vec4<u32>(below,below_far,below_base,far_base);
            answers[id.x*4u+3u]=vec4<u32>(snow_disabled,u32(material_rock_base_id==M_AIR),basin,bitcast<u32>(stone_coverage));
        }}
    "#
    );
    // Match the complete current TerrainConstants ABI, including the ridge LUT.
    let mut constants = [0i32; 404];
    constants[1] = 100;
    constants[2] = 7;
    constants[3] = 123;
    constants[6] = 2_000_000;
    constants[9] = 16;
    constants[12 + 6 * 4] = 10;
    let points: Vec<[i32; 4]> = (0..16384i32)
        .map(|i| {
            // Actual cube-face/plane slices: one domain coordinate is fixed.
            let x = (helio_pass_voxel_planet::noise::hash3(i, 0, 0, 123) & 0xffffff) as i32 * 2 + 1;
            let z = (helio_pass_voxel_planet::noise::hash3(i, 2, 0, 123) & 0xffffff) as i32 * 2 + 1;
            [x, 1 << 27, z, 0]
        })
        .collect();
    let terrain = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&constants),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let input = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&points),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (points.len() * 64) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let coverage_output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (points.len() * 64) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("material appearance filtering"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: terrain.as_entire_binding(),
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
                binding: 3,
                resource: coverage_output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((points.len() as u32 + 63) / 64, 1, 1);
    }
    gpu.queue.submit([encoder.finish()]);
    let bytes = read_buffer(&gpu, &output, (points.len() * 64) as u64);
    let answers: &[[u32; 4]] = bytemuck::cast_slice(&bytes);
    let coverage_bytes = read_buffer(&gpu, &coverage_output, (points.len() * 64) as u64);
    let coverage: &[[f32; 4]] = bytemuck::cast_slice(&coverage_bytes);
    let colours = [
        [200., 0., 200.],
        [91., 125., 65.],
        [120., 87., 61.],
        [133., 139., 142.],
        [203., 188., 151.],
        [217., 228., 236.],
        [28., 72., 92.],
        [116., 111., 102.],
        [185., 142., 104.],
        [82., 88., 95.],
        [101., 75., 53.],
        [59., 102., 52.],
        [155., 113., 89.],
        [148., 77., 63.],
        [158., 119., 79.],
        [121., 126., 130.],
    ];
    let linear = |id: u32, channel: usize| (colours[id as usize][channel] / 255f32).powf(2.2);
    let mut canonical_mean = [0f32; 3];
    let mut filtered_mean = [[0f32; 3]; 2];
    let mut below_exact = [0f32; 3];
    let mut below_filtered = [0f32; 3];
    let mut below_count = 0usize;
    let mut below_dirt = 0usize;
    let mut below_bases = [false; 2];
    let mut canonical_snow = 0usize;
    let mut filtered_snow = 0f32;
    let mut middle_range = [1f32, 0f32];
    for index in 0..points.len() {
        let ids = answers[index * 4];
        let meta = answers[index * 4 + 1];
        assert_eq!(
            ids, [ids[0]; 4],
            "canonical material changed at point {index}"
        );
        assert_eq!(
            meta[2..],
            [1, 1],
            "near/default queries leaked coverage at point {index}"
        );
        let below = answers[index * 4 + 2];
        let reset = answers[index * 4 + 3];
        assert_eq!(
            below[0], below[1],
            "below-snow canonical ID changed at point {index}"
        );
        assert_eq!(
            below[2], below[3],
            "rock base changed with footprint at point {index}"
        );
        assert_eq!(
            reset[..2],
            [1, 1],
            "below snow metadata failed to reset at point {index}"
        );
        for channel in 0..3 {
            assert!(
                (coverage[index * 4 + 2][channel] - linear(below[0], channel)).abs() < 1e-6,
                "near rock colour changed at point {index}"
            );
            if below[2] != 0 {
                assert!(below[2] == 3 || below[2] == 9);
                let stone = f32::from_bits(reset[3]);
                assert!((0.0..=1.0).contains(&stone));
                let expected = 0.875 * (stone * linear(3, channel) + (1.0 - stone) * linear(9, channel)) + 0.125 * linear(2, channel);
                assert!(
                    (coverage[index * 4 + 3][channel] - expected).abs() < 1e-6,
                    "unresolved rock failed to preserve dirt coverage at point {index}"
                );
                below_exact[channel] += linear(below[0], channel);
                below_filtered[channel] += coverage[index * 4 + 3][channel];
            } else {
                assert!(
                    (coverage[index * 4 + 3][channel] - linear(below[0], channel)).abs() < 1e-6
                );
            }
        }
        if below[2] != 0 {
            below_count += 1;
            below_dirt += usize::from(below[0] == 2);
            below_bases[usize::from(below[2] == 9)] = true;
        }
        canonical_snow += usize::from(ids[0] == 5);
        filtered_snow += coverage[index * 4][0];
        middle_range[0] = middle_range[0].min(coverage[index * 4 + 1][0]);
        middle_range[1] = middle_range[1].max(coverage[index * 4 + 1][0]);
        for channel in 0..3 {
            canonical_mean[channel] += linear(ids[0] & 255, channel);
            for filter in 0..2 {
                let w = coverage[index * 4 + filter];
                assert!(w.iter().all(|x| x.is_finite() && *x >= 0.0 && *x <= 1.0));
                assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
                filtered_mean[filter][channel] += w[0] * linear(5, channel)
                    + w[1] * linear(meta[filter], channel)
                    + w[2] * linear(9, channel)
                    + w[3] * linear(2, channel);
            }
        }
    }
    assert!(
        below_count > 1000 && below_dirt > 0 && below_bases == [true, true],
        "fixture must exercise dirt flecks and both broad rock strata below snowline"
    );
    for channel in 0..3 {
        assert!(
            (below_exact[channel] - below_filtered[channel]).abs() / (below_count as f32) < 0.02,
            "filtered rock flecks must preserve their canonical palette mean"
        );
    }
    let n = points.len() as f32;
    assert!(canonical_snow > 0 && canonical_snow < points.len());
    assert!(
        (filtered_snow / n - canonical_snow as f32 / n).abs() < 0.02,
        "unresolved snow/rock coverage must retain the canonical class mean"
    );
    assert!(
        middle_range[1] - middle_range[0] > 0.5,
        "resolved broad outcrops must keep spatial variation"
    );
    for mean in filtered_mean {
        for channel in 0..3 {
            assert!(
                (mean[channel] / n - canonical_mean[channel] / n).abs() < 0.02,
                "filtered palette mean differs: {} vs {}",
                mean[channel] / n,
                canonical_mean[channel] / n
            );
        }
    }
}


fn shader_function<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("fn {name}(")).unwrap();
    &source[start..start + source[start..].find("\n}").unwrap() + 2]
}

// Independent integration: integrate the measured noise histogram by dense
// midpoint sampling; each sample uses an exact geometric stripe/window overlap.
fn integrated_stone_oracle(phase: f64, span: f64, deviation: f64, cutoff: f64, cdf: &[f64]) -> f64 {
    let primitive = |x: f64| {
        let periods = (x / 9000.0).floor();
        4500.0 * periods + (x - periods * 9000.0).min(4500.0)
    };
    let window = |centre: f64| {
        if span == 0.0 { return f64::from(u8::from(centre.rem_euclid(9000.0) < 4500.0)); }
        (primitive(centre + span * 0.5) - primitive(centre - span * 0.5)) / span
    };
    if deviation == 0.0 { return window(phase); }
    let mut total = 0.0;
    let mut mass = 0.0;
    for i in 0..32 {
        let lo = (-65536.0 + i as f64 * 4096.0).max(cutoff);
        let hi = -65536.0 + (i + 1) as f64 * 4096.0;
        if hi <= lo { continue; }
        let weight = (cdf[i + 1] - cdf[i]) * (hi - lo) / 4096.0;
        let mut sum = 0.0;
        for j in 0..512 {
            let noise = lo + (j as f64 + 0.5) / 512.0 * (hi - lo);
            sum += window(phase + noise * deviation * 3000.0 / 65536.0);
        }
        total += weight * sum / 512.0;
        mass += weight;
    }
    total / mass
}

#[test]
fn stone_strata_box_matches_integrated_oracle_and_respects_surface_guards() {
    let Some(gpu) = gpu() else { return };
    let landform = include_str!("../shaders/landform.wgsl");
    let surface = include_str!("../shaders/surface.wgsl");
    let begin = landform.find("const MATERIAL_NOISE_CDF").unwrap();
    let end = landform.find("fn snow_material_coverage").unwrap();
    let band_helpers = &landform[begin..end];
    let cdf: Vec<f64> = landform[begin..].split("(").nth(1).unwrap().split(")").next().unwrap()
        .split(',').map(|v| v.trim().parse().unwrap()).collect();
    let guards = shader_function(surface, "natural_material_filter_allowed");
    let flecks = shader_function(surface, "filtered_rock_flecks");
    let shader = format!(r#"
{band_helpers}
const INFO_TOPOLOGY:u32=0x08000000u;
const M_DIRT:u32=2u;
const M_STONE:u32=3u;
const M_DARK_STONE:u32=9u;
struct Column {{ info:u32, fits:u32 }}
fn column_tops_fit(c:Column)->bool {{return c.fits!=0u;}}
{guards}
var<private> material_stone_coverage:f32=-1.0;
// Deliberately non-default authored palette; no hard-coded rock colours.
fn palette(id:u32)->vec3<f32> {{
    if id==M_STONE {{return vec3<f32>(0.8,0.1,0.6);}}
    if id==M_DARK_STONE {{return vec3<f32>(0.1,0.7,0.2);}}
    return vec3<f32>(0.2,0.3,0.4);
}}
{flecks}
@group(0) @binding(0) var<storage,read> probes:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> answers:array<vec4<f32>>;
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{
    if id.x>=arrayLength(&probes) {{return;}}
    let p=probes[id.x];
    material_stone_coverage=rock_band_coverage(p.x,p.y,p.z,p.w);
    answers[id.x*2u]=vec4<f32>(filtered_rock_flecks(palette(M_STONE),0.75,M_STONE,1.0),material_stone_coverage);
    answers[id.x*2u+1u]=vec4<f32>(f32(natural_material_filter_allowed(false,Column(0u,1u))),
        f32(natural_material_filter_allowed(true,Column(0u,1u))),
        f32(natural_material_filter_allowed(false,Column(INFO_TOPOLOGY,1u))),
        f32(natural_material_filter_allowed(false,Column(0u,0u))));
}}
"#);
    let mut probes = Vec::<[f32;4]>::new();
    for phase in [-16000.,-9000.,-4500.,-50.,0.,50.,2200.,4450.,4500.,4550.,8900.,9000.,15000.] {
        for span in [0.,0.0005,0.99,1.01,100.,1200.,4500.,8999.,9000.,12000.,36000.,1_000_000.] {
            for (deviation,cutoff) in [(0.,-65536.),(1.0/3.0,-65536.),((10f32/9.).sqrt(),-65536.),(1.,8192.)] {
                probes.push([phase,span,deviation,cutoff]);
            }
        }
    }
    for cutoff in [48000.,49000.,56000.,65536.,70000.] {
        for phase in [-50.,4450.,4550.] { probes.push([phase,4500.,1.,cutoff]); }
    }
    let input=gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:None,contents:bytemuck::cast_slice(&probes),usage:wgpu::BufferUsages::STORAGE});
    let output=gpu.device.create_buffer(&wgpu::BufferDescriptor{label:None,size:(probes.len()*32) as u64,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
    let module=gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor{label:Some("production rock band coverage"),source:wgpu::ShaderSource::Wgsl(shader.into())});
    let pipeline=gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:None,layout:None,module:&module,entry_point:Some("probe"),compilation_options:Default::default(),cache:None});
    let group=gpu.device.create_bind_group(&wgpu::BindGroupDescriptor{label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[
        wgpu::BindGroupEntry{binding:0,resource:input.as_entire_binding()},wgpu::BindGroupEntry{binding:1,resource:output.as_entire_binding()}]});
    let mut encoder=gpu.device.create_command_encoder(&Default::default());
    {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&group,&[]);pass.dispatch_workgroups((probes.len() as u32+63)/64,1,1);}
    gpu.queue.submit([encoder.finish()]);
    let bytes=read_buffer(&gpu,&output,(probes.len()*32) as u64);
    let answers:&[[f32;4]]=bytemuck::cast_slice(&bytes);
    let mut worst=0f64;
    for (index,p) in probes.iter().enumerate() {
        let answer=answers[index*2];
        let cut_x=((p[3] as f64)/4096.+16.).clamp(0.,32.);
        let ci=(cut_x as usize).min(31);
        let cut_cdf=cdf[ci]+(cdf[ci+1]-cdf[ci])*(cut_x-ci as f64);
        if p[2]>0. && cut_cdf>0.999 {
            assert_eq!(answer[3],-1.,"vanishing rock-exposure probability must disable approximation");
            continue;
        }
        let expected=integrated_stone_oracle(p[0] as f64,p[1] as f64,p[2] as f64,p[3] as f64,&cdf);
        let error=(answer[3] as f64-expected).abs();worst=worst.max(error);
        assert!(error<0.0015,"phase/span/noise {p:?}: {} vs {expected}, error {error}",answer[3]);
        assert_eq!(answers[index*2+1],[1.,0.,0.,0.],"paint/topology/truncated guards");
        for channel in 0..3 {
            let light=[0.8,0.1,0.6][channel];let dark=[0.1,0.7,0.2][channel];let dirt=[0.2,0.3,0.4][channel];
            let colour=0.75*(0.875*(expected*light+(1.-expected)*dark)+0.125*dirt);
            assert!((answer[channel] as f64-colour).abs()<0.0015);
        }
    }
    println!("{} production GPU stripe probes; independent integrated oracle worst coverage error={worst}",probes.len());
}
