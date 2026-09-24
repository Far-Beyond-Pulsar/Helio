use helio_pass_tiny_voxel::landforms::{BoundedLandforms,FieldClearance,LandformSnapshot,SnapshotId,VoxelField};
use glam::DVec3;
use std::{path::PathBuf,sync::Arc};
#[path="fixtures/voxel_field_oracle.rs"]
mod oracle;

fn random(seed:&mut u32)->u32 {
    *seed=seed.wrapping_add(0x9e3779b9);
    let mut h=*seed;h^=h>>16;h=h.wrapping_mul(0x7feb352d);h^=h>>15;h=h.wrapping_mul(0x846ca68b);h^(h>>16)
}
#[test]
#[ignore="requires published sampling fixtures and an explicit evidence directory"]
fn new_field_clearance_excludes_solid_centres_on_five_datasets() {
    let input=PathBuf::from(std::env::var_os("HELIO_CLEARANCE_INPUT").expect("sampling fixtures"));
    let output=PathBuf::from(std::env::var_os("HELIO_CLEARANCE_OUTPUT").expect("fresh evidence directory"));
    assert!(!output.join("clearance-regions.json").exists());
    let previous:serde_json::Value=serde_json::from_slice(&std::fs::read(input.join("results.json")).unwrap()).unwrap();
    let mut results=Vec::new();
    for case in previous["cases"].as_array().unwrap() {
        let id=case["published"]["id"].as_str().unwrap();
        let id=SnapshotId(std::array::from_fn(|i|u8::from_str_radix(&id[i*2..i*2+2],16).unwrap()));
        let snapshot=LandformSnapshot::decode(&std::fs::read(input.join("data/published").join(format!("{id}.hlfm"))).unwrap(),id).unwrap();
        let bound=FieldClearance::from_atlas(snapshot.resolution(),snapshot.height_atlas());
        for constant in [bound.lipschitz,bound.quantization_guard,bound.maximum_radius] {
            assert_eq!(constant,f64::from(constant as f32),"GPU constants must be exactly encoded");
        }
        let field=VoxelField::new(Arc::new(BoundedLandforms::new(snapshot)),419);
        let bytes=std::fs::read(input.join("data").join(case["artifacts"][0]["file"].as_str().unwrap())).unwrap();
        let cells:Vec<[i32;4]>=bytes.chunks_exact(16).map(|b|std::array::from_fn(|a|i32::from_le_bytes(b[a*4..a*4+4].try_into().unwrap()))).collect();
        let mut checked=0u64;let mut nonzero=0u64;let mut air=0u64;let mut seed=419u32;let mut maximum=0.0_f64;
        let mut face_counts=[0u64;6];
        for i in 0..256 {
            let input=cells[514+i*251];
            let c=[input[0],input[1],input[2]];
            let Some(base)=field.height_units(c) else{continue;};
            let radial=(glam::IVec3::from_array(c).as_dvec3()+DVec3::splat(0.5)).normalize();
            for offset in [0.0,0.1,1.0,2.0,4.0,8.0,32.0,128.0,512.0,2048.0,8192.0,65536.0] {
                let p=radial*(6_371_000.0+f64::from(base)*0.05+offset);
                let c=(p*10.0).floor().as_ivec3().to_array();
                let sampled=oracle::sample(&field,c);
                if sampled[7]==0 || sampled[5]!=0 {continue;}
                air+=1;
                let centre=(glam::IVec3::from_array(c).as_dvec3()+DVec3::splat(0.5))*0.1;
                let depth=centre.length()-(127420000.0+f64::from(sampled[4] as i32))*0.05;
                let safe=bound.empty_radius(depth);
                if safe<=0.0 {continue;}
                nonzero+=1;maximum=maximum.max(safe);
                face_counts[field.landforms().snapshot().sample_cell(c).unwrap().address.face as usize]+=1;
                for k in 0..16 {
                    let direction=match k {0=>radial,1=>-radial,_=>DVec3::from_array(std::array::from_fn(|_|f64::from(random(&mut seed))/f64::from(u32::MAX)*2.0-1.0)).normalize()};
                    for fraction in [0.125,0.5,0.999999] {
                        let q=((centre+direction*(safe*fraction))*10.0-DVec3::splat(0.5)).round().as_ivec3().to_array();
                        if q.iter().any(|c|c.unsigned_abs()>100_000_000) {continue;}
                        let actual=(glam::IVec3::from_array(q).as_dvec3()-glam::IVec3::from_array(c).as_dvec3()).length()*0.1;
                        if actual>safe {continue;}
                        assert_eq!(oracle::sample(&field,q)[5],0,"false clearance case={} c={c:?} q={q:?} radius={safe}",case["name"]);
                        checked+=1;
                    }
                }
            }
        }
        assert!(nonzero>512 && checked>20_000 && face_counts.iter().all(|n|*n>0));
        results.push(serde_json::json!({"case":case["name"],"air_points":air,"nonzero_bounds":nonzero,"checked_centres":checked,"face_counts":face_counts,"maximum_clearance_m":maximum,
            "lipschitz":bound.lipschitz,"quantization_guard_m":bound.quantization_guard,"radius_cap_m":bound.maximum_radius,"false_clearances":0}));
    }
    std::fs::write(output.join("clearance-regions.json"),serde_json::to_vec_pretty(&results).unwrap()).unwrap();
    println!("CLEARANCE_REGIONS {}",serde_json::to_string(&results).unwrap());
}
