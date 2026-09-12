use std::mem::{align_of, size_of};

use helio_pass_billboard::{
    BillboardComponent, BillboardInstance, BillboardSceneBinding, SceneGpuRecord,
};
use pulsar_scenedb::GpuColumnSet;

#[test]
fn billboard_component_matches_the_shader_instance_record() {
    assert_eq!(size_of::<BillboardComponent>(), 48);
    assert_eq!(align_of::<BillboardComponent>(), 4);
    assert_eq!(
        size_of::<BillboardInstance>(),
        size_of::<BillboardComponent>()
    );
    assert_eq!(size_of::<BillboardComponent>() % 16, 0);
}

#[test]
fn scene_store_derives_one_packed_gpu_record() {
    let columns = BillboardComponent::gpu_columns();
    assert_eq!(columns.len(), 3);
    assert_eq!(
        BillboardComponent::buffer_key().as_str(),
        "billboard_instances"
    );
    assert_eq!(
        BillboardComponent::packed_component_id(),
        BillboardComponent::packed_gpu_component_id()
    );
}

#[test]
fn consumer_contract_is_generic_over_the_record_type() {
    fn assert_contract<T: SceneGpuRecord>() {}

    assert_contract::<BillboardComponent>();
    let _: std::marker::PhantomData<BillboardSceneBinding> = std::marker::PhantomData;
}
