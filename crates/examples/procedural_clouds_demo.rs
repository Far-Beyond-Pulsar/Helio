//! Procedural Clouds Demo — entire sky with Blender noise + thickness.
//! Thickness is baked per-sample (sampleDensityThick) with 0 extra fetches.
//! Controls: WASD + Space/Shift move, mouse drag look, 1-6 pattern, Esc.

use std::{collections::HashSet, sync::Arc};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Camera { inv_view_proj: [[f32;4];4], position: [f32;3], _pad: f32 }
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    time_pack: [f32;4], alt_pack: [f32;4], scale_pack: [f32;4],
    extra_pack: [f32;4], cache_pack: [f32;4], bounds_pack: [f32;4],
}

const SHADER: &str = include_str!("shaders/procedural_clouds.wgsl");

fn main(){ env_logger::init(); let el=EventLoop::new().unwrap(); let mut app=App{state:None}; el.run_app(&mut app).unwrap(); }
struct App{state: Option<State>}
struct State{
 window: Arc<Window>, surface: wgpu::Surface<'static>, device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>,
 pipeline: wgpu::RenderPipeline, bind: wgpu::BindGroup, cam_buf: wgpu::Buffer, params_buf: wgpu::Buffer,
 density_tex: wgpu::Texture, density_view: wgpu::TextureView, density_sampler: wgpu::Sampler,
 density_bind: wgpu::BindGroup, density_store_bind: wgpu::BindGroup, compute_pipeline: wgpu::ComputePipeline,
 format: wgpu::TextureFormat,
 pos: glam::Vec3, yaw:f32, pitch:f32, keys: HashSet<KeyCode>, grabbed:bool, delta:(f32,f32), time:f32,
}
impl App{fn new()->Self{Self{state:None}}}
impl ApplicationHandler for App{
 fn resumed(&mut self, el:&ActiveEventLoop){
  if self.state.is_some(){return;}
  let window=Arc::new(el.create_window(Window::default_attributes().with_title("Procedural Clouds - Thickness").with_inner_size(winit::dpi::LogicalSize::new(1280,720))).unwrap());
  let instance=wgpu::Instance::default();
  let surface=instance.create_surface(window.clone()).unwrap();
  let adapter=pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions{power_preference:wgpu::PowerPreference::HighPerformance,compatible_surface:Some(&surface),force_fallback_adapter:false,..Default::default()})).unwrap();
  let (device,queue)=pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor{label:Some("proc"),required_features:helio::required_wgpu_features(adapter.features()),required_limits:helio::required_wgpu_limits(adapter.limits()),experimental_features:helio::required_experimental_features(adapter.features()),..Default::default()})).unwrap();
  device.on_uncaptured_error(Arc::new(|e| panic!("{:?}",e)));
  let caps=surface.get_capabilities(&adapter);let fmt=caps.formats.iter().find(|f|f.is_srgb()).copied().unwrap_or(caps.formats[0]);
  let size=window.inner_size();let cfg=wgpu::SurfaceConfiguration{usage:wgpu::TextureUsages::RENDER_ATTACHMENT,format:fmt,width:size.width,height:size.height,present_mode:wgpu::PresentMode::Fifo,alpha_mode:caps.alpha_modes[0],view_formats:vec![],desired_maximum_frame_latency:2,color_space:wgpu::SurfaceColorSpace::Auto};
  surface.configure(&device,&cfg);
  let cam_buf=device.create_buffer(&wgpu::BufferDescriptor{label:Some("cam"),size:80,usage:wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
  let params_buf=device.create_buffer(&wgpu::BufferDescriptor{label:Some("params"),size:96,usage:wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
  let bgl0=device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{label:Some("bgl0"),entries:&[wgpu::BindGroupLayoutEntry{binding:0,visibility:wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Uniform,has_dynamic_offset:false,min_binding_size:None},count:None},wgpu::BindGroupLayoutEntry{binding:1,visibility:wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Uniform,has_dynamic_offset:false,min_binding_size:None},count:None}]});
  let bgl1=device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{label:Some("bgl1"),entries:&[wgpu::BindGroupLayoutEntry{binding:0,visibility:wgpu::ShaderStages::FRAGMENT,ty:wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),count:None},wgpu::BindGroupLayoutEntry{binding:1,visibility:wgpu::ShaderStages::FRAGMENT,ty:wgpu::BindingType::Texture{sample_type:wgpu::TextureSampleType::Float{filterable:true},view_dimension:wgpu::TextureViewDimension::D3,multisampled:false},count:None}]});
  let bgl2=device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{label:Some("bgl2"),entries:&[wgpu::BindGroupLayoutEntry{binding:0,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::StorageTexture{access:wgpu::StorageTextureAccess::WriteOnly,format:wgpu::TextureFormat::Rgba16Float,view_dimension:wgpu::TextureViewDimension::D3},count:None}]});
  let shader=device.create_shader_module(wgpu::ShaderModuleDescriptor{label:Some("proc"),source:wgpu::ShaderSource::Wgsl(SHADER.into())});
  let pl=device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{label:Some("pl"),bind_group_layouts:&[Some(&bgl0),Some(&bgl1),Some(&bgl2)],immediate_size:0});
  let pipeline=device.create_render_pipeline(&wgpu::RenderPipelineDescriptor{label:Some("pipe"),layout:Some(&pl),vertex:wgpu::VertexState{module:&shader,entry_point:Some("vs"),buffers:&[],compilation_options:Default::default()},fragment:Some(wgpu::FragmentState{module:&shader,entry_point:Some("fs"),targets:&[Some(wgpu::ColorTargetState{format:fmt,blend:None,write_mask:wgpu::ColorWrites::ALL})],compilation_options:Default::default()}),primitive:Default::default(),depth_stencil:None,multisample:Default::default(),multiview_mask:None,cache:None});
  let bind=device.create_bind_group(&wgpu::BindGroupDescriptor{label:Some("bg"),layout:&bgl0,entries:&[wgpu::BindGroupEntry{binding:0,resource:cam_buf.as_entire_binding()},wgpu::BindGroupEntry{binding:1,resource:params_buf.as_entire_binding()}]});
  let density_tex=device.create_texture(&wgpu::TextureDescriptor{label:Some("densityTex"),size:wgpu::Extent3d{width:64,height:32,depth_or_array_layers:64},mip_level_count:1,sample_count:1,dimension:wgpu::TextureDimension::D3,format:wgpu::TextureFormat::Rgba16Float,usage:wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::STORAGE_BINDING,view_formats:&[]});
  let density_view=density_tex.create_view(&Default::default());
  let density_sampler=device.create_sampler(&wgpu::SamplerDescriptor{label:Some("densitySampler"),address_mode_u:wgpu::AddressMode::ClampToEdge,address_mode_v:wgpu::AddressMode::ClampToEdge,address_mode_w:wgpu::AddressMode::ClampToEdge,mag_filter:wgpu::FilterMode::Linear,min_filter:wgpu::FilterMode::Linear,..Default::default()});
  let density_bind=device.create_bind_group(&wgpu::BindGroupDescriptor{label:Some("density_bind"),layout:&bgl1,entries:&[wgpu::BindGroupEntry{binding:0,resource:wgpu::BindingResource::Sampler(&density_sampler)},wgpu::BindGroupEntry{binding:1,resource:wgpu::BindingResource::TextureView(&density_view)}]});
  let density_store_bind=device.create_bind_group(&wgpu::BindGroupDescriptor{label:Some("density_store"),layout:&bgl2,entries:&[wgpu::BindGroupEntry{binding:0,resource:wgpu::BindingResource::TextureView(&density_view)}]});
  let compute_pipeline=device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:Some("cs"),layout:Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{label:Some("cs_pl"),bind_group_layouts:&[Some(&bgl0),Some(&bgl1),Some(&bgl2)],immediate_size:0})),module:&shader,entry_point:Some("cs"),compilation_options:Default::default(),cache:None});
  self.state=Some(State{window,surface,device:Arc::new(device),queue:Arc::new(queue),pipeline,bind,cam_buf,params_buf,density_tex,density_view,density_sampler,density_bind,density_store_bind,compute_pipeline,format:fmt,pos:glam::Vec3::new(0.0,2.5,7.0),yaw:0.0,pitch:-0.2,keys:HashSet::new(),grabbed:false,delta:(0.0,0.0),time:0.0});
 }
 fn window_event(&mut self, el:&ActiveEventLoop, _id:WindowId, e:WindowEvent){
  let Some(s)=&mut self.state else{return};
  match e{
   WindowEvent::CloseRequested=>el.exit(),
   WindowEvent::KeyboardInput{event:KeyEvent{state:ElementState::Pressed,physical_key:PhysicalKey::Code(KeyCode::Escape),..},..}=>{
    if s.grabbed{ let _=s.window.set_cursor_grab(CursorGrabMode::None); s.window.set_cursor_visible(true); s.grabbed=false;} else{el.exit();}}
   WindowEvent::KeyboardInput{event:KeyEvent{physical_key:PhysicalKey::Code(k),state, ..},..}=>{match state{ElementState::Pressed=>{s.keys.insert(k);}, ElementState::Released=>{s.keys.remove(&k);}}}
   WindowEvent::MouseInput{state:ElementState::Pressed,button:MouseButton::Left,..}=>{if !s.grabbed{let _=s.window.set_cursor_grab(CursorGrabMode::Locked).or_else(|_|s.window.set_cursor_grab(CursorGrabMode::Confined)); s.window.set_cursor_visible(false); s.grabbed=true;}}
   WindowEvent::Resized(sz)=>{let cfg=wgpu::SurfaceConfiguration{usage:wgpu::TextureUsages::RENDER_ATTACHMENT,format:s.format,width:sz.width,height:sz.height,present_mode:wgpu::PresentMode::Fifo,alpha_mode:wgpu::CompositeAlphaMode::Auto,view_formats:vec![],desired_maximum_frame_latency:2,color_space:wgpu::SurfaceColorSpace::Auto}; s.surface.configure(&s.device,&cfg);}
   WindowEvent::RedrawRequested=>{
    let dt=0.016; s.time+=dt;
    s.yaw+=s.delta.0*0.002; s.pitch=(s.pitch - s.delta.1*0.002).clamp(-1.5,1.5); s.delta=(0.0,0.0);
    let (sy,cy)=s.yaw.sin_cos(); let (sp,cp)=s.pitch.sin_cos();
    let fwd=glam::Vec3::new(sy*cp,sp,-cy*cp); let right=glam::Vec3::new(cy,0.0,sy);
    if s.keys.contains(&KeyCode::KeyW){s.pos+=fwd*5.0*dt;} if s.keys.contains(&KeyCode::KeyS){s.pos-=fwd*5.0*dt;}
    if s.keys.contains(&KeyCode::KeyA){s.pos-=right*5.0*dt;} if s.keys.contains(&KeyCode::KeyD){s.pos+=right*5.0*dt;}
    if s.keys.contains(&KeyCode::Space){s.pos+=glam::Vec3::Y*5.0*dt;} if s.keys.contains(&KeyCode::ShiftLeft){s.pos-=glam::Vec3::Y*5.0*dt;}
    let size=s.window.inner_size(); let aspect=size.width as f32/size.height.max(1) as f32;
    let proj=glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4,aspect,0.1,100.0);
    let view=glam::Mat4::look_at_rh(s.pos, s.pos+fwd, glam::Vec3::Y);
    let inv=(proj*view).inverse();
    let cam=Camera{inv_view_proj:inv.to_cols_array_2d(), position:[s.pos.x,s.pos.y,s.pos.z], _pad:0.0};
    s.queue.write_buffer(&s.cam_buf,0,bytemuck::bytes_of(&cam));
    let params=Params{
     time_pack:[s.time, s.time*0.7, s.time*0.5, 0.22],
     alt_pack:[0.35,0.6,0.7,0.5],
     scale_pack:[0.8,0.35,2.0,0.6],
     extra_pack:[0.75,6.0,18.0,0.0],
     cache_pack:[0.0,3.0,0.25,1.4],
     bounds_pack:[22.0,0.0,0.0,0.0],
    };
    s.queue.write_buffer(&s.params_buf,0,bytemuck::bytes_of(&params));
    // Bake 64x32x64 volume
    {
        let mut enc=s.device.create_command_encoder(&Default::default());
        {
            let mut cpass=enc.begin_compute_pass(&wgpu::ComputePassDescriptor{label:Some("bake"),timestamp_writes:None});
            cpass.set_pipeline(&s.compute_pipeline);
            cpass.set_bind_group(0,&s.bind, &[]);
            cpass.set_bind_group(1,&s.density_bind, &[]);
            cpass.set_bind_group(2,&s.density_store_bind, &[]);
            cpass.dispatch_workgroups(16,8,16);
        }
        s.queue.submit([enc.finish()]);
    }
    let surface_texture = match s.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        _ => return,
    };
    let view=surface_texture.texture.create_view(&Default::default());
    let mut enc=s.device.create_command_encoder(&Default::default());
    {let mut p=enc.begin_render_pass(&wgpu::RenderPassDescriptor{label:Some("proc"),color_attachments:&[Some(wgpu::RenderPassColorAttachment{view:&view,resolve_target:None,depth_slice:None,ops:wgpu::Operations{load:wgpu::LoadOp::Clear(wgpu::Color{r:0.045,g:0.10,b:0.18,a:1.0}),store:wgpu::StoreOp::Store}})],depth_stencil_attachment:None,timestamp_writes:None,occlusion_query_set:None,multiview_mask:None});
     p.set_pipeline(&s.pipeline); p.set_bind_group(0,&s.bind, &[]); p.set_bind_group(1,&s.density_bind, &[]); p.draw(0..3,0..1);}
    s.queue.submit([enc.finish()]); s.queue.present(surface_texture); s.window.request_redraw();
   }
   _=>{}
  }
 }
 fn device_event(&mut self,_el:&ActiveEventLoop,_id:winit::event::DeviceId,e:DeviceEvent){
  if let DeviceEvent::MouseMotion{delta:(dx,dy)}=e{ if let Some(s)=&mut self.state{ if s.grabbed{ s.delta.0+=dx as f32; s.delta.1+=dy as f32; }}}
 }
 fn about_to_wait(&mut self,_el:&ActiveEventLoop){ if let Some(s)=&self.state{ s.window.request_redraw();}}
}
