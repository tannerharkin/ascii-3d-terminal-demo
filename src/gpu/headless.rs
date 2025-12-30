use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

/// Vertex type for 3D models
/// Matches the layout expected by the shader
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

/// Rotation mode for the rendered model
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RotationMode {
    Static,
    AxisX,
    #[default]
    AxisY,
    AxisZ,
    Tumble,
    Orbit,
}

impl RotationMode {
    pub fn name(&self) -> &'static str {
        match self {
            RotationMode::Static => "Static",
            RotationMode::AxisX => "X Axis",
            RotationMode::AxisY => "Y Axis",
            RotationMode::AxisZ => "Z Axis",
            RotationMode::Tumble => "Tumble",
            RotationMode::Orbit => "Orbit",
        }
    }

    pub fn all() -> &'static [RotationMode] {
        &[
            RotationMode::Static,
            RotationMode::AxisX,
            RotationMode::AxisY,
            RotationMode::AxisZ,
            RotationMode::Tumble,
            RotationMode::Orbit,
        ]
    }
}

/// Lighting mode for rendering
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LightingMode {
    Flat,          // No shading, just vertex color
    #[default]
    Diffuse,       // Current multi-light setup
    Specular,      // Diffuse + specular highlights
    Toon,          // Cel-shaded (quantized)
    Gradient,      // Height-based coloring
    Normals,       // Show normals as color
}

impl LightingMode {
    pub fn name(&self) -> &'static str {
        match self {
            LightingMode::Flat => "Flat",
            LightingMode::Diffuse => "Diffuse",
            LightingMode::Specular => "Specular",
            LightingMode::Toon => "Toon",
            LightingMode::Gradient => "Gradient",
            LightingMode::Normals => "Normals",
        }
    }

    pub fn all() -> &'static [LightingMode] {
        &[
            LightingMode::Flat,
            LightingMode::Diffuse,
            LightingMode::Specular,
            LightingMode::Toon,
            LightingMode::Gradient,
            LightingMode::Normals,
        ]
    }

    pub fn to_u32(self) -> u32 {
        match self {
            LightingMode::Flat => 0,
            LightingMode::Diffuse => 1,
            LightingMode::Specular => 2,
            LightingMode::Toon => 3,
            LightingMode::Gradient => 4,
            LightingMode::Normals => 5,
        }
    }
}

// Internal vertex type matching external Vertex layout
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InternalVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

impl InternalVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InternalVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    light_dir: [f32; 4],
    // Lighting mode (0=Flat, 1=Diffuse, 2=Specular, 3=Toon, 4=Gradient, 5=Normals)
    // Pack with padding to ensure 16-byte alignment
    lighting_mode: u32,
    _padding: [u32; 3],
}

pub struct HeadlessGpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    render_texture: wgpu::Texture,
    render_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    num_indices: u32,
    width: u32,
    height: u32,
    gpu_name: String,
    // Skybox rendering
    skybox_pipeline: wgpu::RenderPipeline,
    skybox_bind_group_layout: wgpu::BindGroupLayout,
    skybox_sampler: wgpu::Sampler,
    skybox_texture: Option<wgpu::Texture>,
    skybox_bind_group: Option<wgpu::BindGroup>,
}

impl HeadlessGpu {
    pub async fn new(width: u32, height: u32) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find an appropriate adapter"))?;

        let adapter_info = adapter.get_info();
        let gpu_name = adapter_info.name.clone();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Headless GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await?;

        // Create render texture
        let render_format = wgpu::TextureFormat::Rgba8Unorm;
        let (render_texture, render_view) =
            create_render_texture(&device, width, height, render_format);
        let (depth_texture, depth_view) = create_depth_texture(&device, width, height);

        // Create shader and pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cube Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/cube.wgsl").into()),
        });

        let (vertices, indices) = create_cube_geometry();
        let num_indices = indices.len() as u32;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = Uniforms {
            mvp: Mat4::IDENTITY.to_cols_array_2d(),
            model: Mat4::IDENTITY.to_cols_array_2d(),
            light_dir: [0.5, 1.0, 0.3, 0.0],
            lighting_mode: LightingMode::default().to_u32(),
            _padding: [0, 0, 0],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Uniform Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Headless Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[InternalVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: render_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Create skybox pipeline
        let skybox_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Skybox Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/skybox.wgsl").into()),
        });

        let skybox_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Skybox Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let skybox_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Skybox Pipeline Layout"),
                bind_group_layouts: &[&skybox_bind_group_layout],
                push_constant_ranges: &[],
            });

        let skybox_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Skybox Render Pipeline"),
            layout: Some(&skybox_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &skybox_shader,
                entry_point: Some("vs_main"),
                buffers: &[], // Fullscreen triangle, no vertex buffer needed
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &skybox_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: render_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for fullscreen triangle
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None, // No depth testing for skybox
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let skybox_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Skybox Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Ok(Self {
            device,
            queue,
            render_texture,
            render_view,
            depth_texture,
            depth_view,
            pipeline,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            uniform_bind_group,
            num_indices,
            width,
            height,
            gpu_name,
            skybox_pipeline,
            skybox_bind_group_layout,
            skybox_sampler,
            skybox_texture: None,
            skybox_bind_group: None,
        })
    }

    pub fn gpu_name(&self) -> &str {
        &self.gpu_name
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }

        self.width = width;
        self.height = height;

        let render_format = wgpu::TextureFormat::Rgba8Unorm;
        let (render_texture, render_view) =
            create_render_texture(&self.device, width, height, render_format);
        self.render_texture = render_texture;
        self.render_view = render_view;
        let (depth_texture, depth_view) = create_depth_texture(&self.device, width, height);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
    }

    /// Set new geometry from external model data
    pub fn set_geometry(&mut self, vertices: &[Vertex], indices: &[u32]) {
        // Convert Vertex to InternalVertex (they have the same layout)
        let internal_vertices: Vec<InternalVertex> = vertices
            .iter()
            .map(|v| InternalVertex {
                position: v.position,
                normal: v.normal,
                color: v.color,
            })
            .collect();

        self.vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&internal_vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        self.index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        self.num_indices = indices.len() as u32;
    }

    /// Load a skybox image from file
    pub fn set_skybox(&mut self, path: &std::path::Path) -> Result<()> {
        use image::GenericImageView;

        let img = image::open(path)?;
        let rgba = img.to_rgba8();
        let dimensions = img.dimensions();

        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Skybox Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            texture_size,
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Skybox Bind Group"),
            layout: &self.skybox_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.skybox_sampler),
                },
            ],
        });

        self.skybox_texture = Some(texture);
        self.skybox_bind_group = Some(bind_group);

        Ok(())
    }

    /// Clear the skybox (use solid color background instead)
    pub fn clear_skybox(&mut self) {
        self.skybox_texture = None;
        self.skybox_bind_group = None;
    }

    pub fn render_with_rotation(
        &self,
        time: f32,
        mode: RotationMode,
        speed: f32,
        lighting: LightingMode,
    ) -> wgpu::CommandBuffer {
        let aspect = self.width as f32 / self.height as f32;

        // Compute rotation and camera based on mode
        let (model, view) = match mode {
            RotationMode::Static => (
                Mat4::IDENTITY,
                Mat4::look_at_rh(Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO, Vec3::Y),
            ),
            RotationMode::AxisX => (
                Mat4::from_rotation_x(time * speed),
                Mat4::look_at_rh(Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO, Vec3::Y),
            ),
            RotationMode::AxisY => (
                Mat4::from_rotation_y(time * speed),
                Mat4::look_at_rh(Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO, Vec3::Y),
            ),
            RotationMode::AxisZ => (
                Mat4::from_rotation_z(time * speed),
                Mat4::look_at_rh(Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO, Vec3::Y),
            ),
            RotationMode::Tumble => (
                Mat4::from_rotation_y(time * speed * 0.7)
                    * Mat4::from_rotation_x(time * speed * 0.5)
                    * Mat4::from_rotation_z(time * speed * 0.3),
                Mat4::look_at_rh(Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO, Vec3::Y),
            ),
            RotationMode::Orbit => {
                let angle = time * speed * 0.5;
                let cam_x = 4.0 * angle.cos();
                let cam_z = 4.0 * angle.sin();
                (
                    Mat4::IDENTITY,
                    Mat4::look_at_rh(Vec3::new(cam_x, 1.5, cam_z), Vec3::ZERO, Vec3::Y),
                )
            }
        };

        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
        let mvp = proj * view * model;

        let uniforms = Uniforms {
            mvp: mvp.to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            light_dir: [0.5, 1.0, 0.3, 0.0],
            lighting_mode: lighting.to_u32(),
            _padding: [0, 0, 0],
        };

        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Headless Render Encoder"),
            });

        // Render skybox first if available
        if let Some(ref skybox_bind_group) = self.skybox_bind_group {
            let mut skybox_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Skybox Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.render_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            skybox_pass.set_pipeline(&self.skybox_pipeline);
            skybox_pass.set_bind_group(0, skybox_bind_group, &[]);
            skybox_pass.draw(0..3, 0..1); // Fullscreen triangle
        }

        // Render 3D model
        {
            // Use LoadOp::Load if skybox was rendered, Clear otherwise
            let color_load_op = if self.skybox_bind_group.is_some() {
                wgpu::LoadOp::Load
            } else {
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.02,
                    g: 0.02,
                    b: 0.05,
                    a: 1.0,
                })
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Headless Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.render_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: color_load_op,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        encoder.finish()
    }

    /// Render with manual rotation angles and zoom (for manual control mode)
    pub fn render_manual(
        &self,
        rotation_x: f32,
        rotation_y: f32,
        zoom: f32,
        lighting: LightingMode,
    ) -> wgpu::CommandBuffer {
        let aspect = self.width as f32 / self.height as f32;

        // Apply rotation: Y rotation (yaw) first, then X rotation (pitch)
        let model = Mat4::from_rotation_y(rotation_y) * Mat4::from_rotation_x(rotation_x);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, zoom), Vec3::ZERO, Vec3::Y);

        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
        let mvp = proj * view * model;

        let uniforms = Uniforms {
            mvp: mvp.to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            light_dir: [0.5, 1.0, 0.3, 0.0],
            lighting_mode: lighting.to_u32(),
            _padding: [0, 0, 0],
        };

        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Headless Render Encoder"),
            });

        // Render skybox first if available
        if let Some(ref skybox_bind_group) = self.skybox_bind_group {
            let mut skybox_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Skybox Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.render_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            skybox_pass.set_pipeline(&self.skybox_pipeline);
            skybox_pass.set_bind_group(0, skybox_bind_group, &[]);
            skybox_pass.draw(0..3, 0..1);
        }

        // Render 3D model
        {
            let color_load_op = if self.skybox_bind_group.is_some() {
                wgpu::LoadOp::Load
            } else {
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.02,
                    g: 0.02,
                    b: 0.05,
                    a: 1.0,
                })
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Headless Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.render_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: color_load_op,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        encoder.finish()
    }

    pub fn render_texture_view(&self) -> &wgpu::TextureView {
        &self.render_view
    }

    pub fn depth_texture_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    pub fn render_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

fn create_render_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Render Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_cube_geometry() -> (Vec<InternalVertex>, Vec<u32>) {
    let s = 0.8;

    let vertices = vec![
        // +X face (Red)
        InternalVertex { position: [s, -s, -s], normal: [1.0, 0.0, 0.0], color: [1.0, 0.2, 0.2] },
        InternalVertex { position: [s, s, -s], normal: [1.0, 0.0, 0.0], color: [1.0, 0.2, 0.2] },
        InternalVertex { position: [s, s, s], normal: [1.0, 0.0, 0.0], color: [1.0, 0.2, 0.2] },
        InternalVertex { position: [s, -s, s], normal: [1.0, 0.0, 0.0], color: [1.0, 0.2, 0.2] },
        // -X face (Cyan)
        InternalVertex { position: [-s, -s, s], normal: [-1.0, 0.0, 0.0], color: [0.2, 1.0, 1.0] },
        InternalVertex { position: [-s, s, s], normal: [-1.0, 0.0, 0.0], color: [0.2, 1.0, 1.0] },
        InternalVertex { position: [-s, s, -s], normal: [-1.0, 0.0, 0.0], color: [0.2, 1.0, 1.0] },
        InternalVertex { position: [-s, -s, -s], normal: [-1.0, 0.0, 0.0], color: [0.2, 1.0, 1.0] },
        // +Y face (Green)
        InternalVertex { position: [-s, s, -s], normal: [0.0, 1.0, 0.0], color: [0.2, 1.0, 0.2] },
        InternalVertex { position: [-s, s, s], normal: [0.0, 1.0, 0.0], color: [0.2, 1.0, 0.2] },
        InternalVertex { position: [s, s, s], normal: [0.0, 1.0, 0.0], color: [0.2, 1.0, 0.2] },
        InternalVertex { position: [s, s, -s], normal: [0.0, 1.0, 0.0], color: [0.2, 1.0, 0.2] },
        // -Y face (Magenta)
        InternalVertex { position: [-s, -s, s], normal: [0.0, -1.0, 0.0], color: [1.0, 0.2, 1.0] },
        InternalVertex { position: [-s, -s, -s], normal: [0.0, -1.0, 0.0], color: [1.0, 0.2, 1.0] },
        InternalVertex { position: [s, -s, -s], normal: [0.0, -1.0, 0.0], color: [1.0, 0.2, 1.0] },
        InternalVertex { position: [s, -s, s], normal: [0.0, -1.0, 0.0], color: [1.0, 0.2, 1.0] },
        // +Z face (Blue)
        InternalVertex { position: [-s, -s, s], normal: [0.0, 0.0, 1.0], color: [0.2, 0.2, 1.0] },
        InternalVertex { position: [s, -s, s], normal: [0.0, 0.0, 1.0], color: [0.2, 0.2, 1.0] },
        InternalVertex { position: [s, s, s], normal: [0.0, 0.0, 1.0], color: [0.2, 0.2, 1.0] },
        InternalVertex { position: [-s, s, s], normal: [0.0, 0.0, 1.0], color: [0.2, 0.2, 1.0] },
        // -Z face (Yellow)
        InternalVertex { position: [s, -s, -s], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 0.2] },
        InternalVertex { position: [-s, -s, -s], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 0.2] },
        InternalVertex { position: [-s, s, -s], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 0.2] },
        InternalVertex { position: [s, s, -s], normal: [0.0, 0.0, -1.0], color: [1.0, 1.0, 0.2] },
    ];

    let indices: Vec<u32> = vec![
        0, 1, 2, 0, 2, 3,
        4, 5, 6, 4, 6, 7,
        8, 9, 10, 8, 10, 11,
        12, 13, 14, 12, 14, 15,
        16, 17, 18, 16, 18, 19,
        20, 21, 22, 20, 22, 23,
    ];

    (vertices, indices)
}
