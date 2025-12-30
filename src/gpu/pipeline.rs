use anyhow::Result;
use bytemuck::{Pod, Zeroable};

/// Uniforms for edge detection pass
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct EdgeDetectUniforms {
    width: u32,
    height: u32,
    depth_threshold: f32,
    normal_threshold: f32,
    dog_threshold: f32,
    use_depth: u32,
    use_normals: u32,
    use_dog: u32,
}

/// Uniforms for Sobel pass
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SobelUniforms {
    width: u32,
    height: u32,
    _padding: [u32; 2],
}

/// Uniforms for final ASCII pass
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AsciiUniforms {
    tex_width: u32,
    tex_height: u32,
    cols: u32,
    rows: u32,
    edge_threshold: u32,
    exposure: f32,
    gamma: f32,
    _padding: f32,
}

/// 3-Pass ASCII Pipeline with edge detection
/// Pass 1: Edge detection (depth + normals + DoG)
/// Pass 2: Sobel direction
/// Pass 3: ASCII character selection with tile voting
pub struct AsciiPipeline {
    // Dimensions
    cols: u32,
    rows: u32,
    tex_width: u32,
    tex_height: u32,

    // Compute pipelines
    edge_pipeline: wgpu::ComputePipeline,
    sobel_pipeline: wgpu::ComputePipeline,
    ascii_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    edge_layout: wgpu::BindGroupLayout,
    sobel_layout: wgpu::BindGroupLayout,
    ascii_layout: wgpu::BindGroupLayout,

    // Intermediate textures
    edge_tex: wgpu::Texture,      // R=edge, G=lum, B=depth
    direction_tex: wgpu::Texture, // R=dir, G=edge_flag, B=lum, A=depth

    // Uniform buffers
    edge_uniform_buf: wgpu::Buffer,
    sobel_uniform_buf: wgpu::Buffer,
    ascii_uniform_buf: wgpu::Buffer,

    // Output buffers
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,

    // Bind groups (created when input textures are provided)
    edge_bind_group: Option<wgpu::BindGroup>,
    sobel_bind_group: Option<wgpu::BindGroup>,
    ascii_bind_group: Option<wgpu::BindGroup>,

    // Tunable parameters
    depth_threshold: f32,
    normal_threshold: f32,
    dog_threshold: f32,
    use_depth: bool,
    use_normals: bool,
    use_dog: bool,
    edge_vote_threshold: u32,
    exposure: f32,
    gamma: f32,
}

impl AsciiPipeline {
    pub fn new(
        device: &wgpu::Device,
        cols: u32,
        rows: u32,
        tex_width: u32,
        tex_height: u32,
    ) -> Result<Self> {
        // Tunable parameters - adjusted for cleaner output with loaded models
        let depth_threshold = 0.08;   // Depth discontinuity threshold (higher = less sensitive)
        let normal_threshold = 0.8;   // Normal discontinuity threshold (higher = less sensitive)
        let dog_threshold = 0.08;     // DoG edge threshold (higher = less sensitive)
        let use_depth = true;         // Enable depth-based edges
        let use_normals = true;       // Enable normal-based edges
        let use_dog = true;           // Enable DoG edges - all three are critical
        let edge_vote_threshold = 3;  // Min edge pixels in tile to use edge char
        let exposure = 1.5;           // Luminance boost
        let gamma = 0.8;              // Contrast curve (attenuation)

        // Create shader modules
        let edge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Edge Detection Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/edge_detect.wgsl").into()),
        });

        let sobel_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sobel Direction Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/sobel_edges.wgsl").into()),
        });

        let ascii_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ASCII Edges Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/ascii_edges.wgsl").into()),
        });

        // Create bind group layouts
        let edge_layout = Self::create_edge_layout(device);
        let sobel_layout = Self::create_sobel_layout(device);
        let ascii_layout = Self::create_ascii_layout(device);

        // Create pipelines
        let edge_pipeline = Self::create_pipeline(device, &edge_shader, &edge_layout, "Edge Pipeline");
        let sobel_pipeline = Self::create_pipeline(device, &sobel_shader, &sobel_layout, "Sobel Pipeline");
        let ascii_pipeline = Self::create_pipeline(device, &ascii_shader, &ascii_layout, "ASCII Pipeline");

        // Create intermediate textures (RGBA32Float for flexibility)
        let edge_tex = Self::create_rgba32f_texture(device, tex_width, tex_height, "Edge Texture");
        let direction_tex = Self::create_rgba32f_texture(device, tex_width, tex_height, "Direction Texture");

        // Create uniform buffers
        let edge_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Edge Uniforms"),
            size: std::mem::size_of::<EdgeDetectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sobel_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sobel Uniforms"),
            size: std::mem::size_of::<SobelUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ascii_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ASCII Uniforms"),
            size: std::mem::size_of::<AsciiUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create output buffers
        let buffer_size = (cols * rows * 4) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ASCII Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ASCII Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            cols,
            rows,
            tex_width,
            tex_height,
            edge_pipeline,
            sobel_pipeline,
            ascii_pipeline,
            edge_layout,
            sobel_layout,
            ascii_layout,
            edge_tex,
            direction_tex,
            edge_uniform_buf,
            sobel_uniform_buf,
            ascii_uniform_buf,
            output_buffer,
            staging_buffer,
            edge_bind_group: None,
            sobel_bind_group: None,
            ascii_bind_group: None,
            depth_threshold,
            normal_threshold,
            dog_threshold,
            use_depth,
            use_normals,
            use_dog,
            edge_vote_threshold,
            exposure,
            gamma,
        })
    }

    fn create_edge_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Edge Detect Layout"),
            entries: &[
                // Color texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Output texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_sobel_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sobel Layout"),
            entries: &[
                // Edge texture input
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Direction texture output
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_ascii_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ASCII Layout"),
            entries: &[
                // Direction texture input
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Color texture input
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::BindGroupLayout,
        label: &str,
    ) -> wgpu::ComputePipeline {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} Layout", label)),
            bind_group_layouts: &[layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn create_rgba32f_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
    }

    pub fn resize(&mut self, device: &wgpu::Device, cols: u32, rows: u32, tex_width: u32, tex_height: u32) {
        let size_changed = tex_width != self.tex_width || tex_height != self.tex_height;
        let cols_changed = cols != self.cols || rows != self.rows;

        if size_changed {
            self.tex_width = tex_width;
            self.tex_height = tex_height;

            self.edge_tex = Self::create_rgba32f_texture(device, tex_width, tex_height, "Edge Texture");
            self.direction_tex = Self::create_rgba32f_texture(device, tex_width, tex_height, "Direction Texture");
        }

        if cols_changed {
            self.cols = cols;
            self.rows = rows;

            let buffer_size = (cols * rows * 4) as u64;
            self.output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ASCII Output Buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            self.staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ASCII Staging Buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        if size_changed || cols_changed {
            self.edge_bind_group = None;
            self.sobel_bind_group = None;
            self.ascii_bind_group = None;
        }
    }

    pub fn update_bind_groups(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        // Update uniform buffers
        let edge_uniforms = EdgeDetectUniforms {
            width: self.tex_width,
            height: self.tex_height,
            depth_threshold: self.depth_threshold,
            normal_threshold: self.normal_threshold,
            dog_threshold: self.dog_threshold,
            use_depth: if self.use_depth { 1 } else { 0 },
            use_normals: if self.use_normals { 1 } else { 0 },
            use_dog: if self.use_dog { 1 } else { 0 },
        };
        queue.write_buffer(&self.edge_uniform_buf, 0, bytemuck::cast_slice(&[edge_uniforms]));

        let sobel_uniforms = SobelUniforms {
            width: self.tex_width,
            height: self.tex_height,
            _padding: [0; 2],
        };
        queue.write_buffer(&self.sobel_uniform_buf, 0, bytemuck::cast_slice(&[sobel_uniforms]));

        let ascii_uniforms = AsciiUniforms {
            tex_width: self.tex_width,
            tex_height: self.tex_height,
            cols: self.cols,
            rows: self.rows,
            edge_threshold: self.edge_vote_threshold,
            exposure: self.exposure,
            gamma: self.gamma,
            _padding: 0.0,
        };
        queue.write_buffer(&self.ascii_uniform_buf, 0, bytemuck::cast_slice(&[ascii_uniforms]));

        // Create texture views for intermediate textures
        let edge_view = self.edge_tex.create_view(&Default::default());
        let direction_view = self.direction_tex.create_view(&Default::default());

        // Edge detection bind group
        self.edge_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Edge Bind Group"),
            layout: &self.edge_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(color_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&edge_view) },
                wgpu::BindGroupEntry { binding: 3, resource: self.edge_uniform_buf.as_entire_binding() },
            ],
        }));

        // Sobel bind group
        self.sobel_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sobel Bind Group"),
            layout: &self.sobel_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&edge_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&direction_view) },
                wgpu::BindGroupEntry { binding: 2, resource: self.sobel_uniform_buf.as_entire_binding() },
            ],
        }));

        // ASCII bind group
        self.ascii_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ASCII Bind Group"),
            layout: &self.ascii_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&direction_view) },
                wgpu::BindGroupEntry { binding: 1, resource: self.ascii_uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(color_view) },
            ],
        }));
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        // Workgroup counts for pixel-level passes (16x16 workgroups)
        let pixel_wg_x = (self.tex_width + 15) / 16;
        let pixel_wg_y = (self.tex_height + 15) / 16;

        // Workgroup counts for ASCII pass (1 thread per cell)
        let ascii_wg_x = self.cols;
        let ascii_wg_y = self.rows;

        // Pass 1: Edge detection
        if let Some(bg) = &self.edge_bind_group {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Edge Detection Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.edge_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(pixel_wg_x, pixel_wg_y, 1);
        }

        // Pass 2: Sobel direction
        if let Some(bg) = &self.sobel_bind_group {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sobel Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sobel_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(pixel_wg_x, pixel_wg_y, 1);
        }

        // Pass 3: ASCII character selection
        if let Some(bg) = &self.ascii_bind_group {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ASCII Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ascii_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(ascii_wg_x, ascii_wg_y, 1);
        }
    }

    pub fn copy_to_staging(&self, encoder: &mut wgpu::CommandEncoder) {
        let size = (self.cols * self.rows * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.output_buffer, 0, &self.staging_buffer, 0, size);
    }

    pub async fn read_results(&self, device: &wgpu::Device) -> Result<Vec<u32>> {
        let buffer_slice = self.staging_buffer.slice(..);

        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
        rx.recv()??;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        self.staging_buffer.unmap();

        Ok(result)
    }

    pub fn cols(&self) -> u32 {
        self.cols
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }
}
