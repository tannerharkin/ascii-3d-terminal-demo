use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

// Use Vertex from the gpu module
use crate::gpu::Vertex;

const SUPPORTED_EXTENSIONS: &[&str] = &["obj", "gltf", "glb"];

pub struct ModelData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

/// Discover all supported model files in a directory (including subdirectories)
pub fn discover_models(dir: &Path) -> Vec<PathBuf> {
    let mut models = Vec::new();
    discover_models_recursive(dir, dir, &mut models);
    models.sort_by(|a, b| get_model_display_name(a).cmp(&get_model_display_name(b)));
    models
}

fn discover_models_recursive(base_dir: &Path, dir: &Path, models: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // Recurse into subdirectories
            discover_models_recursive(base_dir, &path, models);
        } else if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if SUPPORTED_EXTENSIONS.contains(&ext.to_lowercase().as_str()) {
                    models.push(path);
                }
            }
        }
    }
}

/// Get a display name for a model path
/// If the model is in a subdirectory, uses the folder name instead of the file name
/// (handles common packaging like "MyModel/scene.gltf" -> "MyModel")
pub fn get_model_display_name(path: &Path) -> String {
    // Get the file name
    let file_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Check if the file has a generic name
    let generic_names = ["scene", "model", "mesh", "object", "untitled"];
    let is_generic = generic_names.iter().any(|&g| file_name.eq_ignore_ascii_case(g));

    if is_generic {
        // Use the parent folder name instead
        if let Some(parent) = path.parent() {
            if let Some(folder_name) = parent.file_name().and_then(|s| s.to_str()) {
                // Don't use "models" as the name
                if !folder_name.eq_ignore_ascii_case("models") {
                    return folder_name.to_string();
                }
            }
        }
    }

    // Use the file name (with extension for clarity)
    path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// Load a model from file, dispatching based on extension
pub fn load_model(path: &Path) -> Result<ModelData> {
    match path.extension().and_then(|e| e.to_str()).map(|s| s.to_lowercase()) {
        Some(ext) if ext == "obj" => load_obj(path),
        Some(ext) if ext == "gltf" || ext == "glb" => load_gltf(path),
        _ => Err(anyhow!("Unsupported model format: {:?}", path)),
    }
}

/// Load an OBJ file using tobj
fn load_obj(path: &Path) -> Result<ModelData> {
    let load_options = tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ..Default::default()
    };

    let (models, materials_result) = tobj::load_obj(path, &load_options)?;

    if models.is_empty() {
        return Err(anyhow!("No meshes found in OBJ file"));
    }

    // Get materials if available
    let materials = materials_result.ok().unwrap_or_default();

    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();

    for model in &models {
        let mesh = &model.mesh;
        let base_index = all_vertices.len() as u32;

        // Get material color if available
        let material_color = mesh
            .material_id
            .and_then(|id| materials.get(id))
            .map(|m| m.diffuse.unwrap_or([0.8, 0.8, 0.8]))
            .unwrap_or([0.8, 0.8, 0.8]);

        // Process vertices
        let num_vertices = mesh.positions.len() / 3;
        let has_normals = !mesh.normals.is_empty();

        for i in 0..num_vertices {
            let px = mesh.positions[i * 3];
            let py = mesh.positions[i * 3 + 1];
            let pz = mesh.positions[i * 3 + 2];

            let (nx, ny, nz) = if has_normals {
                (
                    mesh.normals[i * 3],
                    mesh.normals[i * 3 + 1],
                    mesh.normals[i * 3 + 2],
                )
            } else {
                (0.0, 1.0, 0.0) // Default up normal, will compute later if needed
            };

            // Use vertex colors if available, otherwise material color
            let color = if !mesh.vertex_color.is_empty() && mesh.vertex_color.len() > i * 3 + 2 {
                [
                    mesh.vertex_color[i * 3],
                    mesh.vertex_color[i * 3 + 1],
                    mesh.vertex_color[i * 3 + 2],
                ]
            } else {
                material_color
            };

            all_vertices.push(Vertex {
                position: [px, py, pz],
                normal: [nx, ny, nz],
                color,
            });
        }

        // Process indices
        for &idx in &mesh.indices {
            all_indices.push(base_index + idx);
        }
    }

    // Compute normals if not provided
    if models.iter().all(|m| m.mesh.normals.is_empty()) {
        compute_normals(&mut all_vertices, &all_indices);
    }

    // Normalize model to fit in view
    normalize_model(&mut all_vertices);

    Ok(ModelData {
        vertices: all_vertices,
        indices: all_indices,
    })
}

/// Load a glTF/GLB file
fn load_gltf(path: &Path) -> Result<ModelData> {
    let (document, buffers, _images) = gltf::import(path)?;

    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let base_index = all_vertices.len() as u32;

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            // Read positions (required)
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or_else(|| anyhow!("No positions in mesh"))?
                .collect();

            // Read normals (optional)
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

            // Get material color
            let material = primitive.material();
            let base_color = material
                .pbr_metallic_roughness()
                .base_color_factor();
            let color = [base_color[0], base_color[1], base_color[2]];

            // Read vertex colors if available
            let colors: Option<Vec<[f32; 3]>> = reader.read_colors(0).map(|iter| {
                iter.into_rgb_f32().collect()
            });

            // Build vertices
            for i in 0..positions.len() {
                let vertex_color = colors
                    .as_ref()
                    .and_then(|c| c.get(i).copied())
                    .unwrap_or(color);

                all_vertices.push(Vertex {
                    position: positions[i],
                    normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
                    color: vertex_color,
                });
            }

            // Read indices
            if let Some(indices) = reader.read_indices() {
                for idx in indices.into_u32() {
                    all_indices.push(base_index + idx);
                }
            } else {
                // Non-indexed geometry: generate indices
                for i in 0..positions.len() as u32 {
                    all_indices.push(base_index + i);
                }
            }
        }
    }

    if all_vertices.is_empty() {
        return Err(anyhow!("No geometry found in glTF file"));
    }

    // Compute normals if they were all default
    let needs_normals = all_vertices.iter().all(|v| v.normal == [0.0, 1.0, 0.0]);
    if needs_normals {
        compute_normals(&mut all_vertices, &all_indices);
    }

    // Normalize model to fit in view
    normalize_model(&mut all_vertices);

    Ok(ModelData {
        vertices: all_vertices,
        indices: all_indices,
    })
}

/// Compute face normals and assign to vertices
fn compute_normals(vertices: &mut [Vertex], indices: &[u32]) {
    // Reset all normals
    for v in vertices.iter_mut() {
        v.normal = [0.0, 0.0, 0.0];
    }

    // Accumulate face normals
    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }

        let v0 = vertices[i0].position;
        let v1 = vertices[i1].position;
        let v2 = vertices[i2].position;

        // Edge vectors
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        // Cross product
        let nx = e1[1] * e2[2] - e1[2] * e2[1];
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        let nz = e1[0] * e2[1] - e1[1] * e2[0];

        // Accumulate (area-weighted)
        for &i in &[i0, i1, i2] {
            vertices[i].normal[0] += nx;
            vertices[i].normal[1] += ny;
            vertices[i].normal[2] += nz;
        }
    }

    // Normalize
    for v in vertices.iter_mut() {
        let len = (v.normal[0].powi(2) + v.normal[1].powi(2) + v.normal[2].powi(2)).sqrt();
        if len > 1e-6 {
            v.normal[0] /= len;
            v.normal[1] /= len;
            v.normal[2] /= len;
        } else {
            v.normal = [0.0, 1.0, 0.0];
        }
    }
}

/// Normalize model to fit in a unit cube centered at origin
fn normalize_model(vertices: &mut [Vertex]) {
    if vertices.is_empty() {
        return;
    }

    // Find bounding box
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];

    for v in vertices.iter() {
        for i in 0..3 {
            min[i] = min[i].min(v.position[i]);
            max[i] = max[i].max(v.position[i]);
        }
    }

    // Center
    let center = [
        (min[0] + max[0]) / 2.0,
        (min[1] + max[1]) / 2.0,
        (min[2] + max[2]) / 2.0,
    ];

    // Scale to fit in ~1.6 unit cube (matching original cube size)
    let size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let max_dim = size[0].max(size[1]).max(size[2]);
    let scale = if max_dim > 1e-6 { 1.6 / max_dim } else { 1.0 };

    // Apply transform
    for v in vertices.iter_mut() {
        v.position[0] = (v.position[0] - center[0]) * scale;
        v.position[1] = (v.position[1] - center[1]) * scale;
        v.position[2] = (v.position[2] - center[2]) * scale;
    }
}
