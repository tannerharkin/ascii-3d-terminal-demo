struct Uniforms {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    light_dir: vec4<f32>,
    lighting_mode: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvp * vec4<f32>(in.position, 1.0);
    // Transform normal by model matrix (assuming no non-uniform scaling)
    out.world_normal = (uniforms.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.color = in.color;
    out.world_pos = (uniforms.model * vec4<f32>(in.position, 1.0)).xyz;
    return out;
}

// Lighting mode values:
// 0 = Flat, 1 = Diffuse, 2 = Specular, 3 = Toon, 4 = Gradient, 5 = Normals

// Calculate diffuse lighting (shared by multiple modes)
fn calc_diffuse(normal: vec3<f32>) -> f32 {
    let key_light_dir = normalize(uniforms.light_dir.xyz);
    let key_diffuse = max(dot(normal, key_light_dir), 0.0);

    let fill_light_dir = normalize(vec3<f32>(-0.5, 0.3, -0.7));
    let fill_diffuse = max(dot(normal, fill_light_dir), 0.0) * 0.4;

    let rim_light_dir = normalize(vec3<f32>(0.0, 0.0, -1.0));
    let rim_diffuse = max(dot(normal, rim_light_dir), 0.0) * 0.3;

    let up = vec3<f32>(0.0, 1.0, 0.0);
    let hemisphere_factor = dot(normal, up) * 0.5 + 0.5;
    let ambient = mix(0.2, 0.4, hemisphere_factor);

    return ambient + key_diffuse * 0.5 + fill_diffuse + rim_diffuse;
}

// Calculate specular highlight
fn calc_specular(normal: vec3<f32>, view_dir: vec3<f32>) -> f32 {
    let key_light_dir = normalize(uniforms.light_dir.xyz);
    let half_dir = normalize(key_light_dir + view_dir);
    let spec = pow(max(dot(normal, half_dir), 0.0), 32.0);
    return spec * 0.5;
}

// Quantize for toon shading
fn toon_shade(intensity: f32) -> f32 {
    if (intensity > 0.95) { return 1.0; }
    else if (intensity > 0.5) { return 0.7; }
    else if (intensity > 0.25) { return 0.4; }
    else { return 0.2; }
}

// HSV to RGB conversion for gradient mode
fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let x = c * (1.0 - abs(fract(h * 6.0) * 2.0 - 1.0));
    let m = v - c;

    var rgb: vec3<f32>;
    let hi = u32(floor(h * 6.0)) % 6u;
    if (hi == 0u) { rgb = vec3<f32>(c, x, 0.0); }
    else if (hi == 1u) { rgb = vec3<f32>(x, c, 0.0); }
    else if (hi == 2u) { rgb = vec3<f32>(0.0, c, x); }
    else if (hi == 3u) { rgb = vec3<f32>(0.0, x, c); }
    else if (hi == 4u) { rgb = vec3<f32>(x, 0.0, c); }
    else { rgb = vec3<f32>(c, 0.0, x); }

    return rgb + vec3<f32>(m, m, m);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);

    // View direction (assuming camera at origin looking at -Z)
    let view_dir = normalize(-in.world_pos);

    var final_color: vec3<f32>;
    let mode = uniforms.lighting_mode;

    if (mode == 0u) {
        // Flat: No shading, just vertex color
        final_color = in.color;
    } else if (mode == 2u) {
        // Specular: Diffuse + specular highlights
        let diffuse = calc_diffuse(normal);
        let spec = calc_specular(normal, view_dir);
        final_color = in.color * diffuse + vec3<f32>(spec);
    } else if (mode == 3u) {
        // Toon: Cel-shaded with quantized bands
        let raw_lighting = calc_diffuse(normal);
        let toon_lighting = toon_shade(raw_lighting);
        final_color = in.color * toon_lighting;
        // Add subtle outline darkening at grazing angles
        let edge_factor = 1.0 - pow(1.0 - abs(dot(normal, view_dir)), 2.0);
        final_color = final_color * mix(0.3, 1.0, edge_factor);
    } else if (mode == 4u) {
        // Gradient: Height-based rainbow gradient
        let height = in.world_pos.y * 0.3 + 0.5;
        let hue = fract(height);
        let base_gradient = hsv2rgb(hue, 0.8, 0.9);
        let lighting = calc_diffuse(normal) * 0.5 + 0.5;
        final_color = base_gradient * lighting;
    } else if (mode == 5u) {
        // Normals: Visualize normals as colors
        final_color = normal * 0.5 + vec3<f32>(0.5);
    } else {
        // Default (mode == 1u): Diffuse multi-light setup
        let lighting = calc_diffuse(normal);
        final_color = in.color * lighting;
    }

    return vec4<f32>(final_color, 1.0);
}
