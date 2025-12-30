// Sobel Edge Direction Pass
// Calculates edge direction from luminance gradients
// Only processes pixels marked as edges

struct Uniforms {
    width: u32,
    height: u32,
    _padding: vec2<u32>,
};

@group(0) @binding(0)
var edge_texture: texture_2d<f32>;  // From edge_detect: R=edge, G=luminance, B=depth

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba32float, write>;

@group(0) @binding(2)
var<uniform> uniforms: Uniforms;

const PI: f32 = 3.14159265359;

// Sample luminance from edge texture (stored in G channel)
fn sample_lum(coords: vec2<i32>) -> f32 {
    let clamped = clamp(coords, vec2<i32>(0), vec2<i32>(i32(uniforms.width) - 1, i32(uniforms.height) - 1));
    return textureLoad(edge_texture, clamped, 0).g;
}

// Sample edge strength (stored in R channel)
fn sample_edge(coords: vec2<i32>) -> f32 {
    let clamped = clamp(coords, vec2<i32>(0), vec2<i32>(i32(uniforms.width) - 1, i32(uniforms.height) - 1));
    return textureLoad(edge_texture, clamped, 0).r;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coords = vec2<i32>(gid.xy);

    // Bounds check
    if (coords.x >= i32(uniforms.width) || coords.y >= i32(uniforms.height)) {
        return;
    }

    // Read edge info
    let edge_data = textureLoad(edge_texture, coords, 0);
    let is_edge = edge_data.r;
    let luminance = edge_data.g;
    let depth = edge_data.b;

    // Default output: no direction, not an edge
    var direction: f32 = -1.0;
    var edge_flag: f32 = 0.0;

    if (is_edge > 0.5) {
        edge_flag = 1.0;

        // Sample 3x3 neighborhood for Sobel
        // Layout:
        // [nw] [n ] [ne]
        // [w ] [c ] [e ]
        // [sw] [s ] [se]
        let lum_nw = sample_lum(coords + vec2<i32>(-1, -1));
        let lum_n  = sample_lum(coords + vec2<i32>(0, -1));
        let lum_ne = sample_lum(coords + vec2<i32>(1, -1));
        let lum_w  = sample_lum(coords + vec2<i32>(-1, 0));
        let lum_e  = sample_lum(coords + vec2<i32>(1, 0));
        let lum_sw = sample_lum(coords + vec2<i32>(-1, 1));
        let lum_s  = sample_lum(coords + vec2<i32>(0, 1));
        let lum_se = sample_lum(coords + vec2<i32>(1, 1));

        // Sobel kernels
        // Gx (horizontal gradient - detects vertical edges):
        // [-1  0  1]
        // [-2  0  2]
        // [-1  0  1]
        let gx = -lum_nw - 2.0 * lum_w - lum_sw + lum_ne + 2.0 * lum_e + lum_se;

        // Gy (vertical gradient - detects horizontal edges):
        // [-1 -2 -1]
        // [ 0  0  0]
        // [ 1  2  1]
        let gy = -lum_nw - 2.0 * lum_n - lum_ne + lum_sw + 2.0 * lum_s + lum_se;

        // Calculate angle
        let theta = atan2(gy, gx);

        // Quantize to 4 directions (matching AcerolaFX exactly)
        // Direction encoding: 0=| 1=- 2=/ 3=\
        let abs_theta = abs(theta) / PI;  // Normalize to [0, 1]

        // AcerolaFX direction thresholds:
        if (abs_theta < 0.05 || abs_theta > 0.9) {
            // VERTICAL |
            direction = 0.0;
        } else if (abs_theta > 0.45 && abs_theta < 0.55) {
            // HORIZONTAL -
            direction = 1.0;
        } else if (abs_theta > 0.05 && abs_theta < 0.45) {
            // DIAGONAL 1
            if (theta > 0.0) {
                direction = 3.0;  // \
            } else {
                direction = 2.0;  // /
            }
        } else if (abs_theta > 0.55 && abs_theta < 0.9) {
            // DIAGONAL 2
            if (theta > 0.0) {
                direction = 2.0;  // /
            } else {
                direction = 3.0;  // \
            }
        }
    }

    // Output: R = direction (0-3, or -1 if not edge), G = edge flag, B = luminance, A = depth
    textureStore(output_texture, coords, vec4<f32>(direction, edge_flag, luminance, depth));
}
