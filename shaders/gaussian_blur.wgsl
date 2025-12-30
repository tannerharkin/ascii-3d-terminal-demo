// Pass 2: Separable Gaussian blur
// Used for both horizontal and vertical passes
// Direction: (1,0) for horizontal, (0,1) for vertical

struct Uniforms {
    width: u32,
    height: u32,
    direction_x: i32,  // 1 for horizontal, 0 for vertical
    direction_y: i32,  // 0 for horizontal, 1 for vertical
    sigma: f32,        // Gaussian sigma (e.g., 2.0 or 3.2)
    _padding: vec3<f32>,
};

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<r32float, write>;

@group(0) @binding(2)
var<uniform> uniforms: Uniforms;

const PI: f32 = 3.14159265359;
const KERNEL_RADIUS: i32 = 5;  // -5 to +5 = 11 samples

// Compute Gaussian weight
fn gaussian(x: f32, sigma: f32) -> f32 {
    let coeff = 1.0 / (sqrt(2.0 * PI) * sigma);
    let exponent = -(x * x) / (2.0 * sigma * sigma);
    return coeff * exp(exponent);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coords = vec2<i32>(gid.xy);

    // Bounds check
    if (coords.x >= i32(uniforms.width) || coords.y >= i32(uniforms.height)) {
        return;
    }

    let direction = vec2<i32>(uniforms.direction_x, uniforms.direction_y);
    let sigma = uniforms.sigma;

    var blur_sum: f32 = 0.0;
    var weight_sum: f32 = 0.0;

    // Sample along the blur direction
    for (var i: i32 = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
        let sample_coords = coords + direction * i;

        // Clamp to texture bounds
        let clamped_coords = clamp(
            sample_coords,
            vec2<i32>(0, 0),
            vec2<i32>(i32(uniforms.width) - 1, i32(uniforms.height) - 1)
        );

        let sample_value = textureLoad(input_texture, clamped_coords, 0).r;
        let weight = gaussian(f32(i), sigma);

        blur_sum += sample_value * weight;
        weight_sum += weight;
    }

    // Normalize
    let result = blur_sum / weight_sum;

    // Write to output
    textureStore(output_texture, coords, vec4<f32>(result, 0.0, 0.0, 1.0));
}
