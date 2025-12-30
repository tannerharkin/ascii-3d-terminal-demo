// Edge Detection Pass
// Combines depth discontinuities + normal discontinuities + DoG luminance edges
// Following AcerolaFX approach

struct Uniforms {
    width: u32,
    height: u32,
    depth_threshold: f32,    // Depth discontinuity threshold (default 0.02)
    normal_threshold: f32,   // Normal discontinuity threshold (default 0.5)
    dog_threshold: f32,      // DoG edge threshold (default 0.02)
    use_depth: u32,          // Enable depth edges (1 = true)
    use_normals: u32,        // Enable normal edges (1 = true)
    use_dog: u32,            // Enable DoG edges (1 = true)
};

@group(0) @binding(0)
var color_texture: texture_2d<f32>;

@group(0) @binding(1)
var depth_texture: texture_depth_2d;

@group(0) @binding(2)
var output_texture: texture_storage_2d<rgba32float, write>;

@group(0) @binding(3)
var<uniform> uniforms: Uniforms;

// Luminance coefficients (Rec. 709)
const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

// Get luminance from color
fn get_luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(LUMA_R, LUMA_G, LUMA_B));
}

// Sample depth at coordinates (with bounds check)
fn sample_depth(coords: vec2<i32>) -> f32 {
    let clamped = clamp(coords, vec2<i32>(0), vec2<i32>(i32(uniforms.width) - 1, i32(uniforms.height) - 1));
    return textureLoad(depth_texture, clamped, 0);
}

// Sample color at coordinates (with bounds check)
fn sample_color(coords: vec2<i32>) -> vec3<f32> {
    let clamped = clamp(coords, vec2<i32>(0), vec2<i32>(i32(uniforms.width) - 1, i32(uniforms.height) - 1));
    return textureLoad(color_texture, clamped, 0).rgb;
}

// Calculate screen-space normal from depth buffer
// Uses cross product of depth gradients
fn calculate_normal(coords: vec2<i32>) -> vec3<f32> {
    let center_depth = sample_depth(coords);
    let left_depth = sample_depth(coords + vec2<i32>(-1, 0));
    let right_depth = sample_depth(coords + vec2<i32>(1, 0));
    let up_depth = sample_depth(coords + vec2<i32>(0, -1));
    let down_depth = sample_depth(coords + vec2<i32>(0, 1));

    // Calculate depth gradients
    let dx = (right_depth - left_depth) * 0.5;
    let dy = (down_depth - up_depth) * 0.5;

    // Construct normal from gradients
    // The normal points towards the camera (positive Z)
    let normal = normalize(vec3<f32>(-dx * 100.0, -dy * 100.0, 1.0));
    return normal;
}

// Simple DoG approximation using 3x3 vs 5x5 box filter difference
fn calculate_dog(coords: vec2<i32>) -> f32 {
    // 3x3 average (narrow)
    var sum3x3: f32 = 0.0;
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let color = sample_color(coords + vec2<i32>(dx, dy));
            sum3x3 += get_luminance(color);
        }
    }
    let avg3x3 = sum3x3 / 9.0;

    // 5x5 average (wide)
    var sum5x5: f32 = 0.0;
    for (var dy: i32 = -2; dy <= 2; dy++) {
        for (var dx: i32 = -2; dx <= 2; dx++) {
            let color = sample_color(coords + vec2<i32>(dx, dy));
            sum5x5 += get_luminance(color);
        }
    }
    let avg5x5 = sum5x5 / 25.0;

    // Difference of averages (approximates DoG)
    return abs(avg3x3 - avg5x5);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coords = vec2<i32>(gid.xy);

    // Bounds check
    if (coords.x >= i32(uniforms.width) || coords.y >= i32(uniforms.height)) {
        return;
    }

    var edge_strength: f32 = 0.0;

    // Sample center values
    let center_depth = sample_depth(coords);
    let center_color = sample_color(coords);
    let center_lum = get_luminance(center_color);

    // === Depth-based edge detection ===
    if (uniforms.use_depth == 1u) {
        // Sample 8 neighbors for depth
        let depth_n  = sample_depth(coords + vec2<i32>(0, -1));
        let depth_s  = sample_depth(coords + vec2<i32>(0, 1));
        let depth_e  = sample_depth(coords + vec2<i32>(1, 0));
        let depth_w  = sample_depth(coords + vec2<i32>(-1, 0));
        let depth_ne = sample_depth(coords + vec2<i32>(1, -1));
        let depth_nw = sample_depth(coords + vec2<i32>(-1, -1));
        let depth_se = sample_depth(coords + vec2<i32>(1, 1));
        let depth_sw = sample_depth(coords + vec2<i32>(-1, 1));

        // Sum of absolute differences (like AcerolaFX)
        var depth_sum: f32 = 0.0;
        depth_sum += abs(depth_n - center_depth);
        depth_sum += abs(depth_s - center_depth);
        depth_sum += abs(depth_e - center_depth);
        depth_sum += abs(depth_w - center_depth);
        depth_sum += abs(depth_ne - center_depth);
        depth_sum += abs(depth_nw - center_depth);
        depth_sum += abs(depth_se - center_depth);
        depth_sum += abs(depth_sw - center_depth);

        if (depth_sum > uniforms.depth_threshold) {
            edge_strength = 1.0;
        }
    }

    // === Normal-based edge detection ===
    if (uniforms.use_normals == 1u && edge_strength < 1.0) {
        let center_normal = calculate_normal(coords);

        // Sample neighbor normals
        let normal_n = calculate_normal(coords + vec2<i32>(0, -1));
        let normal_s = calculate_normal(coords + vec2<i32>(0, 1));
        let normal_e = calculate_normal(coords + vec2<i32>(1, 0));
        let normal_w = calculate_normal(coords + vec2<i32>(-1, 0));

        // Sum of normal differences
        var normal_sum: f32 = 0.0;
        normal_sum += length(normal_n - center_normal);
        normal_sum += length(normal_s - center_normal);
        normal_sum += length(normal_e - center_normal);
        normal_sum += length(normal_w - center_normal);

        if (normal_sum > uniforms.normal_threshold) {
            edge_strength = 1.0;
        }
    }

    // === DoG-based edge detection ===
    if (uniforms.use_dog == 1u && edge_strength < 1.0) {
        let dog = calculate_dog(coords);
        if (dog > uniforms.dog_threshold) {
            edge_strength = 1.0;
        }
    }

    // Output: R = edge strength, G = luminance, B = depth, A = unused
    textureStore(output_texture, coords, vec4<f32>(edge_strength, center_lum, center_depth, 1.0));
}
