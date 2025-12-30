// Pass 3: Difference of Gaussians (DoG) edge detection
// Input: Two blurred luminance textures at different sigmas
// Output: Binary edge map

struct Uniforms {
    width: u32,
    height: u32,
    tau: f32,           // Typically 1.0, controls edge sensitivity
    threshold: f32,     // Edge detection threshold (e.g., 0.05)
};

@group(0) @binding(0)
var blur1_texture: texture_2d<f32>;  // Blur with smaller sigma (e.g., 2.0)

@group(0) @binding(1)
var blur2_texture: texture_2d<f32>;  // Blur with larger sigma (e.g., 3.2)

@group(0) @binding(2)
var output_texture: texture_storage_2d<r32float, write>;

@group(0) @binding(3)
var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coords = vec2<i32>(gid.xy);

    // Bounds check
    if (coords.x >= i32(uniforms.width) || coords.y >= i32(uniforms.height)) {
        return;
    }

    // Sample both blurred textures
    let blur1 = textureLoad(blur1_texture, coords, 0).r;
    let blur2 = textureLoad(blur2_texture, coords, 0).r;

    // Difference of Gaussians
    // DoG approximates the Laplacian of Gaussian (LoG) edge detector
    // Subtracting a more blurred version from a less blurred version
    // highlights regions of rapid intensity change (edges)
    let dog = blur1 - uniforms.tau * blur2;

    // Apply threshold to create binary edge map
    // Using absolute value since edges can be positive or negative
    let edge = select(0.0, 1.0, abs(dog) > uniforms.threshold);

    // Write edge result
    textureStore(output_texture, coords, vec4<f32>(edge, 0.0, 0.0, 1.0));
}
