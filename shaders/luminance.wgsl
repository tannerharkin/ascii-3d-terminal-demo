// Pass 1: Extract luminance from color texture
// Input: Color texture (RGBA)
// Output: Luminance texture (R16F)

struct Uniforms {
    width: u32,
    height: u32,
    _padding: vec2<u32>,
};

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<r32float, write>;

@group(0) @binding(2)
var<uniform> uniforms: Uniforms;

// Luminance coefficients (Rec. 709)
const LUMA_R: f32 = 0.2127;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coords = gid.xy;

    // Bounds check
    if (coords.x >= uniforms.width || coords.y >= uniforms.height) {
        return;
    }

    // Sample color
    let color = textureLoad(input_texture, vec2<i32>(coords), 0);

    // Calculate luminance
    let luminance = LUMA_R * color.r + LUMA_G * color.g + LUMA_B * color.b;

    // Write to output
    textureStore(output_texture, vec2<i32>(coords), vec4<f32>(luminance, 0.0, 0.0, 1.0));
}
