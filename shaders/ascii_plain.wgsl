struct Uniforms {
    width: u32,
    height: u32,
    cols: u32,
    rows: u32,
};

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var<uniform> uniforms: Uniforms;

@group(0) @binding(2)
var<storage, read_write> output: array<u32>;

// Exposure and gamma controls for better visibility
const EXPOSURE: f32 = 1.5;      // Boost brightness
const GAMMA: f32 = 0.8;         // Adjust contrast curve

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;

    // Bounds check
    if (col >= uniforms.cols || row >= uniforms.rows) {
        return;
    }

    // Calculate pixel coordinates for this cell
    let cell_width = f32(uniforms.width) / f32(uniforms.cols);
    let cell_height = f32(uniforms.height) / f32(uniforms.rows);

    // Sample multiple points in the cell for better coverage
    var total_luminance = 0.0;
    let samples = 4u;

    for (var sy = 0u; sy < 2u; sy++) {
        for (var sx = 0u; sx < 2u; sx++) {
            let px = u32(f32(col) * cell_width + cell_width * (0.25 + f32(sx) * 0.5));
            let py = u32(f32(row) * cell_height + cell_height * (0.25 + f32(sy) * 0.5));

            let color = textureLoad(input_texture, vec2<i32>(i32(px), i32(py)), 0);
            let lum = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
            total_luminance += lum;
        }
    }

    var luminance = total_luminance / f32(samples);

    // Apply exposure and gamma for better visibility
    luminance = saturate(pow(luminance * EXPOSURE, GAMMA));

    // Map luminance to character index (0-9)
    // Characters: " .:-=+*#%@" (index 0 = space = darkest, index 9 = @ = brightest)
    let char_index = u32(clamp(luminance * 10.0, 0.0, 9.0));

    // Write to output buffer
    let output_index = row * uniforms.cols + col;
    output[output_index] = char_index;
}
