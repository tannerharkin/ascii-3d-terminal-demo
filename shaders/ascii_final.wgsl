// Pass 5: Final ASCII character selection
// Simplified version without workgroup voting to avoid shared memory issues
// Each workgroup (1 thread) processes one ASCII character cell

struct Uniforms {
    width: u32,          // Texture width in pixels
    height: u32,         // Texture height in pixels
    cols: u32,           // Number of ASCII columns
    rows: u32,           // Number of ASCII rows
    tile_edge_min: u32,  // Minimum edge pixels to use edge char (e.g., 8)
    exposure: f32,       // Luminance boost (e.g., 1.5)
    gamma: f32,          // Contrast curve (e.g., 0.8)
    _padding: f32,
};

@group(0) @binding(0)
var direction_texture: texture_2d<f32>;  // From Sobel pass: R = direction, G = is_edge

@group(0) @binding(1)
var color_texture: texture_2d<f32>;      // Original color for luminance calc

@group(0) @binding(2)
var<uniform> uniforms: Uniforms;

@group(0) @binding(3)
var<storage, read_write> output: array<u32>;

// Constants for character indices
const CHAR_FILL_MAX: u32 = 9u;  // Fill chars are indices 0-9
const CHAR_EDGE_VERTICAL: u32 = 10u;    // |
const CHAR_EDGE_HORIZONTAL: u32 = 11u;  // -
const CHAR_EDGE_DIAG_FWD: u32 = 12u;    // /
const CHAR_EDGE_DIAG_BACK: u32 = 13u;   // \

// One thread per ASCII character cell
@compute @workgroup_size(1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_col = gid.x;
    let tile_row = gid.y;

    // Bounds check
    if (tile_col >= uniforms.cols || tile_row >= uniforms.rows) {
        return;
    }

    // Calculate tile bounds in pixels
    let tile_width = f32(uniforms.width) / f32(uniforms.cols);
    let tile_height = f32(uniforms.height) / f32(uniforms.rows);
    let tile_start_x = f32(tile_col) * tile_width;
    let tile_start_y = f32(tile_row) * tile_height;

    // Vote counts for each direction: [none, |, -, /, \]
    var direction_votes = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
    var luminance_sum: f32 = 0.0;
    var sample_count: u32 = 0u;

    // Sample pixels in the tile (use 4x4 grid for efficiency)
    let samples_x = 4u;
    let samples_y = 4u;
    let step_x = tile_width / f32(samples_x);
    let step_y = tile_height / f32(samples_y);

    for (var sy = 0u; sy < samples_y; sy++) {
        for (var sx = 0u; sx < samples_x; sx++) {
            let px = i32(tile_start_x + (f32(sx) + 0.5) * step_x);
            let py = i32(tile_start_y + (f32(sy) + 0.5) * step_y);

            // Bounds check
            if (px >= 0 && px < i32(uniforms.width) && py >= 0 && py < i32(uniforms.height)) {
                let pixel_coords = vec2<i32>(px, py);

                // Read direction info
                let dir_sample = textureLoad(direction_texture, pixel_coords, 0);
                let direction = u32(dir_sample.r);
                let is_edge = dir_sample.g > 0.5;

                // Vote for edge direction
                if (is_edge && direction >= 1u && direction <= 4u) {
                    direction_votes[direction] += 1u;
                }

                // Read color and calculate luminance
                let color = textureLoad(color_texture, pixel_coords, 0);
                let luminance = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
                luminance_sum += luminance;
                sample_count += 1u;
            }
        }
    }

    // Find dominant direction and total edge votes
    var total_edge_votes = 0u;
    var max_votes = 0u;
    var dominant_dir = 0u;

    for (var i = 1u; i < 5u; i++) {
        total_edge_votes += direction_votes[i];
        if (direction_votes[i] > max_votes) {
            max_votes = direction_votes[i];
            dominant_dir = i;
        }
    }

    var char_index: u32;

    // Check if we have enough edge votes (scaled for sample count)
    let edge_threshold = max(1u, uniforms.tile_edge_min / 4u);  // Adjusted for 4x4 samples
    if (total_edge_votes >= edge_threshold && max_votes > 0u) {
        // Use edge character based on dominant direction
        char_index = CHAR_EDGE_VERTICAL + dominant_dir - 1u;
    } else {
        // Use fill character based on average luminance
        var avg_luminance = 0.0;
        if (sample_count > 0u) {
            avg_luminance = luminance_sum / f32(sample_count);
        }

        // Apply exposure and gamma
        avg_luminance = saturate(pow(avg_luminance * uniforms.exposure, uniforms.gamma));

        // Map to character index (0-9)
        char_index = u32(clamp(avg_luminance * 10.0, 0.0, 9.0));
    }

    // Write to output buffer
    let output_index = tile_row * uniforms.cols + tile_col;
    output[output_index] = char_index;
}
