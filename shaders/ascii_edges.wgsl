// Final ASCII Pass with Tile Voting
// Samples direction texture across each character cell
// Votes on dominant edge direction
// Outputs character index + packed RGB color

struct Uniforms {
    tex_width: u32,          // Texture width in pixels
    tex_height: u32,         // Texture height in pixels
    cols: u32,               // Number of ASCII columns
    rows: u32,               // Number of ASCII rows
    edge_threshold: u32,     // Min edge pixels to use edge char (e.g., 2 out of 16 samples)
    exposure: f32,           // Luminance boost (e.g., 1.5)
    gamma: f32,              // Contrast curve (e.g., 0.8)
    _padding: f32,
};

@group(0) @binding(0)
var direction_texture: texture_2d<f32>;  // R=direction, G=edge_flag, B=luminance, A=depth

@group(0) @binding(1)
var<uniform> uniforms: Uniforms;

@group(0) @binding(2)
var<storage, read_write> output: array<u32>;

@group(0) @binding(3)
var color_texture: texture_2d<f32>;  // Original rendered color

// Character indices:
// 0-9: Fill characters (luminance: dark to bright)
// 10: Vertical edge |
// 11: Horizontal edge -
// 12: Diagonal edge /
// 13: Diagonal edge \

const CHAR_EDGE_VERTICAL: u32 = 10u;
const CHAR_EDGE_HORIZONTAL: u32 = 11u;
const CHAR_EDGE_DIAG_FWD: u32 = 12u;
const CHAR_EDGE_DIAG_BACK: u32 = 13u;

// Fill factors for each character (0-1, how much of the cell the character covers)
// Characters: ' ', '.', ';', 'c', 'o', 'P', 'O', '?', '@', '#'
const CHAR_FILL: array<f32, 14> = array<f32, 14>(
    0.01,  // 0: space (nearly invisible, avoid div by 0)
    0.08,  // 1: .
    0.12,  // 2: ;
    0.30,  // 3: c
    0.40,  // 4: o
    0.55,  // 5: P
    0.60,  // 6: O
    0.50,  // 7: ?
    0.75,  // 8: @
    0.85,  // 9: #
    0.45,  // 10: | (vertical edge)
    0.45,  // 11: - (horizontal edge)
    0.40,  // 12: / (diagonal)
    0.40,  // 13: \ (diagonal)
);

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
    let tile_width = f32(uniforms.tex_width) / f32(uniforms.cols);
    let tile_height = f32(uniforms.tex_height) / f32(uniforms.rows);
    let tile_start_x = f32(tile_col) * tile_width;
    let tile_start_y = f32(tile_row) * tile_height;

    // Vote counts for each direction: [vertical, horizontal, diag_fwd, diag_back]
    var direction_votes = array<u32, 4>(0u, 0u, 0u, 0u);
    var total_edge_pixels: u32 = 0u;
    var luminance_sum: f32 = 0.0;
    var color_sum: vec3<f32> = vec3<f32>(0.0);
    var sample_count: u32 = 0u;

    // Sample 4x4 grid within the tile
    let samples_x = 4u;
    let samples_y = 4u;
    let step_x = tile_width / f32(samples_x);
    let step_y = tile_height / f32(samples_y);

    for (var sy = 0u; sy < samples_y; sy++) {
        for (var sx = 0u; sx < samples_x; sx++) {
            let px = i32(tile_start_x + (f32(sx) + 0.5) * step_x);
            let py = i32(tile_start_y + (f32(sy) + 0.5) * step_y);

            // Bounds check
            if (px >= 0 && px < i32(uniforms.tex_width) && py >= 0 && py < i32(uniforms.tex_height)) {
                let pixel_coords = vec2<i32>(px, py);
                let data = textureLoad(direction_texture, pixel_coords, 0);
                let color = textureLoad(color_texture, pixel_coords, 0).rgb;

                let direction = i32(data.r);
                let is_edge = data.g > 0.5;
                let luminance = data.b;

                // Accumulate luminance and color
                luminance_sum += luminance;
                color_sum += color;
                sample_count += 1u;

                // Vote for edge direction
                if (is_edge && direction >= 0 && direction <= 3) {
                    direction_votes[direction] += 1u;
                    total_edge_pixels += 1u;
                }
            }
        }
    }

    // Find dominant direction
    var max_votes: u32 = 0u;
    var dominant_dir: i32 = -1;

    for (var i = 0; i < 4; i++) {
        if (direction_votes[i] > max_votes) {
            max_votes = direction_votes[i];
            dominant_dir = i;
        }
    }

    var char_index: u32;
    var avg_color: vec3<f32> = vec3<f32>(0.5);

    // Calculate average color
    if (sample_count > 0u) {
        avg_color = color_sum / f32(sample_count);
    }

    // Check if we have enough edge votes
    if (total_edge_pixels >= uniforms.edge_threshold && max_votes > 0u) {
        // Use edge character based on dominant direction
        char_index = CHAR_EDGE_VERTICAL + u32(dominant_dir);
    } else {
        // Use fill character based on average luminance
        var avg_luminance: f32 = 0.0;
        if (sample_count > 0u) {
            avg_luminance = luminance_sum / f32(sample_count);
        }

        // Apply exposure and gamma
        avg_luminance = saturate(pow(avg_luminance * uniforms.exposure, uniforms.gamma));

        // Map to character index (0-9)
        char_index = u32(clamp(avg_luminance * 10.0, 0.0, 9.0));
    }

    // Compensate color for character fill factor
    // Darker characters (low fill) need brighter colors to achieve the same perceived brightness
    let fill_factor = CHAR_FILL[char_index];
    // Boost = 1/fill, but clamped to avoid extreme values
    // We use sqrt to soften the compensation (full compensation would be too aggressive)
    let boost = min(1.0 / sqrt(fill_factor), 3.0);
    let compensated = avg_color * boost;

    // Pack output: char_index in lower 8 bits, RGB in upper 24 bits
    // Format: 0xRRGGBBCC where CC=char, BB=blue, GG=green, RR=red
    let r = u32(clamp(compensated.r * 255.0, 0.0, 255.0));
    let g = u32(clamp(compensated.g * 255.0, 0.0, 255.0));
    let b = u32(clamp(compensated.b * 255.0, 0.0, 255.0));
    let packed = (r << 24u) | (g << 16u) | (b << 8u) | char_index;

    // Write to output buffer
    let output_index = tile_row * uniforms.cols + tile_col;
    output[output_index] = packed;
}
