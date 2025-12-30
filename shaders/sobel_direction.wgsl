// Pass 4: Sobel gradient direction calculation
// Input: Edge texture from DoG pass
// Output: Direction texture (angle quantized to 4 directions + edge flag)

struct Uniforms {
    width: u32,
    height: u32,
    _padding: vec2<u32>,
};

@group(0) @binding(0)
var edge_texture: texture_2d<f32>;

@group(0) @binding(1)
var luminance_texture: texture_2d<f32>;  // Original luminance for gradient calc

@group(0) @binding(2)
var output_texture: texture_storage_2d<rg32float, write>;

@group(0) @binding(3)
var<uniform> uniforms: Uniforms;

const PI: f32 = 3.14159265359;

// Sobel kernels
// Gx: horizontal gradient (detects vertical edges)
// [-1  0  1]
// [-2  0  2]
// [-1  0  1]
//
// Gy: vertical gradient (detects horizontal edges)
// [-1 -2 -1]
// [ 0  0  0]
// [ 1  2  1]

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coords = vec2<i32>(gid.xy);

    // Bounds check
    if (coords.x >= i32(uniforms.width) || coords.y >= i32(uniforms.height)) {
        return;
    }

    // Check if this pixel is an edge
    let is_edge = textureLoad(edge_texture, coords, 0).r;

    if (is_edge < 0.5) {
        // Not an edge, output zero direction
        textureStore(output_texture, coords, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Sample 3x3 neighborhood from luminance texture
    var samples: array<f32, 9>;
    var idx = 0;
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let sample_coords = clamp(
                coords + vec2<i32>(dx, dy),
                vec2<i32>(0, 0),
                vec2<i32>(i32(uniforms.width) - 1, i32(uniforms.height) - 1)
            );
            samples[idx] = textureLoad(luminance_texture, sample_coords, 0).r;
            idx++;
        }
    }

    // samples layout:
    // [0] [1] [2]
    // [3] [4] [5]
    // [6] [7] [8]

    // Sobel Gx (horizontal gradient)
    let gx = -samples[0] + samples[2]
           - 2.0 * samples[3] + 2.0 * samples[5]
           - samples[6] + samples[8];

    // Sobel Gy (vertical gradient)
    let gy = -samples[0] - 2.0 * samples[1] - samples[2]
           + samples[6] + 2.0 * samples[7] + samples[8];

    // Calculate gradient angle
    // atan2 returns angle in [-PI, PI]
    let angle = atan2(gy, gx);

    // Quantize angle to 4 directions:
    // 0: No edge (not used here since we already checked)
    // 1: Vertical edge (|) - gradient is horizontal, edge is vertical
    // 2: Horizontal edge (-) - gradient is vertical, edge is horizontal
    // 3: Diagonal (/) - gradient at -45 deg, edge at 45 deg
    // 4: Diagonal (\) - gradient at 45 deg, edge at -45 deg
    //
    // The edge direction is perpendicular to the gradient direction
    // Gradient angle -> Edge character:
    // ~0 or ~PI (horizontal gradient) -> | (vertical edge)
    // ~PI/2 or ~-PI/2 (vertical gradient) -> - (horizontal edge)
    // ~PI/4 (45 deg gradient) -> \ (135 deg edge)
    // ~-PI/4 or ~3PI/4 (-45 deg gradient) -> / (45 deg edge)

    // Normalize angle to [0, PI) since we only care about direction, not sign
    var norm_angle = angle;
    if (norm_angle < 0.0) {
        norm_angle += PI;
    }

    // Quantize to 4 directions (each spans PI/4 = 45 degrees)
    // Direction centers: 0, PI/4, PI/2, 3PI/4
    var direction: f32;

    // Angle ranges:
    // [0, PI/8) or [7PI/8, PI) -> horizontal gradient -> vertical edge (|) -> 1
    // [PI/8, 3PI/8) -> diagonal gradient -> \ edge -> 4
    // [3PI/8, 5PI/8) -> vertical gradient -> horizontal edge (-) -> 2
    // [5PI/8, 7PI/8) -> diagonal gradient -> / edge -> 3

    let eighth_pi = PI / 8.0;

    if (norm_angle < eighth_pi || norm_angle >= 7.0 * eighth_pi) {
        direction = 1.0;  // Vertical edge |
    } else if (norm_angle < 3.0 * eighth_pi) {
        direction = 4.0;  // Diagonal edge \
    } else if (norm_angle < 5.0 * eighth_pi) {
        direction = 2.0;  // Horizontal edge -
    } else {
        direction = 3.0;  // Diagonal edge /
    }

    // Output: R = direction (1-4), G = is_edge flag (1.0)
    textureStore(output_texture, coords, vec4<f32>(direction, 1.0, 0.0, 1.0));
}
