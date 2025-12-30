mod config;
mod gpu;
mod model;
mod terminal;

use anyhow::Result;
use crossterm::cursor::Hide;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io::stdout;
use std::path::Path;
use std::time::{Duration, Instant};

use arboard::Clipboard;
use config::{run_config_ui, ConfigState};
use gpu::{AsciiPipeline, HeadlessGpu};
use model::load_model;
use terminal::{RenderMode, TerminalRenderer};

const MODELS_DIR: &str = "assets/models";
const SKYBOXES_DIR: &str = "assets/skyboxes";

/// Application mode
enum AppMode {
    Rendering,
    Config,
}

/// Manual control state for spacecraft-like rotation
struct ManualControls {
    /// Whether manual control is active (vs auto rotation)
    active: bool,
    /// Current rotation angles (pitch, yaw) in radians
    rotation: (f32, f32),
    /// Angular velocity (pitch/sec, yaw/sec)
    velocity: (f32, f32),
    /// Camera zoom distance
    zoom: f32,
    /// Default zoom distance
    default_zoom: f32,
}

impl ManualControls {
    fn new() -> Self {
        Self {
            active: false,
            rotation: (0.0, 0.0),
            velocity: (0.0, 0.0),
            zoom: 4.0,
            default_zoom: 4.0,
        }
    }

    /// Reset to default state
    fn reset(&mut self) {
        self.active = false;
        self.rotation = (0.0, 0.0);
        self.velocity = (0.0, 0.0);
        self.zoom = self.default_zoom;
    }

    /// Apply thrust in a direction (like a thruster)
    /// Each call adds velocity - hold key to accelerate more
    fn thrust(&mut self, pitch: f32, yaw: f32) {
        const THRUST_IMPULSE: f32 = 0.15; // velocity added per keypress/repeat
        self.velocity.0 += pitch * THRUST_IMPULSE;
        self.velocity.1 += yaw * THRUST_IMPULSE;

        // Clamp max velocity
        const MAX_VELOCITY: f32 = 3.0;
        self.velocity.0 = self.velocity.0.clamp(-MAX_VELOCITY, MAX_VELOCITY);
        self.velocity.1 = self.velocity.1.clamp(-MAX_VELOCITY, MAX_VELOCITY);

        self.active = true;
    }

    /// Adjust zoom
    fn zoom_in(&mut self) {
        self.zoom = (self.zoom - 0.15).max(1.5);
        self.active = true;
    }

    fn zoom_out(&mut self) {
        self.zoom = (self.zoom + 0.15).min(15.0);
        self.active = true;
    }

    /// Update physics (apply velocity to rotation, apply damping)
    fn update(&mut self, dt: f32) {
        if !self.active {
            return;
        }

        // Apply velocity to rotation
        self.rotation.0 += self.velocity.0 * dt;
        self.rotation.1 += self.velocity.1 * dt;

        // Apply damping (smooth deceleration)
        const DAMPING: f32 = 0.97;
        self.velocity.0 *= DAMPING;
        self.velocity.1 *= DAMPING;

        // Stop very small velocities to avoid drift
        const MIN_VELOCITY: f32 = 0.01;
        if self.velocity.0.abs() < MIN_VELOCITY {
            self.velocity.0 = 0.0;
        }
        if self.velocity.1.abs() < MIN_VELOCITY {
            self.velocity.1 = 0.0;
        }
    }
}

/// Calculate pipeline dimensions and pixel size based on render mode
/// Returns (data_cols, data_rows, pixels_per_cell_x, pixels_per_cell_y)
fn get_pipeline_dims(term_cols: u16, term_rows: u16, mode: RenderMode) -> (u32, u32, u32, u32) {
    match mode {
        RenderMode::PlainAscii | RenderMode::ColoredAscii => {
            // Each terminal cell = one data cell, rendered at 8x16 (char aspect ratio)
            (term_cols as u32, term_rows as u32, 8, 16)
        }
        RenderMode::HalfBlock => {
            // Each terminal row displays 2 data rows
            // Each "pixel" is square (8x8) since ▀ splits the cell in half vertically
            (term_cols as u32, term_rows as u32 * 2, 8, 8)
        }
    }
}

/// Load a model and update GPU geometry
fn load_model_into_gpu(gpu: &mut HeadlessGpu, path: &Path) -> Result<()> {
    let model_data = load_model(path)?;
    gpu.set_geometry(&model_data.vertices, &model_data.indices);
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
    eprintln!("Starting terminal demo...");

    // Initialize terminal renderer
    let mut term = TerminalRenderer::new()?;
    eprintln!("Terminal initialized");
    let (term_cols, term_rows) = term.content_size();

    // Initialize config state
    let mut config = ConfigState::new();
    config.refresh_models(Path::new(MODELS_DIR));
    config.refresh_skyboxes(Path::new(SKYBOXES_DIR));

    // Current render mode
    let mut render_mode = RenderMode::PlainAscii;
    let mut prev_mode = render_mode;

    // GPU info display toggle
    let mut show_gpu_info = true;

    // App mode
    let mut app_mode = AppMode::Rendering;

    // Manual control state
    let mut controls = ManualControls::new();

    // Calculate initial pipeline dimensions based on mode
    let (pipe_cols, pipe_rows, px_x, px_y) = get_pipeline_dims(term_cols, term_rows, render_mode);
    let render_width = pipe_cols * px_x;
    let render_height = pipe_rows * px_y;

    // Initialize headless GPU
    eprintln!("Creating HeadlessGpu...");
    let mut gpu = pollster::block_on(HeadlessGpu::new(render_width, render_height))?;
    eprintln!("HeadlessGpu created");

    // Load initial model if available
    if let Some(ref model_path) = config.model_path {
        eprintln!("Loading model: {:?}", model_path);
        if let Err(e) = load_model_into_gpu(&mut gpu, model_path) {
            eprintln!("Failed to load model: {}", e);
        }
    }

    // Initialize edge-aware ASCII pipeline
    eprintln!("Creating AsciiPipeline...");
    let mut pipeline = AsciiPipeline::new(
        &gpu.device,
        pipe_cols,
        pipe_rows,
        render_width,
        render_height,
    )?;
    eprintln!("AsciiPipeline created");

    let start_time = Instant::now();
    let mut last_frame = Instant::now();
    let mut frame_count = 0u32;
    let mut fps = 0.0f32;
    let mut fps_update_time = Instant::now();

    // Track current model path for change detection
    let mut current_model_path = config.model_path.clone();

    loop {
        match app_mode {
            AppMode::Rendering => {
                // Handle input - process all pending events for responsive controls
                let mut should_quit = false;
                let mut copy_to_clipboard = false;
                while event::poll(Duration::from_millis(0))? {
                    if let Event::Key(key_event) = event::read()? {
                        // Handle Press and Repeat for smooth controls
                        if key_event.kind == KeyEventKind::Press
                            || key_event.kind == KeyEventKind::Repeat
                        {
                            match key_event.code {
                                // WASD for rotation (thruster-style)
                                KeyCode::Char('w') | KeyCode::Char('W') => controls.thrust(-1.0, 0.0),
                                KeyCode::Char('s') | KeyCode::Char('S') => controls.thrust(1.0, 0.0),
                                KeyCode::Char('a') | KeyCode::Char('A') => controls.thrust(0.0, -1.0),
                                KeyCode::Char('d') | KeyCode::Char('D') => controls.thrust(0.0, 1.0),
                                // Q/E for zoom
                                KeyCode::Char('e') | KeyCode::Char('E') => controls.zoom_in(),
                                KeyCode::Char('q') | KeyCode::Char('Q') => controls.zoom_out(),
                                _ => {}
                            }
                        }

                        // Only handle Press for non-repeating actions
                        if key_event.kind == KeyEventKind::Press {
                            match key_event.code {
                                KeyCode::Esc => should_quit = true,
                                KeyCode::Char('1') => render_mode = RenderMode::PlainAscii,
                                KeyCode::Char('2') => render_mode = RenderMode::ColoredAscii,
                                KeyCode::Char('3') => render_mode = RenderMode::HalfBlock,
                                KeyCode::Char('g') | KeyCode::Char('G') => show_gpu_info = !show_gpu_info,
                                // R to reset view
                                KeyCode::Char('r') | KeyCode::Char('R') => controls.reset(),
                                // F to copy frame to clipboard
                                KeyCode::Char('f') | KeyCode::Char('F') => copy_to_clipboard = true,
                                KeyCode::Char('c') | KeyCode::Char('C') => {
                                    // Refresh model and skybox lists before opening config
                                    config.refresh_models(Path::new(MODELS_DIR));
                                    config.refresh_skyboxes(Path::new(SKYBOXES_DIR));
                                    app_mode = AppMode::Config;
                                }
                                KeyCode::Tab => render_mode = render_mode.next(),
                                _ => {}
                            }
                        }
                    }
                }
                if should_quit {
                    break;
                }

                // Update manual controls physics
                let frame_dt = last_frame.elapsed().as_secs_f32();
                controls.update(frame_dt);

                // Check for terminal resize or mode change
                let mode_changed = render_mode != prev_mode;
                let resized = term.check_resize()?;

                if resized || mode_changed {
                    let (new_term_cols, new_term_rows) = term.content_size();
                    let (new_pipe_cols, new_pipe_rows, new_px_x, new_px_y) =
                        get_pipeline_dims(new_term_cols, new_term_rows, render_mode);
                    let new_width = new_pipe_cols * new_px_x;
                    let new_height = new_pipe_rows * new_px_y;
                    gpu.resize(new_width, new_height);
                    pipeline.resize(
                        &gpu.device,
                        new_pipe_cols,
                        new_pipe_rows,
                        new_width,
                        new_height,
                    );
                    prev_mode = render_mode;
                }

                let elapsed = start_time.elapsed().as_secs_f32();

                // Time GPU operations
                let gpu_start = Instant::now();

                // Render 3D scene - use manual controls if active, otherwise auto rotation
                let render_cmd = if controls.active {
                    gpu.render_manual(
                        controls.rotation.0,
                        controls.rotation.1,
                        controls.zoom,
                        config.lighting_mode,
                    )
                } else {
                    gpu.render_with_rotation(
                        elapsed,
                        config.rotation_mode,
                        config.rotation_speed,
                        config.lighting_mode,
                    )
                };
                gpu.queue.submit(std::iter::once(render_cmd));

                // Update pipeline bind groups with color and depth textures
                pipeline.update_bind_groups(
                    &gpu.device,
                    &gpu.queue,
                    gpu.render_texture_view(),
                    gpu.depth_texture_view(),
                );

                // Run edge-aware compute pipeline
                let mut encoder = gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Pipeline Encoder"),
                    });

                pipeline.dispatch(&mut encoder);
                pipeline.copy_to_staging(&mut encoder);

                gpu.queue.submit(std::iter::once(encoder.finish()));

                // Read results (includes GPU sync)
                let ascii_data = pollster::block_on(pipeline.read_results(&gpu.device))?;

                let gpu_time_ms = gpu_start.elapsed().as_secs_f32() * 1000.0;

                // Calculate mask region if GPU info is shown
                let mask = if show_gpu_info {
                    Some(term.gpu_info_mask(gpu.gpu_name()))
                } else {
                    None
                };

                // Render to terminal using current mode
                term.render(
                    &ascii_data,
                    pipeline.cols(),
                    pipeline.rows(),
                    render_mode,
                    mask,
                )?;

                // Copy frame to clipboard if requested
                if copy_to_clipboard {
                    let ansi_string = term.frame_to_ansi_string(
                        &ascii_data,
                        pipeline.cols(),
                        pipeline.rows(),
                        render_mode,
                    );
                    if let Ok(mut clipboard) = Clipboard::new() {
                        let _ = clipboard.set_text(ansi_string);
                    }
                }

                // Update FPS
                frame_count += 1;
                if fps_update_time.elapsed() >= Duration::from_secs(1) {
                    fps = frame_count as f32 / fps_update_time.elapsed().as_secs_f32();
                    frame_count = 0;
                    fps_update_time = Instant::now();
                }

                // Show mode name with manual indicator
                let mode_display = if controls.active {
                    format!("{} [Manual]", render_mode.name())
                } else {
                    render_mode.name().to_string()
                };
                term.render_status(fps, &mode_display)?;
                if show_gpu_info {
                    term.render_gpu_info(
                        gpu.gpu_name(),
                        gpu_time_ms,
                        gpu.render_size(),
                        (pipeline.cols(), pipeline.rows()),
                    )?;
                }

                // Frame timing (target ~30 fps to reduce CPU usage)
                let frame_time = last_frame.elapsed();
                let target_frame_time = Duration::from_millis(33);
                if frame_time < target_frame_time {
                    std::thread::sleep(target_frame_time - frame_time);
                }
                last_frame = Instant::now();
            }

            AppMode::Config => {
                // Create a temporary ratatui terminal for the config UI
                // We need to temporarily take over stdout
                let backend = CrosstermBackend::new(stdout());
                let mut ratatui_terminal = Terminal::new(backend)?;
                ratatui_terminal.clear()?;

                // Run config UI (blocks until user applies or cancels)
                let result = run_config_ui(&mut ratatui_terminal, config.clone())?;

                // Restore terminal state
                drop(ratatui_terminal);

                // Re-hide cursor (config UI may have shown it)
                execute!(stdout(), Hide)?;

                // Clear and redraw
                term.check_resize()?;

                if let Some(new_config) = result {
                    // Check if model changed
                    if new_config.model_path != current_model_path {
                        if let Some(ref model_path) = new_config.model_path {
                            if let Err(e) = load_model_into_gpu(&mut gpu, model_path) {
                                eprintln!("Failed to load model: {}", e);
                            } else {
                                current_model_path = new_config.model_path.clone();
                            }
                        }
                    }

                    // Check if skybox changed
                    if new_config.skybox_path != config.skybox_path {
                        match &new_config.skybox_path {
                            Some(skybox_path) => {
                                if let Err(e) = gpu.set_skybox(skybox_path) {
                                    eprintln!("Failed to load skybox: {}", e);
                                }
                            }
                            None => {
                                gpu.clear_skybox();
                            }
                        }
                    }

                    config = new_config;
                }

                // Return to rendering mode
                app_mode = AppMode::Rendering;
            }
        }
    }

    Ok(())
}
