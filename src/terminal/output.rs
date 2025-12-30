use anyhow::Result;
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    execute, queue,
    style::{Color, Print, ResetColor, SetBackgroundColor, SetForegroundColor},
    terminal::{
        disable_raw_mode, enable_raw_mode, size as terminal_size, Clear, ClearType,
        EnterAlternateScreen, LeaveAlternateScreen,
    },
};
use std::io::{stdout, Stdout, Write};

// Fill characters matching AcerolaFX (dark to bright)
const ASCII_RAMP: &[char] = &[' ', '.', ';', 'c', 'o', 'P', 'O', '?', '@', '#'];

// Edge characters for direction-based edge rendering
// Index 10 = vertical (|), 11 = horizontal (-), 12 = back (\), 13 = forward (/)
const EDGE_CHARS: &[char] = &['|', '-', '\\', '/'];

/// Render mode for terminal output
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderMode {
    PlainAscii,
    ColoredAscii,
    HalfBlock,
}

impl RenderMode {
    pub fn name(&self) -> &'static str {
        match self {
            RenderMode::PlainAscii => "Plain ASCII",
            RenderMode::ColoredAscii => "Colored ASCII",
            RenderMode::HalfBlock => "Half Block",
        }
    }

    pub fn next(&self) -> RenderMode {
        match self {
            RenderMode::PlainAscii => RenderMode::ColoredAscii,
            RenderMode::ColoredAscii => RenderMode::HalfBlock,
            RenderMode::HalfBlock => RenderMode::PlainAscii,
        }
    }
}

pub struct TerminalRenderer {
    stdout: Stdout,
    buffer: String,
    cols: u16,
    rows: u16,
}

/// Unpack color and char index from packed u32
/// Format: 0xRRGGBBCC where CC=char, BB=blue, GG=green, RR=red
fn unpack_data(packed: u32) -> (u8, u8, u8, u8) {
    let char_index = (packed & 0xFF) as u8;
    let b = ((packed >> 8) & 0xFF) as u8;
    let g = ((packed >> 16) & 0xFF) as u8;
    let r = ((packed >> 24) & 0xFF) as u8;
    (r, g, b, char_index)
}

/// Get ASCII character from index
fn get_char(char_index: u8) -> char {
    let idx = char_index as usize;
    if idx < ASCII_RAMP.len() {
        ASCII_RAMP[idx]
    } else if idx < ASCII_RAMP.len() + EDGE_CHARS.len() {
        EDGE_CHARS[idx - ASCII_RAMP.len()]
    } else {
        ' '
    }
}

impl TerminalRenderer {
    pub fn new() -> Result<Self> {
        let mut stdout = stdout();

        enable_raw_mode()?;
        execute!(stdout, EnterAlternateScreen, Hide, Clear(ClearType::All))?;

        let (cols, rows) = terminal_size()?;

        Ok(Self {
            stdout,
            buffer: String::with_capacity((cols as usize + 1) * rows as usize * 20), // Extra for ANSI codes
            cols,
            rows,
        })
    }

    /// Returns usable size for ASCII content (reserves row 0 for status bar)
    pub fn content_size(&self) -> (u16, u16) {
        (self.cols, self.rows.saturating_sub(1))
    }

    pub fn check_resize(&mut self) -> Result<bool> {
        let (new_cols, new_rows) = terminal_size()?;
        if new_cols != self.cols || new_rows != self.rows {
            self.cols = new_cols;
            self.rows = new_rows;
            self.buffer = String::with_capacity((new_cols as usize + 1) * new_rows as usize * 20);
            execute!(self.stdout, Clear(ClearType::All))?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Render using current mode, with optional mask region to skip
    /// mask: Option<(start_col, start_row, width, height)> in terminal coordinates
    pub fn render(&mut self, data: &[u32], cols: u32, rows: u32, mode: RenderMode, mask: Option<(u16, u16, u16, u16)>) -> Result<()> {
        match mode {
            RenderMode::PlainAscii => self.render_plain_ascii(data, cols, rows, mask),
            RenderMode::ColoredAscii => self.render_colored_ascii(data, cols, rows, mask),
            RenderMode::HalfBlock => self.render_half_block(data, cols, rows, mask),
        }
    }

    /// Check if a terminal position is inside the mask region
    fn is_masked(&self, col: u16, row: u16, mask: Option<(u16, u16, u16, u16)>) -> bool {
        if let Some((mask_col, mask_row, mask_w, mask_h)) = mask {
            col >= mask_col && col < mask_col + mask_w && row >= mask_row && row < mask_row + mask_h
        } else {
            false
        }
    }

    /// Plain ASCII mode - no colors
    pub fn render_plain_ascii(&mut self, data: &[u32], cols: u32, rows: u32, mask: Option<(u16, u16, u16, u16)>) -> Result<()> {
        let max_rows = rows.min(self.rows.saturating_sub(1) as u32);
        let max_cols = cols.min(self.cols as u32);

        queue!(self.stdout, MoveTo(0, 1))?;

        for row in 0..max_rows {
            let term_row = row as u16 + 1; // +1 for status bar
            for col in 0..max_cols {
                let term_col = col as u16;
                if self.is_masked(term_col, term_row, mask) {
                    queue!(self.stdout, Print(' '))?;
                } else {
                    let idx = (row * cols + col) as usize;
                    if idx < data.len() {
                        let (_, _, _, char_index) = unpack_data(data[idx]);
                        queue!(self.stdout, Print(get_char(char_index)))?;
                    }
                }
            }
            if row < max_rows - 1 {
                queue!(self.stdout, Print("\r\n"))?;
            }
        }

        self.stdout.flush()?;
        Ok(())
    }

    /// Colored ASCII mode - ANSI 24-bit color
    pub fn render_colored_ascii(&mut self, data: &[u32], cols: u32, rows: u32, mask: Option<(u16, u16, u16, u16)>) -> Result<()> {
        let max_rows = rows.min(self.rows.saturating_sub(1) as u32);
        let max_cols = cols.min(self.cols as u32);

        queue!(self.stdout, MoveTo(0, 1))?;

        let mut last_color: Option<(u8, u8, u8)> = None;

        for row in 0..max_rows {
            let term_row = row as u16 + 1; // +1 for status bar
            for col in 0..max_cols {
                let term_col = col as u16;
                if self.is_masked(term_col, term_row, mask) {
                    queue!(self.stdout, ResetColor, Print(' '))?;
                    last_color = None;
                } else {
                    let idx = (row * cols + col) as usize;
                    if idx < data.len() {
                        let (r, g, b, char_index) = unpack_data(data[idx]);
                        let ch = get_char(char_index);

                        // Only change color if different from last
                        if last_color != Some((r, g, b)) {
                            queue!(self.stdout, SetForegroundColor(Color::Rgb { r, g, b }))?;
                            last_color = Some((r, g, b));
                        }
                        queue!(self.stdout, Print(ch))?;
                    }
                }
            }
            if row < max_rows - 1 {
                queue!(self.stdout, Print("\r\n"))?;
            }
        }

        queue!(self.stdout, ResetColor)?;
        self.stdout.flush()?;
        Ok(())
    }

    /// Half-block mode - uses ▀ with fg/bg colors for 2x vertical resolution
    pub fn render_half_block(&mut self, data: &[u32], cols: u32, rows: u32, mask: Option<(u16, u16, u16, u16)>) -> Result<()> {
        let max_rows = (rows / 2).min(self.rows.saturating_sub(1) as u32);
        let max_cols = cols.min(self.cols as u32);

        queue!(self.stdout, MoveTo(0, 1))?;

        for term_row in 0..max_rows {
            let actual_term_row = term_row as u16 + 1; // +1 for status bar
            let top_row = term_row * 2;
            let bottom_row = top_row + 1;

            for col in 0..max_cols {
                let term_col = col as u16;
                if self.is_masked(term_col, actual_term_row, mask) {
                    queue!(self.stdout, ResetColor, Print(' '))?;
                } else {
                    let top_idx = (top_row * cols + col) as usize;
                    let bottom_idx = (bottom_row * cols + col) as usize;

                    // Get colors for top and bottom pixels
                    let (tr, tg, tb, _) = if top_idx < data.len() {
                        unpack_data(data[top_idx])
                    } else {
                        (0, 0, 0, 0)
                    };

                    let (br, bg, bb, _) = if bottom_idx < data.len() && bottom_row < rows {
                        unpack_data(data[bottom_idx])
                    } else {
                        (0, 0, 0, 0)
                    };

                    // ▀ (upper half block): foreground = top color, background = bottom color
                    queue!(
                        self.stdout,
                        SetForegroundColor(Color::Rgb { r: tr, g: tg, b: tb }),
                        SetBackgroundColor(Color::Rgb { r: br, g: bg, b: bb }),
                        Print('▀')
                    )?;
                }
            }

            queue!(self.stdout, ResetColor)?;
            if term_row < max_rows - 1 {
                queue!(self.stdout, Print("\r\n"))?;
            }
        }

        queue!(self.stdout, ResetColor)?;
        self.stdout.flush()?;
        Ok(())
    }

    /// Generate frame as ANSI-colored string (for clipboard export)
    pub fn frame_to_ansi_string(&self, data: &[u32], cols: u32, rows: u32, mode: RenderMode) -> String {
        match mode {
            RenderMode::PlainAscii => self.frame_to_plain_string(data, cols, rows),
            RenderMode::ColoredAscii => self.frame_to_colored_string(data, cols, rows),
            RenderMode::HalfBlock => self.frame_to_halfblock_string(data, cols, rows),
        }
    }

    fn frame_to_plain_string(&self, data: &[u32], cols: u32, rows: u32) -> String {
        let max_rows = rows.min(self.rows.saturating_sub(1) as u32);
        let max_cols = cols.min(self.cols as u32);
        let mut output = String::new();

        for row in 0..max_rows {
            for col in 0..max_cols {
                let idx = (row * cols + col) as usize;
                if idx < data.len() {
                    let (_, _, _, char_index) = unpack_data(data[idx]);
                    output.push(get_char(char_index));
                }
            }
            output.push('\n');
        }
        output
    }

    fn frame_to_colored_string(&self, data: &[u32], cols: u32, rows: u32) -> String {
        let max_rows = rows.min(self.rows.saturating_sub(1) as u32);
        let max_cols = cols.min(self.cols as u32);
        let mut output = String::new();
        let mut last_color: Option<(u8, u8, u8)> = None;

        for row in 0..max_rows {
            for col in 0..max_cols {
                let idx = (row * cols + col) as usize;
                if idx < data.len() {
                    let (r, g, b, char_index) = unpack_data(data[idx]);
                    let ch = get_char(char_index);

                    if last_color != Some((r, g, b)) {
                        // ANSI 24-bit color: ESC[38;2;R;G;Bm
                        output.push_str(&format!("\x1b[38;2;{};{};{}m", r, g, b));
                        last_color = Some((r, g, b));
                    }
                    output.push(ch);
                }
            }
            output.push_str("\x1b[0m\n"); // Reset at end of line
            last_color = None;
        }
        output
    }

    fn frame_to_halfblock_string(&self, data: &[u32], cols: u32, rows: u32) -> String {
        let max_rows = (rows / 2).min(self.rows.saturating_sub(1) as u32);
        let max_cols = cols.min(self.cols as u32);
        let mut output = String::new();

        for term_row in 0..max_rows {
            let top_row = term_row * 2;
            let bottom_row = top_row + 1;

            for col in 0..max_cols {
                let top_idx = (top_row * cols + col) as usize;
                let bottom_idx = (bottom_row * cols + col) as usize;

                let (tr, tg, tb, _) = if top_idx < data.len() {
                    unpack_data(data[top_idx])
                } else {
                    (0, 0, 0, 0)
                };

                let (br, bg, bb, _) = if bottom_idx < data.len() && bottom_row < rows {
                    unpack_data(data[bottom_idx])
                } else {
                    (0, 0, 0, 0)
                };

                // ANSI: fg=top, bg=bottom, char=▀
                output.push_str(&format!(
                    "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀",
                    tr, tg, tb, br, bg, bb
                ));
            }
            output.push_str("\x1b[0m\n");
        }
        output
    }

    pub fn render_status(&mut self, fps: f32, mode: &str) -> Result<()> {
        let status = format!(" {} | {:.1} FPS | 1-3: modes | c: config | g: gpu | q: quit ", mode, fps);
        execute!(
            self.stdout,
            MoveTo(0, 0),
            ResetColor,
            Print(&status)
        )?;
        Ok(())
    }

    /// Calculate the mask region for GPU info display
    /// Returns (start_col, start_row, width, height) in terminal coordinates
    pub fn gpu_info_mask(&self, gpu_name: &str) -> (u16, u16, u16, u16) {
        const NUM_LINES: u16 = 4;
        // Estimate max line length based on GPU name + fixed formatting
        let max_len = (gpu_name.len() + 12).max(30) as u16; // "      GPU: " prefix + name
        let start_row = self.rows.saturating_sub(NUM_LINES + 1);
        let start_col = self.cols.saturating_sub(max_len + 1);
        (start_col, start_row, max_len + 1, NUM_LINES)
    }

    /// Render GPU/performance info in bottom right corner
    /// Uses fixed-width formatting so labels stay in place while values change
    pub fn render_gpu_info(
        &mut self,
        gpu_name: &str,
        gpu_time_ms: f32,
        render_res: (u32, u32),
        pipeline_res: (u32, u32),
    ) -> Result<()> {
        // Format each line with fixed-width values (right-aligned numbers)
        let lines = [
            format!("      GPU: {}", gpu_name),
            format!("  GPU Time: {:>6.2} ms", gpu_time_ms),
            format!("   Render: {:>4} x {:>4} px", render_res.0, render_res.1),
            format!(" Pipeline: {:>4} x {:>4} cells", pipeline_res.0, pipeline_res.1),
        ];

        // Find the longest line to align everything to the right
        let max_len = lines.iter().map(|l| l.len()).max().unwrap_or(0) as u16;

        // Draw from bottom up, leaving last row clear
        let start_row = self.rows.saturating_sub(lines.len() as u16 + 1);
        let start_col = self.cols.saturating_sub(max_len + 1);

        for (i, line) in lines.iter().enumerate() {
            // Pad line to max_len for consistent clearing
            let padded = format!("{:>width$}", line, width = max_len as usize);
            queue!(
                self.stdout,
                MoveTo(start_col, start_row + i as u16),
                ResetColor,
                Print(&padded)
            )?;
        }

        self.stdout.flush()?;
        Ok(())
    }
}

impl Drop for TerminalRenderer {
    fn drop(&mut self) {
        let _ = execute!(self.stdout, ResetColor, Show, LeaveAlternateScreen);
        let _ = disable_raw_mode();
    }
}
