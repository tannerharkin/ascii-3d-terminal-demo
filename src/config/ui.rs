use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph},
    Frame, Terminal,
};
use std::io::Stdout;
use std::time::Duration;

use super::{get_skybox_display_name, ConfigState};
use crate::gpu::{LightingMode, RotationMode};
use crate::model::get_model_display_name;

/// Which section of the UI is currently focused
#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    Models,
    Rotation,
    Lighting,
    Skybox,
    Speed,
    Buttons,
}

impl Focus {
    fn next(self) -> Self {
        match self {
            Focus::Models => Focus::Rotation,
            Focus::Rotation => Focus::Lighting,
            Focus::Lighting => Focus::Skybox,
            Focus::Skybox => Focus::Speed,
            Focus::Speed => Focus::Buttons,
            Focus::Buttons => Focus::Models,
        }
    }

    fn prev(self) -> Self {
        match self {
            Focus::Models => Focus::Buttons,
            Focus::Rotation => Focus::Models,
            Focus::Lighting => Focus::Rotation,
            Focus::Skybox => Focus::Lighting,
            Focus::Speed => Focus::Skybox,
            Focus::Buttons => Focus::Speed,
        }
    }
}

/// UI state for the config screen
struct ConfigUI {
    config: ConfigState,
    focus: Focus,
    model_list_state: ListState,
    rotation_index: usize,
    lighting_index: usize,
    skybox_index: usize,
    button_index: usize, // 0 = Apply, 1 = Cancel
}

impl ConfigUI {
    fn new(config: ConfigState) -> Self {
        let rotation_index = RotationMode::all()
            .iter()
            .position(|&m| m == config.rotation_mode)
            .unwrap_or(0);

        let lighting_index = LightingMode::all()
            .iter()
            .position(|&m| m == config.lighting_mode)
            .unwrap_or(0);

        let skybox_index = config.selected_skybox_index();

        let mut model_list_state = ListState::default();
        model_list_state.select(config.selected_model_index());

        Self {
            config,
            focus: Focus::Models,
            model_list_state,
            rotation_index,
            lighting_index,
            skybox_index,
            button_index: 0,
        }
    }

    fn handle_key(&mut self, key: KeyCode) -> Option<bool> {
        match key {
            KeyCode::Esc => return Some(false), // Cancel
            KeyCode::Tab => self.focus = self.focus.next(),
            KeyCode::BackTab => self.focus = self.focus.prev(),
            KeyCode::Enter => {
                if self.focus == Focus::Buttons {
                    return Some(self.button_index == 0); // Apply or Cancel
                }
            }
            KeyCode::Up => self.move_up(),
            KeyCode::Down => self.move_down(),
            KeyCode::Left => self.move_left(),
            KeyCode::Right => self.move_right(),
            _ => {}
        }
        None
    }

    fn move_up(&mut self) {
        match self.focus {
            Focus::Models => {
                if let Some(i) = self.model_list_state.selected() {
                    if i > 0 {
                        self.model_list_state.select(Some(i - 1));
                        self.config.select_model(i - 1);
                    }
                }
            }
            Focus::Rotation => {
                if self.rotation_index > 0 {
                    self.rotation_index -= 1;
                    self.config.rotation_mode = RotationMode::all()[self.rotation_index];
                }
            }
            Focus::Lighting => {
                if self.lighting_index > 0 {
                    self.lighting_index -= 1;
                    self.config.lighting_mode = LightingMode::all()[self.lighting_index];
                }
            }
            Focus::Skybox => {
                let total = self.config.available_skyboxes.len() + 1; // +1 for "None"
                if self.skybox_index > 0 {
                    self.skybox_index -= 1;
                    self.config.select_skybox(self.skybox_index);
                } else {
                    // Wrap around
                    self.skybox_index = total - 1;
                    self.config.select_skybox(self.skybox_index);
                }
            }
            _ => {}
        }
    }

    fn move_down(&mut self) {
        match self.focus {
            Focus::Models => {
                if let Some(i) = self.model_list_state.selected() {
                    if i + 1 < self.config.available_models.len() {
                        self.model_list_state.select(Some(i + 1));
                        self.config.select_model(i + 1);
                    }
                } else if !self.config.available_models.is_empty() {
                    self.model_list_state.select(Some(0));
                    self.config.select_model(0);
                }
            }
            Focus::Rotation => {
                if self.rotation_index + 1 < RotationMode::all().len() {
                    self.rotation_index += 1;
                    self.config.rotation_mode = RotationMode::all()[self.rotation_index];
                }
            }
            Focus::Lighting => {
                if self.lighting_index + 1 < LightingMode::all().len() {
                    self.lighting_index += 1;
                    self.config.lighting_mode = LightingMode::all()[self.lighting_index];
                }
            }
            Focus::Skybox => {
                let total = self.config.available_skyboxes.len() + 1; // +1 for "None"
                if self.skybox_index + 1 < total {
                    self.skybox_index += 1;
                    self.config.select_skybox(self.skybox_index);
                } else {
                    // Wrap around
                    self.skybox_index = 0;
                    self.config.select_skybox(self.skybox_index);
                }
            }
            _ => {}
        }
    }

    fn move_left(&mut self) {
        match self.focus {
            Focus::Speed => self.config.adjust_speed(-0.1),
            Focus::Buttons => self.button_index = 0,
            Focus::Rotation => self.move_up(),
            Focus::Lighting => self.move_up(),
            Focus::Skybox => self.move_up(),
            _ => {}
        }
    }

    fn move_right(&mut self) {
        match self.focus {
            Focus::Speed => self.config.adjust_speed(0.1),
            Focus::Buttons => self.button_index = 1,
            Focus::Rotation => self.move_down(),
            Focus::Lighting => self.move_down(),
            Focus::Skybox => self.move_down(),
            _ => {}
        }
    }
}

/// Run the config UI, blocking until user applies or cancels
/// Returns Some(config) if applied, None if cancelled
pub fn run_config_ui(
    terminal: &mut Terminal<ratatui::backend::CrosstermBackend<Stdout>>,
    config: ConfigState,
) -> Result<Option<ConfigState>> {
    let mut ui = ConfigUI::new(config);

    loop {
        terminal.draw(|f| draw_config_ui(f, &mut ui))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    if let Some(apply) = ui.handle_key(key.code) {
                        if apply {
                            return Ok(Some(ui.config));
                        } else {
                            return Ok(None);
                        }
                    }
                }
            }
        }
    }
}

fn draw_config_ui(f: &mut Frame, ui: &mut ConfigUI) {
    let area = f.area();

    // Clear the screen
    f.render_widget(Clear, area);

    // Calculate centered popup area (taller to accommodate new sections)
    let popup_width = 70.min(area.width.saturating_sub(4));
    let popup_height = 28.min(area.height.saturating_sub(2));
    let popup_x = (area.width.saturating_sub(popup_width)) / 2;
    let popup_y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);

    // Main border
    let block = Block::default()
        .title(" Configuration ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    f.render_widget(block, popup_area);

    // Inner area
    let inner = Rect::new(
        popup_area.x + 2,
        popup_area.y + 1,
        popup_area.width.saturating_sub(4),
        popup_area.height.saturating_sub(2),
    );

    // Layout: Models list, Rotation, Lighting, Skybox, Speed, Buttons
    let chunks = Layout::vertical([
        Constraint::Length(1),  // Model label
        Constraint::Length(5),  // Model list
        Constraint::Length(1),  // Rotation label
        Constraint::Length(2),  // Rotation options
        Constraint::Length(1),  // Lighting label
        Constraint::Length(2),  // Lighting options
        Constraint::Length(1),  // Skybox label
        Constraint::Length(1),  // Skybox selector
        Constraint::Length(1),  // Speed label
        Constraint::Length(1),  // Speed slider
        Constraint::Min(1),     // Spacer
        Constraint::Length(1),  // Buttons
    ])
    .split(inner);

    // Model section
    let model_style = if ui.focus == Focus::Models {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::White)
    };
    f.render_widget(
        Paragraph::new("Model:").style(model_style),
        chunks[0],
    );

    let model_items: Vec<ListItem> = ui
        .config
        .available_models
        .iter()
        .map(|p| {
            let name = get_model_display_name(p);
            ListItem::new(format!("  {}", name))
        })
        .collect();

    let model_list = List::new(model_items)
        .block(Block::default().borders(Borders::ALL).border_style(
            if ui.focus == Focus::Models {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::DarkGray)
            },
        ))
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));

    f.render_stateful_widget(model_list, chunks[1], &mut ui.model_list_state);

    // Rotation section
    let rotation_style = if ui.focus == Focus::Rotation {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::White)
    };
    f.render_widget(
        Paragraph::new("Rotation Mode: (arrows to select)").style(rotation_style),
        chunks[2],
    );

    let rotation_modes: Vec<Span> = RotationMode::all()
        .iter()
        .enumerate()
        .map(|(i, mode)| {
            let selected = i == ui.rotation_index;
            let prefix = if selected { ">" } else { " " };
            let style = if selected {
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            Span::styled(format!("{}{:<8}", prefix, mode.name()), style)
        })
        .collect();

    let row1: Vec<Span> = rotation_modes.iter().take(3).cloned().collect();
    let row2: Vec<Span> = rotation_modes.iter().skip(3).cloned().collect();

    let rotation_text = vec![Line::from(row1), Line::from(row2)];
    f.render_widget(Paragraph::new(rotation_text), chunks[3]);

    // Lighting section
    let lighting_style = if ui.focus == Focus::Lighting {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::White)
    };
    f.render_widget(
        Paragraph::new("Lighting Mode: (arrows to select)").style(lighting_style),
        chunks[4],
    );

    let lighting_modes: Vec<Span> = LightingMode::all()
        .iter()
        .enumerate()
        .map(|(i, mode)| {
            let selected = i == ui.lighting_index;
            let prefix = if selected { ">" } else { " " };
            let style = if selected {
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            Span::styled(format!("{}{:<10}", prefix, mode.name()), style)
        })
        .collect();

    let lrow1: Vec<Span> = lighting_modes.iter().take(3).cloned().collect();
    let lrow2: Vec<Span> = lighting_modes.iter().skip(3).cloned().collect();

    let lighting_text = vec![Line::from(lrow1), Line::from(lrow2)];
    f.render_widget(Paragraph::new(lighting_text), chunks[5]);

    // Skybox section
    let skybox_style = if ui.focus == Focus::Skybox {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::White)
    };
    f.render_widget(
        Paragraph::new("Skybox: (arrows to cycle)").style(skybox_style),
        chunks[6],
    );

    // Skybox selector display
    let skybox_name = if ui.skybox_index == 0 {
        "None (solid color)".to_string()
    } else if ui.skybox_index <= ui.config.available_skyboxes.len() {
        get_skybox_display_name(&ui.config.available_skyboxes[ui.skybox_index - 1])
    } else {
        "None".to_string()
    };

    let skybox_display_style = if ui.focus == Focus::Skybox {
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Gray)
    };

    let total_skyboxes = ui.config.available_skyboxes.len() + 1;
    let skybox_text = format!(
        "  < {} > ({}/{})",
        skybox_name,
        ui.skybox_index + 1,
        total_skyboxes
    );
    f.render_widget(
        Paragraph::new(skybox_text).style(skybox_display_style),
        chunks[7],
    );

    // Speed section
    let speed_style = if ui.focus == Focus::Speed {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::White)
    };
    f.render_widget(
        Paragraph::new(format!("Speed: {:.1}x (arrows to adjust)", ui.config.rotation_speed))
            .style(speed_style),
        chunks[8],
    );

    // Speed slider
    let slider_width = chunks[9].width.saturating_sub(2) as usize;
    let speed_normalized = ((ui.config.rotation_speed - 0.1) / 2.9).clamp(0.0, 1.0);
    let filled = (speed_normalized * slider_width as f32) as usize;
    let slider = format!(
        "[{}{}]",
        "=".repeat(filled),
        " ".repeat(slider_width.saturating_sub(filled))
    );
    let slider_style = if ui.focus == Focus::Speed {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    f.render_widget(Paragraph::new(slider).style(slider_style), chunks[9]);

    // Buttons
    let apply_style = if ui.focus == Focus::Buttons && ui.button_index == 0 {
        Style::default().fg(Color::Black).bg(Color::Green)
    } else {
        Style::default().fg(Color::Green)
    };
    let cancel_style = if ui.focus == Focus::Buttons && ui.button_index == 1 {
        Style::default().fg(Color::Black).bg(Color::Red)
    } else {
        Style::default().fg(Color::Red)
    };

    let buttons = Line::from(vec![
        Span::raw("        "),
        Span::styled(" Apply ", apply_style),
        Span::raw("    "),
        Span::styled(" Cancel ", cancel_style),
    ]);
    f.render_widget(Paragraph::new(buttons), chunks[11]);
}
