use crate::gpu::{LightingMode, RotationMode};
use std::path::{Path, PathBuf};

/// Supported skybox image extensions
const SKYBOX_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp"];

/// Configuration state for the demo
#[derive(Clone)]
pub struct ConfigState {
    /// Currently selected model path
    pub model_path: Option<PathBuf>,
    /// List of available model files
    pub available_models: Vec<PathBuf>,
    /// Current rotation mode
    pub rotation_mode: RotationMode,
    /// Rotation speed multiplier (0.1 to 3.0)
    pub rotation_speed: f32,
    /// Current lighting mode
    pub lighting_mode: LightingMode,
    /// Currently selected skybox path (None = solid color background)
    pub skybox_path: Option<PathBuf>,
    /// List of available skybox images
    pub available_skyboxes: Vec<PathBuf>,
}

impl Default for ConfigState {
    fn default() -> Self {
        Self {
            model_path: None,
            available_models: Vec::new(),
            rotation_mode: RotationMode::default(),
            rotation_speed: 1.0,
            lighting_mode: LightingMode::default(),
            skybox_path: None,
            available_skyboxes: Vec::new(),
        }
    }
}

impl ConfigState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Refresh the list of available models from the given directory
    pub fn refresh_models(&mut self, models_dir: &std::path::Path) {
        self.available_models = crate::model::discover_models(models_dir);

        // If no model is selected and models are available, select the first one
        if self.model_path.is_none() && !self.available_models.is_empty() {
            self.model_path = Some(self.available_models[0].clone());
        }

        // If current model is not in list, reset selection
        if let Some(ref path) = self.model_path {
            if !self.available_models.contains(path) {
                self.model_path = self.available_models.first().cloned();
            }
        }
    }

    /// Get the index of the currently selected model
    pub fn selected_model_index(&self) -> Option<usize> {
        self.model_path
            .as_ref()
            .and_then(|p| self.available_models.iter().position(|m| m == p))
    }

    /// Select model by index
    pub fn select_model(&mut self, index: usize) {
        if index < self.available_models.len() {
            self.model_path = Some(self.available_models[index].clone());
        }
    }

    /// Adjust rotation speed (clamped to 0.1 - 3.0)
    pub fn adjust_speed(&mut self, delta: f32) {
        self.rotation_speed = (self.rotation_speed + delta).clamp(0.1, 3.0);
    }

    /// Refresh the list of available skyboxes from the given directory
    pub fn refresh_skyboxes(&mut self, skyboxes_dir: &Path) {
        self.available_skyboxes = discover_skyboxes(skyboxes_dir);

        // If current skybox is not in list, reset selection
        if let Some(ref path) = self.skybox_path {
            if !self.available_skyboxes.contains(path) {
                self.skybox_path = None;
            }
        }
    }

    /// Get the index of the currently selected skybox (0 = None)
    pub fn selected_skybox_index(&self) -> usize {
        match &self.skybox_path {
            None => 0,
            Some(path) => self
                .available_skyboxes
                .iter()
                .position(|s| s == path)
                .map(|i| i + 1)
                .unwrap_or(0),
        }
    }

    /// Select skybox by index (0 = None, 1+ = skybox index)
    pub fn select_skybox(&mut self, index: usize) {
        if index == 0 {
            self.skybox_path = None;
        } else if index <= self.available_skyboxes.len() {
            self.skybox_path = Some(self.available_skyboxes[index - 1].clone());
        }
    }
}

/// Discover skybox images in a directory
fn discover_skyboxes(dir: &Path) -> Vec<PathBuf> {
    let mut skyboxes = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if SKYBOX_EXTENSIONS.contains(&ext.to_lowercase().as_str()) {
                        skyboxes.push(path);
                    }
                }
            }
        }
    }
    skyboxes.sort();
    skyboxes
}

/// Get a display name for a skybox path
pub fn get_skybox_display_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}
