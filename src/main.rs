mod gui;
mod inference;
mod inputs;
mod nms;
mod wincap;

use bevy::{
    prelude::*,
    window::{CursorOptions, WindowLevel, WindowMode},
};

type BBOX = (f32, f32, f32, f32, f32); // x_center, y_center, width, height, confidence

pub struct RingBuffer<const N: usize> {
    buffer: [f32; N],
    ptr: usize,
    len: usize,
}

impl<const N: usize> RingBuffer<N> {
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn push(&mut self, item: f32) {
        self.buffer[self.ptr] = item;
        self.ptr = (self.ptr + 1) & (N - 1);
        self.len = N.min(self.len + 1)
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.buffer[0..self.len].iter()
    }
}

impl<const N: usize> Default for RingBuffer<N> {
    fn default() -> Self {
        Self {
            buffer: [0.0; N],
            ptr: 0,
            len: 0,
        }
    }
}

#[derive(Resource, serde::Deserialize, Clone)]
struct AppConfig {
    kp: f32,
    ki: f32,
    kd: f32,

    max_range: f32,
    max_speed: i32,

    tracking_on_left: bool,
    tracking_on_right: bool,

    trigger_bot_mode: bool,

    trigger_on_right: bool,
    trigger_on_side: bool,

    model_path: String,
    conf_threshold: f32,
    iou_threshold: f32,

    window_name: String,
}

fn main() {
    // Spawn
    std::thread::spawn(inputs::init_input_state);

    // Read config
    let config = std::fs::read_to_string("config.toml").unwrap();
    let config: AppConfig = toml::from_str(&config).unwrap();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Screen Capture".to_string(),
                mode: WindowMode::BorderlessFullscreen(MonitorSelection::Current),
                transparent: true,
                decorations: false,
                window_level: WindowLevel::AlwaysOnTop,
                composite_alpha_mode: bevy::window::CompositeAlphaMode::Auto,
                cursor_options: CursorOptions {
                    hit_test: false,
                    ..Default::default()
                },
                ..Default::default()
            }),
            ..Default::default()
        }))
        .insert_resource(ClearColor(Color::NONE))
        .add_plugins(gui::MyPlugin { config })
        .run();
}
