use std::sync::atomic::{AtomicBool, Ordering};

use tfc::MouseContext;

#[derive(bevy::prelude::Resource)]
pub struct MousePID {
    kp: f32,
    ki: f32,
    kd: f32,
    last_error: f32, // (x, y)
    last_time: std::time::Instant,
    recent_error_mul_dts: crate::RingBuffer<128>,
    tfc_context: tfc::Context,

    max_speed: i32,
}

impl MousePID {
    pub fn new(kp: f32, ki: f32, kd: f32, max_speed: i32) -> Self {
        Self {
            kp,
            ki,
            kd,
            last_error: 0.0,
            recent_error_mul_dts: crate::RingBuffer::default(),
            last_time: std::time::Instant::now(),
            tfc_context: tfc::Context::new().unwrap(),
            max_speed,
        }
    }

    fn norm(pos: (f32, f32)) -> f32 {
        let (x, y) = pos;
        (x * x + y * y).sqrt()
    }

    fn dist_to_x_y(dist: f32, direction: (f32, f32)) -> (f32, f32) {
        let (dx, dy) = direction;
        (dist * dx, dist * dy)
    }

    fn update_pid(&mut self, error: f32) -> f32 {
        let dt = self.last_time.elapsed().as_secs_f32();

        let p = error * self.kp;
        let i = self.ki * self.recent_error_mul_dts.iter().sum::<f32>();
        let d = self.kd * (error - self.last_error) / dt;

        self.last_time = std::time::Instant::now();
        self.recent_error_mul_dts.push(error * dt);
        self.last_error = error;
        p + i + d
    }

    pub fn update(&mut self, target: (f32, f32)) {
        let error = MousePID::norm(target);
        let direction = (target.0 / error, target.1 / error);
        let dist = self.update_pid(error);
        let (dx, dy) = MousePID::dist_to_x_y(dist, direction);
        // move mouse
        let dx = dx as i32;
        let dy = dy as i32;

        // clamp
        // dbg!(dx, dy);
        let dx = dx.clamp(-self.max_speed, self.max_speed);
        let dy = dy.clamp(-self.max_speed, self.max_speed);

        self.tfc_context.mouse_move_rel(dx, dy).ok();
    }

    pub fn left_click(&mut self) {
        self.tfc_context.mouse_click(tfc::MouseButton::Left).ok();
    }
}

struct InputState {
    mouse_left_pressed: AtomicBool,
    mouse_right_pressed: AtomicBool,
}

// make a global mutex variable
static mut INPUT_STATE: Option<InputState> = None;

fn global_event_callback(event: rdev::Event) {
    match event.event_type {
        rdev::EventType::ButtonPress(rdev::Button::Left) => unsafe {
            INPUT_STATE
                .as_ref()
                .unwrap()
                .mouse_left_pressed
                .store(true, Ordering::Relaxed);
        },
        rdev::EventType::ButtonRelease(rdev::Button::Left) => unsafe {
            INPUT_STATE
                .as_ref()
                .unwrap()
                .mouse_left_pressed
                .store(false, Ordering::Relaxed);
        },

        rdev::EventType::ButtonPress(rdev::Button::Right) => unsafe {
            INPUT_STATE
                .as_ref()
                .unwrap()
                .mouse_right_pressed
                .store(true, Ordering::Relaxed);
        },
        rdev::EventType::ButtonRelease(rdev::Button::Right) => unsafe {
            INPUT_STATE
                .as_ref()
                .unwrap()
                .mouse_right_pressed
                .store(false, Ordering::Relaxed);
        },

        rdev::EventType::KeyPress(rdev::Key::End) => {
            // Shutdown the program
            std::process::exit(0);
        }

        _ => {}
    }
}

pub fn init_input_state() {
    unsafe {
        INPUT_STATE = Some(InputState {
            mouse_left_pressed: AtomicBool::new(false),
            mouse_right_pressed: AtomicBool::new(false),
        });
    }

    rdev::listen(global_event_callback).unwrap();
}

pub fn is_mouse_left_pressed() -> bool {
    unsafe {
        INPUT_STATE
            .as_ref()
            .unwrap()
            .mouse_left_pressed
            .load(Ordering::Relaxed)
    }
}

pub fn is_mouse_right_pressed() -> bool {
    unsafe {
        INPUT_STATE
            .as_ref()
            .unwrap()
            .mouse_right_pressed
            .load(Ordering::Relaxed)
    }
}
