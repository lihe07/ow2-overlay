use bevy::{input::common_conditions::input_just_pressed, prelude::*, window::PrimaryWindow};

use core::fmt::Write;

use crate::BBOX;

#[derive(Resource)]
struct Boxes {
    wincap: crate::wincap::WinCap,
    model: crate::inference::Model,
}

fn entity_count(world: &World) -> usize {
    world.entities().total_count()
}

#[allow(clippy::too_many_arguments)]
fn diagnostic_system(
    entity_count: In<usize>,
    mut commands: Commands,
    mut refresh_timer: Local<f32>,
    mut time_buffer: Local<crate::RingBuffer<128>>,
    cached: Local<std::cell::OnceCell<[Entity; 3]>>,
    time: Res<Time>,
    mut text: Query<&mut Text>,
) {
    let [fps, max_ft, entities] = *cached.get_or_init(|| {
        let mut result = [Entity::PLACEHOLDER; 3];
        const OUTER_MARGIN: f32 = 14.;
        const INNER_MARGIN: f32 = 12.;
        let font = TextFont {
            font_size: 42.0,
            ..default()
        };
        commands
            .spawn((
                Node {
                    margin: UiRect {
                        left: Val::Auto,
                        right: Val::Px(OUTER_MARGIN),
                        top: Val::Px(OUTER_MARGIN),
                        bottom: Val::Auto,
                    },
                    ..Default::default()
                },
                BackgroundColor(Color::srgb(0.2, 0.2, 0.2)),
            ))
            .with_children(|c| {
                c.spawn(Node {
                    flex_direction: FlexDirection::Column,
                    margin: UiRect {
                        left: Val::Px(INNER_MARGIN),
                        right: Val::Px(INNER_MARGIN),
                        top: Val::Px(INNER_MARGIN),
                        bottom: Val::Px(INNER_MARGIN),
                    },
                    align_items: AlignItems::FlexStart,
                    ..Default::default()
                })
                .with_children(|c| {
                    result = [
                        c.spawn((Node::default(), font.clone(), Text::new("FPS:")))
                            .id(),
                        c.spawn((Node::default(), font.clone(), Text::new("Max Frametime:")))
                            .id(),
                        c.spawn((Node::default(), font.clone(), Text::new("Entities:")))
                            .id(),
                    ];
                });
                c.spawn(Node {
                    flex_direction: FlexDirection::Column,
                    margin: UiRect {
                        left: Val::Px(INNER_MARGIN),
                        right: Val::Px(INNER_MARGIN),
                        top: Val::Px(INNER_MARGIN),
                        bottom: Val::Px(INNER_MARGIN),
                    },
                    min_width: Val::Px(80.),
                    align_items: AlignItems::FlexEnd,
                    ..Default::default()
                })
                .with_children(|c| {
                    result = [
                        c.spawn((Node::default(), font.clone(), Text::default()))
                            .id(),
                        c.spawn((Node::default(), font.clone(), Text::default()))
                            .id(),
                        c.spawn((Node::default(), font.clone(), Text::default()))
                            .id(),
                    ];
                });
            });
        result
    });

    time_buffer.push(time.delta_secs());

    *refresh_timer += time.delta_secs();
    if *refresh_timer > 0.05 {
        *refresh_timer -= 0.05;

        if let Ok(mut text) = text.get_mut(fps) {
            text.0.clear();
            let fps = time_buffer.len() as f32 / time_buffer.iter().sum::<f32>();
            let _ = write!(&mut text.0, "{:.0}", fps);
        };
        if let Ok(mut text) = text.get_mut(max_ft) {
            text.0.clear();
            let max_ft = time_buffer
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(::core::cmp::Ordering::Equal))
                .unwrap_or(&0.0);
            let _ = write!(&mut text.0, "{:.2}ms", max_ft * 1000.);
        };
        if let Ok(mut text) = text.get_mut(entities) {
            text.0.clear();
            let _ = write!(&mut text.0, "{}", *entity_count);
        };
    }
}

pub struct MyPlugin {
    pub config: crate::AppConfig,
}

impl Plugin for MyPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.config.clone());

        app.add_systems(Startup, setup);
        // app.add_systems(First, entity_count.pipe(diagnostic_system));

        if self.config.trigger_bot_mode {
            app.add_systems(Update, detect_and_draw_boxes.pipe(trigger_bot));
        } else {
            app.add_systems(Update, detect_and_draw_boxes.pipe(tracking_bot));
        }

        let window_id = crate::wincap::find_window_id(&self.config.window_name).unwrap();

        app.insert_resource(Boxes {
            wincap: crate::wincap::WinCap::new(window_id).unwrap(),
            model: crate::inference::Model::from_path(
                &self.config.model_path,
                self.config.conf_threshold,
                self.config.iou_threshold,
            )
            .unwrap(),
        });

        app.insert_resource(crate::inputs::MousePID::new(
            self.config.kp,
            self.config.ki,
            self.config.kd,
            self.config.max_speed,
        ));

        app.add_systems(
            Update,
            exit_system.run_if(input_just_pressed(KeyCode::Insert)),
        );
    }
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);

    let x = 100.0;
    let y = 100.0;
    let w = 100.0;
    let h = 100.0;
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(x - w / 2.0),
            top: Val::Px(y - h / 2.0),
            width: Val::Px(w),
            height: Val::Px(h),

            ..Default::default()
        },
        Outline {
            color: Color::WHITE,
            width: Val::Px(10.0),
            ..Default::default()
        },
        BoxEntity,
    ));
}

fn exit_system(mut exit: EventWriter<AppExit>) {
    exit.send(AppExit::Success);
}

#[derive(Component)]
struct BoxEntity;

/// This system will capture the window and draw bbox
fn detect_and_draw_boxes(
    mut commands: Commands,
    mut boxes: ResMut<Boxes>,
    primary_window: Single<&mut Window, With<PrimaryWindow>>,
    box_entities: Query<Entity, With<BoxEntity>>,
) -> Vec<BBOX> {
    for e in box_entities.iter() {
        commands.entity(e).despawn();
    }
    let resolution = primary_window.resolution.clone();

    let im = boxes.wincap.capture().unwrap();

    let boxes = boxes.model.process_img(im).unwrap();

    let boxes = boxes
        .into_iter()
        .map(|(x, y, w, h, c)| {
            (
                x * resolution.width(),
                y * resolution.height(),
                w * resolution.width(),
                h * resolution.height(),
                c,
            )
        })
        .collect::<Vec<_>>();

    for (x, y, w, h, _) in boxes.iter() {
        commands.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(x - w / 2.0),
                top: Val::Px(y - h / 2.0),
                width: Val::Px(*w),
                height: Val::Px(*h),

                ..Default::default()
            },
            Outline {
                color: Color::WHITE,
                width: Val::Px(10.0),
                ..Default::default()
            },
            BoxEntity,
        ));
    }

    boxes
}

fn tracking_bot(
    boxes: In<Vec<BBOX>>,
    mut commands: Commands,
    primary_window: Single<&mut Window, With<PrimaryWindow>>,
    mut mouse_pid: ResMut<crate::inputs::MousePID>,
    config: Res<crate::AppConfig>,
) {
    let mut closest = None;
    let mut closest_distance = (config.max_range).powi(2); // Disallow distant targets

    let resolution = primary_window.resolution.clone();
    let screen_width_half = resolution.width() / 2.0;
    let screen_height_half = resolution.height() / 2.0;

    for (x, y, w, h, _) in boxes.iter() {
        let head_line_y = y - 0.1 * h;
        let head_line_x = *x;

        // Calculate the dist to the center of the screen
        let dx = head_line_x - screen_width_half;
        let dy = head_line_y - screen_height_half;
        let distance = dx * dx + dy * dy;

        if distance < closest_distance {
            closest_distance = distance;
            closest = Some((head_line_x, head_line_y));
        }
    }

    if let Some((head_line_x, head_line_y)) = closest {
        // Spawn a dot at the head line
        commands.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(head_line_x),
                top: Val::Px(head_line_y),
                width: Val::Px(10.0),
                height: Val::Px(10.0),
                ..Default::default()
            },
            Outline {
                color: Color::xyz(1.0, 0.0, 0.0),
                width: Val::Px(10.0),
                ..Default::default()
            },
            BoxEntity,
        ));
        let dx = head_line_x - screen_width_half;
        let dy = head_line_y - screen_height_half;

        if (crate::inputs::is_mouse_left_pressed() && config.tracking_on_left)
            || (crate::inputs::is_mouse_right_pressed() && config.tracking_on_right)
        {
            mouse_pid.update((dx, dy));
        }
    }
}

fn trigger_bot(
    boxes: In<Vec<BBOX>>,
    primary_window: Single<&mut Window, With<PrimaryWindow>>,
    mut mouse_pid: ResMut<crate::inputs::MousePID>,
    config: Res<crate::AppConfig>,
) {
    if config.trigger_on_right && !crate::inputs::is_mouse_right_pressed() {
        return;
    }

    if config.trigger_on_side
        && !(crate::inputs::is_mouse_side_up_pressed()
            || crate::inputs::is_mouse_side_down_pressed())
    {
        return;
    }

    let resolution = primary_window.resolution.clone();
    let screen_width_half = resolution.width() / 2.0;
    let screen_height_half = resolution.height() / 2.0;

    for (x, y, w, h, _) in boxes.iter() {
        let left = x - w / 2.0 + config.trigger_box_padding;
        let right = x + w / 2.0 - config.trigger_box_padding;
        let top = y - h / 2.0 + config.trigger_box_padding;
        let bottom = y + h / 2.0 - config.trigger_box_padding;

        // Check if the target is in the center of the screen
        if left < screen_width_half
            && right > screen_width_half
            && top < screen_height_half
            && bottom > screen_height_half
        {
            mouse_pid.left_click();
        }
    }
}
