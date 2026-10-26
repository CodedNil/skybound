//! A development aid: with `SKYBOUND_SNAP` set to a path, the game saves its window there after
//! `SKYBOUND_SNAP_AFTER` seconds of ordinary play (three by default) and quits, so what a player
//! would see can be checked without a person at the window.

use bevy::{
    prelude::*,
    render::view::screenshot::{Screenshot, save_to_disk},
};
use std::{env, path::Path};

pub struct SnapshotPlugin;

impl Plugin for SnapshotPlugin {
    fn build(&self, app: &mut App) {
        if let Ok(path) = env::var("SKYBOUND_SNAP") {
            let after = env::var("SKYBOUND_SNAP_AFTER")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(3.0);
            app.insert_resource(Snapshot {
                path,
                after,
                taken: None,
            })
            .add_systems(Update, snap);
        }
    }
}

#[derive(Resource)]
struct Snapshot {
    path: String,
    after: f32,
    /// When the screenshot was asked for; the game quits once it is saved.
    taken: Option<f32>,
}

fn snap(
    mut commands: Commands,
    mut snapshot: ResMut<Snapshot>,
    time: Res<Time<Real>>,
    mut exit: MessageWriter<AppExit>,
) {
    let now = time.elapsed_secs();
    match snapshot.taken {
        None if now >= snapshot.after => {
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(snapshot.path.clone()));
            snapshot.taken = Some(now);
        }
        Some(_) if Path::new(&snapshot.path).exists() => {
            exit.write(AppExit::Success);
        }
        _ => {}
    }
}
