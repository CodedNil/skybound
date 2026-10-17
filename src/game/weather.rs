use bevy::prelude::*;
use bevy::render::{RenderApp, extract_resource::ExtractResource};
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future::poll_once};
use skybound::scene::{
    climate::simulation::{PlanetWeather, TICK},
    cloud::Cloud,
};
use std::{env, sync::Arc};

#[derive(Resource)]
pub struct WeatherSimulation(pub Arc<PlanetWeather>, Option<Task<PlanetWeather>>);
impl Default for WeatherSimulation {
    fn default() -> Self {
        let mut arguments = env::args();
        let seed = arguments
            .find(|arg| arg == "--seed")
            .and_then(|_| arguments.next())
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        Self(Arc::new(PlanetWeather::new(seed, 0.0)), None)
    }
}
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct CloudSnapshot {
    pub clouds: Arc<[Cloud]>,
    pub grid: Arc<[u32]>,
    pub time: f32,
    pub revision: u64,
}
impl FromWorld for CloudSnapshot {
    fn from_world(world: &mut World) -> Self {
        let model = &world.resource::<WeatherSimulation>().0;
        Self {
            clouds: model.clouds.clone().into(),
            grid: model.grid.clone().into(),
            time: model.time,
            revision: model.revision,
        }
    }
}
pub struct WeatherPlugin;
impl Plugin for WeatherPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WeatherSimulation>()
            .init_resource::<CloudSnapshot>()
            .add_systems(Update, simulate);
    }
}
fn simulate(
    time: Res<Time<Virtual>>,
    mut simulation: ResMut<WeatherSimulation>,
    mut snapshot: ResMut<CloudSnapshot>,
) {
    let now = time.elapsed_secs();
    if let Some(mut pending) = simulation.1.take() {
        if let Some(model) = block_on(poll_once(&mut pending)) {
            simulation.0 = Arc::new(model);
        } else {
            simulation.1 = Some(pending);
        }
    }
    if simulation.1.is_none() && (now - simulation.0.time >= TICK || now < simulation.0.time) {
        let current = Arc::clone(&simulation.0);
        simulation.1 = Some(AsyncComputeTaskPool::get().spawn(async move {
            let mut model = (*current).clone();
            if now < model.time || now - model.time > 120.0 {
                model.time = now;
                model.revision += 1;
            } else {
                model.tick();
            }
            model
        }));
    }
    if snapshot.revision != simulation.0.revision {
        snapshot.clouds = simulation.0.clouds.clone().into();
        snapshot.grid = simulation.0.grid.clone().into();
        snapshot.time = simulation.0.time;
        snapshot.revision = simulation.0.revision;
    }
}
