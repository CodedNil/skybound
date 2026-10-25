use crate::{
    game::{
        flight::{Mount, Player, SPAWN},
        weather::CloudField,
        world::{PlanetPosition, WorldAnchor, WorldData},
    },
    math::{ToEngine, ToScene},
};
use bevy::{
    math::DVec3,
    prelude::*,
    render::{RenderApp, extract_resource::ExtractResource},
    transform::TransformSystems,
};
use rayon::prelude::*;
use skybound::scene::{
    PLANET_RADIUS,
    climate::{Climate, layer_altitude},
    life::{
        CLUTCH, CreatureInstance, FULL_POUCH, HERD, KINDS, Kind,
        behaviour::{Activity, Motion, Senses},
        biology::{Anatomy, Brood, Feeding, Lift, Reproduction, Social, edible},
        physiology::Vitals,
    },
};
use std::{collections::HashMap, fmt::Write};

const CELL: f32 = 300.0;

const STEP: f32 = 0.25;

const POPULATION_LIMIT: usize = 512;

#[derive(Component, Clone, Copy)]
pub struct Species(pub Kind);

#[derive(Component, Clone, Copy)]
pub struct Physiology(pub Vitals);

#[derive(Component)]
pub struct Individual(pub u32);

#[derive(Component, Default)]
pub struct EcologicalMotion(pub Vec3);

#[derive(Component, Default)]
pub struct Relations {
    pub activity: Activity,
    pub senses: Senses,
    host: Option<Entity>,
    /// The parent that bore it, while it still depends on that parent.
    parent: Option<Entity>,
    remembered_food: Option<(Entity, DVec3, f32)>,
    pollen: Option<(Entity, Kind)>,
    pollinated_until: f32,
    last_birth: f32,
}

impl Relations {
    pub fn rotate(&mut self, rotation: Quat) {
        let rotation = rotation.to_scene();
        let senses = &mut self.senses;
        for direction in [
            &mut senses.wind,
            &mut senses.up,
            &mut senses.separation,
            &mut senses.alignment,
        ] {
            *direction = rotation * *direction;
        }
        for target in [
            &mut senses.food,
            &mut senses.danger,
            &mut senses.companion,
            &mut senses.habitat,
            &mut senses.storm,
            &mut senses.water,
        ] {
            *target = target.map(|direction| rotation * direction);
        }
    }
}

#[derive(Clone, Copy)]
struct Agent {
    entity: Entity,
    kind: Kind,
    vitals: Vitals,
    position: Vec3,
    velocity: Vec3,
    rotation: Quat,
}

struct Pool {
    nutrients: f32,
    detritus: f32,
    bound_aur: f32,
    recycled_at: f32,
}

impl Pool {
    /// Detritus breaking down into nutrients since it was last recycled.
    fn recycle(&mut self, now: f32) {
        let recycled = self.detritus * (1.0 - (-(now - self.recycled_at) * 0.001).exp());
        self.detritus -= recycled;
        self.nutrients += recycled;
        self.recycled_at = now;
    }
}

impl Default for Pool {
    fn default() -> Self {
        Self {
            nutrients: 100.0,
            detritus: 2.0,
            bound_aur: 0.0,
            recycled_at: 0.0,
        }
    }
}

/// The world's creatures drawn this frame, besides the player and their
/// mount: those covering the most of the screen.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct Herd(pub Vec<CreatureInstance>);

#[derive(Resource, Default)]
pub struct Ecosystem {
    pub living: usize,
    pub births: u64,
    pub deaths: u64,
    patches: HashMap<IVec3, Pool>,
}

pub struct EcosystemPlugin;

impl Plugin for EcosystemPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Ecosystem>()
            .add_systems(Startup, seed)
            .add_systems(Update, (attach_players, identify))
            .add_systems(FixedUpdate, simulate)
            .add_systems(PostUpdate, gather_herd.after(TransformSystems::Propagate))
            .add_systems(Update, census);
    }
}

type UnregisteredActors = (Or<(With<Player>, With<Mount>)>, Without<Species>);

type Organisms<'a> = (
    Entity,
    &'a Species,
    &'a mut Physiology,
    &'a mut Relations,
    &'a mut Transform,
    &'a mut EcologicalMotion,
    Has<Player>,
    Has<Mount>,
);

struct Meal {
    eater: Entity,
    food: Entity,
    amount: f32,
}

fn population(kind: Kind, position: Vec3, mass_fraction: f32) -> impl Bundle {
    let mut vitals = Vitals::new(kind.biology());
    vitals.mass *= mass_fraction;
    if mass_fraction < 0.5 {
        vitals.age_days = 0.0;
    }
    (
        Species(kind),
        Physiology(vitals),
        EcologicalMotion::default(),
        Relations::default(),
        WorldAnchor,
        // Level, heading along the local horizon until the first step of
        // the simulation steers it.
        Transform::from_translation(position).looking_to(Vec3::X, Vec3::Z),
    )
}

fn seed(mut commands: Commands) {
    for (index, kind) in KINDS.iter().copied().enumerate() {
        if matches!(kind, Kind::Nae | Kind::Naetu) {
            continue;
        }
        let biology = kind.biology();
        // Six of each, scattered over a wide disc about the start, so a few
        // losses cannot end a lineage and every kind is met near the player.
        for cohort in 0..6 {
            let angle = (index * 6 + cohort) as f32 * 2.399_963;
            let reach = 160.0 + (cohort as f32 * 0.37 + index as f32 * 0.61).fract() * 520.0;
            let altitude = SPAWN
                .z
                .clamp(biology.altitude.0 + 60.0, biology.altitude.1 - 60.0);
            let position = vec3(
                angle.cos() * reach,
                angle.sin() * reach,
                altitude + angle.sin() * 50.0,
            );
            commands.spawn(population(kind, position, 0.9));
        }
    }
}

fn identify(
    mut commands: Commands,
    organisms: Query<Entity, (With<Species>, Without<Individual>)>,
) {
    for entity in &organisms {
        commands.entity(entity).insert(Individual(
            (entity.to_bits() as u32) ^ ((entity.to_bits() >> 32) as u32).wrapping_mul(0x9e37_79b9),
        ));
    }
}

fn attach_players(
    mut commands: Commands,
    actors: Query<(Entity, Has<Player>), UnregisteredActors>,
) {
    for (entity, player) in &actors {
        let kind = if player { Kind::Nae } else { Kind::Naetu };
        commands.entity(entity).insert((
            Species(kind),
            Physiology(Vitals::new(kind.biology())),
            EcologicalMotion::default(),
            Relations::default(),
        ));
    }
}

fn climate_at(world: &WorldData, clouds: &CloudField, local: Vec3, now: f32) -> Climate {
    let position = world.position(local).as_vec3().to_scene();
    let weather = Climate::weather_on_planet(position, now);
    let mut climate = Climate::on_planet(position, now, weather);
    climate.wind = world.rotate_to_local(climate.wind.to_engine()).to_scene();
    // Creatures live among the clouds that are drawn: in them the air is
    // saturated, and tall towers mean storms.
    let [lower, upper] = clouds.cover(position.normalize());
    let altitude = position.length() - PLANET_RADIUS;
    let cumulus = layer_altitude(0);
    let within = if altitude > cumulus && altitude < upper.w {
        lower.x
    } else {
        0.0
    };
    climate.low_cloud = lower.x.max(lower.y);
    climate.middle_cloud = lower.z.max(lower.w).max(upper.x);
    climate.high_cloud = upper.y.max(upper.z);
    climate.humidity = climate.humidity.max(within);
    climate.storm = climate
        .storm
        .max(lower.x * ((upper.w - cumulus - 32_000.0) / 24_000.0).clamp(0.0, 1.0));
    climate
}

#[expect(
    clippy::too_many_arguments,
    reason = "the ecosystem step reads the whole world"
)]
fn simulate(
    mut commands: Commands,
    clock: Res<Time<Fixed>>,
    time: Res<Time>,
    world: Res<WorldData>,
    clouds: Res<CloudField>,
    mut ecosystem: ResMut<Ecosystem>,
    (mut organisms, mut coordinates): (Query<Organisms>, Query<&mut PlanetPosition>),
    mut accumulated: Local<f32>,
) {
    *accumulated += clock.delta_secs();
    if *accumulated < STEP {
        return;
    }
    let dt = *accumulated;
    *accumulated = 0.0;
    let now = time.elapsed_secs();
    let agents = snapshot(&organisms);
    let index: HashMap<_, _> = agents.iter().map(|agent| (agent.entity, *agent)).collect();
    let grid = grid(&agents);
    let mut meals = Vec::new();
    ecosystem.living = agents
        .iter()
        .filter(|agent| agent.vitals.health > 0.0)
        .count();
    for (entity, species, mut body, mut relations, mut transform, mut motion, player, mount) in
        &mut organisms
    {
        let biology = species.0.biology();
        let climate = climate_at(&world, &clouds, transform.translation, now);
        let patch = ecosystem
            .patches
            .entry(cell(world.position(transform.translation).as_vec3()))
            .or_default();
        patch.recycle(now);
        if body.0.health <= 0.0 {
            decompose(&mut body.0, patch, dt);
            if !player && !mount && body.0.mass < biology.adult_mass * 0.001 {
                commands.entity(entity).despawn();
            }
            continue;
        }
        let neighbours = neighbours(transform.translation, biology.perception, &grid);
        let agent = index[&entity];
        let mut senses = perceive(agent, &mut relations, climate, &neighbours, now, &world);
        let stage = nurture(agent, &mut relations, &index, now);
        stage.steer(&mut senses);
        relations.senses = senses;
        let mut movement = moving(motion.0, transform.rotation);
        let controls = movement.advance(species.0, &senses, dt);
        relations.activity = controls.activity;
        let effort = (movement.velocity - climate.wind).length() / biology.speed.max(1.0);
        let waste = body.0.metabolize(biology, climate, effort.min(2.0), dt);
        patch.detritus += waste.mass;
        patch.bound_aur += waste.aur;
        let growth = body.0.grow(biology, climate, patch.nutrients, dt);
        patch.nutrients -= growth;
        if body.0.health <= 0.0 {
            ecosystem.deaths += 1;
            continue;
        }
        meals.extend(if stage.forages() {
            forage(agent, &mut body.0, &relations, &neighbours, patch, dt)
        } else {
            stage.provide(entity, &mut body.0, dt)
        });
        if !player && !mount {
            if let Some(host) = relations
                .host
                .and_then(|host| index.get(&host))
                .filter(|host| host.vitals.health > 0.0)
            {
                let carried = relations.parent == relations.host;
                transform.translation = held_position(agent, host, carried, &world);
                if carried && biology.brood == Brood::Strings {
                    transform.rotation = host.rotation;
                }
                motion.0 = host.velocity;
            } else if stage == Stage::Egg {
                // An egg cast loose goes where the air takes it.
                let drift = climate.wind.to_engine();
                travel(entity, drift * dt, &mut transform, &mut coordinates, &world);
                motion.0 = drift;
            } else {
                relations.host = None;
                let offset = movement.position.to_engine();
                travel(entity, offset, &mut transform, &mut coordinates, &world);
                transform.rotation = movement.rotation.to_engine();
                motion.0 = movement.velocity.to_engine();
            }
        }
        if ecosystem.living < POPULATION_LIMIT
            && reproduce(agent, &mut body.0, &mut relations, &neighbours, now)
        {
            let host = nest(agent, &neighbours);
            commands
                .spawn(population(
                    species.0,
                    transform.translation + Vec3::X * 3.0,
                    0.15,
                ))
                .insert(Relations {
                    host,
                    parent: Some(entity),
                    ..Relations::default()
                });
            ecosystem.living += 1;
            ecosystem.births += 1;
        }
    }
    ecosystem.deaths += transfer_meals(&mut organisms, &meals, now);
}

type Drawn<'a> = (
    Entity,
    &'a Species,
    &'a Physiology,
    &'a Transform,
    &'a EcologicalMotion,
    Option<&'a Individual>,
    &'a Relations,
);

/// Every minute, how many of each species are alive, so a lineage dying
/// out shows in the log.
fn census(organisms: Query<(&Species, &Physiology)>, time: Res<Time>, mut last: Local<f32>) {
    if time.elapsed_secs() - *last < 60.0 {
        return;
    }
    *last = time.elapsed_secs();
    let mut counts: HashMap<Kind, usize> = HashMap::new();
    for (species, body) in &organisms {
        if body.0.health > 0.0 {
            *counts.entry(species.0).or_default() += 1;
        }
    }
    let mut line = String::new();
    for kind in KINDS {
        let _ = write!(line, " {kind:?}={}", counts.get(kind).copied().unwrap_or(0));
    }
    info!("census{line}");
}

/// Pose the creatures that will cover the most of the screen this frame,
/// carrying each one's last pose along for motion vectors.
fn gather_herd(
    mut herd: ResMut<Herd>,
    mut last: Local<HashMap<Entity, CreatureInstance>>,
    camera: Single<&GlobalTransform, With<Camera3d>>,
    organisms: Query<Drawn, (Without<Player>, Without<Mount>)>,
    time: Res<Time>,
) {
    let eye = camera.translation();
    let forward = camera.forward();
    // Young carried inside or glued onto a parent are drawn as part of it.
    let mut carrying: HashMap<Entity, u32> = HashMap::new();
    for (entity, species, body, _, _, _, relations) in &organisms {
        let biology = species.0.biology();
        let young = body.0.age_days / f64::from(biology.maturity_days);
        if relations.host.is_some() && relations.host == relations.parent {
            let brood = match biology.brood {
                Brood::Carried if young < 0.3 => CLUTCH,
                Brood::Pouch => FULL_POUCH,
                _ => continue,
            };
            carrying.insert(relations.host.unwrap_or(entity), brood);
            carrying.insert(entity, 0);
        }
    }
    let mut ranked: Vec<(f32, Entity)> = organisms
        .iter()
        .filter_map(|(entity, species, _, transform, _, _, _)| {
            if carrying.get(&entity) == Some(&0) {
                return None;
            }
            let radius = species.0.render_radius();
            let offset = transform.translation - eye;
            let distance = offset.length().max(1.0);
            // Behind the camera, or too small to cover a pixel.
            if offset.dot(*forward) < -radius || radius / distance < 0.0015 {
                return None;
            }
            Some((radius / distance, entity))
        })
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    ranked.truncate(HERD);
    let now = time.elapsed_secs();
    // What each pose needs, read out of the world so the poses, each with
    // its own tentacle and cloth simulation, can be built across every core.
    let chosen: Vec<_> = ranked
        .into_iter()
        .filter_map(|(_, entity)| {
            let (_, species, body, transform, motion, individual, relations) =
                organisms.get(entity).ok()?;
            Some((
                entity,
                species.0,
                body.0.age_days as f32,
                *transform,
                motion.0.length(),
                individual.map_or_else(|| entity.to_bits() as u32, |identity| identity.0),
                relations.activity,
                carrying.get(&entity).copied().unwrap_or(0),
            ))
        })
        .collect();
    let poses: Vec<(Entity, CreatureInstance)> = chosen
        .par_iter()
        .map(
            |&(entity, kind, age, transform, speed, seed, activity, held)| {
                // Each individual beats its wings at its own phase and pace,
                // harder the faster it flies.
                let phase = (seed % 997) as f32 * 0.37;
                let rate = 3.0 + (seed % 13) as f32 * 0.12;
                // Resting animals fold their wings and stop beating them.
                let resting = matches!(activity, Activity::Rest | Activity::Attach);
                let fold = if resting { 0.9 } else { 0.0 };
                let stroke = ((now + phase) * rate).sin()
                    * (0.08 + (speed / 40.0).min(1.0) * 0.25)
                    * (1.0 - fold);
                let mut pose = CreatureInstance::specimen(
                    transform
                        .translation
                        .extend(kind.render_radius())
                        .to_scene(),
                    transform.rotation.to_array().into(),
                    Vec4::new(stroke, fold, now + phase, speed).to_scene(),
                    kind as u32,
                    age,
                    seed,
                );
                if held != 0 {
                    pose.kind.z = held;
                }
                (entity, pose)
            },
        )
        .collect();
    let mut drawn = HashMap::with_capacity(poses.len());
    herd.0.clear();
    for (entity, mut pose) in poses {
        if let Some(old) = last.get(&entity)
            && old.position.truncate().distance(pose.position.truncate()) < 100.0
        {
            pose.previous_position = old.position;
            pose.previous_rotation = old.rotation;
            pose.previous_animation = old.animation;
            pose.previous_proportions = old.proportions;
            pose.previous_tail = old.tail;
            pose.previous_cloth = old.cloth;
        }
        drawn.insert(entity, pose);
        herd.0.push(pose);
    }
    *last = drawn;
}

fn snapshot(organisms: &Query<Organisms>) -> Vec<Agent> {
    organisms
        .iter()
        .map(
            |(entity, species, body, _, transform, motion, _, _)| Agent {
                entity,
                kind: species.0,
                vitals: body.0,
                position: transform.translation,
                velocity: motion.0,
                rotation: transform.rotation,
            },
        )
        .collect()
}

/// The organisms sorted into cells, so each finds its neighbours quickly.
fn grid(agents: &[Agent]) -> HashMap<IVec3, Vec<Agent>> {
    let mut grid: HashMap<IVec3, Vec<Agent>> = HashMap::new();
    for agent in agents {
        grid.entry(cell(agent.position)).or_default().push(*agent);
    }
    grid
}

fn cell(position: Vec3) -> IVec3 {
    (position / CELL).floor().as_ivec3()
}

fn neighbours(position: Vec3, range: f32, grid: &HashMap<IVec3, Vec<Agent>>) -> Vec<Agent> {
    let origin = cell(position);
    let radius = (range / CELL).ceil() as i32;
    let mut found = Vec::new();
    for x in -radius..=radius {
        for y in -radius..=radius {
            for z in -radius..=radius {
                if let Some(agents) = grid.get(&(origin + ivec3(x, y, z))) {
                    found.extend(
                        agents.iter().copied().filter(|agent| {
                            agent.position.distance_squared(position) < range * range
                        }),
                    );
                }
            }
        }
    }
    found.sort_by(|a, b| {
        a.position
            .distance_squared(position)
            .total_cmp(&b.position.distance_squared(position))
    });
    found
}

fn companion(kind: Kind, other: Kind) -> bool {
    match kind {
        Kind::Nae => other == Kind::Naetu,
        Kind::Naetu => other == Kind::Nae,
        Kind::Lucinani => other == Kind::Susurrus,
        Kind::Luminiflora
        | Kind::Aurocorallum
        | Kind::Aeromuscus
        | Kind::Flexilora
        | Kind::Chromacorallum => other.biology().adult_mass > kind.biology().adult_mass * 20.0,
        _ => kind == other,
    }
}

fn perceive(
    agent: Agent,
    relations: &mut Relations,
    climate: Climate,
    neighbours: &[Agent],
    now: f32,
    world: &WorldData,
) -> Senses {
    let Agent {
        entity,
        kind,
        vitals: body,
        position,
        ..
    } = agent;
    let biology = kind.biology();
    let mut senses = Senses {
        wind: climate.wind,
        up: world.up(position).to_scene(),
        aur: climate.aur,
        time: now,
        energy: body.energy,
        hydration: body.hydration,
        fatigue: body.fatigue,
        ..Senses::default()
    };
    for other in neighbours.iter().filter(|other| other.entity != entity) {
        let offset = (other.position - position).to_scene();
        if senses.food.is_none()
            && ((edible(kind, other.kind) && other.vitals.health > 0.0)
                || (biology.feeding == Feeding::Scavenger && other.vitals.health <= 0.0))
        {
            senses.food = Some(offset);
            relations.remembered_food = Some((other.entity, world.position(other.position), now));
        }
        if senses.danger.is_none()
            && other.vitals.health > 0.0
            && other.kind.biology().feeding == Feeding::Predator
            && edible(other.kind, kind)
        {
            senses.danger = Some(offset);
        }
        if senses.water.is_none() && other.kind == Kind::Lacufolium && other.vitals.health > 0.0 {
            senses.water = Some(offset);
        }
        if senses.companion.is_none() && companion(kind, other.kind) && other.vitals.health > 0.0 {
            senses.companion = Some(offset);
            if offset.length_squared() < 25.0
                && (biology.organs.lift == Lift::Attached || kind == Kind::Lucinani)
            {
                relations.host = Some(other.entity);
            }
        }
        if other.kind == kind && other.vitals.health > 0.0 {
            senses.separation -= offset / offset.length_squared().max(1.0);
            if matches!(biology.social, Social::Herd | Social::Swarm) {
                senses.alignment += other.velocity.to_scene() * 0.02;
            }
        }
    }
    if senses.food.is_none()
        && let Some((remembered, location, seen)) = relations.remembered_food
        && now - seen < 8.0
    {
        if neighbours
            .iter()
            .any(|other| other.entity == remembered && other.vitals.health <= 0.0)
        {
            relations.remembered_food = None;
        } else {
            senses.food = Some((world.local(location) - position).to_scene());
        }
    }
    let target_altitude = if biology.organs.aur_capacity > 0.0 && body.aur < 0.15 {
        25.0
    } else {
        climate
            .altitude
            .clamp(biology.altitude.0 + 20.0, biology.altitude.1 - 20.0)
    };
    if (target_altitude - climate.altitude).abs() > 5.0 {
        senses.habitat = Some(senses.up * (target_altitude - climate.altitude));
    }
    if climate.storm > 0.1 {
        senses.storm = Some(Vec3::Y.to_scene() * 100.0);
    }
    senses.buoyancy = match biology.organs.lift {
        Lift::Bladders | Lift::Membrane | Lift::Chimney => {
            -climate.wind.dot(senses.up)
                + ((target_altitude - climate.altitude) * 0.01).clamp(-2.0, 2.0)
        }
        Lift::Wings => -climate.wind.dot(senses.up) - body.fatigue * 0.2,
        Lift::Attached | Lift::Aerosol => 0.0,
    };
    if matches!(kind, Kind::Aerohirudo | Kind::Ptilohirudo) && body.energy > 0.8 {
        relations.host = None;
    }
    senses
}

fn forage(
    agent: Agent,
    body: &mut Vitals,
    relations: &Relations,
    neighbours: &[Agent],
    pool: &mut Pool,
    dt: f32,
) -> Option<Meal> {
    let Agent {
        entity,
        kind,
        position,
        ..
    } = agent;
    let senses = &relations.senses;
    let biology = kind.biology();
    if relations.activity == Activity::Drink
        && senses
            .water
            .is_some_and(|offset| offset.length_squared() < 100.0)
    {
        body.hydration = (body.hydration + dt * 0.2).min(1.0);
    }
    if biology.feeding == Feeding::Scavenger && body.energy < 0.85 {
        let amount = pool.detritus.min(body.mass * dt * 0.003);
        let aur = pool.bound_aur * amount / pool.detritus.max(1e-10);
        pool.detritus -= amount;
        pool.bound_aur -= aur;
        body.ingest(amount, aur);
    }
    if kind == Kind::Luminiflora
        && let Some(host) = relations.host
    {
        return Some(Meal {
            eater: host,
            food: entity,
            amount: body.mass * dt * 0.0001,
        });
    }
    if !matches!(
        relations.activity,
        Activity::Feed | Activity::Clean | Activity::Attach | Activity::Drift
    ) {
        return None;
    }
    let reach = biology.adult_mass.cbrt() * 2.0 + 2.0;
    neighbours
        .iter()
        .find(|other| {
            other.entity != entity
                && ((edible(kind, other.kind) && other.vitals.health > 0.0)
                    || (biology.feeding == Feeding::Scavenger && other.vitals.health <= 0.0))
                && other.position.distance_squared(position) < reach * reach
        })
        .map(|food| Meal {
            eater: entity,
            food: food.entity,
            amount: body.mass * dt * 0.003,
        })
}

fn transfer_meals(organisms: &mut Query<Organisms>, meals: &[Meal], now: f32) -> u64 {
    let mut deaths = 0;
    for meal in meals {
        let Ok(
            [
                (_, eater_kind, mut eater, mut visitor, _, _, _, _),
                (_, food_kind, mut food, mut flower, _, _, _, _),
            ],
        ) = organisms.get_many_mut([meal.eater, meal.food])
        else {
            continue;
        };
        if eater.0.health <= 0.0 || food.0.mass <= 0.0 {
            continue;
        }
        let nectar = eater_kind.0.biology().feeding == Feeding::Nectar;
        if !nectar && food.0.health <= 0.0 && eater_kind.0.biology().feeding != Feeding::Scavenger {
            continue;
        }
        let rate = if nectar { 0.1 } else { 1.0 };
        let amount = (meal.amount * rate).min(food.0.mass * 0.05);
        food.0.mass -= amount;
        eater.0.ingest(
            amount,
            if food_kind.0.biology().anatomy == Anatomy::Plant {
                amount * 0.1
            } else {
                0.0
            },
        );
        if nectar {
            if let Some((donor, kind)) = visitor.pollen
                && donor != meal.food
                && kind == food_kind.0
            {
                flower.pollinated_until = now + 120.0;
            }
            visitor.pollen = Some((meal.food, food_kind.0));
        }
        if eater_kind.0.biology().feeding == Feeding::Parasite {
            visitor.host = Some(meal.food);
        }
        if food.0.mass < food_kind.0.biology().adult_mass * 0.25 && food.0.health > 0.0 {
            food.0.health = 0.0;
            deaths += 1;
        }
    }
    deaths
}

/// An organism's motion this step in the scene's frame, starting from
/// where it is, at its velocity and heading.
fn moving(velocity: Vec3, rotation: Quat) -> Motion {
    Motion {
        position: Vec3::ZERO.to_scene(),
        velocity: velocity.to_scene(),
        rotation: rotation.to_scene(),
    }
}

/// Where a newborn is first kept, by its lineage's way of keeping them: on
/// its parent, tethered to the nearest living plant (its food plant if
/// there is one in reach), or nowhere.
fn nest(parent: Agent, neighbours: &[Agent]) -> Option<Entity> {
    match parent.kind.biology().brood {
        Brood::Carried | Brood::Strings | Brood::Pouch => Some(parent.entity),
        Brood::Tethered => {
            let rank = |other: &Agent| {
                let preference = if edible(parent.kind, other.kind) {
                    0.2
                } else {
                    1.0
                };
                other.position.distance(parent.position) * preference
            };
            neighbours
                .iter()
                .filter(|other| {
                    other.kind.biology().anatomy == Anatomy::Plant && other.vitals.health > 0.0
                })
                .min_by(|a, b| rank(a).total_cmp(&rank(b)))
                .map(|plant| plant.entity)
        }
        Brood::Live | Brood::Drifting => None,
    }
}

/// Where an organism held by another sits. Carried by its parent, it is in
/// its pouch or glued to its body, or on a string trailing behind it;
/// clinging to a plant or animal, it is somewhere on it, each at its own
/// place.
fn held_position(agent: Agent, host: &Agent, carried: bool, world: &WorldData) -> Vec3 {
    if carried {
        return if agent.kind.biology().brood == Brood::Strings {
            host.position + host.rotation * Vec3::Z * host.kind.render_radius() * 0.5
        } else {
            host.position
        };
    }
    let bits = agent.entity.to_bits();
    let around = Vec3::new(
        (bits % 7) as f32 - 3.0,
        (bits % 5) as f32 - 2.0,
        (bits % 3) as f32 - 1.0,
    ) * 0.3;
    host.position + (world.up(host.position) + around) * host.kind.biology().adult_mass.cbrt() * 0.3
}

/// Move an organism through the world, on the planet's surface frame when
/// it has one.
fn travel(
    entity: Entity,
    offset: Vec3,
    transform: &mut Transform,
    coordinates: &mut Query<&mut PlanetPosition>,
    world: &WorldData,
) {
    if let Ok(mut coordinate) = coordinates.get_mut(entity) {
        coordinate.translate(offset, transform, world);
    } else {
        transform.translation += offset;
    }
}

/// How far a young organism still depends on its parent.
#[derive(Clone, Copy, PartialEq)]
enum Stage {
    /// Grown, or never cared for.
    Free,
    /// An egg, carried or cast loose, living on its yolk.
    Egg,
    /// In its parent's pouch, fed there.
    Pouch(Entity),
    /// Out of the pouch and due back: the way to the parent.
    Homing(Vec3),
}

impl Stage {
    /// Back to the pouch, whatever else it might rather be doing.
    fn steer(self, senses: &mut Senses) {
        if let Self::Homing(offset) = self {
            senses.habitat = Some(offset.to_scene());
            senses.danger = None;
            senses.separation *= 0.0;
        }
    }

    const fn forages(self) -> bool {
        !matches!(self, Self::Egg | Self::Pouch(_))
    }

    /// What a young organism that does not forage lives on: an egg on its
    /// yolk, and pouch young on what the parent feeds them.
    fn provide(self, entity: Entity, body: &mut Vitals, dt: f32) -> Option<Meal> {
        if let Self::Pouch(parent) = self {
            return Some(Meal {
                eater: entity,
                food: parent,
                amount: body.mass * dt * 0.002,
            });
        }
        body.energy = body.energy.max(0.6);
        body.hydration = body.hydration.max(0.6);
        None
    }
}

/// Where a young organism is kept, by its lineage's way of keeping them.
/// Eggs stay where they were laid until they hatch, except strings, which
/// the parent cuts loose to drift soon after laying them. Pouch young stay
/// in the pouch until they are formed, then make trips out to feed, at
/// first brief and then longer, homing on the parent when each is over,
/// until they no longer come back.
fn nurture(
    agent: Agent,
    relations: &mut Relations,
    index: &HashMap<Entity, Agent>,
    now: f32,
) -> Stage {
    let biology = agent.kind.biology();
    let young = (agent.vitals.age_days / f64::from(biology.maturity_days)) as f32;
    let Some(parent) = relations
        .parent
        .and_then(|parent| index.get(&parent))
        .filter(|parent| parent.vitals.health > 0.0)
    else {
        relations.parent = None;
        let egg = biology.anatomy == Anatomy::Tetralate
            && biology.reproduction == Reproduction::Eggs
            && young < 0.3;
        // Orphaned or hatched, a young tetralate flies free of whatever it
        // was laid on.
        if biology.anatomy == Anatomy::Tetralate && !egg {
            relations.host = None;
        }
        return if egg { Stage::Egg } else { Stage::Free };
    };
    let parent = *parent;
    match biology.brood {
        Brood::Pouch if young < 0.2 => {
            relations.host = Some(parent.entity);
            Stage::Pouch(parent.entity)
        }
        Brood::Pouch if young < 0.7 => {
            let independence = (young - 0.2) / 0.5;
            let cycle = 120.0;
            let phase = ((now + (agent.entity.to_bits() % 997) as f32) / cycle).fract();
            if phase < 0.1 + independence * 0.85 {
                relations.host = None;
                return Stage::Free;
            }
            let offset = parent.position - agent.position;
            if relations.host == Some(parent.entity)
                || offset.length() < parent.kind.render_radius() * 0.4 + 3.0
            {
                relations.host = Some(parent.entity);
                Stage::Pouch(parent.entity)
            } else {
                Stage::Homing(offset)
            }
        }
        Brood::Pouch | Brood::Live => {
            relations.parent = None;
            relations.host = None;
            Stage::Free
        }
        _ if young < 0.3 => {
            if biology.brood == Brood::Strings && young > 0.08 {
                relations.host = None;
            }
            Stage::Egg
        }
        _ => {
            relations.parent = None;
            relations.host = None;
            Stage::Free
        }
    }
}

fn decompose(body: &mut Vitals, pool: &mut Pool, dt: f32) {
    let amount = body.mass * (1.0 - (-dt * 0.005).exp());
    body.mass -= amount;
    pool.detritus += amount + body.gut;
    pool.bound_aur += body.gut_aur;
    body.gut = 0.0;
    body.gut_aur = 0.0;
}

fn reproduce(
    agent: Agent,
    body: &mut Vitals,
    relations: &mut Relations,
    neighbours: &[Agent],
    now: f32,
) -> bool {
    let kind = agent.kind;
    let biology = kind.biology();
    let maturity = f64::from(biology.maturity_days);
    let mature = body.age_days >= maturity && body.mass > biology.adult_mass * 1.15;
    let fertile = match biology.reproduction {
        Reproduction::Pollination => relations.pollinated_until > now,
        Reproduction::Eggs | Reproduction::LiveBirth => neighbours.iter().any(|other| {
            other.entity != agent.entity
                && other.kind == kind
                && other.vitals.age_days >= maturity
                && other.vitals.health > 0.0
                && other.position.distance_squared(agent.position) < 400.0
        }),
        _ => true,
    };
    if mature
        && fertile
        && body.energy > 0.8
        && body.hydration > 0.5
        && now - relations.last_birth > 90.0
    {
        body.mass -= biology.adult_mass * 0.15;
        body.energy -= 0.2;
        relations.last_birth = now;
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::SystemState;

    #[test]
    fn live_birth_requires_another_living_adult_and_conserves_mass() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let mate = world.spawn_empty().id();
        let kind = Kind::Naetu;
        let mut body = Vitals::new(kind.biology());
        body.age_days = f64::from(kind.biology().maturity_days);
        body.mass *= 1.2;
        body.energy = 1.0;
        let agent = Agent {
            entity,
            kind,
            vitals: body,
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        let mut relations = Relations::default();
        assert!(!reproduce(
            agent,
            &mut body,
            &mut relations,
            &[agent],
            200.0
        ));
        let partner = Agent {
            entity: mate,
            ..agent
        };
        let before = body.mass;
        assert!(reproduce(
            agent,
            &mut body,
            &mut relations,
            &[partner],
            200.0
        ));
        assert!((before - body.mass - kind.biology().adult_mass * 0.15).abs() < 1e-4);
        assert!(!reproduce(
            agent,
            &mut body,
            &mut relations,
            &[partner],
            201.0
        ));
    }

    #[test]
    fn competing_grazers_cannot_create_food_or_eat_a_carcass() {
        let mut world = World::new();
        let first = world.spawn(population(Kind::Naetu, Vec3::ZERO, 1.0)).id();
        let second = world.spawn(population(Kind::Naetu, Vec3::ZERO, 1.0)).id();
        let plant = world
            .spawn(population(Kind::Aerothallus, Vec3::ZERO, 0.3))
            .id();
        let before = world.get::<Physiology>(plant).expect("plant").0.mass;
        let meals = [
            Meal {
                eater: first,
                food: plant,
                amount: 1e6,
            },
            Meal {
                eater: second,
                food: plant,
                amount: 1e6,
            },
        ];
        let mut state = SystemState::<Query<Organisms>>::new(&mut world);
        transfer_meals(
            &mut state.get_mut(&mut world).expect("organisms query"),
            &meals,
            0.0,
        );
        let remaining = world.get::<Physiology>(plant).expect("plant").0.mass;
        let eaten = world.get::<Physiology>(first).expect("grazer").0.gut
            + world.get::<Physiology>(second).expect("grazer").0.gut;
        assert!((before - remaining - eaten).abs() < 1e-5);
        world.get_mut::<Physiology>(plant).expect("plant").0.health = 0.0;
        transfer_meals(
            &mut state.get_mut(&mut world).expect("organisms query"),
            &meals,
            1.0,
        );
        assert!((world.get::<Physiology>(plant).expect("plant").0.mass - remaining).abs() < 1e-6);
    }

    #[test]
    fn young_stay_with_their_parent_as_their_lineage_keeps_them() {
        let mut world = World::new();
        let mother = world.spawn_empty().id();
        let child = world.spawn_empty().id();
        let agent = |entity, kind: Kind, young: f32, position| {
            let mut vitals = Vitals::new(kind.biology());
            vitals.age_days = f64::from(kind.biology().maturity_days * young);
            Agent {
                entity,
                kind,
                vitals,
                position,
                velocity: Vec3::ZERO,
                rotation: Quat::IDENTITY,
            }
        };
        let far = Vec3::new(500.0, 0.0, 0.0);
        let kind = Kind::Herbivolus;
        let index: HashMap<_, _> = [(mother, agent(mother, kind, 2.0, Vec3::ZERO))].into();
        let mut relations = Relations {
            parent: Some(mother),
            ..Relations::default()
        };
        assert!(
            nurture(agent(child, kind, 0.1, far), &mut relations, &index, 0.0)
                == Stage::Pouch(mother),
            "a newborn left the pouch"
        );
        assert_eq!(relations.host, Some(mother));
        // Half grown, it spends some of each cycle away and comes home.
        let stages: Vec<_> = (0..120)
            .map(|second| {
                let mut relations = Relations {
                    parent: Some(mother),
                    ..Relations::default()
                };
                nurture(
                    agent(child, kind, 0.45, far),
                    &mut relations,
                    &index,
                    second as f32,
                )
            })
            .collect();
        assert!(stages.contains(&Stage::Free), "pouch young never left");
        assert!(
            stages.iter().any(|stage| matches!(stage, Stage::Homing(_))),
            "pouch young never came home"
        );
        let mut relations = Relations {
            parent: Some(mother),
            ..Relations::default()
        };
        assert!(nurture(agent(child, kind, 0.8, far), &mut relations, &index, 0.0) == Stage::Free);
        assert!(
            relations.parent.is_none(),
            "grown young still depend on the parent"
        );
        // A string of eggs is cut loose to drift before it hatches.
        let kind = Kind::Manduca;
        let index: HashMap<_, _> = [(mother, agent(mother, kind, 2.0, Vec3::ZERO))].into();
        let mut relations = Relations {
            parent: Some(mother),
            host: Some(mother),
            ..Relations::default()
        };
        assert!(
            nurture(
                agent(child, kind, 0.05, Vec3::ZERO),
                &mut relations,
                &index,
                0.0
            ) == Stage::Egg
        );
        assert_eq!(relations.host, Some(mother));
        assert!(
            nurture(
                agent(child, kind, 0.2, Vec3::ZERO),
                &mut relations,
                &index,
                0.0
            ) == Stage::Egg
        );
        assert!(relations.host.is_none(), "egg string never cut loose");
    }

    #[test]
    fn food_memory_survives_origin_shifts_and_expires() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let food = world.spawn_empty().id();
        let kind = Kind::Herbivolus;
        let agent = Agent {
            entity,
            kind,
            vitals: Vitals::new(kind.biology()),
            position: vec3(0.0, 0.0, 2400.0),
            velocity: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        let mut relations = Relations {
            remembered_food: Some((
                food,
                WorldData::default().position(agent.position + Vec3::X * 80.0),
                1.0,
            )),
            ..Relations::default()
        };
        let origin = Vec3::X * 50_000.0;
        let mut coordinates = WorldData::default();
        coordinates.origin = coordinates.position(origin);
        let shifted = Agent {
            position: agent.position - origin,
            ..agent
        };
        let climate = Climate::sample(agent.position.to_scene(), 2.0, 0.0);
        let senses = perceive(shifted, &mut relations, climate, &[], 2.0, &coordinates);
        assert!(
            senses
                .food
                .expect("remembered food")
                .distance(Vec3::X.to_scene() * 80.0)
                < 1e-5
        );
        let expired = perceive(shifted, &mut relations, climate, &[], 10.0, &coordinates);
        assert!(expired.food.is_none());
    }
}
