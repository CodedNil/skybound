use crate::{
    game::{
        flight::{Mount, Player},
        weather::CloudField,
        world::{PlanetPosition, WorldAnchor, WorldData},
    },
    math::{ToEngine, ToScene},
};
use bevy::{
    ecs::system::ParallelCommands,
    math::DVec3,
    prelude::*,
    render::{RenderApp, extract_resource::ExtractResource},
    transform::TransformSystems,
    utils::Parallel,
};
use skybound::scene::{
    PLANET_RADIUS,
    climate::{Climate, layer_altitude},
    life::{
        CreatureInstance, Kind,
        animalia::mind::Mind,
        behaviour::{Activity, Senses},
        organs::Strike,
        plantae::physiology::grow,
        vitals::Vitals,
    },
};
use std::{
    collections::HashMap,
    mem,
    sync::atomic::{AtomicUsize, Ordering},
};

mod breeding;
mod conduct;
mod drawing;
mod feeding;
mod movement;
mod senses;

use self::{
    breeding::{bear, decompose, reproduce},
    conduct::{Conduct, conduct, land},
    drawing::{census, gather_herd},
    feeding::{Meal, forage, transfer_meals},
    movement::{Move, moving, relocate},
    senses::{Surroundings, orient},
};

const CELL: f32 = 300.0;

/// Seconds between steps of the simulation, twenty to the second: often enough that the fastest
/// hunters cannot pass through their prey between steps, and drawn smoothly between them.
const STEP: f32 = 0.05;

const POPULATION_LIMIT: usize = 1600;

/// A call given this step: whose voice gave it, where, and how far it carries.
#[derive(Clone, Copy)]
pub struct Call {
    caller: Kind,
    at: Vec3,
    reach: f32,
}

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
    /// What it has set itself to do, and its temper.
    pub mind: Mind,
    minded: bool,
    /// Until when it carries fertile eggs or young, having mated.
    fertile_until: f32,
    /// When it can strike again.
    ready: f32,
    /// The plant it roosts on, or a Weaver's web is strung from.
    home: Option<Entity>,
    /// The nearest partner it keeps company with, which a voiceless animal can ask to call.
    companion: Option<Entity>,
    /// A seed carried in its gut from a fruit, and when it will be passed.
    seed: Option<(Kind, f32)>,
    /// The hunter carrying it off, while it is helpless to escape.
    captor: Option<Entity>,
    /// Where it was before the last step moved it, as an offset from where it is, how it was
    /// turned, and when that step was, so it is drawn gliding between steps rather than jumping.
    lag: Vec3,
    turned: Quat,
    stepped: f32,
}

impl Relations {
    pub fn rotate(&mut self, turn: Quat) {
        self.lag = turn * self.lag;
        self.turned = turn * self.turned;
        let rotation = turn.to_scene();
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
    activity: Activity,
    /// Whether it is the player or their mount, which nothing hunts or eats: the game has no death
    /// for them yet.
    protected: bool,
}

impl Agent {
    fn alive(&self) -> bool {
        self.vitals.health > 0.0
    }

    fn grown(&self) -> f32 {
        (self.vitals.age_days / f64::from(self.kind.biology().maturity_days)) as f32
    }
}

#[derive(Clone, Copy)]
struct Pool {
    nutrients: f32,
    detritus: f32,
    bound_aur: f32,
    recycled_at: f32,
}

impl Pool {
    /// Nothing at all, from which changes are summed.
    const NONE: Self = Self {
        nutrients: 0.0,
        detritus: 0.0,
        bound_aur: 0.0,
        recycled_at: 0.0,
    };

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

/// The world's creatures drawn this frame, besides the player and their mount: those covering the
/// most of the screen.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct Herd(pub Vec<CreatureInstance>);

#[derive(Resource, Default)]
pub struct Ecosystem {
    pub living: usize,
    pub births: u64,
    pub deaths: u64,
    patches: HashMap<IVec3, Pool>,
    /// Calls made last step, which carry to the callers' kin far off.
    calls: Vec<Call>,
}

impl Ecosystem {
    /// The patch of air at a place, its detritus recycled up to now.
    fn patch(&mut self, position: DVec3, now: f32) -> &mut Pool {
        let patch = self.patches.entry(cell(position.as_vec3())).or_default();
        patch.recycle(now);
        patch
    }

    const fn born(&mut self) {
        self.living += 1;
        self.births += 1;
    }
}

pub struct EcosystemPlugin;

impl Plugin for EcosystemPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Ecosystem>()
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
    Option<&'a mut PlanetPosition>,
);

/// One organism as the step finds it.
type Organism<'a> = (
    Entity,
    &'a Species,
    Mut<'a, Physiology>,
    Mut<'a, Relations>,
    Mut<'a, Transform>,
    Mut<'a, EcologicalMotion>,
    bool,
    bool,
    Option<Mut<'a, PlanetPosition>>,
);

/// A blow dealt this step, landed once every organism has moved.
struct Attack {
    victim: Entity,
    strike: Option<Strike>,
    /// Seconds the victim is held: tangled in silk or paralysed by venom.
    stun: f32,
    /// The hunter carrying the victim off in its mouth tentacles.
    carrier: Option<Entity>,
}

pub fn population(kind: Kind, position: Vec3, mass_fraction: f32) -> impl Bundle {
    let mut vitals = Vitals::born(kind);
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
        // Level, heading along the local horizon until the first step of the simulation steers it.
        Transform::from_translation(position).looking_to(Vec3::X, Vec3::Z),
    )
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
            Physiology(Vitals::born(kind)),
            EcologicalMotion::default(),
            Relations::default(),
        ));
    }
}

fn climate_at(world: &WorldData, clouds: &CloudField, local: Vec3, now: f32) -> Climate {
    let position = world.position(local).as_vec3().to_scene();
    let weather = Climate::weather_on_planet(position, now);
    let mut climate = Climate::on_planet(position, weather);
    climate.wind = world.rotate_to_local(climate.wind.to_engine()).to_scene();
    climate.sunward = world
        .rotate_to_local(climate.sunward.to_engine())
        .to_scene();
    // Creatures live among the clouds that are drawn: in them the air is saturated, and tall towers
    // mean storms.
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

/// What one thread's share of a step changed beyond the organisms themselves: the meals, blows,
/// calls and seeds to settle, what the air about them gained or lost, and the births and deaths.
#[derive(Default)]
struct Outcome {
    meals: Vec<Meal>,
    attacks: Vec<Attack>,
    calls: Vec<Call>,
    seedlings: Vec<(Kind, Vec3)>,
    patches: HashMap<IVec3, Pool>,
    births: u64,
    deaths: u64,
}

impl Outcome {
    /// Gather what the air of a patch gained and lost from one organism's step.
    fn exchange(&mut self, cell: IVec3, before: Pool, after: Pool) {
        let change = self.patches.entry(cell).or_insert(Pool::NONE);
        change.nutrients += after.nutrients - before.nutrients;
        change.detritus += after.detritus - before.detritus;
        change.bound_aur += after.bound_aur - before.bound_aur;
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the ecosystem step reads the whole world"
)]
fn simulate(
    commands: ParallelCommands,
    clock: Res<Time<Fixed>>,
    time: Res<Time>,
    world: Res<WorldData>,
    clouds: Res<CloudField>,
    mut ecosystem: ResMut<Ecosystem>,
    mut organisms: Query<Organisms>,
    mut accumulated: Local<f32>,
) {
    *accumulated += clock.delta_secs();
    if *accumulated < STEP {
        return;
    }
    let dt = mem::take(&mut *accumulated);
    let now = time.elapsed_secs();
    let agents = snapshot(&organisms);
    let index: HashMap<_, _> = agents.iter().map(|agent| (agent.entity, *agent)).collect();
    let grid = grid(&agents);
    let calls = mem::take(&mut ecosystem.calls);
    let players = players(&organisms);
    ecosystem.living = agents.iter().filter(|agent| agent.alive()).count();
    // The air about each organism as the step begins, which each works on a copy of; what they
    // all changed is settled afterwards.
    for agent in &agents {
        ecosystem.patch(world.position(agent.position), now);
    }
    let living = AtomicUsize::new(ecosystem.living);
    let mut outcomes: Parallel<Outcome> = Parallel::default();
    let shared = Shared {
        world: &world,
        clouds: &clouds,
        index: &index,
        grid: &grid,
        players: &players,
        calls: &calls,
        patches: &ecosystem.patches,
        living: &living,
        commands: &commands,
        now,
        dt,
    };
    organisms
        .par_iter_mut()
        .for_each(|organism| live_out(organism, &shared, &mut outcomes.borrow_local_mut()));
    let mut outcome = Outcome::default();
    for part in outcomes.iter_mut() {
        let mut part = mem::take(part);
        outcome.meals.append(&mut part.meals);
        outcome.attacks.append(&mut part.attacks);
        outcome.calls.append(&mut part.calls);
        outcome.seedlings.append(&mut part.seedlings);
        outcome.births += part.births;
        outcome.deaths += part.deaths;
        #[expect(
            clippy::iter_over_hash_type,
            reason = "the changes are summed, so their order does not matter"
        )]
        for (cell, change) in part.patches {
            let patch = ecosystem.patches.entry(cell).or_default();
            patch.nutrients = (patch.nutrients + change.nutrients).max(0.0);
            patch.detritus = (patch.detritus + change.detritus).max(0.0);
            patch.bound_aur = (patch.bound_aur + change.bound_aur).max(0.0);
        }
    }
    ecosystem.calls = outcome.calls;
    ecosystem.births += outcome.births;
    ecosystem.living += outcome.births as usize;
    ecosystem.deaths += outcome.deaths;
    ecosystem.deaths += transfer_meals(&mut organisms, &outcome.meals, now);
    ecosystem.deaths += land(&mut organisms, &outcome.attacks);
    commands.command_scope(|mut commands| {
        for (kind, position) in outcome.seedlings {
            if ecosystem.living < POPULATION_LIMIT {
                commands.spawn(population(kind, position, 0.1));
                ecosystem.born();
            }
        }
    });
}

/// What every organism's step reads of the world, shared across the threads.
struct Shared<'a> {
    world: &'a WorldData,
    clouds: &'a CloudField,
    index: &'a HashMap<Entity, Agent>,
    grid: &'a HashMap<IVec3, Vec<Agent>>,
    players: &'a [Vec3],
    calls: &'a [Call],
    patches: &'a HashMap<IVec3, Pool>,
    /// How many are alive, counting the births so far this step.
    living: &'a AtomicUsize,
    commands: &'a ParallelCommands<'a, 'a>,
    now: f32,
    dt: f32,
}

/// One organism's step: it senses and decides, lives a step of its body's life, eats, acts, moves
/// and perhaps gives birth, changing only itself and noting the rest in `outcome`.
fn live_out(
    (
        entity,
        species,
        mut body,
        mut relations,
        mut transform,
        mut motion,
        player,
        mount,
        coordinate,
    ): Organism,
    shared: &Shared,
    outcome: &mut Outcome,
) {
    let Shared {
        world,
        clouds,
        index,
        grid,
        players,
        calls,
        patches,
        living,
        commands,
        now,
        dt,
    } = *shared;
    let kind = species.0;
    let climate = climate_at(world, clouds, transform.translation, now);
    let cell = cell(world.position(transform.translation).as_vec3());
    let before = patches.get(&cell).copied().unwrap_or_default();
    let mut patch = before;
    if body.0.health <= 0.0 {
        if decompose(&mut body.0, &mut patch, dt, kind.biology()) && !player && !mount {
            commands.command_scope(|mut commands| commands.entity(entity).despawn());
        }
        outcome.exchange(cell, before, patch);
        return;
    }
    let neighbours = neighbours(transform.translation, kind.senses(), grid);
    let agent = index[&entity];
    let (senses, stage, targets) = orient(
        agent,
        &mut relations,
        !player && !mount,
        &Surroundings {
            climate,
            neighbours: &neighbours,
            players,
            calls,
            now,
        },
        (index, world),
    );
    let mut movement = moving(motion.0, transform.rotation);
    let controls = movement.advance(kind, &senses, dt);
    relations.activity = controls.activity;
    let effort = (movement.velocity - climate.wind).length() / kind.speed().max(1.0);
    let alive = live(&mut body.0, kind, climate, effort.min(2.0), &mut patch, dt);
    if !alive {
        outcome.deaths += 1;
        outcome.exchange(cell, before, patch);
        return;
    }
    let meal = if stage.forages() {
        forage(agent, &mut body.0, &relations, &neighbours, &mut patch, dt)
    } else {
        stage.provide(entity, &mut body.0, dt)
    };
    outcome.exchange(cell, before, patch);
    outcome.meals.extend(meal);
    conduct(
        agent,
        &mut body.0,
        &mut relations,
        &targets,
        index,
        Conduct {
            now,
            dt,
            attacks: &mut outcome.attacks,
            calls: &mut outcome.calls,
        },
    );
    outcome
        .seedlings
        .extend(pass_seed(&mut relations, now, transform.translation));
    if !player && !mount {
        relocate(
            agent,
            stage,
            &mut relations,
            (&mut transform, &mut motion),
            &movement,
            index,
            Move {
                coordinate: coordinate.map(Mut::into_inner),
                world,
                wind: climate.wind.to_engine(),
                dt,
                now,
            },
        );
    }
    if living.load(Ordering::Relaxed) < POPULATION_LIMIT
        && reproduce(agent, &mut body.0, &mut relations, &neighbours, now)
    {
        living.fetch_add(1, Ordering::Relaxed);
        commands.command_scope(|mut commands| {
            bear(&mut commands, agent, &neighbours, transform.translation);
        });
        outcome.births += 1;
    }
}

/// Where the players and their mounts are.
fn players(organisms: &Query<Organisms>) -> Vec<Vec3> {
    organisms
        .iter()
        .filter(|(.., player, mount, _)| *player || *mount)
        .map(|(_, _, _, _, transform, ..)| transform.translation)
        .collect()
}

/// A step of the body's own work: metabolism and, in plants, growth, trading waste and nutrients
/// with the patch of air around it. Returns whether it is still alive.
fn live(
    body: &mut Vitals,
    kind: Kind,
    climate: Climate,
    effort: f32,
    patch: &mut Pool,
    dt: f32,
) -> bool {
    let waste = body.metabolize(kind, climate, effort, dt);
    patch.detritus += waste.mass;
    patch.bound_aur += waste.aur;
    let growth = grow(body, kind, climate, patch.nutrients, dt);
    patch.nutrients -= growth;
    body.health > 0.0
}

/// A seed carried from a fruit, passed once it has ridden long enough, to sprout where its carrier
/// has flown.
fn pass_seed(relations: &mut Relations, now: f32, position: Vec3) -> Option<(Kind, Vec3)> {
    let (kind, at) = relations.seed?;
    if now <= at {
        return None;
    }
    relations.seed = None;
    Some((kind, position))
}

fn snapshot(organisms: &Query<Organisms>) -> Vec<Agent> {
    organisms
        .iter()
        .map(
            |(entity, species, body, relations, transform, motion, player, mount, _)| Agent {
                entity,
                kind: species.0,
                vitals: body.0,
                position: transform.translation,
                velocity: motion.0,
                rotation: transform.rotation,
                activity: relations.activity,
                protected: player || mount,
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

#[cfg(test)]
mod tests {
    use super::{
        breeding::{Stage, nurture},
        feeding::Meal,
        senses::perceive,
        *,
    };
    use bevy::ecs::system::SystemState;

    #[test]
    fn birth_requires_mating_and_conserves_mass() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
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
            activity: Activity::Drift,
            protected: false,
        };
        let mut relations = Relations::default();
        assert!(
            !reproduce(agent, &mut body, &mut relations, &[agent], 200.0),
            "bred without mating"
        );
        relations.fertile_until = 1000.0;
        let before = body.mass;
        assert!(reproduce(agent, &mut body, &mut relations, &[agent], 200.0));
        assert!((before - body.mass - kind.biology().adult_mass * 0.15).abs() < 1e-4);
        assert!(!reproduce(
            agent,
            &mut body,
            &mut relations,
            &[agent],
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
                activity: Activity::Drift,
                protected: false,
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
            activity: Activity::Drift,
            protected: false,
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
