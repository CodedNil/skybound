use crate::{
    game::{
        flight::{Mount, Player},
        weather::CloudField,
        world::{PlanetPosition, WorldAnchor, WorldData},
    },
    math::{ToEngine, ToScene},
};
use bevy::{
    ecs::{entity::EntityHashMap, system::ParallelCommands},
    math::DVec3,
    platform::collections::HashMap as FastMap,
    prelude::*,
    render::{RenderApp, extract_resource::ExtractResource},
    transform::TransformSystems,
    utils::Parallel,
};
use skybound::scene::{
    PLANET_RADIUS,
    climate::{Climate, layer_altitude},
    features::nearest_geyser,
    life::{
        Drawn, Kind,
        animalia::mind::Mind,
        behaviour::{Activity, Motion, Senses},
        genome::{FECUNDITY, Genome},
        organs::Strike,
        plantae::physiology::grow,
        vitals::Vitals,
    },
};
use spirv_std::glam::Vec3 as SceneVec3;
use std::{
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
    movement::{Move, flex, moving, relocate},
    senses::{Surroundings, orient},
};

const CELL: f32 = 300.0;

/// Seconds between steps of the simulation, twenty to the second: often enough that the fastest
/// hunters cannot pass through their prey between steps, and drawn smoothly between them.
const STEP: f32 = 0.05;

const POPULATION_LIMIT: usize = 25_600;

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
    pollen: Option<(Entity, Kind, Genome)>,
    pollinated_until: f32,
    last_birth: f32,
    /// What it has set itself to do, and what it has learned.
    pub mind: Mind,
    /// The genes it was born with, and those of the mate or pollen that will father its young.
    pub genes: Genome,
    sire: Option<Genome>,
    /// Where its wings are in their beat, in radians, and how hard it is beating them.
    wingbeat: f32,
    effort: f32,
    /// How far the air bends a plant at its top, in its own frame, and how fast the bend is
    /// changing: a spring the moving air pushes against.
    bend: Vec3,
    bending: Vec3,
    /// Until when it carries fertile eggs or young, having mated.
    fertile_until: f32,
    /// When it can strike again.
    ready: f32,
    /// The plant it roosts on, or a Weaver's web is strung from.
    home: Option<Entity>,
    /// The nearest partner it keeps company with, which a voiceless animal can ask to call.
    companion: Option<Entity>,
    /// A seed carried in its gut from a fruit, when it will be passed, and the genes it carries.
    seed: Option<(Kind, f32, Genome)>,
    /// The hunter carrying it off, while it is helpless to escape.
    captor: Option<Entity>,
    /// How fast others pushed it along last step: a guardian nudging a stray egg home,
    /// tree-dwellers drawing their tree into rising air.
    nudge: Vec3,
    /// The plant it is tied to by silk, and how long the tie is: a Weaver's work, or the strands
    /// of a kelp forest grown into one another. It holds the two together against the wind
    /// until it is stretched too far and snaps.
    pub tether: Option<(Entity, f32)>,
    /// The climate about it when last sampled, when and where: the air changes over seconds and
    /// kilometres, not from one step to the next.
    climate: Option<(Climate, f32, Vec3)>,
    /// Where it was before the last step moved it, as an offset from where it is, how it was
    /// turned, and when that step was, so it is drawn gliding between steps rather than jumping.
    lag: Vec3,
    turned: Quat,
    stepped: f32,
}

impl Relations {
    pub fn rotate(&mut self, turn: Quat) {
        // The cached climate's winds and bearings are in the old axes.
        self.climate = None;
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
            &mut senses.minerals,
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
    genes: Genome,
    /// How practised it is, from what it has learned.
    skill: f32,
    /// The plant it is tied to by silk, if it is.
    tether: Option<Entity>,
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
pub struct Herd(pub Vec<Drawn>);

#[derive(Resource, Default)]
pub struct Ecosystem {
    pub living: usize,
    pub births: u64,
    pub deaths: u64,
    patches: FastMap<IVec3, Pool>,
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

pub fn population(kind: Kind, position: Vec3, mass_fraction: f32, genes: Genome) -> impl Bundle {
    let mut vitals = Vitals::born(kind);
    vitals.mass *= mass_fraction;
    let seed = u64::from(position.x.to_bits()) ^ u64::from(position.z.to_bits()).rotate_left(29);
    // A grown one met in the wild is any age from just grown to old, the old the fewer.
    let biology = kind.biology();
    if mass_fraction < 0.5 {
        vitals.age_days = 0.0;
    } else {
        let span = (biology.lifespan_days * 0.7 - biology.maturity_days).max(0.0);
        let age = biology.maturity_days + span * breeding::unit(seed ^ 0xa9).powi(3);
        vitals.age_days = f64::from(age);
    }
    // Its sex falls out at conception, as even odds between the two. A grown one met in the wild
    // may already be carrying, as many of its kind are at any time, somewhere through it.
    vitals.female = breeding::unit(seed) < 0.5;
    if mass_fraction >= 0.5 && vitals.bears(kind) && breeding::unit(seed ^ 0xb1) < 0.3 {
        let litter = biology.litter;
        vitals.expecting = litter.draw(breeding::unit(seed ^ 0xc3), genes.get(FECUNDITY));
        vitals.gestation = breeding::unit(seed ^ 0xd7);
    }
    (
        Species(kind),
        Physiology(vitals),
        EcologicalMotion::default(),
        Relations {
            genes,
            ..Relations::default()
        },
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
            Relations {
                genes: Genome::wild(entity.to_bits()),
                ..Relations::default()
            },
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
    // A hurricane's winds carry whatever is caught in them, and everything knows which way its
    // heart lies.
    let local = |direction: SceneVec3| world.rotate_to_local(direction.to_engine()).to_scene();
    let (gale, toward, severity) =
        clouds.gale(position.normalize(), position.length() - PLANET_RADIUS);
    climate.wind += local(gale);
    climate.storm = climate.storm.max(severity.min(1.0));
    climate.storm_toward = local(toward);
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
    // Rain falls from heavy convective cloud onto whatever is beneath it, and freezes out higher
    // up.
    let beneath = 1.0 - altitude.smoothstep(upper.w, upper.w + 500.0);
    climate.precipitation = climate
        .precipitation
        .max(lower.x.smoothstep(0.45, 0.9) * beneath * (1.0 - climate.ice * 0.6));
    climate.storm = climate
        .storm
        .max(lower.x * ((upper.w - cumulus - 32_000.0) / 24_000.0).clamp(0.0, 1.0));
    climate
}

/// What one thread's share of a step changed beyond the organisms themselves: the meals, blows,
/// calls, pushes, gifts and seeds to settle, what the air about them gained or lost, and the births
/// and deaths.
#[derive(Default)]
struct Outcome {
    meals: Vec<Meal>,
    attacks: Vec<Attack>,
    calls: Vec<Call>,
    nudges: Vec<(Entity, Vec3)>,
    gifts: Vec<(Entity, f32)>,
    ties: Vec<(Entity, Entity, f32)>,
    seedlings: Vec<(Kind, Vec3, Genome)>,
    patches: FastMap<IVec3, Pool>,
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
    let index: Index = agents.iter().map(|agent| (agent.entity, *agent)).collect();
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
        outcome.nudges.append(&mut part.nudges);
        outcome.gifts.append(&mut part.gifts);
        outcome.ties.append(&mut part.ties);
        outcome.seedlings.append(&mut part.seedlings);
        outcome.births += part.births;
        outcome.deaths += part.deaths;
        // The changes are summed, so their order does not matter.
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
    hand_over(&mut organisms, &outcome.nudges, &outcome.gifts);
    for &(plant, anchor, rest) in &outcome.ties {
        if let Ok((_, _, _, mut relations, ..)) = organisms.get_mut(plant)
            && relations.tether.is_none()
        {
            relations.tether = Some((anchor, rest));
        }
    }
    commands.command_scope(|mut commands| {
        for (kind, position, genes) in outcome.seedlings {
            if ecosystem.living < POPULATION_LIMIT {
                commands.spawn(population(kind, position, 0.1, genes));
                ecosystem.born();
            }
        }
    });
}

/// Settle the step's pushes and gifts: each push moves its ward next step, the pushes on one
/// summed, and each gift is eaten by the guardian it was brought to.
fn hand_over(organisms: &mut Query<Organisms>, nudges: &[(Entity, Vec3)], gifts: &[(Entity, f32)]) {
    for &(ward, push) in nudges {
        if let Ok((_, _, _, mut relations, ..)) = organisms.get_mut(ward) {
            relations.nudge += push;
        }
    }
    for &(guardian, gift) in gifts {
        if let Ok((_, _, mut body, ..)) = organisms.get_mut(guardian) {
            body.0.ingest(gift, 0.0);
        }
    }
}

/// What every organism's step reads of the world, shared across the threads.
struct Shared<'a> {
    world: &'a WorldData,
    clouds: &'a CloudField,
    index: &'a Index,
    grid: &'a Grid<'a>,
    players: &'a [Vec3],
    calls: &'a [Call],
    patches: &'a FastMap<IVec3, Pool>,
    /// How many are alive, counting the births so far this step.
    living: &'a AtomicUsize,
    commands: &'a ParallelCommands<'a, 'a>,
    now: f32,
    dt: f32,
}

/// One organism's step: it senses and decides, lives a step of its body's life, eats, acts, moves
/// and perhaps gives birth, changing only itself and noting the rest in `outcome`.
#[expect(
    clippy::too_many_lines,
    reason = "one organism's step reads best as the sequence of what happens to it"
)]
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
        index,
        players,
        now,
        dt,
        ..
    } = *shared;
    let kind = species.0;
    // Sampled afresh every fifth of a second or thirty metres, each organism on its own beat so the
    // sampling is spread across the steps.
    let beat = 0.15 + (entity.to_bits() % 4) as f32 * 0.025;
    let climate = match relations.climate {
        Some((climate, at, place))
            if now - at < beat && place.distance(transform.translation) < 30.0 =>
        {
            climate
        }
        _ => {
            let climate = climate_at(world, shared.clouds, transform.translation, now);
            relations.climate = Some((climate, now, transform.translation));
            climate
        }
    };
    let cell = cell(world.position(transform.translation).as_vec3());
    let before = shared.patches.get(&cell).copied().unwrap_or_default();
    let mut patch = before;
    if body.0.health <= 0.0 {
        if decompose(&mut body.0, &mut patch, dt, kind.biology()) && !player && !mount {
            shared
                .commands
                .command_scope(|mut commands| commands.entity(entity).despawn());
        }
        outcome.exchange(cell, before, patch);
        return;
    }
    // A geyser's jet kills whatever it strikes, but for the player and their mount, which the game
    // has no death for yet.
    if !player && !mount && in_geyser(world, kind, transform.translation) {
        body.0.health = 0.0;
        outcome.deaths += 1;
        outcome.exchange(cell, before, patch);
        return;
    }
    let neighbours = neighbours(transform.translation, kind.senses(), shared.grid);
    let agent = index[&entity];
    let (senses, stage, targets) = orient(
        agent,
        &mut relations,
        !player && !mount,
        &Surroundings {
            climate,
            neighbours: &neighbours,
            players,
            calls: shared.calls,
            now,
        },
        (index, world),
    );
    let mut movement = moving(motion.0, transform.rotation);
    let controls = movement.advance(kind, &senses, dt);
    relations.activity = controls.activity;
    settle(
        &mut relations,
        &mut body.0,
        (kind, &movement, climate),
        players,
        transform.translation,
        dt,
    );
    // Every second of beating wings is paid for, three times over at full effort.
    let alive = live(
        &mut body.0,
        kind,
        climate,
        movement.effort * 3.0,
        &mut patch,
        dt,
    );
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
            neighbours: &neighbours,
            attacks: &mut outcome.attacks,
            calls: &mut outcome.calls,
            nudges: &mut outcome.nudges,
            gifts: &mut outcome.gifts,
            ties: &mut outcome.ties,
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
    if breed(
        agent,
        &mut body.0,
        &mut relations,
        &neighbours,
        shared,
        transform.translation,
    ) {
        outcome.births += 1;
    }
}

/// Whether a body at `local` is inside a geyser's jet.
fn in_geyser(world: &WorldData, kind: Kind, local: Vec3) -> bool {
    let planet = world.position(local);
    let up = planet.normalize().as_vec3().to_scene();
    let altitude = (planet.length() - f64::from(PLANET_RADIUS)) as f32;
    nearest_geyser(up).is_some_and(|(geyser, distance)| {
        distance < geyser.radius + kind.render_radius() * 0.5 && altitude < geyser.height
    })
}

/// Bear a litter if one is due, as many of it as there is room in the world for.
fn breed(
    agent: Agent,
    body: &mut Vitals,
    relations: &mut Relations,
    neighbours: &[&Agent],
    shared: &Shared,
    position: Vec3,
) -> bool {
    let litter = reproduce(agent, body, relations, shared.now);
    let mut born = 0;
    for index in 0..litter {
        if shared.living.load(Ordering::Relaxed) >= POPULATION_LIMIT {
            break;
        }
        shared.living.fetch_add(1, Ordering::Relaxed);
        // Each of a litter its own: the same parents, but its own share of their genes.
        let now = shared.now + index as f32 * 1e-3;
        shared.commands.command_scope(|mut commands| {
            bear(
                &mut commands,
                agent,
                relations.sire,
                neighbours,
                position,
                now,
            );
        });
        born += 1;
    }
    born > 0
}

/// What moving did to the body and the mind: wings beat on or a plant bends, the hide gathers what
/// grows on it, an Iron Spine gathers iron, and calm time near the player wears away an animal's
/// unease with them.
fn settle(
    relations: &mut Relations,
    body: &mut Vitals,
    (kind, movement, climate): (Kind, &Motion, Climate),
    players: &[Vec3],
    position: Vec3,
    dt: f32,
) {
    let wind = climate.wind;
    flex(relations, kind, movement, wind, dt);
    body.overgrow(kind, (movement.velocity - wind).length(), dt);
    body.gather_iron(kind, climate, relations.activity == Activity::Hone, dt);
    let near_player = players
        .iter()
        .any(|player| player.distance(position) < 60.0);
    let afraid = relations.activity == Activity::Flee;
    relations.mind.experience.settle(near_player, afraid, dt);
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
fn pass_seed(relations: &mut Relations, now: f32, position: Vec3) -> Option<(Kind, Vec3, Genome)> {
    let (kind, at, genes) = relations.seed?;
    if now <= at {
        return None;
    }
    relations.seed = None;
    Some((kind, position, genes))
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
                genes: relations.genes,
                skill: relations.mind.experience.skill,
                tether: relations.tether.map(|(anchor, _)| anchor),
            },
        )
        .collect()
}

/// The organisms sorted into cells, so each finds its neighbours quickly.
/// The organisms sorted into cells, keyed by a fast hash: every organism looks up many cells a
/// step.
type Grid<'a> = FastMap<IVec3, Vec<&'a Agent>>;

/// Every organism as the step found it, by entity.
type Index = EntityHashMap<Agent>;

fn grid(agents: &[Agent]) -> Grid<'_> {
    let mut grid = Grid::default();
    for agent in agents {
        grid.entry(cell(agent.position)).or_default().push(agent);
    }
    grid
}

fn cell(position: Vec3) -> IVec3 {
    (position / CELL).floor().as_ivec3()
}

fn neighbours<'a>(position: Vec3, range: f32, grid: &Grid<'a>) -> Vec<&'a Agent> {
    let origin = cell(position);
    let radius = (range / CELL).ceil() as i32;
    // Only the cells some part of which lies within range: a ball of them, not the cube about it.
    let reach = (range / CELL + 1.8).powi(2);
    let mut found: Vec<&Agent> = Vec::new();
    for x in -radius..=radius {
        for y in -radius..=radius {
            for z in -radius..=radius {
                if (x * x + y * y + z * z) as f32 > reach {
                    continue;
                }
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
    found
}

#[cfg(test)]
mod tests {
    use super::{
        breeding::{Stage, nurture},
        feeding::Meal,
        senses::{perceive, sense},
        *,
    };
    use bevy::ecs::system::SystemState;
    use skybound::scene::life::KINDS;
    use std::iter::once;

    #[test]
    fn birth_requires_mating_and_conserves_mass() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let kind = Kind::Naetu;
        let mut body = Vitals::new(kind.biology());
        body.age_days = f64::from(kind.biology().maturity_days);
        body.mass *= 1.2;
        body.energy = 1.0;
        body.female = true;
        let agent = Agent {
            entity,
            kind,
            vitals: body,
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            activity: Activity::Drift,
            protected: false,
            genes: Genome::TYPICAL,
            skill: 0.0,
            tether: None,
        };
        let mut relations = Relations::default();
        reproduce(agent, &mut body, &mut relations, 200.0);
        assert_eq!(body.expecting, 0, "conceived without mating");
        relations.fertile_until = 1000.0;
        let mut male = body;
        male.female = false;
        reproduce(agent, &mut male, &mut relations, 200.0);
        assert_eq!(male.expecting, 0, "a male conceived");
        assert_eq!(reproduce(agent, &mut body, &mut relations, 200.0), 0);
        let litter = body.expecting;
        assert!(litter > 0, "a mated female did not conceive");
        // Nothing is born until the carrying is done, and then the whole litter at once.
        assert_eq!(reproduce(agent, &mut body, &mut relations, 300.0), 0);
        body.gestation = 1.0;
        let before = body.mass;
        assert_eq!(reproduce(agent, &mut body, &mut relations, 400.0), litter);
        let cost = kind.biology().adult_mass * 0.15 * (litter as f32).sqrt();
        assert!((before - body.mass - cost).abs() < 1e-3);
        assert_eq!(body.expecting, 0);
    }

    #[test]
    fn litters_follow_their_weights_and_fecundity() {
        let litter = Kind::Sputator.biology().litter;
        let draws = |fecundity: f32| -> f32 {
            (0..1000)
                .map(|index| litter.draw(index as f32 / 1000.0, fecundity) as f32)
                .sum::<f32>()
                / 1000.0
        };
        assert!((draws(0.5) - litter.mean()).abs() < 0.1);
        assert!(draws(0.9) > draws(0.5) && draws(0.5) > draws(0.1));
        // A single calf is the rule for the giants.
        let calf = Kind::Antiquus.biology().litter;
        assert!((calf.mean() - 1.0).abs() < 0.05);
    }

    #[test]
    fn competing_grazers_cannot_create_food_or_eat_a_carcass() {
        let mut world = World::new();
        let first = world
            .spawn(population(Kind::Naetu, Vec3::ZERO, 1.0, Genome::TYPICAL))
            .id();
        let second = world
            .spawn(population(Kind::Naetu, Vec3::ZERO, 1.0, Genome::TYPICAL))
            .id();
        let plant = world
            .spawn(population(
                Kind::Aerothallus,
                Vec3::ZERO,
                0.3,
                Genome::TYPICAL,
            ))
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
                genes: Genome::TYPICAL,
                skill: 0.0,
                tether: None,
            }
        };
        let far = Vec3::new(500.0, 0.0, 0.0);
        let kind = Kind::Herbivolus;
        let index: Index = once((mother, agent(mother, kind, 2.0, Vec3::ZERO))).collect();
        let mut relations = Relations {
            parent: Some(mother),
            ..Relations::default()
        };
        assert!(
            nurture(agent(child, kind, 0.1, far), &mut relations, &index) == Stage::Pouch(mother),
            "a newborn left the pouch"
        );
        assert_eq!(relations.host, Some(mother));
        // Half grown, it goes out while it is fed and comes home as it grows hungry.
        let stages: Vec<_> = (0..=10)
            .map(|fed| {
                let mut relations = Relations {
                    parent: Some(mother),
                    ..Relations::default()
                };
                let mut young = agent(child, kind, 0.45, far);
                young.vitals.energy = fed as f32 / 10.0;
                nurture(young, &mut relations, &index)
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
        assert!(nurture(agent(child, kind, 0.8, far), &mut relations, &index) == Stage::Free);
        assert!(
            relations.parent.is_none(),
            "grown young still depend on the parent"
        );
        // A string of eggs is cut loose to drift before it hatches.
        let kind = Kind::Manduca;
        let index: Index = once((mother, agent(mother, kind, 2.0, Vec3::ZERO))).collect();
        let mut relations = Relations {
            parent: Some(mother),
            host: Some(mother),
            ..Relations::default()
        };
        assert!(
            nurture(agent(child, kind, 0.05, Vec3::ZERO), &mut relations, &index) == Stage::Egg
        );
        assert_eq!(relations.host, Some(mother));
        assert!(nurture(agent(child, kind, 0.2, Vec3::ZERO), &mut relations, &index) == Stage::Egg);
        assert!(relations.host.is_none(), "egg string never cut loose");
    }

    #[test]
    fn hunters_leave_a_sanctuary_alone_unless_desperate() {
        let mut world = World::new();
        let hunter_kind = Kind::Ferrispina;
        let prey_kind = *KINDS
            .iter()
            .find(|kind| {
                kind.animal().is_some()
                    && hunter_kind.eats(**kind)
                    && kind.biology().adult_mass < hunter_kind.biology().adult_mass * 0.5
            })
            .expect("something a Shredder hunts");
        let mut agent = |kind: Kind, position| Agent {
            entity: world.spawn_empty().id(),
            kind,
            vitals: Vitals::new(kind.biology()),
            position,
            velocity: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            activity: Activity::Drift,
            protected: false,
            genes: Genome::TYPICAL,
            skill: 0.0,
            tether: None,
        };
        let hunter = agent(hunter_kind, Vec3::ZERO);
        let prey = agent(prey_kind, Vec3::X * 60.0);
        let guardian = agent(Kind::Custos, Vec3::X * 80.0);
        let climate = Climate::sample(Vec3::ZERO.to_scene(), 2.0, 0.0);
        let prey_in = |hunter: Agent, neighbours: &[&Agent]| {
            let around = Surroundings {
                climate,
                neighbours,
                players: &[],
                calls: &[],
                now: 0.0,
            };
            sense(hunter, &Relations::default(), &Senses::default(), &around)
                .1
                .prey
                .is_some()
        };
        assert!(prey_in(hunter, &[&prey]), "the hunter overlooked its prey");
        assert!(
            !prey_in(hunter, &[&prey, &guardian]),
            "the hunter hunted in a sanctuary"
        );
        let mut starving = hunter;
        starving.vitals.energy = 0.02;
        assert!(
            prey_in(starving, &[&prey, &guardian]),
            "a starving hunter would not test the sanctuary"
        );
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
            genes: Genome::TYPICAL,
            skill: 0.0,
            tether: None,
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
