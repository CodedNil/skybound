use crate::{
    game::{
        flight::{Mount, Player, SPAWN},
        weather::CloudField,
        world::{PlanetPosition, WorldAnchor, WorldData},
    },
    math::{ToEngine, ToScene},
};
use bevy::{math::DVec3, prelude::*};
use skybound::scene::{
    FEATURE_SCALE, PLANET_RADIUS,
    climate::Climate,
    life::{
        KINDS, Kind,
        behaviour::{Activity, Motion, Senses},
        biology::{Anatomy, Feeding, Lift, Reproduction, Social, edible},
        physiology::Vitals,
    },
};
use std::collections::HashMap;

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
}

struct Pool {
    nutrients: f32,
    detritus: f32,
    bound_aur: f32,
    recycled_at: f32,
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
            .add_systems(FixedUpdate, simulate);
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
        Transform::from_translation(position),
    )
}

fn seed(mut commands: Commands) {
    for (index, kind) in KINDS.iter().copied().enumerate() {
        if matches!(kind, Kind::Nae | Kind::Naetu) {
            continue;
        }
        let biology = kind.biology();
        for cohort in 0..3 {
            let angle = (index * 3 + cohort) as f32 * 2.399_963;
            let altitude = SPAWN
                .z
                .clamp(biology.altitude.0 + 60.0, biology.altitude.1 - 60.0);
            let position = vec3(
                angle.cos() * 220.0,
                angle.sin() * 220.0,
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
    let [cover, layers] = clouds.cover(position.normalize());
    // Cloud heights are in feature space.
    let altitude = (position.length() - PLANET_RADIUS) * FEATURE_SCALE;
    let within = if altitude > layers.x && altitude < cover.y {
        cover.x
    } else {
        0.0
    };
    climate.low_cloud = cover.x;
    climate.middle_cloud = cover.z;
    climate.high_cloud = cover.w;
    climate.humidity = climate.humidity.max(within);
    climate.storm = climate
        .storm
        .max(cover.x * ((cover.y - 20_000.0) / 20_000.0).clamp(0.0, 1.0));
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
    let mut grid: HashMap<IVec3, Vec<Agent>> = HashMap::new();
    for agent in &agents {
        grid.entry(cell(agent.position)).or_default().push(*agent);
    }
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
        let recycled = patch.detritus * (1.0 - (-(now - patch.recycled_at) * 0.001).exp());
        patch.detritus -= recycled;
        patch.nutrients += recycled;
        patch.recycled_at = now;
        if body.0.health <= 0.0 {
            decompose(&mut body.0, patch, dt);
            if !player && !mount && body.0.mass < biology.adult_mass * 0.001 {
                commands.entity(entity).despawn();
            }
            continue;
        }
        let neighbours = neighbours(transform.translation, biology.perception, &grid);
        let agent = index[&entity];
        let senses = perceive(agent, &mut relations, climate, &neighbours, now, &world);
        relations.senses = senses;
        let mut movement = Motion {
            position: Vec3::ZERO.to_scene(),
            velocity: motion.0.to_scene(),
            rotation: transform.rotation.to_scene(),
        };
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
        if let Some(meal) = forage(agent, &mut body.0, &relations, &neighbours, patch, dt) {
            meals.push(meal);
        }
        if !player && !mount {
            if let Some(host) = relations
                .host
                .and_then(|host| index.get(&host))
                .filter(|host| host.vitals.health > 0.0)
            {
                transform.translation = host.position
                    + world.up(host.position) * host.kind.biology().adult_mass.cbrt() * 0.3;
                motion.0 = host.velocity;
            } else {
                relations.host = None;
                if let Ok(mut coordinate) = coordinates.get_mut(entity) {
                    coordinate.translate(movement.position.to_engine(), &mut transform, &world);
                } else {
                    transform.translation += movement.position.to_engine();
                }
                transform.rotation = movement.rotation.to_engine();
                motion.0 = movement.velocity.to_engine();
            }
        }
        if ecosystem.living < POPULATION_LIMIT
            && reproduce(agent, &mut body.0, &mut relations, &neighbours, now)
        {
            commands.spawn(population(
                species.0,
                transform.translation + Vec3::X * 3.0,
                0.15,
            ));
            ecosystem.living += 1;
            ecosystem.births += 1;
        }
    }
    ecosystem.deaths += transfer_meals(&mut organisms, &meals, now);
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
            },
        )
        .collect()
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
