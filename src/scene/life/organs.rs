//! The organs every organism is built from, where each species carries them, and what a wound does
//! to them.
//!
//! Life above the aur needs what life anywhere needs: a way to take in and store food, to exchange
//! gases, to move fluid round the body, to keep water, to coordinate itself and to reproduce. This
//! world adds two needs of its own: lift, for there is nothing to stand on, and a guard against the
//! aur, which some lineages have turned into a store of energy. Each organ answers one of those
//! needs and is declared in the taxon where it arose: the pump and nerve centre at the root of the
//! animals, flight muscles separately in each lineage that took to the air, the aur sac in the
//! Nætu lineage, fruit in the fruiting plants. A species' organs are placed in its own body frame,
//! where its surface encloses them, so a strike that reaches one through the body does what losing
//! that organ would.

#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{Drawn, KINDS, Kind};
use crate::scene::life::{Injury, ORGAN_SLOTS};
use core::f32::consts::{PI, TAU};
use spirv_std::glam::{Vec3, Vec4, vec2, vec3};
#[cfg(not(target_arch = "spirv"))]
use std::{sync::OnceLock, thread};

/// An organ, declared in the taxon where it arose: its name, the need it meets, the colour it is
/// drawn in when the body is opened, and the tissue it is made of.
///
/// Organs that meet the same need in different lineages are different organs, as the separately
/// evolved flight muscles of the tetralates and the winged blood-feeders are, and the body's
/// workings ask only how well each need is met.
#[derive(Clone, Copy, Debug)]
pub struct Organ {
    pub name: &'static str,
    pub function: Function,
    pub colour: Vec3,
    pub tissue: Tissue,
}

impl Organ {
    pub const fn new(name: &'static str, function: Function, colour: Vec3) -> Self {
        Self {
            name,
            function,
            colour,
            tissue: Tissue::Soft,
        }
    }

    /// The same organ, made of `tissue`.
    pub const fn of(self, tissue: Tissue) -> Self {
        Self { tissue, ..self }
    }
}

impl PartialEq for Organ {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

/// What an organ is made of, which sets how a wound takes it and how it looks cut open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tissue {
    /// Glandular, hollow or spongy tissue: crops, pumps, folds and stores.
    Soft,
    /// Contractile fibre in bundles, which a cut parts cleanly and a burn barely reaches.
    Muscle,
    /// Spring fibre: no animal above the aur has rigid bone, for a rigid frame snaps where a
    /// flexible one gives, and nothing stands on anything. Those that have a frame at all wind it
    /// from coiled, elastic fibre that bends with each stroke and gives the stroke back.
    Spring,
    /// Spring fibre laced with iron the body has eaten, stiffer and heavier, in the metal-eaters.
    Ironbound,
}

/// The needs an organ can meet. Which lineages meet each, and with what organ, is theirs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Function {
    /// Driving fluid through the body. Lose it and every other organ starves within moments.
    Circulation,
    /// Coordinating the body and the senses. Damage leaves it erratic; destroy it and the body
    /// stops.
    Coordination,
    /// Holding and digesting food. Torn, it leaks its contents and digests poorly.
    Digestion,
    /// Exchanging gases, and in a dry world wringing the moisture back out of every breath.
    /// Damaged, the body tires quickly and dries out.
    Breathing,
    /// Storing lifting gas. Punctured, it leaks, and the body sinks towards the aur.
    Lift,
    /// Storing water against dry air.
    Water,
    /// Moving the body: crawling, gripping, squeezing, steering.
    Movement,
    /// Powering wings. Torn, the animal is slow and tires.
    Flight,
    /// Taking hold of food and working it: suckers, jaws and the muscle that crushes. Damaged,
    /// the body feeds slowly.
    Handling,
    /// Holding the body's shape and springing back the work of each stroke. Snapped, the muscles
    /// pull against nothing and every stroke costs its full price.
    Support,
    /// Making and keeping young. Damaged, it cannot breed, and loses any young it carries.
    Breeding,
    /// Holding aur safely, to spend on healing and protection.
    AurStore,
    /// Secreting something spent on each use and slowly made again: acid, silk, venom, light.
    Secretion,
    /// Gathering the light, aur, wind or lightning a plant lives on.
    Gathering,
    /// Carrying sap and holding a plant together.
    Sap,
    /// Drawing water and aur from the air below.
    Drawing,
    /// Making nectar for pollinators.
    Nectar,
    /// Ripening fruit to be eaten and carry its seed away.
    Fruit,
}

/// No organ: the filler of a body's unused sites.
const NONE: Organ = Organ::new("", Function::Circulation, Vec3::ZERO);

/// The form an organ takes in place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// A rounded mass, its half extents the site's `radii`.
    Lobe,
    /// A rod from the site's centre to `end`, `radius` thick.
    Rod { end: Vec3, radius: f32 },
    /// A hoop round the body across its length, its half spans across and up the site's
    /// `radii.x` and `radii.y` and its own thickness `radii.z`.
    Hoop,
}

/// One organ in place: its kind, its centre in the body's own frame and its half extents, whether
/// a matching one lies on the other side, and the form it takes.
#[derive(Clone, Copy, Debug)]
pub struct Site {
    pub organ: Organ,
    pub centre: Vec3,
    pub radii: Vec3,
    pub paired: bool,
    pub shape: Shape,
}

impl Site {
    const fn lobe(organ: Organ, centre: Vec3, radii: Vec3, paired: bool) -> Self {
        Self {
            organ,
            centre,
            radii,
            paired,
            shape: Shape::Lobe,
        }
    }

    /// The organ's volume, counting both of a pair.
    pub fn volume(&self) -> f32 {
        let one = match self.shape {
            Shape::Lobe => self.radii.x * self.radii.y * self.radii.z * PI * 4.0 / 3.0,
            Shape::Rod { end, radius } => (end - self.centre).length() * radius * radius * PI,
            Shape::Hoop => (self.radii.x + self.radii.y) * PI * self.radii.z * self.radii.z * PI,
        };
        one * if self.paired { 2.0 } else { 1.0 }
    }

    /// The middle of the organ.
    pub fn middle(&self) -> Vec3 {
        match self.shape {
            Shape::Rod { end, .. } => (self.centre + end) * 0.5,
            _ => self.centre,
        }
    }

    /// How the cutaway draws it: where it lies and whether it is paired; its extents, or the far
    /// end of a rod and its thickness, and its glow; and its colour and form.
    pub fn encode(&self) -> [Vec4; 3] {
        let glow = if self.organ.function == Function::AurStore {
            1.5
        } else {
            0.0
        };
        let (extent, form) = match self.shape {
            Shape::Lobe => (self.radii.extend(glow), 0),
            Shape::Rod { end, radius } => (end.extend(radius), 1),
            Shape::Hoop => (self.radii.extend(0.0), 2),
        };
        let tissue = match self.organ.tissue {
            Tissue::Soft => 0,
            Tissue::Muscle => 1,
            Tissue::Spring => 2,
            Tissue::Ironbound => 3,
        };
        [
            self.centre.extend(if self.paired { 1.0 } else { 0.0 }),
            extent,
            self.organ.colour.extend((form + tissue * 3) as f32),
        ]
    }

    /// How squarely a strike reaches one of this organ, in shares of it: a lobe is harmed by how
    /// near the wound's path comes to its centre in units of its own size, so a wound must reach
    /// into a small organ to harm it but grazes a large one; a rod or hoop by how near the path
    /// comes to its core, for it snaps wherever it is cut through.
    fn reached(&self, strike: Strike, side: f32) -> f32 {
        let mirror = vec3(side, 1.0, 1.0);
        let centre = self.centre * mirror;
        match self.shape {
            Shape::Lobe => {
                let near = closest_on_segment(centre, strike.from, strike.to);
                let reach = ((near - centre) / self.radii.max(Vec3::splat(1e-3))).length();
                let tolerance = 1.0 + strike.radius / self.radii.min_element().max(1e-3);
                (1.0 - reach / tolerance).max(0.0)
            }
            Shape::Rod { end, radius } => {
                let end = end * mirror;
                let mut nearest = f32::MAX;
                for step in 0..9 {
                    let point = centre.lerp(end, step as f32 / 8.0);
                    let near = closest_on_segment(point, strike.from, strike.to);
                    nearest = nearest.min((near - point).length());
                }
                (1.0 - (nearest - strike.radius).max(0.0) / radius.max(1e-3)).max(0.0) * 0.7
            }
            Shape::Hoop => {
                let mut nearest = f32::MAX;
                for step in 0..16 {
                    let angle = step as f32 / 16.0 * TAU;
                    let point =
                        centre + vec3(angle.cos() * self.radii.x, angle.sin() * self.radii.y, 0.0);
                    let near = closest_on_segment(point, strike.from, strike.to);
                    nearest = nearest.min((near - point).length());
                }
                (1.0 - (nearest - strike.radius).max(0.0) / self.radii.z.max(1e-3)).max(0.0) * 0.5
            }
        }
    }
}

/// How many organ sites a species may have.
pub const MAX: usize = ORGAN_SLOTS;

/// A species' organs, in its body's own frame.
#[derive(Clone, Copy, Debug)]
pub struct Viscera {
    sites: [Site; MAX],
    count: usize,
}

impl Viscera {
    pub const NONE: Self = Self {
        sites: [Site::lobe(NONE, Vec3::ZERO, Vec3::ZERO, false); MAX],
        count: 0,
    };

    /// # Panics
    ///
    /// If there are more than [`MAX`] sites.
    pub const fn new(sites: &[Site]) -> Self {
        assert!(sites.len() <= MAX, "too many organ sites");
        let mut organs = Self::NONE;
        let mut index = 0;
        while index < sites.len() {
            organs.sites[index] = sites[index];
            index += 1;
        }
        organs.count = sites.len();
        organs
    }

    pub fn sites(&self) -> &[Site] {
        &self.sites[..self.count]
    }

    /// The half extents of the space the organs fill, roughly the body's.
    pub fn extent(&self) -> Vec3 {
        let mut extent = Vec3::splat(0.05);
        for site in self.sites() {
            let reach = match site.shape {
                Shape::Rod { end, radius } => site.centre.abs().max(end.abs()) + radius,
                _ => site.centre.abs() + site.radii,
            };
            extent = extent.max(reach);
        }
        extent
    }

    /// Where the first organ meeting a need lies; `None` if there is none.
    pub fn find(&self, function: Function) -> Option<Vec3> {
        self.sites()
            .iter()
            .find(|site| site.organ.function == function)
            .map(Site::middle)
    }

    pub fn has(&self, function: Function) -> bool {
        self.sites()
            .iter()
            .any(|site| site.organ.function == function)
    }

    /// How well a need is still met, from the condition of each organ meeting it weighted by its
    /// size; `None` if the body has none, so that some other part of it meets that need.
    pub fn function(&self, condition: &[f32; MAX], function: Function) -> Option<f32> {
        let mut total = 0.0;
        let mut working = 0.0;
        for (index, site) in self.sites().iter().enumerate() {
            if site.organ.function == function {
                let volume = site.volume();
                total += volume;
                working += volume * condition[index];
            }
        }
        (total > 0.0).then(|| working / total)
    }

    /// What a strike does to each organ: the share of it destroyed, from how squarely the wound's
    /// path reaches it, how hard it was dealt, and how its kind of wound takes that tissue.
    pub fn struck(&self, strike: Strike) -> [f32; MAX] {
        let mut damage = [0.0; MAX];
        for (index, site) in self.sites().iter().enumerate() {
            let mirrors = if site.paired { 2 } else { 1 };
            for mirror in 0..mirrors {
                let side = if mirror == 0 { 1.0 } else { -1.0 };
                let share = site.reached(strike, side) / mirrors as f32;
                damage[index] += share * strike.force * strike.injury.severity(site.organ);
            }
        }
        damage
    }

    /// Room for one more site, if there is any.
    #[cfg(not(target_arch = "spirv"))]
    fn push(&mut self, site: Site) {
        assert!(self.count < MAX, "too many organ sites");
        self.sites[self.count] = site;
        self.count += 1;
    }
}

/// Where a part of a body plan lies, in terms of the body it is in.
///
/// It is not in any one species' measurements, so a lineage can declare its anatomy once and every
/// descendant carries it in its own build.
///
/// A place in the trunk is a `Vec3` of how far `along` it, from -1 at its front to 1 at its back;
/// how high in its cross-section, from -1 at the belly to 1 at the back; and how far `out`, from 0
/// on the midline to 1 at the flank, any part set out from the midline being one of a pair.
/// Sizes are shares of the cross-section's half width and height and of the trunk's half length.
#[derive(Clone, Copy, Debug)]
pub enum Place {
    /// A rounded mass at a place in the trunk.
    Lobe { at: Vec3, size: Vec3 },
    /// A pair of masses at the roots of wing pair `wing`, inside the shoulders.
    Shoulder { wing: usize, size: Vec3 },
    /// A rod from one place in the trunk to another, following its curve, `thickness` a share of
    /// the cross-section.
    Rod {
        from: Vec3,
        to: Vec3,
        thickness: f32,
    },
    /// A hoop round the trunk `along` it.
    Hoop { along: f32, thickness: f32 },
    /// A hoop round the trunk at the roots of wing pair `wing`, where the flight muscles pull.
    Girdle { wing: usize, thickness: f32 },
    /// Exactly where a species' own design puts it.
    At {
        centre: Vec3,
        radii: Vec3,
        paired: bool,
    },
    /// Lost in this lineage.
    Lost,
}

/// One part of a body plan, declared in the taxon where it arose and carried by every taxon below
/// it, which may change or lose it: a part `replacing` an organ takes that organ's place.
#[derive(Clone, Copy, Debug)]
pub struct Part {
    pub organ: Organ,
    pub place: Place,
    pub replaces: Option<&'static str>,
}

impl Part {
    const fn new(organ: Organ, place: Place) -> Self {
        Self {
            organ,
            place,
            replaces: None,
        }
    }

    pub const fn lobe(organ: Organ, at: Vec3, size: Vec3) -> Self {
        Self::new(organ, Place::Lobe { at, size })
    }

    pub const fn shoulder(organ: Organ, wing: usize, size: Vec3) -> Self {
        Self::new(organ, Place::Shoulder { wing, size })
    }

    pub const fn rod(organ: Organ, from: Vec3, to: Vec3, thickness: f32) -> Self {
        Self::new(
            organ,
            Place::Rod {
                from,
                to,
                thickness,
            },
        )
    }

    pub const fn hoop(organ: Organ, along: f32, thickness: f32) -> Self {
        Self::new(organ, Place::Hoop { along, thickness })
    }

    pub const fn girdle(organ: Organ, wing: usize, thickness: f32) -> Self {
        Self::new(organ, Place::Girdle { wing, thickness })
    }

    /// An organ the lineage no longer has.
    pub const fn lost(organ: Organ) -> Self {
        Self {
            organ,
            place: Place::Lost,
            replaces: Some(organ.name),
        }
    }

    /// This part, in the place of every part of an ancestor's organ.
    pub const fn replacing(self, organ: Organ) -> Self {
        Self {
            replaces: Some(organ.name),
            ..self
        }
    }
}

/// One organ exactly where a species' design puts it, `radii` in half extents.
pub const fn site(organ: Organ, centre: Vec3, radii: Vec3) -> Part {
    Part::new(
        organ,
        Place::At {
            centre,
            radii,
            paired: false,
        },
    )
}

/// A pair of organs mirrored across the body, one at `centre` and the other at its reflection.
pub const fn pair(organ: Organ, centre: Vec3, radii: Vec3) -> Part {
    Part::new(
        organ,
        Place::At {
            centre,
            radii,
            paired: true,
        },
    )
}

/// A species' organs: its lineage's body plan, root to species, laid out in its own body.
///
/// Each species' anatomy is worked out the first time it is asked for and kept.
#[cfg(not(target_arch = "spirv"))]
pub fn anatomy(kind: Kind) -> &'static Viscera {
    static PACKED: OnceLock<Vec<OnceLock<Viscera>>> = OnceLock::new();
    PACKED.get_or_init(|| KINDS.iter().map(|_| OnceLock::new()).collect())[kind as usize]
        .get_or_init(|| pack(kind))
}

/// Work out every species' anatomy at once, in parallel, so that none is worked out mid-step.
#[cfg(not(target_arch = "spirv"))]
pub fn prepare() {
    thread::scope(|scope| {
        for &kind in KINDS {
            scope.spawn(move || anatomy(kind));
        }
    });
}

/// The body plan a species inherits, root first, with every change its lineage made on the way
/// down.
#[cfg(not(target_arch = "spirv"))]
pub fn plan(kind: Kind) -> Vec<Part> {
    let mut lineage = Vec::new();
    let mut taxon = kind.taxon().parent;
    while let Some(current) = taxon {
        lineage.push(current.anatomy);
        taxon = current.parent;
    }
    let mut parts: Vec<Part> = Vec::new();
    for declared in lineage.iter().rev().chain([&kind.parts()]) {
        for part in *declared {
            let Some(name) = part.replaces else {
                parts.push(*part);
                continue;
            };
            let first = parts.iter().position(|old| old.organ.name == name);
            parts.retain(|old| old.organ.name != name);
            if !matches!(part.place, Place::Lost) {
                parts.insert(first.unwrap_or(parts.len()).min(parts.len()), *part);
            }
        }
    }
    parts
}

/// One cross-section of a body: how much of it there is, where its middle lies and how far it
/// spreads across and up from there, and a point surely inside it.
#[cfg(not(target_arch = "spirv"))]
#[derive(Clone, Copy, Debug, Default)]
struct Section {
    area: f32,
    middle: f32,
    width: f32,
    height: f32,
    core: Vec3,
}

/// A body measured in cross-sections along its length, the trunk being the run of them stout
/// enough to hold organs, which leaves out thin tails, tentacles and frills.
#[cfg(not(target_arch = "spirv"))]
struct Trunk {
    sections: Vec<Section>,
    /// The stoutest section.
    girth: Section,
    start: f32,
    step: f32,
    front: f32,
    back: f32,
}

#[cfg(not(target_arch = "spirv"))]
impl Trunk {
    fn measure(body: &dyn Fn(Vec3) -> f32, reach: f32) -> Self {
        // Find the body's bounds coarsely, counting anywhere within half a cell of its surface.
        let cells = 28;
        let cell = reach * 2.0 / cells as f32;
        let mut low = Vec3::splat(f32::MAX);
        let mut high = Vec3::splat(f32::MIN);
        for i in 0..cells / 2 {
            for j in 0..cells {
                for k in 0..cells {
                    let p = vec3(
                        (i as f32 + 0.5) * cell,
                        -reach + (j as f32 + 0.5) * cell,
                        -reach + (k as f32 + 0.5) * cell,
                    );
                    if body(p) < cell * 0.9 {
                        low = low.min(p);
                        high = high.max(p);
                    }
                }
            }
        }
        if low.x > high.x {
            low = Vec3::splat(-0.1);
            high = Vec3::splat(0.1);
        }
        let low = (low - Vec3::splat(cell)).max(Vec3::splat(-reach)) * vec3(0.0, 1.0, 1.0);
        let high = (high + Vec3::splat(cell)).min(Vec3::splat(reach));
        // Then measure it slice by slice.
        let slices = 56;
        let (across, up) = (16, 32);
        let step = (high.z - low.z) / slices as f32;
        let (dx, dy) = (high.x / across as f32, (high.y - low.y) / up as f32);
        let mut sections = Vec::with_capacity(slices);
        for k in 0..slices {
            let z = low.z + (k as f32 + 0.5) * step;
            let (mut count, mut sum_y, mut sum_xx, mut sum_yy) = (0.0f32, 0.0, 0.0, 0.0);
            let mut deepest = (vec3(0.0, 0.0, z), f32::MAX);
            for i in 0..across {
                for j in 0..up {
                    let p = vec3((i as f32 + 0.5) * dx, low.y + (j as f32 + 0.5) * dy, z);
                    let distance = body(p);
                    if distance < 0.0 {
                        count += 1.0;
                        sum_y += p.y;
                        sum_xx += p.x * p.x;
                        sum_yy += p.y * p.y;
                    }
                    if distance < deepest.1 {
                        deepest = (p, distance);
                    }
                }
            }
            // The deepest point on the midline, which organs are drawn towards to fit, or, where
            // the body parts round the midline, the deepest point of all.
            let core = (0..up)
                .map(|j| vec3(0.0, low.y + (j as f32 + 0.5) * dy, z))
                .map(|p| (p, body(p)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .filter(|(_, distance)| *distance < 0.0)
                .unwrap_or(deepest)
                .0;
            let section = if count > 0.0 {
                let middle = sum_y / count;
                // A filled ellipse's mean square spread is a quarter of its half extent squared.
                Section {
                    area: count * dx * dy * 2.0,
                    middle,
                    width: (sum_xx / count).sqrt() * 2.0,
                    height: middle.mul_add(-middle, sum_yy / count).max(0.0).sqrt() * 2.0,
                    core,
                }
            } else {
                Section::default()
            };
            sections.push(section);
        }
        let largest = (0..slices)
            .max_by(|&a, &b| sections[a].area.total_cmp(&sections[b].area))
            .unwrap_or(0);
        let stout = sections[largest].area * 0.12;
        let (mut first, mut last) = (largest, largest);
        while first > 0 && sections[first - 1].area >= stout {
            first -= 1;
        }
        while last + 1 < slices && sections[last + 1].area >= stout {
            last += 1;
        }
        let start = low.z;
        Self {
            girth: sections[largest],
            sections,
            start,
            step,
            front: start + first as f32 * step,
            back: start + (last + 1) as f32 * step,
        }
    }

    fn half_length(&self) -> f32 {
        (self.back - self.front) * 0.5
    }

    fn z(&self, along: f32) -> f32 {
        f32::midpoint(self.front, self.back) + along.clamp(-1.0, 1.0) * self.half_length()
    }

    /// The trunk's cross-section at `z`, between its slices.
    fn section(&self, z: f32) -> Section {
        let z = z.clamp(self.front + self.step * 0.5, self.back - self.step * 0.5);
        let at = ((z - self.start) / self.step - 0.5).max(0.0);
        let index = (at as usize).min(self.sections.len() - 1);
        let next = (index + 1).min(self.sections.len() - 1);
        let (a, b) = (self.sections[index], self.sections[next]);
        if b.area <= 0.0 {
            return a;
        }
        if a.area <= 0.0 {
            return b;
        }
        let t = at - index as f32;
        Section {
            area: a.area + (b.area - a.area) * t,
            middle: a.middle + (b.middle - a.middle) * t,
            width: a.width + (b.width - a.width) * t,
            height: a.height + (b.height - a.height) * t,
            core: if t < 0.5 { a.core } else { b.core },
        }
    }

    /// A place in the trunk, in the body's own frame.
    fn locate(&self, at: Vec3) -> Vec3 {
        let z = self.z(at.x);
        let section = self.section(z);
        vec3(
            at.z * section.width,
            section.middle + at.y * section.height,
            z,
        )
    }

    /// A part's half extents, from its shares of the trunk where it lies, though never less than
    /// half what they would be at its stoutest, so a part declared where the body narrows is sized
    /// for the body and moves to where it fits rather than shrinking to a point.
    fn extent(&self, z: f32, size: Vec3) -> Vec3 {
        let section = self.section(z);
        vec3(
            size.x * section.width.max(self.girth.width * 0.5),
            size.y * section.height.max(self.girth.height * 0.5),
            size.z * self.half_length(),
        )
        .max(Vec3::splat(1e-3))
    }
}

/// Directions to test a mass's fit in, over its whole surface.
#[cfg(not(target_arch = "spirv"))]
fn around() -> impl Iterator<Item = Vec3> {
    (0..27)
        .filter(|&index| index != 13)
        .map(|index| {
            vec3(
                (index % 3) as f32 - 1.0,
                (index / 3 % 3) as f32 - 1.0,
                (index / 9) as f32 - 1.0,
            )
        })
        .map(Vec3::normalize)
}

/// Whether a point lies within a placed mass, or its mirror.
#[cfg(not(target_arch = "spirv"))]
fn within(point: Vec3, site: &Site) -> Option<Vec3> {
    let mirrors: &[f32] = if site.paired { &[1.0, -1.0] } else { &[1.0] };
    mirrors.iter().find_map(|&side| {
        let centre = site.centre * vec3(side, 1.0, 1.0);
        (((point - centre) / site.radii).length() < 1.0).then_some(centre)
    })
}

/// Fit a mass into the body: shrink it a little at a time and draw it towards the trunk's core
/// and its middle until it lies wholly inside, easing it away from the masses already placed, so
/// that a part declared where the body is thin moves to where there is room for it rather than
/// shrinking to nothing.
#[cfg(not(target_arch = "spirv"))]
fn settle(body: &dyn Fn(Vec3) -> f32, trunk: &Trunk, mut site: Site, placed: &[Site]) -> Site {
    let margin = site.radii.min_element() * 0.15;
    let middle = f32::midpoint(trunk.front, trunk.back);
    let sides: &[f32] = if site.paired { &[1.0, -1.0] } else { &[1.0] };
    for _ in 0..80 {
        // Bodies are not quite symmetric, so each of a pair is fitted on its own side.
        let fits = |point: Vec3| {
            sides
                .iter()
                .all(|&side| body(point * vec3(side, 1.0, 1.0)) <= -margin)
        };
        let mut outside = !fits(site.centre);
        let mut away = Vec3::ZERO;
        for direction in around() {
            outside |= !fits(site.centre + direction * site.radii);
            for other in placed.iter().filter(|other| other.shape == Shape::Lobe) {
                for &side in sides {
                    let mine = site.centre * vec3(side, 1.0, 1.0);
                    let probe = Site {
                        centre: mine,
                        ..site
                    };
                    let theirs = other.centre + direction * other.radii;
                    let theirs_mirrored =
                        other.centre * vec3(-1.0, 1.0, 1.0) + direction * other.radii;
                    if let Some(centre) = within(mine + direction * site.radii, other) {
                        away += (mine - centre) * vec3(side, 1.0, 1.0);
                    }
                    if within(theirs, &probe).is_some()
                        || (other.paired && within(theirs_mirrored, &probe).is_some())
                    {
                        away += (mine - other.centre) * vec3(side, 1.0, 1.0);
                    }
                }
            }
        }
        // A single organ stays on the midline, and the two of a pair must not meet there.
        if !site.paired {
            away.x = 0.0;
        }
        let crossing = site.paired && site.centre.x < site.radii.x * 1.05;
        if !outside && away.length_squared() < 1e-12 && !crossing {
            break;
        }
        if outside {
            let core = trunk.section(site.centre.z).core;
            site.centre = site.centre.lerp(vec3(site.centre.x, core.y, core.z), 0.08);
            site.centre.z += (middle - site.centre.z) * 0.02;
        }
        if away.length_squared() >= 1e-12 {
            site.centre += away.normalize() * site.radii.min_element() * 0.12;
        }
        if crossing {
            site.radii.x = site.radii.x.min(site.centre.x / 1.05).max(1e-3);
        }
        site.radii *= 0.97;
    }
    if !sides
        .iter()
        .all(|&side| body(site.centre * vec3(side, 1.0, 1.0)) < 0.0)
    {
        let core = trunk.section(site.centre.z).core;
        site.centre = if site.paired {
            vec3(site.radii.x * 1.05, core.y, core.z)
        } else {
            core
        };
    }
    site
}

/// Lay out a species' body plan in its own body, measured from the shape it grows into.
#[cfg(not(target_arch = "spirv"))]
#[expect(
    clippy::too_many_lines,
    reason = "each kind of place is laid out in turn, in one pass over the plan"
)]
fn pack(kind: Kind) -> Viscera {
    let index = KINDS.iter().position(|&other| other == kind).unwrap_or(0) as u32;
    let specimen = Drawn::specimen(
        Vec4::new(0.0, 0.0, 0.0, kind.render_radius()),
        Vec4::W,
        Vec4::ZERO,
        index,
        kind.biology().maturity_days * 1.25,
        0,
    );
    let body = |point: Vec3| specimen.surface(point).distance;
    let parts = plan(kind);
    let trunk = Trunk::measure(&body, kind.render_radius().max(0.1));
    let look = kind.look();
    let wing = |index: usize| {
        // A lineage drawn without wing sheets of its own, as the Næ's are, or whose wings are
        // fused into its body, as a Grave's are, still keeps the muscles where the pairs once
        // rooted.
        if look.wings[index].x > 0.0 {
            (look.hinges[index], look.wings[index].z)
        } else {
            let z = trunk.z(if index == 0 { -0.3 } else { 0.3 });
            let section = trunk.section(z);
            (
                vec2(section.width, section.middle + section.height * 0.5),
                z,
            )
        }
    };
    let mut sites: Vec<(usize, Site)> = Vec::new();
    let placed =
        |sites: &Vec<(usize, Site)>| sites.iter().map(|(_, site)| *site).collect::<Vec<_>>();
    // Fixed parts first, then frames, which run through everything, then the masses, the largest
    // first so the small fit round them.
    let mut order: Vec<usize> = (0..parts.len()).collect();
    let rank = |part: &Part| match part.place {
        Place::At { .. } => (0, 0.0),
        Place::Rod { .. } | Place::Hoop { .. } | Place::Girdle { .. } => (1, 0.0),
        Place::Lobe { size, .. } | Place::Shoulder { size, .. } => (2, -(size.x * size.y * size.z)),
        Place::Lost => (3, 0.0),
    };
    order.sort_by(|&a, &b| {
        let (ra, rb) = (rank(&parts[a]), rank(&parts[b]));
        ra.0.cmp(&rb.0).then(ra.1.total_cmp(&rb.1))
    });
    for index in order {
        let part = parts[index];
        let organ = part.organ;
        match part.place {
            Place::At {
                centre,
                radii,
                paired,
            } => sites.push((index, Site::lobe(organ, centre, radii, paired))),
            Place::Lobe { at, size } => {
                let centre = trunk.locate(at);
                let site = Site::lobe(organ, centre, trunk.extent(centre.z, size), at.z > 0.0);
                sites.push((index, settle(&body, &trunk, site, &placed(&sites))));
            }
            Place::Shoulder { wing: pair, size } => {
                let (hinge, z) = wing(pair);
                let section = trunk.section(z);
                let centre = vec3(
                    hinge.x.min(section.width) * 0.5,
                    section.middle + (hinge.y - section.middle) * 0.45,
                    z,
                );
                let site = Site::lobe(organ, centre, trunk.extent(z, size), true);
                sites.push((index, settle(&body, &trunk, site, &placed(&sites))));
            }
            Place::Rod {
                from,
                to,
                thickness,
            } => {
                let middle = trunk.locate((from + to) * 0.5);
                let section = trunk.section(middle.z);
                let radius = (section
                    .width
                    .max(trunk.girth.width * 0.5)
                    .min(section.height.max(trunk.girth.height * 0.5))
                    * thickness)
                    .max(1e-3);
                // Each end and the middle drawn inside the body, so the rod follows its curve.
                let inward = |point: Vec3| {
                    let core = trunk.section(point.z).core;
                    let mut point = point;
                    for _ in 0..60 {
                        if body(point) < -radius * 1.5
                            && body(point * vec3(-1.0, 1.0, 1.0)) < -radius * 1.5
                        {
                            break;
                        }
                        point = point.lerp(vec3(point.x, core.y, core.z), 0.12);
                        point.x *= 0.9;
                    }
                    if body(point) >= -radius {
                        point = core;
                    }
                    point
                };
                let mut points = [
                    inward(trunk.locate(from)),
                    inward(middle),
                    inward(trunk.locate(to)),
                ];
                // Where the body is not symmetric, as round a curled tail, the ends of a pair of
                // rods are drawn back towards the middle until both lie inside it.
                let paired = from.z > 0.0;
                let both = |point: Vec3| {
                    body(point) < 0.0 && (!paired || body(point * vec3(-1.0, 1.0, 1.0)) < 0.0)
                };
                for end in [0, 2] {
                    for _ in 0..20 {
                        if both(points[end]) {
                            break;
                        }
                        points[end] = points[end].lerp(points[1], 0.15);
                    }
                }
                for segment in 0..2 {
                    let site = Site {
                        organ,
                        centre: points[segment],
                        radii: Vec3::splat(radius),
                        paired,
                        shape: Shape::Rod {
                            end: points[segment + 1],
                            radius,
                        },
                    };
                    sites.push((index, site));
                }
            }
            Place::Hoop { .. } | Place::Girdle { .. } => {
                let (z, thickness) = match part.place {
                    Place::Hoop { along, thickness } => (trunk.z(along), thickness),
                    Place::Girdle {
                        wing: pair,
                        thickness,
                    } => (wing(pair).1, thickness),
                    _ => continue,
                };
                let section = trunk.section(z);
                let tube = (section.width.min(section.height) * thickness).max(1e-3);
                let centre = section.core;
                let mut span = vec3(section.width, section.height, 0.0);
                // Shrink the hoop until it runs wholly inside the body wall.
                for _ in 0..40 {
                    let outside = (0..16).any(|step| {
                        let angle = step as f32 / 16.0 * TAU;
                        body(centre + vec3(angle.cos() * span.x, angle.sin() * span.y, 0.0))
                            > -tube * 1.5
                    });
                    if !outside {
                        break;
                    }
                    span *= 0.94;
                }
                let site = Site {
                    organ,
                    centre,
                    radii: vec3(span.x, span.y, tube),
                    paired: false,
                    shape: Shape::Hoop,
                };
                sites.push((index, site));
            }
            Place::Lost => {}
        }
    }
    // Kept in the order the plan declares them.
    sites.sort_by_key(|(index, _)| *index);
    let mut viscera = Viscera::NONE;
    for (_, site) in sites {
        viscera.push(site);
    }
    viscera
}

/// How each kind of wound harms the body, which only the simulation asks.
impl Injury {
    /// How much this kind of wound harms an organ: a burn ruins the delicate breathing folds
    /// and gas stores near the skin but barely touches muscle, and a drain harms nothing but
    /// takes fluid.
    const fn severity(self, organ: Organ) -> f32 {
        use Function::{
            Breathing, Circulation, Coordination, Flight, Gathering, Lift, Movement, Nectar, Sap,
        };
        match (self, organ.tissue) {
            (Self::Drain, _) => return 0.0,
            // A spring frame gives under a cut or a thrust but snaps under a crushing blow, and
            // laced with iron it turns most of what it does not stop.
            (Self::Crush, Tissue::Spring | Tissue::Ironbound) => return 1.6,
            (_, Tissue::Ironbound) => return 0.3,
            (_, Tissue::Spring) => return 0.6,
            // Muscle parts cleanly along a cut.
            (Self::Slash, Tissue::Muscle) => return 1.3,
            _ => {}
        }
        match (self, organ.function) {
            (Self::Burn, Breathing | Lift | Gathering | Nectar) => 1.5,
            (Self::Burn, Movement | Flight | Sap) => 0.4,
            (Self::Slash | Self::Tear, Lift | Gathering) => 1.4,
            (Self::Pierce, Circulation | Coordination) => 1.6,
            _ => 1.0,
        }
    }

    /// How freely a wound of this kind bleeds, for its size.
    pub const fn bleeding(self) -> f32 {
        match self {
            Self::Slash => 1.4,
            Self::Pierce => 1.0,
            Self::Bite => 1.2,
            Self::Tear => 1.3,
            Self::Burn => 0.3,
            Self::Crush => 0.2,
            Self::Drain => 0.15,
        }
    }
}

/// A wound's path through a body, in its own frame: from where it entered to where it stopped, how
/// wide, how hard, and of what kind.
#[derive(Clone, Copy, Debug)]
pub struct Strike {
    pub from: Vec3,
    pub to: Vec3,
    pub radius: f32,
    pub force: f32,
    pub injury: Injury,
}

fn closest_on_segment(point: Vec3, from: Vec3, to: Vec3) -> Vec3 {
    let axis = to - from;
    let t = ((point - from).dot(axis) / axis.length_squared().max(1e-8)).clamp(0.0, 1.0);
    from + axis * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::life::{Drawn, KINDS};
    use spirv_std::glam::{Vec4, vec3};

    #[test]
    fn every_organ_lies_inside_its_body() {
        for (index, kind) in KINDS.iter().enumerate() {
            let biology = kind.biology();
            let c = Drawn::specimen(
                Vec4::new(0.0, 0.0, 0.0, kind.render_radius()),
                Vec4::W,
                Vec4::ZERO,
                index as u32,
                biology.maturity_days * 1.25,
                0,
            );
            for site in kind.organs().sites() {
                let sides: &[f32] = if site.paired { &[1.0, -1.0] } else { &[1.0] };
                // A rod is drawn only within the body, so only its ends must lie inside it.
                let points = match site.shape {
                    Shape::Rod { end, .. } => vec![site.centre, end],
                    _ => vec![site.centre],
                };
                for (&side, point) in sides
                    .iter()
                    .flat_map(|side| points.iter().map(move |point| (side, point)))
                {
                    let centre = *point * vec3(side, 1.0, 1.0);
                    let inside = c.surface(centre).distance;
                    assert!(
                        inside < 0.0,
                        "{kind:?}'s {} at {centre} lies outside its body by {inside}",
                        site.organ.name
                    );
                }
            }
        }
    }

    #[test]
    fn a_lance_through_the_pump_destroys_it_and_spares_the_far_side() {
        let pump = Organ::new("pump", Function::Circulation, Vec3::ONE);
        let folds = Organ::new("breathing folds", Function::Breathing, Vec3::ONE);
        let organs = Viscera::new(&[
            Site::lobe(pump, vec3(0.0, 0.0, 0.0), Vec3::splat(0.5), false),
            Site::lobe(folds, vec3(1.0, 0.0, 0.0), Vec3::splat(0.3), true),
        ]);
        let damage = organs.struck(Strike {
            from: vec3(0.0, 3.0, 0.0),
            to: vec3(0.0, -3.0, 0.0),
            radius: 0.1,
            force: 1.0,
            injury: Injury::Pierce,
        });
        assert!(damage[0] > 1.0, "the lance missed the pump: {damage:?}");
        assert!(damage[1] < 1e-6, "the lance reached the breathing folds");
        let mut condition = [1.0; MAX];
        condition[1] = 0.0;
        assert_eq!(organs.function(&condition, Function::Breathing), Some(0.0));
        assert!(organs.function(&condition, Function::Digestion).is_none());
    }
}
