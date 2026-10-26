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

use spirv_std::glam::Vec3;

/// An organ, declared in the taxon where it arose: its name, the need it meets, and the colour it
/// is drawn in when the body is opened.
///
/// Organs that meet the same need in different lineages are different organs, as the separately
/// evolved flight muscles of the tetralates and the winged blood-feeders are, and the body's
/// workings ask only how well each need is met.
#[derive(Clone, Copy, Debug)]
pub struct Organ {
    pub name: &'static str,
    pub function: Function,
    pub colour: Vec3,
}

impl Organ {
    pub const fn new(name: &'static str, function: Function, colour: Vec3) -> Self {
        Self {
            name,
            function,
            colour,
        }
    }
}

impl PartialEq for Organ {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
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
    /// Moving the body: crawling, gripping, squeezing.
    Movement,
    /// Powering wings. Torn, the animal is slow and tires.
    Flight,
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

/// One organ in place: its kind, its centre in the body's own frame and its half extents, and
/// whether a matching one lies on the other side.
#[derive(Clone, Copy, Debug)]
pub struct Site {
    pub organ: Organ,
    pub centre: Vec3,
    pub radii: Vec3,
    pub paired: bool,
}

/// One organ at `centre`, `radii` in half extents.
pub const fn site(organ: Organ, centre: Vec3, radii: Vec3) -> Site {
    Site {
        organ,
        centre,
        radii,
        paired: false,
    }
}

/// A pair of organs mirrored across the body, one at `centre` and the other at its reflection.
pub const fn pair(organ: Organ, centre: Vec3, radii: Vec3) -> Site {
    Site {
        organ,
        centre,
        radii,
        paired: true,
    }
}

/// How many organ sites a species may have.
pub const MAX: usize = 12;

/// A species' organs, in its body's own frame.
#[derive(Clone, Copy, Debug)]
pub struct Viscera {
    sites: [Site; MAX],
    count: usize,
}

impl Viscera {
    pub const NONE: Self = Self {
        sites: [site(NONE, Vec3::ZERO, Vec3::ZERO); MAX],
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
            extent = extent.max(site.centre.abs() + site.radii);
        }
        extent
    }

    /// Where the first organ meeting a need lies; `None` if there is none.
    pub fn find(&self, function: Function) -> Option<Vec3> {
        self.sites()
            .iter()
            .find(|site| site.organ.function == function)
            .map(|site| site.centre)
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

    /// What a strike does to each organ: the share of it destroyed, from how near the wound's path
    /// passes to its centre and how hard it was dealt.
    pub fn struck(&self, strike: Strike) -> [f32; MAX] {
        let mut damage = [0.0; MAX];
        for (index, site) in self.sites().iter().enumerate() {
            let mirrors = if site.paired { 2 } else { 1 };
            for mirror in 0..mirrors {
                let centre = if mirror == 0 {
                    site.centre
                } else {
                    site.centre * Vec3::new(-1.0, 1.0, 1.0)
                };
                let near = closest_on_segment(centre, strike.from, strike.to);
                // In units of the organ's own size, so a wound must reach into a small organ to
                // harm it but grazes a large one.
                let reach = ((near - centre) / site.radii.max(Vec3::splat(1e-3))).length();
                let tolerance = 1.0 + strike.radius / site.radii.min_element().max(1e-3);
                let share = (1.0 - reach / tolerance).max(0.0) / mirrors as f32;
                damage[index] += share * strike.force * strike.wound.severity(site.organ);
            }
        }
        damage
    }
}

impl Site {
    /// The organ's volume, counting both of a pair.
    pub fn volume(&self) -> f32 {
        self.radii.x * self.radii.y * self.radii.z * if self.paired { 2.0 } else { 1.0 }
    }
}

/// The ways a body can be wounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wound {
    /// A lance or bite driven deep along a line.
    Pierce,
    /// A blade drawn across the body: long and shallow.
    Slash,
    /// Acid eating into the surface.
    Burn,
    /// Jaws or a rasp tearing out a mouthful.
    Bite,
    /// A parasite's sucker drawing fluid, leaving the organs whole.
    Drain,
}

impl Wound {
    /// How much this kind of wound harms an organ: a burn ruins the delicate breathing folds
    /// and gas stores near the skin but barely touches muscle, and a drain harms nothing but
    /// takes fluid.
    const fn severity(self, organ: Organ) -> f32 {
        use Function::{
            Breathing, Circulation, Coordination, Flight, Gathering, Lift, Movement, Nectar, Sap,
        };
        match (self, organ.function) {
            (Self::Drain, _) => 0.0,
            (Self::Burn, Breathing | Lift | Gathering | Nectar) => 1.5,
            (Self::Burn, Movement | Flight | Sap) => 0.4,
            (Self::Slash, Lift | Gathering) => 1.4,
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
            Self::Burn => 0.3,
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
    pub wound: Wound,
}

fn closest_on_segment(point: Vec3, from: Vec3, to: Vec3) -> Vec3 {
    let axis = to - from;
    let t = ((point - from).dot(axis) / axis.length_squared().max(1e-8)).clamp(0.0, 1.0);
    from + axis * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::life::{CreatureInstance, KINDS, body_surface};
    use spirv_std::glam::{Vec4, vec3};

    #[test]
    fn every_organ_lies_inside_its_body() {
        for (index, kind) in KINDS.iter().enumerate() {
            let biology = kind.biology();
            let c = CreatureInstance::specimen(
                Vec4::new(0.0, 0.0, 0.0, kind.render_radius()),
                Vec4::W,
                Vec4::ZERO,
                index as u32,
                biology.maturity_days * 1.25,
                0,
            );
            for site in kind.organs().sites() {
                let sides: &[f32] = if site.paired { &[1.0, -1.0] } else { &[1.0] };
                for &side in sides {
                    let centre = site.centre * vec3(side, 1.0, 1.0);
                    let inside = body_surface(centre, &c).distance;
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
            site(pump, vec3(0.0, 0.0, 0.0), Vec3::splat(0.5)),
            pair(folds, vec3(1.0, 0.0, 0.0), Vec3::splat(0.3)),
        ]);
        let damage = organs.struck(Strike {
            from: vec3(0.0, 3.0, 0.0),
            to: vec3(0.0, -3.0, 0.0),
            radius: 0.1,
            force: 1.0,
            wound: Wound::Pierce,
        });
        assert!(damage[0] > 1.0, "the lance missed the pump: {damage:?}");
        assert!(damage[1] < 1e-6, "the lance reached the breathing folds");
        let mut condition = [1.0; MAX];
        condition[1] = 0.0;
        assert_eq!(organs.function(&condition, Function::Breathing), Some(0.0));
        assert!(organs.function(&condition, Function::Digestion).is_none());
    }
}
