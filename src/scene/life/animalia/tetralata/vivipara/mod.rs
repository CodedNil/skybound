pub use super::render_wings;
use crate::scene::life::{biology::Biology, genome::Gene};
pub mod ferrilaminata;
pub mod grandivora;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 12.0,
    ..super::BIOLOGY
};

/// What the live-bearers are born with beyond the tetralates': how devotedly they tend their young,
/// and how freely they call.
pub const DEVOTION: Gene = Gene(26);
pub const CHATTINESS: Gene = Gene(27);

#[cfg(not(target_arch = "spirv"))]
pub use nature::{
    CALL, ETHOLOGY, PHYSIOLOGY, SOCIALISE, YOUNG, herd, keeps_watch, notice, sociable, song,
};

/// The three pairs of mouth tentacles the live-bearers grew from the ancestral one, to handle
/// food without losing their grip or their contact with companions.
pub const ORAL_PAIRS: u32 = 3;

taxon!(
    Class,
    "Vivipara",
    "Live Bearers",
    [
        "Members of this class bear live young within a protected internal brood chamber.",
        "Three pairs of mouth tentacles evolved from the ancestral tetralate pair, allowing food handling without sacrificing a secure grip or close contact with companions.",
        "Expanded respiratory folds and moisture-recovery chambers support larger bodies and sustained flight.",
        "Gas-storage compartments remain separate from the breathing passages, providing buoyancy without interrupting oxygen exchange.",
        "Prolonged development favours social recognition, vocal communication and moving nursery groups that protect young without a fixed nesting surface.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The live-bearers took to herding. Their young are few and long in growing, so they travel
    //! together, keep track of one another by call, and court by song that carries across open sky.
    //! A resting herd sleeps together, but one in five keeps watch, the watch passing round.

    use crate::scene::life::{
        animalia::{
            CURIOSITY, SOCIABILITY, VIGILANCE,
            mind::{Ethology, Glimpse, Intent, Interest, Mind, Perception, Situation, Urges},
            nature::{COURT, SLEEP},
            physiology::{Brood, Physiology},
            tetralata::{ETHOLOGY as TETRALATE, FOLLOW, PLAY, PLAYFULNESS, shade},
        },
        behaviour::{Activity, Goal},
    };
    use spirv_std::glam::Vec3;

    pub const PHYSIOLOGY: Physiology = Physiology {
        brood: Brood::Live,
        senses: 700.0,
        flocks: true,
        // Song carries across open sky.
        voice: 3000.0,
        form: |_, _| 0,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges: herd,
        court: song,
        sleep: raft,
        act,
        notice,
        clings: false,
        ..TETRALATE
    };

    /// The call of one of its kind, far off.
    pub const CALL: Interest = Interest("call");
    /// Young of its kind, which a live-bearer, having so few, watches over.
    pub const YOUNG: Interest = Interest("young");

    /// Grooming, greeting and keeping close to its fellows.
    pub const SOCIALISE: Intent = Intent {
        name: "socialise",
        act: |_, s| {
            Goal::new(
                Activity::Groom,
                s.seen.kin.map(|kin| kin - kin.normalize_or_zero() * 2.0),
                0.4,
            )
            .hovering()
        },
    };

    /// A live-bearer hears its kind calling, and keeps track of the young among them.
    pub fn notice(glimpse: &Glimpse, seen: &mut Perception) {
        (TETRALATE.notice)(glimpse, seen);
        if glimpse.other == glimpse.kind {
            if glimpse.heard {
                seen.note(CALL, glimpse.offset);
            } else if glimpse.grown < 0.7 {
                seen.note(YOUNG, glimpse.offset);
            }
        }
    }

    /// A herd animal follows a call from beyond sight as readily as the herd it can see.
    fn act(mind: &mut Mind, s: &Situation) -> Option<Goal> {
        if mind.intent == FOLLOW {
            let call = s.seen.mark(CALL);
            return Some(Goal::new(
                Activity::Drift,
                call.or(s.seen.kin).map(|kin| kin + s.seen.heading * 4.0),
                0.5,
            ));
        }
        (TETRALATE.act)(mind, s)
    }

    pub fn herd(s: &Situation, urges: &mut Urges) {
        (TETRALATE.urges)(s, urges);
        // Following the herd matters less once among it.
        if let Some(kin) = s.seen.kin {
            let apart = (kin.length() / 30.0).min(1.0);
            urges.add(FOLLOW, (0.2 + s.temper(SOCIABILITY) * 0.3) * apart);
        }
        if s.seen.mark(CALL).is_some() {
            urges.add(FOLLOW, 0.25 + s.drives.desire * 0.4);
        }
        // Song draws a mate from beyond sight.
        if s.seen.mate.is_none() && s.drives.desire > 0.6 {
            urges.add(COURT, s.drives.desire);
        }
        if keeps_watch(s) {
            urges.scale(SLEEP, 0.28);
        }
        sociable(s, urges, 1.0);
    }

    /// Play and company, which live-bearers keep up all their lives, hunters as much as grazers.
    ///
    /// The young chase and mock-fight, the grown play when fed and unafraid, and all of them groom
    /// and greet their fellows. `wits` scales it, for the cleverest lineages play the most.
    pub fn sociable(s: &Situation, urges: &mut Urges, wits: f32) {
        if s.seen.kin_count == 0 {
            return;
        }
        let content = (1.0 - s.drives.hunger) * (1.0 - s.fear).max(0.0);
        let young = (1.0 - s.drives.youth) * 0.6;
        let grown = 0.1 + s.temper(PLAYFULNESS) * 0.25 + s.temper(CURIOSITY) * 0.05;
        urges.add(PLAY, (young + grown) * content * wits);
        urges.add(
            SOCIALISE,
            (0.1 + s.temper(SOCIABILITY) * 0.3) * content * wits,
        );
    }

    /// Whether this animal keeps watch over a resting herd: the most watchful of them, while they
    /// are still fresh enough to; as a sentinel tires, the next most watchful takes over.
    pub fn keeps_watch(s: &Situation) -> bool {
        s.seen.kin_count >= 3 && s.temper(VIGILANCE) * (1.0 - s.drives.sleepiness * 0.5) > 0.55
    }

    /// Song: calling across the sky, and going to a mate that answers.
    pub fn song(s: &Situation) -> Goal {
        let mate = s.seen.mate.map(|(mate, _)| mate);
        Goal::new(
            Activity::Display,
            mate.or_else(|| s.seen.mark(CALL)),
            if mate.is_some() { 0.5 } else { 0.0 },
        )
        .hovering()
        .calling()
    }

    /// Sleeping adrift among the herd.
    fn raft(s: &Situation) -> Goal {
        Goal::new(
            Activity::Sleep,
            Some(s.seen.kin.unwrap_or(Vec3::ZERO) + shade(s)),
            0.35,
        )
        .folded(0.9)
        .hovering()
    }
}
