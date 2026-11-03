//! Seating what grows from a body onto the body itself.
//!
//! A wing's root and a mouth tentacle's are laid out by each species' design, but the body they
//! grow from is this individual's own: as plump or lean as its genes have it, as grown as its age,
//! as swollen as its belly. So, once a body is posed, each is seated on it, as a shrinkwrap would:
//! the body is measured bare, with nothing grown on it drawn, and a wing's root is brought to where
//! the line through its hinge meets the skin, and a tentacle's root to the nearest point of the
//! skin, the whole tentacle moved with it. Eyes and the rest a species anchors (see
//! [`crate::scene::life::ANCHORS`]) are each moved to lie against this body's nearest skin just as
//! the design sets them against its species' reference body. A body's shape changes over its
//! life and not from one frame to the next, so what is measured is kept for that individual at
//! that stage of its life.

use crate::scene::life::{
    CreatureInstance, Drawn, KINDS, Look,
    animalia::tetralata::{KNOTS, Tetralate},
    blood_feeder_surface, look_of, tetralate_surface,
};
use spirv_std::glam::{Vec3, Vec4, vec3, vec4};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

/// Where an individual's wings and mouth tentacles are seated: its wings' measured roots, how far
/// each tentacle's root is moved to meet its skin, and how far each anchor is moved.
#[derive(Clone, Copy)]
struct Seats {
    wings: Vec4,
    tentacles: [Vec3; 9],
    anchors: [Vec4; ANCHOR_SEATS],
}

/// The most anchors a body has room for.
pub const ANCHOR_SEATS: usize = 6;

/// Seat the wings and mouth tentacles of a newly posed body on the body itself.
pub fn seat(drawn: &mut Drawn, kind: u32) {
    match drawn {
        Drawn::Tetralate(c) if c.life.w <= 0.0 => {
            let seats = remembered(key(c), || measure(c, kind));
            c.wings.seats = seats.wings;
            c.animal.anchors = seats.anchors;
            for index in 0..9 {
                let (sheet, column) = (index / 3, index % 3);
                if c.tentacles[sheet][0][column].w <= 0.0 {
                    continue;
                }
                let shift = seats.tentacles[index].extend(0.0);
                for knot in 0..KNOTS {
                    c.tentacles[sheet][knot][column] += shift;
                    c.previous_tentacles[sheet][knot][column] += shift;
                }
            }
        }
        Drawn::BloodFeeder(c) if c.life.w <= 0.0 => {
            let look = look_of(kind);
            let seats = remembered(key(c), || {
                let mut bare = *c;
                bare.wings.seats = vec4(0.0, 0.0, 0.0, -1.0);
                let body = |p: Vec3| blood_feeder_surface(kind, p, &bare).distance;
                Seats {
                    wings: wings(look, c.opaque_bounds.x, body),
                    tentacles: [Vec3::ZERO; 9],
                    anchors: anchors(kind, c, body),
                }
            });
            c.wings.seats = seats.wings;
            c.animal.anchors = seats.anchors;
        }
        _ => {}
    }
}

/// Measure a tetralate's seats on its bare body.
fn measure(c: &Tetralate, kind: u32) -> Seats {
    let mut bare = *c;
    bare.wings.seats = vec4(0.0, 0.0, 0.0, -1.0);
    let body = |p: Vec3| tetralate_surface(kind, p, &bare).distance;
    let size = size(c);
    let mut tentacles = [Vec3::ZERO; 9];
    for (index, shift) in tentacles.iter_mut().enumerate() {
        let root = c.tentacles[index / 3][0][index % 3];
        if root.w <= 0.0 {
            continue;
        }
        // Brought to the nearest point of the skin, a little within it, so it grows out of it.
        let mut at = root.truncate();
        for _ in 0..6 {
            let distance = body(at);
            let normal = gradient(body, at, size * 0.002);
            at -= normal * (distance + root.w * 0.4);
        }
        // A root a long way from any skin is the design's own doing; leave it be.
        if (at - root.truncate()).length() < size * 0.15 {
            *shift = at - root.truncate();
        }
    }
    Seats {
        wings: wings(look_of(kind), c.opaque_bounds.x, body),
        tentacles,
        anchors: anchors(kind, c, body),
    }
}

/// How large a body is in its design's frame.
fn size(c: &CreatureInstance) -> f32 {
    c.opaque_bounds.truncate().length().max(0.01)
}

/// How far each of a species' anchors is moved on this bare body, to lie as far below its nearest
/// skin as the species sets it.
fn anchors(kind: u32, c: &CreatureInstance, body: impl Fn(Vec3) -> f32) -> [Vec4; ANCHOR_SEATS] {
    let species = KINDS[kind as usize];
    let size = size(c);
    let mut shifts = [Vec4::ZERO; ANCHOR_SEATS];
    for (index, seat) in shifts
        .iter_mut()
        .enumerate()
        .take(species.anchors() as usize)
    {
        let anchor = species.anchor(index as u32, c);
        let point = anchor.truncate();
        if let Some(skin) = nearest(&body, point, size) {
            let inward = -gradient(&body, skin, size * 0.002);
            let shift = skin + inward * anchor.w - point;
            // A design point a long way from any skin is the design's own doing; leave it be.
            if shift.length() < size * 0.15 {
                *seat = shift.extend(0.0);
            }
        }
    }
    shifts
}

/// The nearest point of the bare body's skin to a point near it, or none if it is far from any.
fn nearest(body: impl Fn(Vec3) -> f32, point: Vec3, size: f32) -> Option<Vec3> {
    let mut at = point;
    for _ in 0..12 {
        let distance = body(at);
        if distance.abs() < size * 0.0002 {
            break;
        }
        at -= gradient(&body, at, size * 0.002) * distance;
    }
    ((at - point).length() < size * 0.2 && body(at).abs() < size * 0.005).then_some(at)
}

/// Where each pair of wings meets the bare body: in from outside along the line through its hinge,
/// to the skin, and a little within it.
fn wings(look: Look, reach: f32, body: impl Fn(Vec3) -> f32) -> Vec4 {
    let mut seats = vec4(look.hinges[0].x, look.hinges[1].x, 1.0, 0.0);
    for index in 0..2 {
        let wing = look.wings[index];
        if wing.x <= 0.0 {
            continue;
        }
        let hinge = look.hinges[index];
        let mut at = vec3(reach.max(hinge.x * 2.0), hinge.y, wing.z);
        let mut found = None;
        for _ in 0..64 {
            let distance = body(at);
            if distance < wing.y * 0.005 {
                found = Some(at.x);
                break;
            }
            at.x -= distance.max(wing.y * 0.01);
            if at.x <= 0.0 {
                break;
            }
        }
        let seat = found.map_or(hinge.x, |x| x - wing.y * 0.04);
        if index == 0 {
            seats.x = seat;
        } else {
            seats.y = seat;
        }
    }
    seats
}

/// Which way the bare body's skin faces at a point.
fn gradient(body: impl Fn(Vec3) -> f32, p: Vec3, e: f32) -> Vec3 {
    let n = vec3(
        body(p + Vec3::X * e) - body(p - Vec3::X * e),
        body(p + Vec3::Y * e) - body(p - Vec3::Y * e),
        body(p + Vec3::Z * e) - body(p - Vec3::Z * e),
    );
    n / n.length().max(1e-9)
}

/// What a body's seats depend on: its species, its own seed and genes, and how far through its
/// growing it is, in steps fine enough that no step shows.
type Key = (u32, u32, u64, u32);

fn key(c: &CreatureInstance) -> Key {
    let genes = c.genes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, four| {
        four.to_array().iter().fold(hash, |hash, gene| {
            (hash ^ u64::from(gene.to_bits())).wrapping_mul(0x0100_0000_01b3)
        })
    });
    (c.kind.x, c.kind.y, genes, (c.life.y * 48.0) as u32)
}

/// The seats measured for a body, measured once and remembered.
fn remembered(key: Key, measure: impl FnOnce() -> Seats) -> Seats {
    static SEATS: OnceLock<Mutex<HashMap<Key, Seats>>> = OnceLock::new();
    let seats = SEATS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(found) = seats.lock().expect("seats").get(&key) {
        return *found;
    }
    let measured = measure();
    let mut seats = seats.lock().expect("seats");
    // Bodies come and go as the player travels; forget them all now and then rather than keep
    // every one ever met.
    if seats.len() > 20_000 {
        seats.clear();
    }
    seats.insert(key, measured);
    measured
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::life::KINDS;

    /// After seating, every grown tetralate's wing roots and mouth tentacle roots lie on its own
    /// bare body, and its eyes as deep in its skin as its species sets them, whatever its genes.
    #[test]
    fn wings_tentacles_and_eyes_grow_from_the_skin() {
        let mut failures = Vec::new();
        for &kind in KINDS {
            for seed in [0, 3, 11, 17, 23] {
                let drawn = Drawn::specimen(
                    Vec4::new(0.0, 0.0, 0.0, kind.render_radius()),
                    Vec4::W,
                    Vec4::new(0.0, 0.0, 1.0, 0.0),
                    kind as u32,
                    kind.biology().maturity_days * 1.5,
                    seed,
                );
                let Drawn::Tetralate(c) = drawn else {
                    continue;
                };
                let mut bare = c;
                bare.wings.seats.w = -1.0;
                let body = |p: Vec3| tetralate_surface(kind as u32, p, &bare).distance;
                let look = look_of(kind as u32);
                for index in 0..2 {
                    let wing = look.wings[index];
                    if wing.x <= 0.0 {
                        continue;
                    }
                    let root = vec3(
                        c.wings.seat(index, look.hinges[index].x),
                        look.hinges[index].y,
                        wing.z,
                    );
                    assert!(
                        body(root).abs() < wing.y * 0.15,
                        "{kind:?} ({seed}) wing {index} is rooted {} off its body",
                        body(root)
                    );
                }
                // Each anchor lies as far below its nearest skin as its species sets it.
                for index in 0..kind.anchors() {
                    let anchor = kind.anchor(index, &c);
                    let seated = c.seated(index, anchor.truncate());
                    let Some(skin) = nearest(body, seated, size(&c)) else {
                        failures.push(format!("{kind:?} ({seed}) anchor {index} is far from skin"));
                        continue;
                    };
                    let depth =
                        (seated - skin).length() * if body(seated) < 0.0 { 1.0 } else { -1.0 };
                    if (depth - anchor.w).abs() > size(&c) * 0.002 {
                        failures.push(format!(
                            "{kind:?} ({seed}) anchor {index} lies {depth} below its skin, not {}",
                            anchor.w
                        ));
                    }
                }
                for index in 0..9 {
                    let root = c.tentacles[index / 3][0][index % 3];
                    if root.w <= 0.0 {
                        continue;
                    }
                    assert!(
                        body(root.truncate()) < root.w,
                        "{kind:?} ({seed}) tentacle {index} floats {} off its body",
                        body(root.truncate())
                    );
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
