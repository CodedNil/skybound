//! Planet-fixed lightning seeds. Cameras select visible events, never create them.
use crate::scene::PLANET_RADIUS;
use crate::scene::math::fourth;
use crate::scene::planet::surface_frame;
use core::f32::consts::TAU;
use spirv_std::glam::{FloatExt, Vec3, Vec4, vec3};
use std::sync::OnceLock;

struct Seed {
    up: Vec3,
    period: f32,
    offset: f32,
    duration: f32,
    random: f32,
}
#[derive(Clone, Copy)]
pub struct Event {
    pub id: u32,
    pub position: Vec3,
    pub up: Vec3,
    pub strength: f32,
    pub reach: f32,
    pub radius: f32,
    pub random: f32,
}
impl Event {
    /// A rising discharge rooted inside the veil. Geometry is fixed during a
    /// flash; only its intensity flickers, so the channel never waves like a rope.
    pub fn segments(self) -> [[Vec4; 2]; 8] {
        let basis = surface_frame(self.up);
        let root = self.up * (PLANET_RADIUS - 900.0);
        let seed = self.id ^ self.random.to_bits();
        let mut points = [root; 7];
        let mut segments = [[Vec4::ZERO; 2]; 8];
        for index in 1..=6 {
            let t = index as f32 / 6.0;
            let east = random(seed ^ (index as u32 * 17)) * 2.0 - 1.0;
            let north = random(seed ^ (index as u32 * 31)) * 2.0 - 1.0;
            points[index] = root
                + self.up * (self.reach * t)
                + (basis.x_axis * east + basis.y_axis * north) * (self.reach * 0.045);
            segments[index - 1] = [
                points[index - 1].extend((24.0 + self.random * 18.0) * (1.0 - t * 0.45)),
                points[index].extend(self.strength * 12.0 * (1.0 - t * 0.25)),
            ];
        }
        for branch in 0..2 {
            let index = 2 + branch * 2;
            let angle = random(seed ^ (branch as u32 + 113)) * TAU;
            let sideways = basis.x_axis * angle.cos() + basis.y_axis * angle.sin();
            let end = points[index] + (self.up * 0.13 + sideways * 0.10) * self.reach;
            segments[6 + branch] = [
                points[index].extend(10.0 + self.random * 6.0),
                end.extend(self.strength * 4.0),
            ];
        }
        segments
    }
}

fn random(id: u32) -> f32 {
    let mut value = id.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    value = ((value >> ((value >> 28) + 4)) ^ value).wrapping_mul(277_803_737);
    ((value >> 22) ^ value) as f32 / u32::MAX as f32
}
fn catalog() -> &'static [Seed] {
    static SEEDS: OnceLock<Vec<Seed>> = OnceLock::new();
    SEEDS.get_or_init(|| {
        (0..8192)
            .map(|id| {
                let z = 1.0 - 2.0 * (id as f32 + 0.5) / 8192.0;
                let longitude = id as f32 * 2.399_963_1 + random(id ^ 0x117) * 0.1;
                let radius = (1.0 - z * z).sqrt();
                let period = 24.0 + random(id ^ 0x319) * 30.0;
                Seed {
                    up: vec3(radius * longitude.cos(), radius * longitude.sin(), z),
                    period,
                    offset: random(id ^ 0x713) * period,
                    duration: 0.7 + random(id ^ 0x151) * 1.6,
                    random: random(id ^ 0xa17),
                }
            })
            .collect()
    })
}
pub fn events(time: f32) -> impl Iterator<Item = Event> {
    catalog().iter().enumerate().filter_map(move |(id, seed)| {
        let age = (time + seed.offset).rem_euclid(seed.period);
        if age >= seed.duration {
            return None;
        }
        let phase = age / seed.duration;
        let envelope = phase.smoothstep(0.0, 0.12) * (1.0 - phase.smoothstep(0.7, 1.0));
        let flicker = fourth((time * 19.0 + seed.random * TAU).sin().max(0.0));
        // Each recurring storm produces a new channel on its next discharge.
        let cycle = ((time + seed.offset) / seed.period).floor() as u32;
        let strike = random((id as u32) ^ cycle.wrapping_mul(0x9e37_79b9));
        let reach = 8000.0 + strike * 20_000.0;
        Some(Event {
            id: id as u32,
            position: seed.up * (PLANET_RADIUS + reach * 0.5),
            up: seed.up,
            reach,
            radius: 6000.0 + seed.random * 10_000.0,
            strength: envelope * (0.25 + flicker * 0.75),
            random: strike,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn discharges_rise_from_the_veil_and_branches_join_the_main_channel() {
        for event in events(17.5) {
            let segments = event.segments();
            assert_eq!(segments, event.segments());
            let root = segments[0][0].truncate();
            assert!((root.length() - PLANET_RADIUS + 900.0).abs() < 0.5);
            for (index, segment) in segments[..6].iter().enumerate() {
                let direction = segment[1].truncate() - segment[0].truncate();
                assert!(direction.normalize().dot(event.up) > 0.75);
                if index > 0 {
                    assert_eq!(segment[0].truncate(), segments[index - 1][1].truncate());
                }
            }
            assert_eq!(segments[6][0].truncate(), segments[1][1].truncate());
            assert_eq!(segments[7][0].truncate(), segments[3][1].truncate());
            let tip = segments[5][1].truncate() - root;
            assert!((tip.dot(event.up) - event.reach).abs() < 0.5);
        }
    }

    #[test]
    fn lightning_has_reproducible_activity_across_both_hemispheres() {
        let first: Vec<_> = events(17.5).collect();
        let second: Vec<_> = events(17.5).collect();
        assert!(first.len() > 150 && first.len() < 700);
        assert_eq!(
            first.iter().map(|event| event.id).collect::<Vec<_>>(),
            second.iter().map(|event| event.id).collect::<Vec<_>>()
        );
        assert!(first.iter().any(|event| event.up.z > 0.8));
        assert!(first.iter().any(|event| event.up.z < -0.8));
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.position, b.position);
            assert!((a.up.length() - 1.0).abs() < 0.0001);
            assert!((0.0..=1.0).contains(&a.strength));
        }
    }
}
