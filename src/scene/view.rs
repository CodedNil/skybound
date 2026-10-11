use spirv_std::glam::{Mat4, Quat, Vec3, Vec4, Vec4Swizzles, vec3};

use crate::scene::atmosphere::{get_pole_light_color, render_sky};
use crate::scene::geometry::get_pole_position;
use crate::scene::life::CreatureInstance;

pub const PLANET_RADIUS: f32 = 1_000_000.0;

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewUniform {
    pub clip_from_world: Mat4,
    pub world_from_clip: Mat4,
    pub prev_clip_from_world: Mat4,
    pub world_from_clip_unjittered: Mat4,
    pub world_position: Vec4,
    pub camera_position: Vec4,
    pub planet_rotation: Quat,
    pub times: Vec4,
    pub weather: Vec4,
    pub pole_position: Vec4,
    pub pole_color: Vec4,
    pub ambient_color: Vec4,
    pub sky_horizon: Vec4,
    pub aur_flashes: [Vec4; 32],
    pub aur_bolts: [[Vec4; 2]; 32],
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrameUniform {
    pub view: ViewUniform,
    pub players: [CreatureInstance; 2],
    pub viewport: Vec4,
}

impl ViewUniform {
    pub fn prepare_atmosphere(&mut self) {
        let pole = get_pole_position(
            self.planet_center(),
            self.planet_rotation,
            self.ro_relative(),
        );
        let up = (self.world_position.xyz() - self.planet_center()).normalize();
        let offset = pole - self.world_position.xyz();
        let pole_dir = if offset.length_squared() < 0.0001 {
            up
        } else {
            offset.normalize()
        };
        let ro = self.ro_relative();
        self.pole_position = pole.extend(0.0);
        self.pole_color = get_pole_light_color(ro, pole_dir).extend(0.0);
        self.ambient_color =
            (render_sky(up, ro, pole_dir) * 0.7 + render_sky(-up, ro, pole_dir) * 0.15).extend(0.0);
        let position = (self.world_position.xy() + self.camera_offset().truncate())
            .extend(ro.z - PLANET_RADIUS);
        self.ambient_color += super::aur::ambient(position, self.time()).extend(0.0);
    }

    #[cfg(not(target_arch = "spirv"))]
    pub fn prepare_flashes(&mut self) {
        use std::f32::consts::PI;

        use spirv_std::glam::{FloatExt, Vec2, Vec3Swizzles};
        let hash = |value: f32| {
            let mut point = (Vec3::splat(value) * vec3(0.1031, 0.1030, 0.1029)).fract_gl();
            point += point.dot(point.yxz() + 33.33);
            ((point.x + point.y + point.z) * point).fract_gl()
        };
        self.aur_flashes = [Vec4::ZERO; 32];
        self.aur_bolts = [[Vec4::ZERO; 2]; 32];
        let cell =
            ((self.world_position.xy() + self.camera_offset().truncate()) / 10_000.0).floor();
        let time = self.time();
        let mut count = 0;
        let mut bolt_count = 0;
        for x in -1..=1 {
            for y in -1..=1 {
                let cell = cell + Vec2::new(x as f32, y as f32);
                let seed = hash(cell.dot(Vec2::new(127.1, 311.7)));
                let age = (time - seed.x * 20.0).rem_euclid(20.0);
                let duration = 1.0 + seed.y * 3.0;
                if age >= duration {
                    continue;
                }
                let life = age / duration;
                let envelope = life.smoothstep(0.0, 0.12) * (1.0 - life.smoothstep(0.75, 1.0));
                for index in 0..4 {
                    if count == self.aur_flashes.len() {
                        return;
                    }
                    let point = hash(seed.z * 137.0 + index as f32 * 17.0);
                    let motion = Vec2::new(
                        (time * 0.5 + point.x * 10.0).sin(),
                        (time * 0.4 + point.y * 9.0).cos(),
                    );
                    let center = (cell + point.xy() * 0.65 + motion * 0.12) * 10_000.0;
                    let strength =
                        envelope * (0.75 + 0.25 * (time * 17.0 + point.x * 31.0).sin().abs());
                    if strength > 0.0 {
                        self.aur_flashes[count] =
                            center.extend(250.0 + point.z * 900.0).extend(strength);
                        count += 1;
                    }
                    let flicker = (time * 11.0 + point.z * 19.0).fract();
                    if strength <= 0.0 || flicker > 0.28 {
                        continue;
                    }
                    let strike = (time * 11.0).floor();
                    let bolt_seed = hash(point.z * 317.0 + strike);
                    let origin = center.extend(-120.0);
                    let direction =
                        ((bolt_seed.xy() - 0.5) * 650.0).extend(400.0 + bolt_seed.z * 650.0);
                    let mut start = origin;
                    for segment in 1..=7 {
                        if bolt_count == self.aur_bolts.len() {
                            break;
                        }
                        let t = segment as f32 / 7.0;
                        let jitter = (hash(strike + point.z * 97.0 + segment as f32) - 0.5)
                            * (110.0 * (PI * t).sin());
                        let end = origin + direction * t + jitter;
                        self.aur_bolts[bolt_count] = [
                            start.extend(2.5 * (1.0 - t * 0.6)),
                            end.extend(strength * 55.0 * (1.0 - flicker / 0.28)),
                        ];
                        bolt_count += 1;
                        if segment == 3 || segment == 5 {
                            if bolt_count == self.aur_bolts.len() {
                                break;
                            }
                            let branch = (hash(strike + segment as f32 * 37.0) - 0.5) * 350.0
                                + Vec3::Z * 120.0;
                            self.aur_bolts[bolt_count] =
                                [end.extend(0.8), (end + branch).extend(strength * 24.0)];
                            bolt_count += 1;
                        }
                        start = end;
                    }
                }
            }
        }
    }

    pub fn planet_center(&self) -> Vec3 {
        self.world_position
            .xy()
            .extend(-PLANET_RADIUS - self.world_position.w)
    }

    pub fn ro_relative(&self) -> Vec3 {
        vec3(
            0.0,
            0.0,
            self.world_position.z + PLANET_RADIUS + self.world_position.w,
        )
    }

    pub fn latitude(&self) -> f32 {
        self.camera_position.x
    }

    pub fn longitude(&self) -> f32 {
        self.camera_position.y
    }

    pub fn camera_offset(&self) -> Vec3 {
        vec3(
            self.camera_position.z,
            self.camera_position.w,
            self.world_position.w,
        )
    }

    pub fn time(&self) -> f32 {
        self.times.x
    }

    pub fn pole_direction(&self) -> Vec3 {
        let offset = self.pole_position.truncate() - self.world_position.truncate();
        if offset.length_squared() < 0.0001 {
            Vec3::Z
        } else {
            offset.normalize()
        }
    }

    pub fn frame_count(&self) -> f32 {
        self.times.y
    }
}

#[cfg(test)]
mod tests {
    use core::f32::consts::FRAC_PI_2;

    use super::*;

    #[test]
    fn polar_light_rises_symmetrically_from_dark_equatorial_conditions() {
        let light = |latitude: f32| {
            let mut view = ViewUniform {
                world_position: vec3(0.0, 0.0, 2400.0).extend(0.0),
                planet_rotation: Quat::from_rotation_x(FRAC_PI_2 - latitude),
                ..ViewUniform::default()
            };
            view.prepare_atmosphere();
            view.pole_color.truncate().length()
        };
        assert!(light(0.0) < 0.001);
        assert!(light(0.87) > light(0.0) + 0.1);
        assert!(light(1.4) > light(0.87));
        assert!((light(0.87) - light(-0.87)).abs() < 0.01);
    }

    #[test]
    fn atmosphere_stays_finite_across_altitudes_and_latitudes() {
        for altitude in [-1000.0, 0.0, 20_000.0, 400_000.0] {
            for latitude in [-1.0, 0.0, 1.0] {
                let mut view = ViewUniform {
                    world_position: vec3(0.0, 0.0, altitude).extend(0.0),
                    camera_position: Vec4::new(latitude, 0.0, 0.0, 0.0),
                    planet_rotation: Quat::from_rotation_x(latitude),
                    ..ViewUniform::default()
                };
                view.prepare_atmosphere();
                assert!(view.pole_position.is_finite(), "invalid pole position");
                assert!(view.pole_color.is_finite(), "invalid pole color");
                assert!(view.ambient_color.is_finite(), "invalid ambient color");
            }
        }
    }
}
