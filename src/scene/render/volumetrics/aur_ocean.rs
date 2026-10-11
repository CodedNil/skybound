use spirv_std::glam::{FloatExt, Vec3, Vec3Swizzles, Vec4Swizzles, vec3};
use spirv_std::num_traits::Float;

use super::VolumeSample;
use crate::scene::ViewUniform;
use crate::scene::aur::{CURRENT, curl};
use crate::scene::render::utils::Textures;

pub const OCEAN_TOP_HEIGHT: f32 = 2800.0;

pub fn sample_ocean(
    pos: Vec3,
    view: &ViewUniform,
    only_density: bool,
    textures: &Textures,
) -> VolumeSample {
    if pos.z > OCEAN_TOP_HEIGHT {
        return VolumeSample::default();
    }
    let time = view.time();
    let camera = (view.world_position.xy() + view.camera_offset().truncate())
        .extend(view.ro_relative().z - crate::scene::PLANET_RADIUS);
    let distant = pos.distance(camera).smoothstep(8000.0, 80_000.0);
    let flowing = pos - CURRENT * time;
    let flow = textures
        .base(flowing * 0.0004 - Vec3::Z * time * 0.009)
        .yzw()
        * 2.0
        - 1.0;
    let warped = curl(flowing.xy() * 0.0007, time) * 0.25;
    let coordinates = warped.extend(pos.z * 0.0008 - time * 0.025) + flow * 0.4;
    let cells = textures.base(coordinates * 0.8).w;
    let detail = textures.details(coordinates);
    let noise = detail.w * 2.0 - 1.0;
    let wisp = (1.0 - noise.abs() * 2.0)
        .max(0.0)
        .powi(3)
        .lerp(0.12, distant);
    let granules = cells.smoothstep(0.25, 0.75).lerp(0.5, distant);
    let wave = (warped.x * 8.0 + noise * 0.7 + time * 0.9).sin() * 280.0
        + (warped.y * 6.0 - time * 0.7).sin() * 160.0
        + flow.y * 250.0
        - 120.0;
    let altitude = pos.z - wave;
    let body = (-altitude / 350.0).clamp(-12.0, 5.0).exp();
    let plume = (-altitude / 800.0).clamp(-12.0, 5.0).exp();
    let mut sample = VolumeSample {
        density: (body * (0.012 + granules * 0.06 + wisp * 0.12)
            + plume * granules.powi(4) * wisp * 0.045)
            * (1.0 - pos.z.smoothstep(2000.0, OCEAN_TOP_HEIGHT)),
        ..VolumeSample::default()
    };
    if !only_density && sample.density > 0.0 {
        let hue = vec3(0.24, 0.035, 0.55).lerp(vec3(0.48, 0.08, 0.42), granules);
        let hot = (1.0 - detail.x).powi(2).lerp(0.2, distant);
        let pulse = 0.85 + 0.15 * (time * 2.0 + warped.x * 7.0).sin();
        let mut flash = 0.0;
        for index in 0..32 {
            let event = view.aur_flashes[index];
            if event.w <= 0.0 {
                break;
            }
            let distance = (pos.xy() - event.xy()).length_squared() + (pos.z + 120.0).powi(2);
            flash += (-distance / (event.z * event.z)).exp() * event.w;
        }
        let emission = (hue * (0.3 + granules * 0.8) + vec3(0.15, 0.5, 0.9) * hot) * pulse
            + vec3(1.0, 1.8, 3.0) * flash * 4.0;
        sample.color = hue;
        sample.emission = emission * view.sky_horizon.y;
    }
    sample
}
