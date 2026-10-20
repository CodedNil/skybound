use crate::scene::life::{
    CreatureInstance,
    animalia::bullidorsata::caelomorpha::{GasChamber, drift_body, gel, skin, tissue_distance},
    biology::Biology,
    ellipsoid, smooth_min,
    surface::{Ray, Surface, TraceResult, local_ray, make_hit},
};
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 3650.0,
    lifespan_days: 146_000.0,
    adult_mass: 120.0,
    altitude: (50.0, 2500.0),
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 46.0;

taxon!(
    Species,
    "Aerogigas auratus",
    "Air Giants",
    [
        "Air Giants drift through the humid lower Gardens beneath broad golden canopies, their vaulted undersides enclosing sheltered pockets of air.",
        "Large gas chambers support the crown while smaller chambers keep the scalloped fringe spread through the updraft.",
        "Amber veins and pale branching ribs frame translucent tissues stippled with intercepted algae and plant fragments.",
        "Feeding folds face into the passing current, so a slow change in tilt brings fresh food through the arches without powered flight.",
        "During the dim phase of the aur cycle, the canopy settles lower and its moist underside becomes a refuge for smaller drifters.",
    ],
);

const CHAMBERS: [GasChamber; 6] = [
    GasChamber {
        center: vec3(-8.5, 0.5, -3.0),
        radii: vec3(3.8, 3.5, 2.6),
    },
    GasChamber {
        center: vec3(0.0, 2.6, -3.0),
        radii: vec3(4.0, 3.7, 2.7),
    },
    GasChamber {
        center: vec3(8.5, 0.3, -3.0),
        radii: vec3(3.8, 3.3, 2.6),
    },
    GasChamber {
        center: vec3(-8.0, 0.6, 2.8),
        radii: vec3(3.7, 3.3, 2.6),
    },
    GasChamber {
        center: vec3(0.5, 2.7, 2.8),
        radii: vec3(4.0, 3.6, 2.7),
    },
    GasChamber {
        center: vec3(8.9, 0.1, 2.8),
        radii: vec3(3.6, 3.0, 2.5),
    },
];

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(20.0, 13.0, 14.0).extend(0.0);
}

pub fn surface(p: Vec3, _: &CreatureInstance) -> Surface {
    let arch = vec3(p.x, p.y + p.x * p.x * 0.022, p.z);
    let mut distance = ellipsoid(arch - vec3(0.0, -0.7, 0.0), vec3(10.5, 0.9, 4.5));
    for i in -3..=3 {
        let x = i as f32;
        for side in [-1.0, 1.0] {
            let center = vec3(
                x * 3.5,
                -0.8 + (x * 1.7).sin() * 0.7,
                side * (3.0 - x.abs() * 0.25),
            );
            distance = smooth_min(distance, ellipsoid(arch - center, vec3(2.1, 1.5, 1.9)), 1.8);
        }
    }
    Surface::new(distance + tissue_distance(p, 12.0), 0)
}

pub fn envelope(p: Vec3, c: &CreatureInstance) -> Surface {
    skin(p, c, surface, &CHAMBERS, 12.0)
}

fn canopy(result: &mut TraceResult, ray: Ray, c: &CreatureInstance, tint: Vec3) {
    let local = local_ray(ray, c);
    let curvature = vec3(8.5 / 256.0, 0.0, 8.5 / 64.0);
    let quadratic = (curvature * local.direction * local.direction).element_sum();
    let linear =
        local.direction.y + (curvature * local.origin * local.direction).element_sum() * 2.0;
    let constant = local.origin.y - 0.2 + (curvature * local.origin * local.origin).element_sum();
    let discriminant = linear * linear - 4.0 * quadratic * constant;
    if discriminant < 0.0 || (quadratic.abs() < 1e-8 && linear.abs() < 1e-8) {
        return;
    }
    let roots = if quadratic.abs() < 1e-8 {
        [-constant / linear, -1.0]
    } else {
        let root = discriminant.sqrt();
        [
            (-linear - root) / (2.0 * quadratic),
            (-linear + root) / (2.0 * quadratic),
        ]
    };
    for distance in roots {
        if distance <= 0.0 || distance >= ray.max_distance {
            continue;
        }
        let point = local.origin + local.direction * distance;
        let radial = vec3(point.x / 16.0, 0.0, point.z / 8.0);
        let rim = 1.0 + (radial.z.atan2(radial.x) * 10.0).cos() * 0.08;
        if radial.length_squared() > rim * rim {
            continue;
        }
        let mut normal = (curvature * point * 2.0 + Vec3::Y).normalize();
        if normal.dot(local.direction) > 0.0 {
            normal = -normal;
        }
        let cosine = normal.dot(local.direction).abs();
        let mut material = gel(point, 12.0, tint.lerp(vec3(0.95, 0.88, 0.6), 0.45));
        material.transmission = (-(Vec3::ONE - tint) * (0.18 / cosine.max(0.12))).exp() * 0.9;
        let mut hit = make_hit(ray, c, distance, normal, point, material);
        hit.part = 1;
        result.add(hit);
    }
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    let tint = vec3(0.78, 0.65, 0.30);
    let mut result = drift_body(ray, c, envelope, surface, tint, 12.0, &CHAMBERS);
    canopy(&mut result, ray, c, tint);
    result
}
