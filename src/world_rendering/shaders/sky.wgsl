#define_import_path skybound::sky

/// https://github.com/mrdoob/three.js/blob/master/examples/jsm/objects/Sky.js

// Sun and Sky Properties
const SUN_ENERGY: f32 = 1000.0;
const EXPOSURE: f32 = 0.3;

// Rayleigh Scattering (Atmosphere)
const RAYLEIGH: f32 = 1.0;
const RAYLEIGH_ZENITH_LENGTH: f32 = 8.4e3;
const TOTAL_RAYLEIGH: vec3<f32> = vec3<f32>(5.804542996261093E-6, 1.3562911419845635E-5, 3.0265902468824876E-5);

// Mie Scattering (Atmospheric Particles)
const MIE_COEFFICIENT: f32 = 0.005;
const MIE_DIRECTIONAL_G: f32 = 0.8;
const MIE_ZENITH_LENGTH: f32 = 1.25e3;
const TURBIDITY: f32 = 2.0;
const MIE_CONST: vec3<f32> = vec3<f32>(1.8399918514433978E14, 2.7798023919660528E14, 4.0790479543861094E14);

// Earth Shadow Hack Constants
const CUTOFF_ANGLE: f32 = 1.6110731556870734; // pi / 1.95
const STEEPNESS: f32 = 1.5;
const SUN_ANGULAR_DIAMETER_COS: f32 = 0.999956676946448443553574619906976478926848692; // 66 arc seconds -> degrees, and the cosine of that

// Precomputed Constants
const PI: f32 = 3.14159265;
const E: f32 = 2.71828182;
const UP: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
const THREE_OVER_SIXTEEN_PI: f32 = 3.0 / (16.0 * PI);
const ONE_OVER_FOUR_PI: f32 = 1.0 / (4.0 * PI);


struct AtmosphereColors {
    sky: vec3<f32>,
    sun: vec3<f32>,
    ambient: vec3<f32>,
    ground: vec3<f32>,
    phase: f32,
}

fn sun_intensity(zenith_angle_cos: f32) -> f32 {
    let clamped_zenith_angle_cos: f32 = clamp(zenith_angle_cos, -1.0, 1.0);
    return SUN_ENERGY * max(0.0, 1.0 - pow(E, -((CUTOFF_ANGLE - acos(clamped_zenith_angle_cos)) / STEEPNESS)));
}

fn total_mie(t: f32) -> vec3<f32> {
    let c: f32 = (0.2 * t) * 10E-18;
    return 0.434 * c * MIE_CONST;
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return THREE_OVER_SIXTEEN_PI * (1.0 + pow(cos_theta, 2.0));
}

fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2: f32 = pow(g, 2.0);
    let inverse: f32 = 1.0 / pow(1.0 - 2.0 * g * cos_theta + g2, 1.5);
    return ONE_OVER_FOUR_PI * ((1.0 - g2) * inverse);
}


fn render_sky(rd: vec3<f32>, sun_dir: vec3<f32>, altitude: f32) -> vec3<f32> {
    let v_sun_e: f32 = sun_intensity(dot(sun_dir, UP));
    let v_sunfade: f32 = 1.0 - clamp(1.0 - exp((sun_dir.y / 450000.0)), 0.0, 1.0);

    let rayleigh_coefficient: f32 = RAYLEIGH - (1.0 * (1.0 - v_sunfade));

    // Extinction (absorption + out scattering)
    let v_beta_r: vec3<f32> = TOTAL_RAYLEIGH * rayleigh_coefficient;
    let v_beta_m: vec3<f32> = total_mie(TURBIDITY) * MIE_COEFFICIENT;

    // Adjust ray direction for altitude to lower the horizon
    let horizon_offset_factor: f32 = 0.000001;
    let adjusted_rd_y: f32 = rd.y + altitude * horizon_offset_factor;
    let adjusted_ray_direction: vec3<f32> = normalize(vec3<f32>(rd.x, adjusted_rd_y, rd.z));

    // Optical length
    let zenith_angle: f32 = acos(max(0.0, dot(UP, adjusted_ray_direction)));
    let inverse_optical_mass: f32 = 1.0 / (cos(zenith_angle) + 0.15 * pow(93.885 - degrees(zenith_angle), -1.253));
    let s_r: f32 = RAYLEIGH_ZENITH_LENGTH * inverse_optical_mass;
    let s_m: f32 = MIE_ZENITH_LENGTH * inverse_optical_mass;

    // Combined extinction factor
    let f_ex: vec3<f32> = exp(-(v_beta_r * s_r + v_beta_m * s_m));

    // In scattering
    let cos_theta: f32 = dot(rd, sun_dir);

    let r_phase: f32 = rayleigh_phase(cos_theta * 0.5 + 0.5);
    let beta_r_theta: vec3<f32> = v_beta_r * r_phase;

    let m_phase: f32 = hg_phase(cos_theta, MIE_DIRECTIONAL_G);
    let beta_m_theta: vec3<f32> = v_beta_m * m_phase;

    // Lin calculation
    var lin: vec3<f32> = pow(v_sun_e * ((beta_r_theta + beta_m_theta) / (v_beta_r + v_beta_m)) * (1.0 - f_ex), vec3<f32>(1.5));
    lin *= mix(vec3<f32>(1.0), pow(v_sun_e * ((beta_r_theta + beta_m_theta) / (v_beta_r + v_beta_m)) * f_ex, vec3<f32>(0.5)), clamp(pow(1.0 - dot(UP, sun_dir), 5.0), 0.0, 1.0));

    // Night sky
    // for UV mapping a night sky texture, just a base color for now.
    var l0: vec3<f32> = vec3<f32>(0.1) * f_ex;

    // Solar disk and out-scattering
    let sun_disk: f32 = smoothstep(SUN_ANGULAR_DIAMETER_COS, SUN_ANGULAR_DIAMETER_COS + 0.00002, cos_theta);
    l0 += (v_sun_e * 19000.0 * f_ex) * sun_disk;

    // Composition
    let tex_color: vec3<f32> = (lin + l0) * 0.04 + vec3<f32>(0.0, 0.0003, 0.00075);
    var ret_color: vec3<f32> = pow(tex_color, vec3<f32>(1.0 / (1.2 + (1.2 * v_sunfade))));
    ret_color *= EXPOSURE;
    return ret_color;
}
