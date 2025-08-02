#define_import_path skybound::sky

// Phase functions
fn rayleigh_phase(cos2: f32) -> f32 {
    return 0.75 * (1.0 + cos2);
}

// Simple sky shading
fn render_sky(rd: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let sun_dot = sun_dir.y; // Sun elevation
    let elevation = rd.y; // Ray elevation, normalized
    let view_sun_dot = dot(rd, sun_dir); // Cosine of angle between view and sun
    let view_sun_dot2 = view_sun_dot * view_sun_dot; // For phase function
    let t = (elevation + 1.0) / 2.0; // Gradient from horizon to zenith [0, 1]

    // Base colors
    let day_horizon = vec3(0.7, 0.8, 1.0); // Light blue
    let day_zenith = vec3(0.2, 0.4, 0.8); // Deep blue
    let sunset_horizon = vec3(1.0, 0.5, 0.0); // Orange
    let sunset_zenith = vec3(0.2, 0.0, 0.4); // Purple
    let night_horizon = vec3(0.0, 0.0, 0.1); // Dark blue
    let night_zenith = vec3(0.0, 0.0, 0.0); // Black

    // Smooth transitions
    let day_to_sunset = smoothstep(-0.1, 0.1, sun_dot); // Day (1) to sunset (0)
    let sunset_to_night = smoothstep(-0.3, -0.1, sun_dot); // Sunset (1) to night (0)

    // Blend horizon and zenith colors
    let horizon_color = mix(
        mix(night_horizon, sunset_horizon, sunset_to_night),
        day_horizon,
        day_to_sunset
    );
    let zenith_color = mix(
        mix(night_zenith, sunset_zenith, sunset_to_night),
        day_zenith,
        day_to_sunset
    );

    // Rayleigh-like phase for sky scattering
    let rayleigh_phase = 0.75 * (1.0 + view_sun_dot2); // Simplified from Shadertoy

    // Base sky gradient
    let sky_col = mix(horizon_color, zenith_color, t) * rayleigh_phase;

    return clamp(sky_col, vec3(0.0), vec3(1.0));
}
