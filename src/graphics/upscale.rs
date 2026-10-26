use bevy::{
    anti_alias::{
        contrast_adaptive_sharpening::ContrastAdaptiveSharpening,
        dlss::{Dlss, DlssPerfQualityMode, DlssSuperResolutionFeature},
        taa::TemporalAntiAliasing,
    },
    ecs::system::EntityCommands,
    prelude::*,
};
use std::env;

/// The temporal upscalers a camera may carry, for resetting whichever it has.
pub type History = (
    Option<&'static mut Dlss>,
    Option<&'static mut TemporalAntiAliasing>,
);

/// Give a camera DLSS if the GPU supports it, or temporal anti-aliasing.
pub fn add_upscaler(camera: &mut EntityCommands, dlss_supported: bool) {
    if dlss_supported {
        camera.insert(Dlss::<DlssSuperResolutionFeature> {
            perf_quality_mode: quality(),
            reset: true,
            ..default()
        });
    } else {
        camera.insert(TemporalAntiAliasing::default());
    }
    // DLSS no longer sharpens by itself.
    camera.insert(ContrastAdaptiveSharpening {
        sharpening_strength: 0.3,
        ..default()
    });
}

/// Discard the upscaler's history, as after a camera cut.
pub fn reset_history((dlss, taa): (Option<Mut<Dlss>>, Option<Mut<TemporalAntiAliasing>>)) {
    if let Some(mut dlss) = dlss {
        dlss.reset = true;
    }
    if let Some(mut taa) = taa {
        taa.reset = true;
    }
}

/// How far below the window's resolution the scene renders, from `SKYBOUND_UPSCALE`; Quality
/// by default.
fn quality() -> DlssPerfQualityMode {
    match env::var("SKYBOUND_UPSCALE").as_deref() {
        Ok("native") => DlssPerfQualityMode::Dlaa,
        Ok("balanced") => DlssPerfQualityMode::Balanced,
        Ok("performance") => DlssPerfQualityMode::Performance,
        Ok("ultra") => DlssPerfQualityMode::UltraPerformance,
        Ok("auto") => DlssPerfQualityMode::Auto,
        // Auto drops to Ultra Performance at 4K, and the volumes, traced at half the render
        // resolution and never jittered, gain nothing back.
        _ => DlssPerfQualityMode::Quality,
    }
}
