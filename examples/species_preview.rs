use futures_lite::future::block_on;
use image::{DynamicImage, RgbImage, RgbaImage, imageops};
use rayon::prelude::*;
use skybound::scene::{
    life,
    life::{
        CreatureInstance, Rank, SPECIES, SPECIES_MODULES, Taxon, animalia::physiology::Brood,
        organs::Function, surface::tonemap,
    },
    preview::PreviewView,
};
use spirv_std::glam::{Vec2, Vec3, Vec4, vec2, vec3};
use std::{
    cmp::Reverse,
    collections::HashMap,
    env,
    f32::consts::{PI, TAU},
    fs,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::SystemTime,
};
use webp_animation::{Encoder, EncoderOptions, EncodingConfig, EncodingType, LossyEncodingConfig};
use wgpu::util::{BufferInitDescriptor, DeviceExt, make_spirv};

const SEEDS: [u32; 9] = [11, 23, 37, 53, 71, 89, 107, 131, 157];

#[derive(Clone)]
struct View {
    origin: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    center: Vec2,
    field: f32,
    /// The sky behind the creature, across the view from zero to one, and whether it is one even
    /// tone rather than a gradient.
    backdrop: fn(Vec2) -> Vec3,
    even: bool,
    /// How much of the view the creature fills across and up, once fitted.
    filled: Vec2,
}

impl View {
    /// The view as the GPU traces it, for a tile `size` samples across.
    fn on_gpu(&self, size: usize) -> PreviewView {
        PreviewView {
            origin: self.origin.extend(self.field),
            forward: self.forward.extend(self.center.x),
            right: self.right.extend(self.center.y),
            up: self.up.extend(self.field / size as f32),
            tile: Vec4::new(size as f32, 0.0, if self.even { 1.0 } else { 0.0 }, 0.0),
        }
    }

    fn fit(index: usize, c: &CreatureInstance) -> Self {
        let angle = [0.08, -0.25, 0.25, -0.5, 1.0, 0.5, 0.75, 0.0, 0.0][index] * PI;
        let elevation = [0.1f32, 0.2, 0.2, 0.0, 0.1, 0.0, 0.25, 1.25, -0.95][index];
        let origin = vec3(
            angle.sin() * elevation.cos(),
            elevation.sin(),
            -angle.cos() * elevation.cos(),
        ) * c.position.w
            * 1.7;
        let forward = -origin.normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let mut view = Self {
            origin,
            forward,
            right,
            up: right.cross(forward),
            center: Vec2::ZERO,
            field: 1.35,
            backdrop: background,
            even: false,
            filled: Vec2::ONE,
        };
        // Where in the view the creature lies, from a coarse trace of it.
        let mut min = Vec2::ONE;
        let mut max = Vec2::ZERO;
        let coarse = gpu().trace(&[(*c, view.on_gpu(96))]).remove(0);
        for (index, pixel) in coarse.iter().enumerate() {
            if pixel.w > 0.0 {
                let uv = vec2((index % 96) as f32 + 0.5, 95.5 - (index / 96) as f32) / 96.0;
                min = min.min(uv);
                max = max.max(uv);
            }
        }
        if max.cmpge(min).all() {
            view.center = ((min + max) * 0.5 - 0.5) * view.field;
            let span = max - min + 2.0 / 96.0;
            view.field *= span.max_element() * 1.08;
            view.filled = span / (span.max_element() * 1.08);
        }
        view
    }
}

/// The sheets the preview can render for a species.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Output {
    /// The adult from nine angles.
    Render,
    /// The adult cut away to show its organs, from nine angles.
    Anatomy,
    /// One individual from egg or birth to old age.
    Lifecycle,
    /// Nine individuals, to show how they differ.
    Variation,
    /// At every fork in the tree, every species below it side by side at their true sizes.
    Lineup,
}

impl Output {
    const fn name(self) -> &'static str {
        match self {
            Self::Render => "render",
            Self::Anatomy => "anatomy",
            Self::Lifecycle => "lifecycle",
            Self::Variation => "variation",
            Self::Lineup => "lineup",
        }
    }
}

fn adjacent_preview(kind: usize) -> PathBuf {
    let module = SPECIES_MODULES[kind]
        .split_whitespace()
        .collect::<String>()
        .replace("::", "/");
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/scene/life")
        .join(module)
}

/// When anything a species' previews are drawn from last changed: its own file, every module
/// file of the taxa above it, the shared life code, the planet and this previewer.
fn sources(kind: usize) -> SystemTime {
    let species = adjacent_preview(kind).with_extension("rs");
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let life = root.join("src/scene/life");
    let mut files = vec![
        species.clone(),
        root.join("src/scene/planet.rs"),
        root.join("examples/species_preview.rs"),
    ];
    for folder in species.ancestors().skip(1) {
        if let Ok(entries) = fs::read_dir(folder) {
            files.extend(
                entries
                    .flatten()
                    .map(|entry| entry.path())
                    .filter(|path| path.extension().is_some_and(|extension| extension == "rs")),
            );
        }
        if folder == life {
            break;
        }
    }
    files
        .iter()
        .filter_map(|file| fs::metadata(file).and_then(|meta| meta.modified()).ok())
        .max()
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

/// Whether a preview already shows what it would be drawn from now.
fn fresh(path: &Path, since: SystemTime) -> bool {
    fs::metadata(path)
        .and_then(|meta| meta.modified())
        .is_ok_and(|drawn| drawn >= since)
}

fn slug(name: &str) -> String {
    name.to_lowercase()
        .replace('æ', "ae")
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

fn filename(mut taxon: &Taxon) -> String {
    let mut branches = Vec::new();
    while let Some(parent) = taxon.parent {
        branches.push(slug(taxon.name));
        taxon = parent;
    }
    branches.reverse();
    branches.join("__")
}

fn option(name: &str) -> Option<String> {
    let mut args = env::args();
    args.find(|arg| arg == name).and_then(|_| args.next())
}

fn background(uv: Vec2) -> Vec3 {
    vec3(0.025, 0.12, 0.32).lerp(vec3(0.10, 0.22, 0.34), uv.y)
}

/// One even sky behind a whole lineup, so its creatures stand together rather than in boxes.
fn even_sky(_: Vec2) -> Vec3 {
    background(Vec2::splat(0.5))
}

/// One of a species as the previews draw it, as overgrown as one of its kind and age would be.
fn specimen(kind: usize, days: f32, seed: u32, seconds: f32) -> CreatureInstance {
    let species = life::KINDS[kind];
    let mut c = CreatureInstance::specimen(
        Vec3::ZERO.extend(species.render_radius()),
        Vec4::W,
        Vec4::new((seconds * TAU / 4.0).sin() * 0.12, 0.0, seconds, 0.0),
        kind as u32,
        days,
        seed,
    );
    let biology = species.biology();
    let grown = (days / biology.maturity_days).min(2.0) * 0.5 + days / biology.lifespan_days * 2.0;
    c.scars.w = (species.overgrown() * grown).min(1.0);
    // As much iron as one of its age has gathered, for those that gather it.
    if species.animal().is_some_and(|animal| animal.ironclad) {
        c.stores.x = (grown * 0.6).min(1.0);
    }
    c
}

fn label(pixels: &mut [u8], size: usize, text: &str) {
    let scale = (size / 128).max(1);
    for (column, ch) in text.chars().enumerate() {
        let glyph: [u8; 5] = match ch {
            '0' | 'O' => [62, 65, 65, 65, 62],
            '1' => [0, 66, 127, 64, 0],
            '2' => [98, 81, 73, 73, 70],
            '3' => [34, 65, 73, 73, 54],
            '4' => [24, 20, 18, 127, 16],
            '5' => [39, 69, 69, 69, 57],
            '6' => [62, 73, 73, 73, 48],
            '7' => [1, 113, 9, 5, 3],
            '8' => [54, 73, 73, 73, 54],
            '9' => [6, 73, 73, 73, 62],
            '.' => [0, 96, 96, 0, 0],
            'D' => [127, 65, 65, 65, 62],
            'Y' => [7, 8, 112, 8, 7],
            'S' => [38, 73, 73, 73, 50],
            'B' => [127, 73, 73, 73, 54],
            'R' => [127, 9, 25, 41, 70],
            'U' => [63, 64, 64, 64, 63],
            'H' => [127, 8, 8, 8, 127],
            'G' => [62, 65, 73, 73, 58],
            'A' => [126, 9, 9, 9, 126],
            'V' => [31, 32, 64, 32, 31],
            'I' => [0, 65, 127, 65, 0],
            _ => [0; 5],
        };
        for (x, bits) in glyph.into_iter().enumerate() {
            for y in 0..7 {
                for dx in 0..scale {
                    for dy in 0..scale {
                        let px = 6 * scale + (column * 6 + x) * scale + dx;
                        let py = size - 10 * scale + y * scale + dy;
                        if px < size {
                            let start = (py * size + px) * 3;
                            pixels[start..start + 3].copy_from_slice(if bits & (1 << y) != 0 {
                                &[225, 235, 245]
                            } else {
                                &[12, 25, 40]
                            });
                        }
                    }
                }
            }
        }
    }
}

/// The GPU every preview not cut away for anatomy is traced on, set up once.
fn gpu() -> &'static Gpu {
    static GPU: OnceLock<Gpu> = OnceLock::new();
    GPU.get_or_init(Gpu::new)
}

/// Tiles drawn on the GPU, each a creature in its view at its size, cut away to show its organs
/// if `anatomy`, as tonemapped RGB rows from the top.
fn draw(tiles: &[(CreatureInstance, &View, usize)], anatomy: bool) -> Vec<Vec<u8>> {
    let batch: Vec<(CreatureInstance, PreviewView)> = tiles
        .iter()
        .map(|(c, view, size)| {
            // Twice as many samples each way as pixels, averaged down.
            let mut view = view.on_gpu(*size * 2);
            view.tile.w = if anatomy { 1.0 } else { 0.0 };
            (*c, view)
        })
        .collect();
    gpu()
        .trace(&batch)
        .into_par_iter()
        .zip(tiles.par_iter())
        .map(|(samples, (_, _, size))| {
            let size = *size;
            (0..size * size)
                .flat_map(|index| {
                    let (x, y) = (index % size * 2, index / size * 2);
                    let at = |dx: usize, dy: usize| samples[(y + dy) * size * 2 + x + dx];
                    let pixel = (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) * 0.25;
                    let colour = tonemap(pixel.truncate());
                    [
                        (colour.x * 255.0).round() as u8,
                        (colour.y * 255.0).round() as u8,
                        (colour.z * 255.0).round() as u8,
                    ]
                })
                .collect()
        })
        .collect()
}

/// How many frames an animated sheet runs to, and how many it shows a second: a two-second loop,
/// enough to show a wingbeat and the drift of a tentacle.
const fn frames(_: Output) -> (usize, f32) {
    (24, 12.0)
}

/// One panel of a sheet at a moment: the creature as that panel shows it, and its label.
fn panel(kind: usize, output: Output, panel: usize, seconds: f32) -> (CreatureInstance, String) {
    let biology = life::KINDS[kind].biology();
    let age = if output == Output::Lifecycle {
        life::development::grid_age(biology, panel)
    } else {
        biology.maturity_days * 1.25
    };
    let seed = if output == Output::Variation {
        SEEDS[panel]
    } else {
        0
    };
    let mut c = specimen(kind, age, seed, seconds);
    // The render sheet shows each animal feeding once a loop, mouth opening and its tentacles
    // reaching in, while its wings keep their steady beat.
    let species = life::KINDS[kind];
    if output == Output::Render && species.animal().is_some() {
        let feeding = (seconds * PI * 0.5).sin().powi(2);
        c.dynamics = Vec4::new(seconds * species.look().flight.x * TAU, 0.68, feeding, 1.0);
        c.previous_dynamics = c.dynamics;
    }
    // Some hides are grazed bare, others left to grow over.
    if output == Output::Variation {
        c.scars.w = (c.scars.w * (0.3 + (panel as f32 * 0.61).fract() * 1.6)).min(1.0);
    }
    // Every third individual of a variation sheet has lived through attacks: its wings torn, some
    // tears healed to scar, its body scarred where it was struck.
    if output == Output::Variation && panel % 3 == 2 {
        let hard = 0.3 + (panel as f32 * 0.37).fract() * 0.5;
        // The worst of them has lost a hind wing outright.
        let lost = if panel % 9 == 8 { 1.0 } else { hard * 0.6 };
        c.wear = Vec4::new(hard, hard * 0.3, lost, 0.0);
        c.scars = Vec4::new(hard, hard * 0.5, hard * 0.8, c.scars.w);
        let body = c.opaque_bounds.truncate();
        c.wounds = [
            (body * Vec3::new(0.5, 0.3, -0.2)).extend(hard),
            (body * Vec3::new(-0.4, 0.5, 0.35)).extend(hard * 0.7),
            (body * Vec3::new(0.1, -0.4, 0.1)).extend(hard * 0.4),
            Vec4::ZERO,
        ];
    }
    // The first grown panel of a lifecycle shows a parent with its brood, where its lineage
    // carries one.
    let litter = species.biology().litter.mean().round().max(1.0) as u32;
    let brooding = output == Output::Lifecycle && panel == 6;
    if brooding {
        match species.animal().map(|animal| animal.brood) {
            Some(Brood::Carried) => {
                c.kind.z = life::CLUTCH;
                c.stores.w = litter as f32;
            }
            Some(Brood::Pouch) => c.kind.z = life::FULL_POUCH,
            _ => {}
        }
    }
    // The next lifecycle panel, and two individuals of each variation sheet, are carrying young,
    // one near its time and one early on.
    if (output == Output::Lifecycle && panel == 7) || (output == Output::Variation && panel == 4) {
        c.carrying(litter, 0.95);
    } else if output == Output::Variation && panel == 7 {
        c.carrying(litter + 1, 0.55);
    }
    // A plant's water: a Pondplant's basin full in the reference specimen and as its life has left
    // it across a variation sheet.
    if species.plant().is_some() {
        c.stores.y = if output == Output::Variation {
            0.1 + (panel as f32 * 0.37).fract() * 0.9
        } else {
            0.8
        };
    }
    let label = match output {
        Output::Lifecycle => {
            let title = if age >= 365.0 {
                format!("{:.1}Y", age / 365.0)
            } else {
                format!("{age:.1}D")
            };
            if brooding && c.kind.z != 0 {
                format!("{title} BROOD")
            } else if c.stores.z > 0.0 && species.animal().is_some() {
                format!("{title} GRAVID")
            } else {
                title
            }
        }
        Output::Variation => format!("S{seed}"),
        _ => String::new(),
    };
    (c, label)
}

/// A sheet's frames: its nine panels, fitted once so they hold still, then drawn moment by moment
/// as the creatures in them move.
fn render_sheet(kind: usize, size: usize, output: Output, seconds: f32) -> Vec<RgbImage> {
    let biology = life::KINDS[kind].biology();
    let adult = specimen(kind, biology.maturity_days * 1.25, 0, seconds);
    let mut fixed = View::fit(1, &adult);
    fixed.field *= 1.2;
    let views: Vec<View> = (0..9)
        .into_par_iter()
        .map(|index| {
            let (c, _) = panel(kind, output, index, seconds);
            match output {
                Output::Render | Output::Anatomy => View::fit(index, &c),
                // A lifecycle keeps the adult's view, so the young are seen growing at true
                // scale, but closes in on eggs and spores so their forms show, up to three times
                // nearer.
                Output::Lifecycle => {
                    let own = View::fit(1, &c);
                    if c.life.w > 0.0 && own.field < fixed.field {
                        View {
                            field: (own.field * 1.2).max(fixed.field / 3.0),
                            ..own
                        }
                    } else {
                        fixed.clone()
                    }
                }
                Output::Variation | Output::Lineup => fixed.clone(),
            }
        })
        .collect();
    let (count, rate) = frames(output);
    (0..count)
        .map(|frame| {
            let time = seconds + frame as f32 / rate;
            let panels: Vec<(CreatureInstance, String)> = (0..9)
                .map(|index| panel(kind, output, index, time))
                .collect();
            let tiles: Vec<(CreatureInstance, &View, usize)> = panels
                .iter()
                .zip(&views)
                .map(|((c, _), view)| (*c, view, size))
                .collect();
            let mut grid = RgbImage::new((size * 3) as u32, (size * 3) as u32);
            for (index, mut pixels) in draw(&tiles, output == Output::Anatomy)
                .into_iter()
                .enumerate()
            {
                if !panels[index].1.is_empty() {
                    label(&mut pixels, size, &panels[index].1);
                }
                let panel =
                    RgbImage::from_raw(size as u32, size as u32, pixels).expect("tile pixels");
                imageops::replace(
                    &mut grid,
                    &panel,
                    i64::try_from((index % 3) * size).expect("grid width"),
                    i64::try_from((index / 3) * size).expect("grid height"),
                );
            }
            grid
        })
        .collect()
}

/// Frames saved as a looping, lossy animated WebP, shown `rate` a second; a single frame makes a
/// still. Previews need to show what a creature is and how it moves, not every last detail.
fn save_webp(path: &Path, frames: &[RgbaImage], rate: f32) {
    let (width, height) = frames[0].dimensions();
    let options = EncoderOptions {
        encoding_config: Some(EncodingConfig {
            encoding_type: EncodingType::Lossy(LossyEncodingConfig::default()),
            quality: 78.0,
            method: 4,
        }),
        ..EncoderOptions::default()
    };
    let mut encoder = Encoder::new_with_options((width, height), options).expect("WebP encoder");
    let step = (1000.0 / rate) as i32;
    let mut at = 0;
    for frame in frames {
        encoder
            .add_frame(frame.as_raw(), at)
            .expect("encode WebP frame");
        at += step;
    }
    let data = encoder.finalize(at).expect("finish WebP");
    fs::write(path, &*data).expect("save WebP");
}

/// An animated sheet saved as WebP.
fn save_sheet(path: &Path, frames: Vec<RgbImage>, rate: f32) {
    let frames: Vec<RgbaImage> = frames
        .into_iter()
        .map(|frame| DynamicImage::ImageRgb8(frame).to_rgba8())
        .collect();
    save_webp(path, &frames, rate);
}

/// The taxa where the tree forks, those with more than one branch that has species below it, and
/// every phylum, each a body plan of its own worth a sheet even if only one branch of it lives.
fn forks() -> Vec<&'static Taxon> {
    let mut branches: Vec<(&'static Taxon, Vec<&'static str>)> = Vec::new();
    for &species in SPECIES {
        let mut child: &'static Taxon = species;
        while let Some(parent) = child.parent {
            match branches
                .iter_mut()
                .find(|(taxon, _)| taxon.name == parent.name)
            {
                Some((_, children)) if !children.contains(&child.name) => children.push(child.name),
                Some(_) => {}
                None => branches.push((parent, vec![child.name])),
            }
            child = parent;
        }
    }
    branches
        .into_iter()
        .filter(|(taxon, children)| children.len() > 1 || taxon.rank == Rank::Phylum)
        .map(|(taxon, _)| taxon)
        .collect()
}

/// Where a taxon's own sheets live: its module's folder, beside its `mod.rs`.
fn taxon_folder(taxon: &Taxon) -> PathBuf {
    let mut depth = 0;
    let mut ancestor = taxon;
    while let Some(parent) = ancestor.parent {
        depth += 1;
        ancestor = parent;
    }
    let kind = SPECIES
        .iter()
        .position(|species| species.descends_from(taxon))
        .expect("a fork has species below it");
    let module = SPECIES_MODULES[kind].split_whitespace().collect::<String>();
    module.split("::").take(depth).fold(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/scene/life"),
        |path, part| path.join(part),
    )
}

/// Every species below a taxon on a clear square sheet `SHEET` pixels across, drawn as large as
/// will fit: measured once at a trial size, then drawn again at the size that fills the square.
fn render_lineup(taxon: &Taxon, seconds: f32) -> RgbaImage {
    const TRIAL: usize = 400;
    let trial = render_lineup_at(taxon, TRIAL, seconds);
    let fill = SHEET as f32 / trial.width().max(trial.height()) as f32;
    let mut sheet = render_lineup_at(taxon, (TRIAL as f32 * fill * 0.98) as usize, seconds);
    // Packing does not scale exactly, so the last few pixels are taken up or given back.
    let over = sheet.width().max(sheet.height());
    if over > SHEET {
        let shrink = SHEET as f32 / over as f32;
        sheet = imageops::resize(
            &sheet,
            ((sheet.width() as f32 * shrink) as u32).max(1),
            ((sheet.height() as f32 * shrink) as u32).max(1),
            imageops::FilterType::Lanczos3,
        );
    }
    let mut square = RgbaImage::new(SHEET, SHEET);
    imageops::overlay(
        &mut square,
        &sheet,
        i64::from((SHEET - sheet.width()) / 2),
        i64::from((SHEET - sheet.height()) / 2),
    );
    square
}

/// The side of a lineup sheet.
const SHEET: u32 = 1024;

/// Every species below a taxon, grown, seen from the same side and drawn to one scale on a clear
/// sheet. The largest is `size` pixels across, and none is drawn smaller than a few pixels so the
/// smallest can still be found.
///
/// Each creature is cut out by its own silhouette, and they are nested together largest first:
/// each goes in the first place, scanning down the sheet and across, where its silhouette clears
/// all those already placed by a small margin. Nothing is turned or rescaled to fit.
fn render_lineup_at(taxon: &Taxon, size: usize, seconds: f32) -> RgbaImage {
    let members: Vec<(CreatureInstance, View, f32)> = SPECIES
        .iter()
        .enumerate()
        .filter(|(_, species)| species.descends_from(taxon))
        .map(|(kind, _)| {
            let days = life::KINDS[kind].biology().maturity_days * 1.25;
            let c = specimen(kind, days, 0, seconds);
            let mut view = View::fit(1, &c);
            // Room for wing tips and trailing tails the fit's coarse look can miss.
            view.field *= 1.2;
            view.filled /= 1.2;
            view.backdrop = even_sky;
            view.even = true;
            // How wide the view is in metres at the creature, from how far off the camera sits.
            let across = view.field * view.origin.length();
            (c, view, across)
        })
        .collect();
    let widest = members.iter().map(|m| m.2).fold(0.0f32, f32::max).max(1e-3);
    let sprites: Vec<RgbaImage> = members
        .par_iter()
        .map(|(c, view, metres)| {
            let edge = ((metres / widest * size as f32) as usize).max(12);
            let pixels = draw(&[(*c, view, edge)], false).remove(0);
            // The sky as it was drawn, read from the corner the creature never reaches.
            let sky = Vec3::new(
                f32::from(pixels[0]),
                f32::from(pixels[1]),
                f32::from(pixels[2]),
            );
            // Clear where the sky shows through, and partly clear where a membrane lets it.
            let sprite = RgbaImage::from_fn(edge as u32, edge as u32, |x, y| {
                let at = (y as usize * edge + x as usize) * 3;
                let rgb = [pixels[at], pixels[at + 1], pixels[at + 2]];
                let difference = (f32::from(rgb[0]) - sky.x)
                    .abs()
                    .max((f32::from(rgb[1]) - sky.y).abs())
                    .max((f32::from(rgb[2]) - sky.z).abs());
                let alpha = ((difference - 3.0) / 14.0).clamp(0.0, 1.0);
                image::Rgba([rgb[0], rgb[1], rgb[2], (alpha * 255.0) as u8])
            });
            trim(&sprite)
        })
        .collect();
    nest(&sprites)
}

/// A sprite cut down to the box around what is not clear in it.
fn trim(sprite: &RgbaImage) -> RgbaImage {
    let (mut low, mut high) = ((u32::MAX, u32::MAX), (0, 0));
    for (x, y, pixel) in sprite.enumerate_pixels() {
        if pixel[3] > 8 {
            low = (low.0.min(x), low.1.min(y));
            high = (high.0.max(x), high.1.max(y));
        }
    }
    if low.0 > high.0 {
        return RgbaImage::new(1, 1);
    }
    imageops::crop_imm(sprite, low.0, low.1, high.0 - low.0 + 1, high.1 - low.1 + 1).to_image()
}

/// Sprites nested into one clear sheet by their silhouettes, largest first, each in the first
/// place down and across the sheet where it overlaps none already placed. The sheet's width is the
/// one, of many tried, that leaves it nearest to square.
fn nest(sprites: &[RgbaImage]) -> RgbaImage {
    let shapes: Vec<Silhouette> = sprites.iter().map(Silhouette::of).collect();
    let mut order: Vec<usize> = (0..sprites.len()).collect();
    order.sort_by_key(|&index| Reverse(shapes[index].area()));
    let filled: u32 = shapes.iter().map(Silhouette::area).sum();
    let narrowest = shapes.iter().map(|shape| shape.width).max().unwrap_or(1);
    let side = (filled as f32).sqrt();
    let (placed, width, _) = (0..24)
        .map(|step| narrowest.max((side * (0.9 + step as f32 * 0.05)) as u32))
        .map(|width| {
            let (placed, height) = pack(&shapes, &order, width);
            (placed, width, height)
        })
        .min_by_key(|&(_, width, height)| width.max(height))
        .expect("widths were tried");
    let height = sprites
        .iter()
        .zip(&placed)
        .map(|(sprite, &(_, y))| y * SILHOUETTE_CELL + sprite.height())
        .max()
        .unwrap_or(1);
    let mut sheet = RgbaImage::new(
        width * SILHOUETTE_CELL + SILHOUETTE_CELL,
        height + SILHOUETTE_CELL,
    );
    for (sprite, &(x, y)) in sprites.iter().zip(&placed) {
        imageops::overlay(
            &mut sheet,
            sprite,
            i64::from(x * SILHOUETTE_CELL),
            i64::from(y * SILHOUETTE_CELL),
        );
    }
    sheet
}

/// Silhouettes are matched on a coarse grid of cells this many pixels across, each grown by a cell
/// so neighbours keep a margin.
const SILHOUETTE_CELL: u32 = 3;

/// The cells a sprite covers.
struct Silhouette {
    width: u32,
    height: u32,
    cells: Vec<bool>,
}

impl Silhouette {
    fn of(sprite: &RgbaImage) -> Self {
        let width = sprite.width().div_ceil(SILHOUETTE_CELL);
        let height = sprite.height().div_ceil(SILHOUETTE_CELL);
        let mut cells = vec![false; (width * height) as usize];
        for (x, y, pixel) in sprite.enumerate_pixels() {
            if pixel[3] > 24 {
                let (cx, cy) = (x / SILHOUETTE_CELL, y / SILHOUETTE_CELL);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (gx, gy) = (cx + dx, cy + dy);
                    if gx < width && gy < height {
                        cells[(gy * width + gx) as usize] = true;
                    }
                }
            }
        }
        Self {
            width,
            height,
            cells,
        }
    }

    fn area(&self) -> u32 {
        self.cells.iter().filter(|&&filled| filled).count() as u32
    }

    fn covers(&self, x: u32, y: u32) -> bool {
        self.cells[(y * self.width + x) as usize]
    }
}

/// Where each silhouette goes, in cells, on a sheet `width` cells wide, and how many cells tall
/// the sheet comes out.
fn pack(shapes: &[Silhouette], order: &[usize], width: u32) -> (Vec<(u32, u32)>, u32) {
    let mut taken: Vec<bool> = Vec::new();
    let mut placed = vec![(0, 0); shapes.len()];
    let mut height = 0;
    for &index in order {
        let shape = &shapes[index];
        let fits = |taken: &[bool], x: u32, y: u32| {
            (0..shape.height).all(|sy| {
                (0..shape.width).all(|sx| {
                    !shape.covers(sx, sy)
                        || !taken
                            .get(((y + sy) * width + x + sx) as usize)
                            .copied()
                            .unwrap_or(false)
                })
            })
        };
        // Below everything placed so far there is always room.
        let clear = taken.len() as u32 / width + 1;
        let (x, y) = (0..=clear)
            .flat_map(|y| (0..=width - shape.width).map(move |x| (x, y)))
            .find(|&(x, y)| fits(&taken, x, y))
            .expect("a sheet without end has room");
        let rows = ((y + shape.height) * width) as usize;
        if taken.len() < rows {
            taken.resize(rows, false);
        }
        for sy in 0..shape.height {
            for sx in 0..shape.width {
                if shape.covers(sx, sy) {
                    taken[((y + sy) * width + x + sx) as usize] = true;
                }
            }
        }
        placed[index] = (x, y);
        height = height.max(y + shape.height);
    }
    (placed, height)
}

/// A lineup beside every fork in the tree, or in `output`, for the forks `selected` names.
fn lineups(output: Option<&str>, selected: Option<&str>, seconds: f32, all: bool) {
    for taxon in forks() {
        if selected.is_some_and(|filter| {
            !filter
                .split(',')
                .any(|name| slug(taxon.name).contains(name))
        }) {
            continue;
        }
        let path = output.map_or_else(
            || taxon_folder(taxon).join(format!("{}.webp", slug(taxon.name))),
            |output| {
                let name = if taxon.parent.is_some() {
                    filename(taxon)
                } else {
                    slug(taxon.name)
                };
                Path::new(output).join(format!("{name}.webp"))
            },
        );
        let since = SPECIES
            .iter()
            .enumerate()
            .filter(|(_, species)| species.descends_from(taxon))
            .map(|(kind, _)| sources(kind))
            .max()
            .unwrap_or(SystemTime::UNIX_EPOCH);
        if !all && fresh(&path, since) {
            continue;
        }
        save_webp(&path, &[render_lineup(taxon, seconds)], 1.0);
        println!("{}: {}", taxon.name, path.display());
    }
}

fn main() {
    if env::args().any(|arg| arg == "--help") {
        println!(
            "species_preview [--size 768] [--species comma,separated,names] [--out directory] [--time seconds] [--outputs render,anatomy,lifecycle,variation,lineup]\nBy default, writes all four animated sheets (lossy WebP, traced on the GPU but for anatomy) beside each species source file, redrawing only those older than the code they are drawn from (--all redraws everything), and at each fork in the tree a still lineup of every species below it, named for the taxon, beside its mod.rs. --anatomy selects anatomy only."
        );
        return;
    }
    let tile: usize = option("--size")
        .or_else(|| env::args().nth(1).filter(|v| !v.starts_with('-')))
        .map_or(768, |value| {
            value.parse().expect("tile size must be an integer")
        });
    assert!(
        (32..=2048).contains(&tile),
        "tile size must be between 32 and 2048"
    );
    let output = option("--out");
    // Only previews older than what they are drawn from are drawn again, unless all are asked for.
    let all = env::args().any(|arg| arg == "--all");
    let selected = option("--species");
    let seconds =
        option("--time").map_or(1.0, |value| value.parse().expect("time must be a number"));
    if let Some(output) = &output {
        fs::create_dir_all(output).expect("create renders folder");
    }
    let requested = option("--outputs").unwrap_or_else(|| {
        if env::args().any(|arg| arg == "--anatomy") {
            "anatomy".into()
        } else {
            "render,anatomy,lifecycle,variation,lineup".into()
        }
    });
    let outputs: Vec<_> = requested
        .split(',')
        .map(|name| match name {
            "render" => Output::Render,
            "anatomy" => Output::Anatomy,
            "lifecycle" => Output::Lifecycle,
            "variation" => Output::Variation,
            "lineup" => Output::Lineup,
            _ => panic!("unknown output: {name}"),
        })
        .collect();
    if outputs.contains(&Output::Lineup) {
        lineups(output.as_deref(), selected.as_deref(), seconds, all);
    }
    let outputs: Vec<_> = outputs
        .into_iter()
        .filter(|&output| output != Output::Lineup)
        .collect();
    let mut count = 0;
    for (kind, taxon) in SPECIES.iter().enumerate() {
        if selected.as_ref().is_some_and(|filter| {
            !filter.split(',').any(|name| {
                slug(taxon.name).contains(name) || slug(taxon.english_name).contains(name)
            })
        }) {
            continue;
        }
        count += 1;
        let since = sources(kind);
        let base = output.as_ref().map_or_else(
            || adjacent_preview(kind),
            |output| Path::new(output).join(filename(taxon)),
        );
        for &output in &outputs {
            let path = base.with_file_name(format!(
                "{}_{}.webp",
                base.file_stem()
                    .expect("preview stem")
                    .to_string_lossy()
                    .trim_end_matches("_render"),
                output.name()
            ));
            if !all && fresh(&path, since) {
                continue;
            }
            save_sheet(
                &path,
                render_sheet(kind, tile, output, seconds),
                frames(output).1,
            );
            println!("{}: {}", taxon.name, path.display());
        }
    }
    assert!(
        count > 0 || outputs.is_empty(),
        "no species matched the filter"
    );
}

/// A creature carrying its species' organs, so the tracer draws it cut away to show them.
fn cut_away(mut c: CreatureInstance) -> CreatureInstance {
    let viscera = life::KINDS[c.kind.x as usize].organs();
    let sites = viscera.sites();
    c.kind.w = sites.len().min(life::ORGAN_SLOTS) as u32;
    for (index, site) in sites.iter().take(life::ORGAN_SLOTS).enumerate() {
        let glow = if site.organ.function == Function::AurStore {
            1.5
        } else {
            0.0
        };
        c.organs[index] = [
            site.centre.extend(if site.paired { 1.0 } else { 0.0 }),
            site.radii.extend(glow),
            site.organ.colour.extend(0.0),
        ];
    }
    c
}

/// The creature tracer run on the GPU, headless: each call traces a batch of tiles, one creature
/// to a tile, in a single dispatch.
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    module: wgpu::ShaderModule,
    /// A pipeline for each species drawn so far, built with that species fixed in it.
    pipelines: Mutex<HashMap<u32, wgpu::ComputePipeline>>,
}

impl Gpu {
    fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..wgpu::RequestAdapterOptions::default()
        }))
        .expect("a GPU to draw previews on");
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("previews"),
            required_limits: adapter.limits(),
            ..wgpu::DeviceDescriptor::default()
        }))
        .expect("a device to draw previews on");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene"),
            source: make_spirv(include_bytes!(concat!(env!("OUT_DIR"), "/preview.spv"))),
        });

        Self {
            device,
            queue,
            module,
            pipelines: Mutex::new(HashMap::new()),
        }
    }

    /// The pipeline for one species, its body the only one compiled in.
    fn pipeline(&self, species: u32) -> wgpu::ComputePipeline {
        let mut pipelines = self.pipelines.lock().expect("pipelines");
        pipelines
            .entry(species)
            .or_insert_with(|| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("preview"),
                        layout: None,
                        module: &self.module,
                        entry_point: Some("preview"),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &[("0", f64::from(species))],
                            ..wgpu::PipelineCompilationOptions::default()
                        },
                        cache: None,
                    })
            })
            .clone()
    }

    /// Trace each creature in its view at its own size, returning each tile's linear colour and
    /// coverage, row by row from the top. Every tile must be of the same species.
    fn trace(&self, tiles: &[(CreatureInstance, PreviewView)]) -> Vec<Vec<Vec4>> {
        let species = tiles[0].0.kind.x;
        assert!(
            tiles.iter().all(|(c, _)| c.kind.x == species),
            "one species to a dispatch"
        );
        let pipeline = self.pipeline(species);
        let mut views: Vec<PreviewView> = Vec::with_capacity(tiles.len());
        let mut offset = 0usize;
        let mut creatures: Vec<CreatureInstance> = Vec::with_capacity(tiles.len());
        for (c, view) in tiles {
            let size = view.tile.x as usize;
            let mut view = *view;
            view.tile.y = offset as f32;
            creatures.push(if view.tile.w > 0.5 { cut_away(*c) } else { *c });
            views.push(view);
            offset += size * size;
        }
        let storage = |contents: &[u8]| {
            self.device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let views_buffer = storage(bytemuck::cast_slice(&views));
        let creatures_buffer = storage(bytemuck::cast_slice(&creatures));
        let bytes = (offset * size_of::<Vec4>()) as u64;
        let pixels = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: views_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: creatures_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pixels.as_entire_binding(),
                },
            ],
        });
        let widest = views
            .iter()
            .map(|view| view.tile.x as u32)
            .max()
            .unwrap_or(1);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(widest.div_ceil(8), widest.div_ceil(8), tiles.len() as u32);
        }
        encoder.copy_buffer_to_buffer(&pixels, 0, &readback, 0, bytes);
        self.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| {
            result.expect("read previews back");
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("previews to finish");
        let data = slice.get_mapped_range().expect("previews mapped");
        let all: &[Vec4] = bytemuck::cast_slice(&data);
        views
            .iter()
            .map(|view| {
                let size = view.tile.x as usize;
                let start = view.tile.y as usize;
                all[start..start + size * size].to_vec()
            })
            .collect()
    }
}
