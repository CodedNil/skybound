use godot::prelude::*;

mod entity;
mod raymarch;

struct SkyboundExtension;

#[gdextension]
unsafe impl ExtensionLibrary for SkyboundExtension {}
