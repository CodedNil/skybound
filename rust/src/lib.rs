use godot::prelude::*;

mod craft;
mod gpt;
mod vec3;

struct SkyboundExtension;

#[gdextension]
unsafe impl ExtensionLibrary for SkyboundExtension {}
