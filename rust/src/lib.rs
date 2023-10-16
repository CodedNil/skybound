use godot::prelude::*;

mod craft;
mod entity;
mod gpt;

struct SkyboundExtension;

#[gdextension]
unsafe impl ExtensionLibrary for SkyboundExtension {}
