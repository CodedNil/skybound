use godot::prelude::*;

mod craft;
mod gpt;

struct SkyboundExtension;

#[gdextension]
unsafe impl ExtensionLibrary for SkyboundExtension {}
