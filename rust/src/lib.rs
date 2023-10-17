use godot::prelude::*;

mod entity;

struct SkyboundExtension;

#[gdextension]
unsafe impl ExtensionLibrary for SkyboundExtension {}
