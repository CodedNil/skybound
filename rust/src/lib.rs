use godot::prelude::*;

mod craft;

struct SkyboundExtension;

#[gdextension]
unsafe impl ExtensionLibrary for SkyboundExtension {}
