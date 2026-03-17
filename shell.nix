{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    rustc
    cargo
    rustfmt
    clippy

    pkg-config
    clang

    wayland
  ];
  LD_LIBRARY_PATH =
    with pkgs;
    lib.makeLibraryPath [
      udev
      alsa-lib-with-plugins
      vulkan-loader
      libxkbcommon
    ];
}
