{pkgs ? import <nixpkgs> {}}:
with pkgs;
  mkShell rec {
    nativeBuildInputs = [
      pkg-config
    ];
    buildInputs = [
      udev
      alsa-lib-with-plugins
      vulkan-loader
      mold

      libxkbcommon
      wayland
    ];
    LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
  }
