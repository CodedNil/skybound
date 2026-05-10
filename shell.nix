{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    rustup
    spirv-tools
    pkg-config
    clang
    wayland
    vulkan-headers
  ];
  LIBCLANG_PATH = pkgs.lib.makeLibraryPath [ pkgs.llvmPackages_latest.libclang.lib ];
  LD_LIBRARY_PATH =
    with pkgs;
    lib.makeLibraryPath [
      udev
      alsa-lib-with-plugins
      vulkan-loader
      libxkbcommon
    ]
    + ":/run/opengl-driver/lib:/run/lib-opengl-driver-32/lib";
  DLSS_SDK = "${builtins.getEnv "HOME"}/Documents/DLSS";
  VULKAN_SDK = "${pkgs.vulkan-headers}";
}
