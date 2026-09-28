{
  description = "Skybound development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs =
    { nixpkgs, rust-overlay, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      };
      rustToolchain = pkgs.rust-bin.nightly."2026-07-03".default.override {
        extensions = [
          "rust-src"
          "rustc-dev"
          "llvm-tools"
        ];
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          rustToolchain
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
        VULKAN_SDK = "${pkgs.vulkan-headers}";
      };
    };
}
