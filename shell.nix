{pkgs ? import <nixpkgs> {}}: let
  overrides = builtins.fromTOML (builtins.readFile ./rust-toolchain.toml);
  libPath = with pkgs;
    lib.makeLibraryPath [
      udev
      alsa-lib-with-plugins
      vulkan-loader

      libxkbcommon
    ];
in
  pkgs.mkShell {
    buildInputs = with pkgs; [
      pkg-config
      clang
      mold
      llvmPackages.bintools
      rustup

      wayland
    ];
    RUSTC_VERSION = overrides.toolchain.channel;
    LIBCLANG_PATH = pkgs.lib.makeLibraryPath [pkgs.llvmPackages_latest.libclang.lib];
    shellHook = ''
      export PATH=$PATH:''${CARGO_HOME:-~/.cargo}/bin
      export PATH=$PATH:''${RUSTUP_HOME:-~/.rustup}/toolchains/$RUSTC_VERSION-x86_64-unknown-linux-gnu/bin/
    '';
    RUSTFLAGS = builtins.map (a: ''-L ${a}/lib'') [
      # add libraries here (e.g. pkgs.libvmi)
    ];
    LD_LIBRARY_PATH = libPath;
    BINDGEN_EXTRA_CLANG_ARGS =
      (builtins.map (a: ''-I"${a}/include"'') [
        pkgs.glibc.dev
      ])
      ++ [
        ''-I"${pkgs.llvmPackages_latest.libclang.lib}/lib/clang/${pkgs.llvmPackages_latest.libclang.version}/include"''
        ''-I"${pkgs.glib.dev}/include/glib-2.0"''
        ''-I${pkgs.glib.out}/lib/glib-2.0/include/''
      ];
  }
