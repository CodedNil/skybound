# Vendored NVIDIA DLSS SDK

The parts of NVIDIA's DLSS Super Resolution SDK **v310.9.16** that Bevy's
`dlss` feature needs on Linux x86-64, from
<https://github.com/NVIDIA/DLSS/tree/v310.9.16>. Every file is byte for byte
as released (checked against the release's git blob hashes).

It is used under NVIDIA's DLSS SDK license (`LICENSE.txt`). Anything that
ships the game must ship `lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.*` and
`NOTICES.txt`, which holds the copyright and license notices from sections
9.5 and 9.6 of the SDK's programming guide, as that license requires.

## What is kept

- `include/`: the 13 NGX headers that `dlss_wgpu`'s bindings include,
  directly or through each other. The loader, standalone, frame generation
  and deprecated headers are left out.
- `lib/Linux_x86_64/libnvsdk_ngx.a`: the NGX loader, linked in at build time.
- `lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.310.9.16`: DLSS Super
  Resolution itself, loaded at run time.

Ray reconstruction (`dlssd`), frame generation (`dlssg`), the development
builds and the Windows and ARM libraries are left out. To support Windows,
add `lib/Windows_x86_64/x64/nvsdk_ngx_*.lib` and
`lib/Windows_x86_64/rel/nvngx_dlss.dll` from the same release.

## How it is found

`DLSS_SDK` points here: set in `.cargo/config.toml` for cargo, and by the
flake's dev shell for running the built binary. `dlss_wgpu` builds against
the headers and libraries here, and loads the runtime library from
`$DLSS_SDK/lib/Linux_x86_64/rel`.
