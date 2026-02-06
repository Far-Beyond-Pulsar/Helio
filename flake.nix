{
  description = "Helio Rust project with Wayland and X11 dependencies";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            pkgs.wayland
            pkgs.wayland-protocols
            pkgs.libxkbcommon
          ];
          buildInputs = [
            pkgs.wayland
            pkgs.wayland-protocols
            pkgs.xkeyboard_config
            pkgs.rustc
            pkgs.cargo
            pkgs.pkg-config
            pkgs.wayland
            pkgs.wayland-protocols
            pkgs.libxkbcommon
            pkgs.udev
            pkgs.xorg.libX11
            pkgs.xorg.libXcursor
            pkgs.xorg.libXi
            pkgs.xorg.libXrandr
            pkgs.xorg.libXinerama
            pkgs.xorg.libXext
            pkgs.xorg.libxcb
            pkgs.xorg.libXrender
            pkgs.xorg.libXfixes
            pkgs.xorg.libXdmcp
            pkgs.xorg.libXtst
            pkgs.xorg.libXScrnSaver
            pkgs.xorg.libSM
            pkgs.xorg.libICE
            pkgs.vulkan-loader
            pkgs.vulkan-tools
            pkgs.vulkan-headers
            pkgs.mesa
            pkgs.alsa-lib
            pkgs.dbus
            pkgs.fontconfig
            pkgs.freetype
            pkgs.libGL
            pkgs.libGLU
            pkgs.libdrm
            pkgs.libinput
            pkgs.libxkbfile
            pkgs.xorg.xcbutil
            pkgs.xorg.xcbutilwm
            pkgs.xorg.xcbutilimage
            pkgs.xorg.xcbutilkeysyms
            pkgs.xorg.xcbutilrenderutil
            pkgs.xorg.xcbutilcursor
          ];
          shellHook = ''
            export RUST_BACKTRACE=1; \
            export LD_LIBRARY_PATH="${pkgs.wayland}/lib:${pkgs.xorg.libX11}/lib:${pkgs.xorg.libxcb}/lib:${pkgs.xorg.libXcursor}/lib:${pkgs.xorg.libXi}/lib:${pkgs.xorg.libXrandr}/lib:${pkgs.xorg.libXinerama}/lib:${pkgs.xorg.libXext}/lib:${pkgs.libxkbcommon}/lib:${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH"; \
            export XKB_CONFIG_ROOT="${pkgs.xkeyboard_config}/share/X11/xkb"; \
            echo "[nix develop] LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"; \
            echo "[nix develop] XKB_CONFIG_ROOT set to: $XKB_CONFIG_ROOT";
          '';
        };
      }
    );
}
