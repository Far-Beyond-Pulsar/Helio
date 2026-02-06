{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
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
    export RUST_BACKTRACE=1
  '';
}
