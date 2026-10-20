"""Bakes voxel_craft's block textures from ambientCG materials (CC0).

Usage: python tools/bake_textures.py <maps dir> <out dir>

<maps dir> holds one folder per material, the files of ambientCG's 1K-JPG
download (https://ambientcg.com/get?file=<Id>_1K-JPG.zip): <Id>_1K-JPG_Color,
_NormalGL, _Roughness and, where it has them, _AmbientOcclusion and _Opacity.
<out dir> is assets/textures/pbr: per block surface <name>_color.jpg (sRGB),
<name>_normal.jpg (OpenGL tangent space, linear) and <name>_orm.jpg
(occlusion, roughness, metalness: linear), SIZE pixels square.

Every texture is seamless; a 2:1 one is stacked to a square. Log ends are
the middle of a sawn slice, and the grass side is the dirt under a ragged
band of the grass top.
"""

import os
import sys
import random

from PIL import Image, ImageFilter

SIZE = 512

# Block surface -> ambientCG material.
SOURCES = {
    "grass_top": "Grass004",
    "dirt": "Ground106",
    "stone": "Rock060",
    "andesite": "Rock051",
    "sand": "Ground080",
    "snow": "Snow006",
    "gravel": "Gravel040",
    "sandstone": "Bricks084",
    "sandstone_top": "Ground093A",
    "clay": "Clay007",
    "bricks": "Bricks085",
    "cobblestone": "Bricks102",
    "stone_bricks": "Bricks066",
    "oak_planks": "WoodFloor044",
    "dark_oak_planks": "Planks037A",
    "oak_log": "Bark012",
    "birch_log": "Bark007",
    "spruce_log": "Bark006",
    "jungle_log": "Bark015",
    "oak_log_top": "TreeEnd003",
    "birch_log_top": "TreeEnd004",
    "spruce_log_top": "TreeEnd002",
    "jungle_log_top": "TreeEnd005",
}


def load(maps, material, kind):
    path = os.path.join(maps, material, f"{material}_1K-JPG_{kind}.jpg")
    return Image.open(path) if os.path.exists(path) else None


def square(image):
    """A seamless texture made square: a 2:1 one stacked twice."""
    w, h = image.size
    if w == h:
        return image
    assert w == 2 * h, image.size
    out = Image.new(image.mode, (w, w))
    out.paste(image, (0, 0))
    out.paste(image, (0, h))
    return out


def slice_middle(image):
    """The middle of a sawn slice: the square inside its rim."""
    w, h = image.size
    side = int(min(w, h) * 0.5)
    x, y = (w - side) // 2, (h - side) // 2
    return image.crop((x, y, x + side, y + side))


def maps_of(maps_dir, material, log_end):
    colour = load(maps_dir, material, "Color").convert("RGB")
    normal = load(maps_dir, material, "NormalGL").convert("RGB")
    rough = load(maps_dir, material, "Roughness").convert("L")
    ao = load(maps_dir, material, "AmbientOcclusion")
    ao = ao.convert("L") if ao else Image.new("L", colour.size, 255)
    fit = slice_middle if log_end else square
    resize = lambda im: fit(im).resize((SIZE, SIZE), Image.LANCZOS)
    colour, normal, rough, ao = (resize(m) for m in (colour, normal, rough, ao))
    orm = Image.merge("RGB", (ao, rough, Image.new("L", (SIZE, SIZE), 0)))
    return colour, normal, orm


def grass_side(dirt, grass):
    """Dirt under a band of the grass top, its lower edge ragged (a smooth
    random height a column, seamless across the tile's sides)."""
    rng = random.Random(7)
    knots = [rng.uniform(0.16, 0.30) for _ in range(16)]
    mask = Image.new("L", (SIZE, SIZE), 0)
    px = mask.load()
    for x in range(SIZE):
        t = x / SIZE * len(knots)
        i = int(t)
        f = t - i
        f = f * f * (3 - 2 * f)
        h = knots[i % len(knots)] * (1 - f) + knots[(i + 1) % len(knots)] * f
        # Blades: a short jag a few pixels wide.
        h += rng.uniform(-0.025, 0.025)
        for y in range(int(h * SIZE)):
            px[x, y] = 255
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    return tuple(Image.composite(g, d, mask) for d, g in zip(dirt, grass))


def save(out, name, maps):
    for suffix, image in zip(("color", "normal", "orm"), maps):
        image.save(os.path.join(out, f"{name}_{suffix}.jpg"), quality=90, subsampling=0 if suffix != "color" else 2)


def main():
    maps_dir, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    baked = {}
    for name, material in SOURCES.items():
        baked[name] = maps_of(maps_dir, material, name.endswith("_log_top"))
        save(out, name, baked[name])
    save(out, "grass_side", grass_side(baked["dirt"], baked["grass_top"]))
    print(f"baked {len(baked) + 1} surfaces into {out}")


if __name__ == "__main__":
    main()
