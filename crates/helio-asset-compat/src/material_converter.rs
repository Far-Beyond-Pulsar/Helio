//! PBR material mapping from SolidRS to Helio

use crate::{AssetError, Result};
use helio_render_v2::material::{Material, TextureData};
use solid_rs::scene::{Material as SolidMaterial, Scene, AlphaMode};

/// Convert a SolidRS material to Helio's PBR material format
///
/// Note: This function doesn't load GPU textures directly - it extracts TextureData
/// that Helio's material system will upload to the GPU.
pub fn convert_material(
    material: &SolidMaterial,
    scene: &Scene,
    base_dir: &std::path::Path,
) -> Result<Material> {
    let mut helio_mat = Material::new();

    // Base color factor (RGBA)
    helio_mat.base_color = [
        material.base_color_factor.x,
        material.base_color_factor.y,
        material.base_color_factor.z,
        material.base_color_factor.w,
    ];

    log::info!("  Material '{}' base_color: [{:.3}, {:.3}, {:.3}, {:.3}]",
        material.name,
        helio_mat.base_color[0],
        helio_mat.base_color[1],
        helio_mat.base_color[2],
        helio_mat.base_color[3]);

    // Metallic and roughness factors
    helio_mat.metallic = material.metallic_factor;
    helio_mat.roughness = material.roughness_factor;

    // Occlusion strength (Helio uses ao factor)
    helio_mat.ao = material.occlusion_strength;

    // Emissive
    helio_mat.emissive_color = [
        material.emissive_factor.x,
        material.emissive_factor.y,
        material.emissive_factor.z,
    ];
    helio_mat.emissive_factor = 1.0; // SolidRS bakes intensity into emissive_factor

    // Alpha mode
    match material.alpha_mode {
        AlphaMode::Opaque => {
            helio_mat.alpha_cutoff = 0.0;
            helio_mat.transparent_blend = false;
        }
        AlphaMode::Mask => {
            helio_mat.alpha_cutoff = material.alpha_cutoff;
            helio_mat.transparent_blend = false;
        }
        AlphaMode::Blend => {
            helio_mat.alpha_cutoff = 0.0;
            helio_mat.transparent_blend = true;
        }
    }

    // Load textures - CHECK FOR UV TRANSFORMS
    if let Some(tex_ref) = &material.base_color_texture {
        if let Some(ref t) = tex_ref.transform {
            log::warn!("⚠️  Base color texture HAS UV TRANSFORM: offset=({:.3}, {:.3}), scale=({:.3}, {:.3}), rot={:.1}° - NOT YET APPLIED!",
                t.offset.x, t.offset.y, t.scale.x, t.scale.y, t.rotation.to_degrees());
        }
        println!("    Loading texture for material '{}' (texture_index={})", material.name, tex_ref.texture_index);
        helio_mat.base_color_texture = Some(load_texture_data(tex_ref, scene, base_dir, true)?);

        // FIX: When base_color texture is present, override base_color factor to white
        // FBX exports often have incorrect base_color_factor that darkens/tints textures
        println!("    🔧 Overriding base_color from [{:.3}, {:.3}, {:.3}] to [1.0, 1.0, 1.0]",
            helio_mat.base_color[0], helio_mat.base_color[1], helio_mat.base_color[2]);
        helio_mat.base_color = [1.0, 1.0, 1.0, 1.0];
    }

    if let Some(tex_ref) = &material.normal_texture {
        if let Some(ref t) = tex_ref.transform {
            log::warn!("⚠️  Normal texture HAS UV TRANSFORM: offset=({:.3}, {:.3}), scale=({:.3}, {:.3}), rot={:.1}° - NOT YET APPLIED!",
                t.offset.x, t.offset.y, t.scale.x, t.scale.y, t.rotation.to_degrees());
        }
        helio_mat.normal_map = Some(load_texture_data(tex_ref, scene, base_dir, false)?);
    }

    if let Some(orm_texture) = build_orm_texture(material, scene, base_dir)? {
        helio_mat.orm_texture = Some(orm_texture);
    }

    if let Some(tex_ref) = &material.emissive_texture {
        if let Some(ref t) = tex_ref.transform {
            log::warn!("⚠️  Emissive texture HAS UV TRANSFORM: offset=({:.3}, {:.3}), scale=({:.3}, {:.3}), rot={:.1}° - NOT YET APPLIED!",
                t.offset.x, t.offset.y, t.scale.x, t.scale.y, t.rotation.to_degrees());
        }
        helio_mat.emissive_texture = Some(load_texture_data(tex_ref, scene, base_dir, true)?);
    }

    Ok(helio_mat)
}

fn build_orm_texture(
    material: &SolidMaterial,
    scene: &Scene,
    base_dir: &std::path::Path,
) -> Result<Option<TextureData>> {
    let metallic_roughness = if let Some(tex_ref) = &material.metallic_roughness_texture {
        if let Some(ref t) = tex_ref.transform {
            log::warn!("⚠️  Metallic-roughness texture HAS UV TRANSFORM: offset=({:.3}, {:.3}), scale=({:.3}, {:.3}), rot={:.1}° - NOT YET APPLIED!",
                t.offset.x, t.offset.y, t.scale.x, t.scale.y, t.rotation.to_degrees());
        }
        Some(load_texture_data(tex_ref, scene, base_dir, false)?)
    } else {
        None
    };

    let occlusion = if let Some(tex_ref) = &material.occlusion_texture {
        if let Some(ref t) = tex_ref.transform {
            log::warn!("⚠️  Occlusion texture HAS UV TRANSFORM: offset=({:.3}, {:.3}), scale=({:.3}, {:.3}), rot={:.1}° - NOT YET APPLIED!",
                t.offset.x, t.offset.y, t.scale.x, t.scale.y, t.rotation.to_degrees());
        }
        Some(load_texture_data(tex_ref, scene, base_dir, false)?)
    } else {
        None
    };

    let Some((width, height)) = metallic_roughness
        .as_ref()
        .map(|texture| (texture.width, texture.height))
        .or_else(|| occlusion.as_ref().map(|texture| (texture.width, texture.height)))
    else {
        return Ok(None);
    };

    let metallic_roughness = metallic_roughness
        .as_ref()
        .map(|texture| resize_texture_if_needed(texture, width, height));
    let occlusion = occlusion
        .as_ref()
        .map(|texture| resize_texture_if_needed(texture, width, height));

    let mut orm = vec![255u8; width as usize * height as usize * 4];
    for pixel_index in 0..(width as usize * height as usize) {
        let base = pixel_index * 4;
        orm[base] = occlusion.as_ref().map_or(255, |texture| texture.data[base]);
        orm[base + 1] = metallic_roughness
            .as_ref()
            .map_or(255, |texture| texture.data[base + 1]);
        orm[base + 2] = metallic_roughness
            .as_ref()
            .map_or(255, |texture| texture.data[base + 2]);
        orm[base + 3] = 255;
    }

    log::info!(
        "  Packed ORM texture for material '{}' ({}x{}, ao={}, metallic_roughness={})",
        material.name,
        width,
        height,
        occlusion.is_some(),
        metallic_roughness.is_some()
    );

    Ok(Some(TextureData::new(orm, width, height)))
}

fn resize_texture_if_needed(texture: &TextureData, width: u32, height: u32) -> TextureData {
    if texture.width == width && texture.height == height {
        return TextureData::new(texture.data.clone(), texture.width, texture.height);
    }

    let image = image::RgbaImage::from_raw(texture.width, texture.height, texture.data.clone())
        .expect("TextureData should always contain tightly-packed RGBA8 pixels");
    let resized = image::imageops::resize(&image, width, height, image::imageops::FilterType::Triangle);
    TextureData::new(resized.into_raw(), width, height)
}

/// Load texture data from a SolidRS TextureRef
fn load_texture_data(
    tex_ref: &solid_rs::scene::TextureRef,
    scene: &Scene,
    base_dir: &std::path::Path,
    _srgb: bool,
) -> Result<TextureData> {
    // Get the texture
    let texture = scene.textures.get(tex_ref.texture_index)
        .ok_or_else(|| AssetError::InvalidData(format!(
            "Texture index {} out of bounds", tex_ref.texture_index
        )))?;

    // Get the image
    let image = scene.images.get(texture.image_index)
        .ok_or_else(|| AssetError::InvalidData(format!(
            "Image index {} out of bounds", texture.image_index
        )))?;

    // Extract image data
    use solid_rs::scene::ImageSource;
    use std::path::{Path, PathBuf};

    // Log which image we're loading
    println!("      → Image source: {}", match &image.source {
        ImageSource::Embedded { .. } => "<embedded>".to_string(),
        ImageSource::Uri(uri) => uri.clone(),
    });

    let data_vec: Vec<u8>;
    let image_data = match &image.source {
        ImageSource::Embedded { data, .. } => data.as_slice(),
        ImageSource::Uri(uri_path) => {
            // Smart path resolution: try multiple locations
            let resolved_path = match resolve_texture_path(uri_path, base_dir) {
                Some(path) => {
                    log::debug!("Resolved texture '{}' to '{}'", uri_path, path.display());
                    path
                }
                None => {
                    // Build helpful error message showing where we looked
                    let filename = Path::new(uri_path).file_name()
                        .and_then(|f| f.to_str())
                        .unwrap_or("<unknown>");

                    let mut searched_paths = Vec::new();

                    // Strategy 1: Path as-is
                    searched_paths.push(format!("  1. {}", uri_path));

                    // Strategy 2: Filename in base_dir
                    searched_paths.push(format!("  2. {}", base_dir.join(filename).display()));

                    // Strategy 3: base_dir/<stem>.fbm/filename
                    if let Some(stem) = base_dir.file_stem() {
                        searched_paths.push(format!("  3. {}",
                            base_dir.join(format!("{}.fbm", stem.to_string_lossy())).join(filename).display()));
                    }

                    // Strategy 4: Scan for ANY .fbm directory
                    if let Ok(entries) = std::fs::read_dir(base_dir) {
                        let mut found_fbm_dirs = false;
                        for entry in entries.flatten() {
                            if let Ok(name) = entry.file_name().into_string() {
                                if name.ends_with(".fbm") {
                                    found_fbm_dirs = true;
                                    searched_paths.push(format!("  4. {}", entry.path().join(filename).display()));
                                }
                            }
                        }
                        if !found_fbm_dirs {
                            searched_paths.push(format!("  4. (no .fbm directories found in {})", base_dir.display()));
                        }
                    }

                    // Strategy 5: textures/ subdirectory
                    searched_paths.push(format!("  5. {}", base_dir.join("textures").join(filename).display()));

                    return Err(AssetError::InvalidData(format!(
                        "Could not find texture file '{}'\n\nSearched:\n{}\n\nCurrent directory: {}\nBase directory: {}\n\nTip: Copy the .fbm folder to the same directory as the model file.",
                        filename,
                        searched_paths.join("\n"),
                        std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_else(|_| "?".to_string()),
                        base_dir.display()
                    )));
                }
            };

            data_vec = std::fs::read(&resolved_path)
                .map_err(|e| AssetError::InvalidData(format!(
                    "Failed to read image file '{}': {}",
                    resolved_path.display(), e
                )))?;
            data_vec.as_slice()
        }
    };

    // Decode image to RGBA8
    let decoded = image::load_from_memory(image_data)
        .map_err(|e| AssetError::InvalidData(format!("Failed to decode image: {}", e)))?;

    let rgba = decoded.to_rgba8();
    let (width, height) = rgba.dimensions();

    println!("      → Decoded: '{}' ({}x{}, {} KB)",
        texture.name, width, height, rgba.len() / 1024);

    Ok(TextureData::new(rgba.into_raw(), width, height))
}

/// Resolve texture path: FBX/glTF files often contain absolute paths that are only
/// valid on the machine where they were exported. This function tries multiple
/// strategies to find the actual texture file:
///
/// 1. Try the path as-is (in case it's a valid absolute or relative path)
/// 2. Extract just the filename and look in base_dir
/// 3. Look in base_dir/.fbm/ (FBX convention)
/// 4. Look in base_dir/textures/
/// 5. Try alternate extensions (.jpg <-> .jpeg, .png <-> .PNG)
fn resolve_texture_path(uri_path: &str, base_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    use std::path::{Path, PathBuf};

    let uri_path = Path::new(uri_path);

    // Strategy 1: Try the path as-is
    if uri_path.exists() {
        return Some(uri_path.to_path_buf());
    }

    // Extract the filename
    let filename = uri_path.file_name()?;

    // Helper: Try a path with alternate extensions
    let try_with_extensions = |base_path: PathBuf| -> Option<PathBuf> {
        if base_path.exists() {
            return Some(base_path);
        }

        // Try alternate extensions (jpg <-> jpeg, png <-> PNG, etc.)
        if let Some(ext) = base_path.extension() {
            let ext_str = ext.to_str()?;
            let alternate_exts = match ext_str.to_lowercase().as_str() {
                "jpg" => vec!["jpeg", "JPG", "JPEG"],
                "jpeg" => vec!["jpg", "JPG", "JPEG"],
                "png" => vec!["PNG"],
                "tga" => vec!["TGA"],
                _ => vec![],
            };

            for alt_ext in alternate_exts {
                let alt_path = base_path.with_extension(alt_ext);
                if alt_path.exists() {
                    return Some(alt_path);
                }
            }
        }
        None
    };

    // Strategy 2: Look in the base directory
    if let Some(path) = try_with_extensions(base_dir.join(filename)) {
        return Some(path);
    }

    // Strategy 3: Look in .fbm directory (FBX convention)
    if let Some(stem) = base_dir.file_stem() {
        let fbm_path = base_dir.join(format!("{}.fbm", stem.to_string_lossy())).join(filename);
        if let Some(path) = try_with_extensions(fbm_path) {
            return Some(path);
        }
    }

    // Strategy 4: Look for any .fbm directory
    if let Ok(entries) = std::fs::read_dir(base_dir) {
        for entry in entries.flatten() {
            if let Ok(name) = entry.file_name().into_string() {
                if name.ends_with(".fbm") {
                    let fbm_path = entry.path().join(filename);
                    if let Some(path) = try_with_extensions(fbm_path) {
                        return Some(path);
                    }
                }
            }
        }
    }

    // Strategy 5: Look in textures/ subdirectory
    if let Some(path) = try_with_extensions(base_dir.join("textures").join(filename)) {
        return Some(path);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageFormat, RgbaImage};
    use solid_rs::glam::{Vec3, Vec4};
    use solid_rs::scene::{Image, ImageSource, Texture, TextureRef};
    use std::io::Cursor;

    fn encode_png(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
        let image = RgbaImage::from_raw(width, height, rgba.to_vec()).unwrap();
        let mut cursor = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(image)
            .write_to(&mut cursor, ImageFormat::Png)
            .unwrap();
        cursor.into_inner()
    }

    #[test]
    fn test_convert_basic_material() {
        let mut scene = Scene::default();
        let solid_mat = SolidMaterial {
            name: "TestMaterial".to_string(),
            base_color_factor: Vec4::new(1.0, 0.5, 0.25, 1.0),
            metallic_factor: 0.8,
            roughness_factor: 0.6,
            specular_color: Vec3::ONE,
            specular_weight: 1.0,
            ior: 1.5,
            occlusion_strength: 1.0,
            emissive_factor: Vec3::ZERO,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            base_color_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            metallic_roughness_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
            double_sided: false,
            extensions: Default::default(),
        };

        let helio_mat = convert_material(&solid_mat, &scene, std::path::Path::new(".")).unwrap();

        assert_eq!(helio_mat.base_color, [1.0, 0.5, 0.25, 1.0]);
        assert_eq!(helio_mat.metallic, 0.8);
        assert_eq!(helio_mat.roughness, 0.6);
        assert_eq!(helio_mat.alpha_cutoff, 0.0); // Opaque mode
        assert!(!helio_mat.transparent_blend);
    }

    #[test]
    fn test_convert_material_packs_orm_texture() {
        let mut scene = Scene::default();
        scene.images.push(Image {
            name: "mr".to_string(),
            source: ImageSource::Embedded {
                mime_type: "image/png".to_string(),
                data: encode_png(
                    &[
                        0, 32, 64, 255,
                        0, 96, 160, 255,
                    ],
                    2,
                    1,
                ),
            },
            extensions: Default::default(),
        });
        scene.images.push(Image {
            name: "ao".to_string(),
            source: ImageSource::Embedded {
                mime_type: "image/png".to_string(),
                data: encode_png(
                    &[
                        200, 0, 0, 255,
                        120, 0, 0, 255,
                    ],
                    2,
                    1,
                ),
            },
            extensions: Default::default(),
        });
        scene.textures.push(Texture::new("mr_tex", 0));
        scene.textures.push(Texture::new("ao_tex", 1));

        let solid_mat = SolidMaterial {
            name: "OrmMaterial".to_string(),
            base_color_factor: Vec4::ONE,
            metallic_factor: 0.8,
            roughness_factor: 0.6,
            specular_color: Vec3::ONE,
            specular_weight: 1.0,
            ior: 1.5,
            occlusion_strength: 0.75,
            emissive_factor: Vec3::ZERO,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            base_color_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            metallic_roughness_texture: Some(TextureRef::new(0)),
            occlusion_texture: Some(TextureRef::new(1)),
            emissive_texture: None,
            double_sided: false,
            extensions: Default::default(),
        };

        let helio_mat = convert_material(&solid_mat, &scene, std::path::Path::new(".")).unwrap();

        assert_eq!(helio_mat.metallic, 0.8);
        assert_eq!(helio_mat.roughness, 0.6);
        assert_eq!(helio_mat.ao, 0.75);

        let orm = helio_mat.orm_texture.expect("expected packed ORM texture");
        assert_eq!((orm.width, orm.height), (2, 1));
        assert_eq!(
            orm.data,
            vec![
                200, 32, 64, 255,
                120, 96, 160, 255,
            ]
        );
    }
}
