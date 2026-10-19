//! Tunables of the example: the planet, the player and what streams around them.

/// Planet radius (m). Small enough to circle in a few hours on foot, large
/// enough that the horizon visibly curves.
pub const PLANET_RADIUS_M: f64 = 16_000.0;
/// One block is one voxel of this size (m).
pub const VOXEL_M: f64 = 1.0;
/// Seed of the terrain and of every placed feature.
pub const WORLD_SEED: u64 = 7;

/// Side of the square regions features (trees, structures) are decided in, in
/// blocks. Each region is generated from its own hash of the world seed, so a
/// region always comes out the same wherever it is generated from.
pub const REGION_BLOCKS: i32 = 32;
/// Regions decorated around the player, as a radius in regions.
pub const REGION_RADIUS: i32 = 2048;
/// Most regions the world worker decorates in parallel per pass.
pub const DECORATE_BATCH: usize = 48;
/// Regions the generator may hand to the edit worker per frame.
pub const REGIONS_PER_FRAME: usize = 12;

/// Walking, running and flying speeds (m/s).
pub const WALK_SPEED: f64 = 4.3;
pub const SPRINT_SPEED: f64 = 6.6;
pub const FLY_SPEED: f64 = 14.0;
pub const FLY_FAST_SPEED: f64 = 60.0;
/// Gravity (m/s^2) and the jump's launch speed (about 1.3 blocks high).
pub const GRAVITY: f64 = 30.0;
pub const JUMP_SPEED: f64 = 8.8;
/// The player's collision box: width and height (m), and the eye above feet.
pub const PLAYER_WIDTH: f64 = 0.6;
pub const PLAYER_HEIGHT: f64 = 1.8;
pub const EYE_HEIGHT: f64 = 1.62;
/// How far blocks can be reached (m).
pub const REACH: f64 = 6.0;
/// Third person camera distance behind the character (m).
pub const THIRD_PERSON_DISTANCE: f64 = 4.5;

/// A day's length in seconds of play.
pub const DAY_SECONDS: f64 = 600.0;
