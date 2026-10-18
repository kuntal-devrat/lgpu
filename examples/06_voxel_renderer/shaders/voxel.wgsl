// Voxel Raymarching DDA (Digital Differential Analyzer) Shader
// Designed for native Vulkan 1.3 Dynamic Rendering via lgpu

struct PushConstants {
    cam_pos: vec3<f32>,
    time: f32,
    cam_dir: vec3<f32>,
    aspect: f32,
    cam_up: vec3<f32>,
    mode: u32,
    render_mode: u32,
    sun_angle: f32,
    fov: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

var<push_constant> pc: PushConstants;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Fullscreen procedural triangle covering [-1, 3] x [-1, 3]
    let x = f32(i32(in_vertex_index) / 2) * 4.0 - 1.0;
    let y = f32(i32(in_vertex_index) % 2) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

// -------------------------------------------------------------
// Voxel Material Definitions
// -------------------------------------------------------------
const VOXEL_AIR: u32    = 0u;
const VOXEL_GRASS: u32  = 1u;
const VOXEL_DIRT: u32   = 2u;
const VOXEL_STONE: u32  = 3u;
const VOXEL_SAND: u32   = 4u;
const VOXEL_WATER: u32  = 5u;
const VOXEL_WOOD: u32   = 6u;
const VOXEL_LEAVES: u32 = 7u;
const VOXEL_GOLD: u32   = 8u;
const VOXEL_BRICK: u32  = 9u;

// -------------------------------------------------------------
// Procedural Hash & Noise
// -------------------------------------------------------------
fn hash21(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(123.34, 456.21));
    let d = dot(q, q + 45.32);
    return fract(d);
}

fn hash31(p: vec3<f32>) -> f32 {
    let q = fract(p * vec3<f32>(123.34, 456.21, 789.92));
    let d = dot(q, q + 45.32);
    return fract(d);
}

fn noise2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash21(i + vec2<f32>(0.0, 0.0));
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm2d(p: vec2<f32>) -> f32 {
    var total = 0.0;
    var amp = 0.5;
    var pos = p;
    for (var i = 0; i < 4; i = i + 1) {
        total = total + noise2d(pos) * amp;
        pos = pos * 2.02;
        amp = amp * 0.5;
    }
    return total;
}

// -------------------------------------------------------------
// Procedural Voxel World Query
// -------------------------------------------------------------
fn get_voxel(pos: vec3<i32>, mode: u32) -> u32 {
    let x = pos.x;
    let y = pos.y;
    let z = pos.z;

    // Bounds limit for rendering performance
    if (y < 0 || y > 64) {
        return VOXEL_AIR;
    }

    if (mode == 0u) {
        // --- Mode 0: Mountain Highlands, Lakes & Pine Trees ---
        let fx = f32(x) * 0.035;
        let fz = f32(z) * 0.035;
        let height_fbm = fbm2d(vec2<f32>(fx, fz));
        let ground_height = i32(12.0 + height_fbm * 26.0);

        // Water basin at y <= 16
        if (y <= 16 && y > ground_height) {
            return VOXEL_WATER;
        }

        if (y < ground_height) {
            if (y < ground_height - 3) {
                // Gold veins deep underground
                if (y < 8 && hash31(vec3<f32>(f32(x), f32(y), f32(z))) > 0.96) {
                    return VOXEL_GOLD;
                }
                return VOXEL_STONE;
            }
            if (ground_height <= 18) {
                return VOXEL_SAND;
            }
            return VOXEL_DIRT;
        } else if (y == ground_height) {
            if (ground_height <= 18) {
                return VOXEL_SAND;
            }
            return VOXEL_GRASS;
        }

        // Procedural Pine Trees on high dry ground
        if (ground_height > 18 && ground_height < 32) {
            // Check tree grid
            let cell = vec2<i32>(x / 7, z / 7);
            let tree_pos = cell * 7 + vec2<i32>(3, 3);
            if (x == tree_pos.x && z == tree_pos.y) {
                let tree_h = 5;
                if (y > ground_height && y <= ground_height + tree_h) {
                    return VOXEL_WOOD;
                }
            }
            let dx = abs(x - tree_pos.x);
            let dz = abs(z - tree_pos.y);
            let dy = y - (ground_height + 4);
            if (dy >= 0 && dy <= 3) {
                let radius = 2 - dy / 2;
                if (dx <= radius && dz <= radius && (dx + dz < radius * 2 + 1)) {
                    if (!(dx == 0 && dz == 0 && dy < 2)) {
                        return VOXEL_LEAVES;
                    }
                }
            }
        }

        return VOXEL_AIR;
    } else if (mode == 1u) {
        // --- Mode 1: Floating Voxel Islands ---
        let fx = f32(x) * 0.05;
        let fz = f32(z) * 0.05;
        let fy = f32(y - 28) * 0.09;

        let d = length(vec2<f32>(f32(x % 48 - 24), f32(z % 48 - 24))) * 0.08;
        let density = fbm2d(vec2<f32>(fx, fz)) * 2.0 - fy * fy - d;

        if (density > 0.3) {
            let above = fbm2d(vec2<f32>(fx, fz)) * 2.0 - (fy + 0.09) * (fy + 0.09) - d;
            if (above <= 0.3) {
                return VOXEL_GRASS;
            } else if (fy < -0.8) {
                return VOXEL_STONE;
            } else {
                return VOXEL_DIRT;
            }
        }
        return VOXEL_AIR;
    } else {
        // --- Mode 2: Ancient Voxel Citadel / Pyramid ---
        let cx = abs(x % 64 - 32);
        let cz = abs(z % 64 - 32);
        let dist = max(cx, cz);

        // Step pyramid / citadel walls
        let tier = 28 - (dist / 2);
        if (dist <= 26 && y <= tier) {
            if (y == tier) {
                if (dist == 0 || dist == 6 || dist == 12 || dist == 18) {
                    return VOXEL_GOLD;
                }
                return VOXEL_STONE;
            }
            if (dist == 24 || dist == 16 || dist == 8) {
                // Battlements
                if ((x + z) % 2 == 0 && y == tier + 1) {
                    return VOXEL_BRICK;
                }
            }
            return VOXEL_BRICK;
        }

        // Surrounding flat plain with river
        if (y <= 4) {
            if (abs(x) < 3) {
                return VOXEL_WATER;
            }
            if (abs(x) < 5) {
                return VOXEL_SAND;
            }
            if (y == 4) {
                return VOXEL_GRASS;
            }
            return VOXEL_DIRT;
        }

        return VOXEL_AIR;
    }
}

// -------------------------------------------------------------
// Voxel Color & Palette
// -------------------------------------------------------------
fn get_voxel_color(voxel_type: u32, normal: vec3<f32>, p: vec3<i32>) -> vec3<f32> {
    let noise = hash31(vec3<f32>(f32(p.x), f32(p.y), f32(p.z))) * 0.08 - 0.04;

    switch (voxel_type) {
        case VOXEL_GRASS: {
            if (normal.y > 0.5) {
                // Lush green top
                return vec3<f32>(0.32 + noise, 0.65 + noise, 0.22 + noise);
            } else {
                // Dirt side
                return vec3<f32>(0.45 + noise, 0.30 + noise, 0.16 + noise);
            }
        }
        case VOXEL_DIRT: {
            return vec3<f32>(0.48 + noise, 0.32 + noise, 0.18 + noise);
        }
        case VOXEL_STONE: {
            let gray = 0.52 + noise;
            return vec3<f32>(gray, gray * 0.98, gray * 1.02);
        }
        case VOXEL_SAND: {
            return vec3<f32>(0.86 + noise, 0.79 + noise, 0.55 + noise);
        }
        case VOXEL_WATER: {
            return vec3<f32>(0.15, 0.45, 0.82);
        }
        case VOXEL_WOOD: {
            if (abs(normal.y) > 0.5) {
                return vec3<f32>(0.55, 0.40, 0.25);
            }
            return vec3<f32>(0.35 + noise, 0.22 + noise, 0.12 + noise);
        }
        case VOXEL_LEAVES: {
            return vec3<f32>(0.18 + noise, 0.55 + noise, 0.16 + noise);
        }
        case VOXEL_GOLD: {
            return vec3<f32>(0.95, 0.78, 0.18);
        }
        case VOXEL_BRICK: {
            return vec3<f32>(0.68 + noise, 0.36 + noise, 0.28 + noise);
        }
        default: {
            return vec3<f32>(0.8, 0.2, 0.8);
        }
    }
}

// -------------------------------------------------------------
// Voxel Ambient Occlusion (AO)
// -------------------------------------------------------------
fn calc_ambient_occlusion(p: vec3<i32>, normal: vec3<f32>, mode: u32) -> f32 {
    let n = vec3<i32>(normal);
    var u_axis = vec3<i32>(0, 0, 0);
    var v_axis = vec3<i32>(0, 0, 0);

    if (abs(normal.y) > 0.5) {
        u_axis = vec3<i32>(1, 0, 0);
        v_axis = vec3<i32>(0, 0, 1);
    } else if (abs(normal.x) > 0.5) {
        u_axis = vec3<i32>(0, 1, 0);
        v_axis = vec3<i32>(0, 0, 1);
    } else {
        u_axis = vec3<i32>(1, 0, 0);
        v_axis = vec3<i32>(0, 1, 0);
    }

    let origin = p + n;
    var occluded = 0.0;

    let b_u1 = get_voxel(origin + u_axis, mode) != VOXEL_AIR;
    let b_u2 = get_voxel(origin - u_axis, mode) != VOXEL_AIR;
    let b_v1 = get_voxel(origin + v_axis, mode) != VOXEL_AIR;
    let b_v2 = get_voxel(origin - v_axis, mode) != VOXEL_AIR;

    if (b_u1) { occluded = occluded + 1.0; }
    if (b_u2) { occluded = occluded + 1.0; }
    if (b_v1) { occluded = occluded + 1.0; }
    if (b_v2) { occluded = occluded + 1.0; }

    return clamp(1.0 - occluded * 0.18, 0.28, 1.0);
}

// -------------------------------------------------------------
// Digital Differential Analyzer (DDA) Raymarcher
// -------------------------------------------------------------
struct HitRecord {
    hit: bool,
    voxel_pos: vec3<i32>,
    normal: vec3<f32>,
    voxel_type: u32,
    dist: f32,
    water_hit: bool,
    water_dist: f32,
    water_pos: vec3<f32>,
    water_normal: vec3<f32>,
};

fn raymarch_dda(ray_orig: vec3<f32>, ray_dir: vec3<f32>, max_dist: f32, mode: u32) -> HitRecord {
    var rec: HitRecord;
    rec.hit = false;
    rec.dist = max_dist;
    rec.water_hit = false;
    rec.water_dist = max_dist;
    rec.water_pos = vec3<f32>(0.0);
    rec.water_normal = vec3<f32>(0.0, 1.0, 0.0);

    var pos = vec3<i32>(floor(ray_orig));
    let step_dir = vec3<i32>(sign(ray_dir));

    let delta_dist = abs(vec3<f32>(1.0, 1.0, 1.0) / ray_dir);

    var side_dist: vec3<f32>;
    let cell_frac = fract(ray_orig);

    if (step_dir.x > 0) {
        side_dist.x = (1.0 - cell_frac.x) * delta_dist.x;
    } else {
        side_dist.x = cell_frac.x * delta_dist.x;
    }
    if (step_dir.y > 0) {
        side_dist.y = (1.0 - cell_frac.y) * delta_dist.y;
    } else {
        side_dist.y = cell_frac.y * delta_dist.y;
    }
    if (step_dir.z > 0) {
        side_dist.z = (1.0 - cell_frac.z) * delta_dist.z;
    } else {
        side_dist.z = cell_frac.z * delta_dist.z;
    }

    var mask = vec3<f32>(0.0, 0.0, 0.0);
    var d = 0.0;

    // Check if ray origin is inside water
    let start_v = get_voxel(pos, mode);
    if (start_v == VOXEL_WATER) {
        rec.water_hit = true;
        rec.water_dist = 0.0;
        rec.water_pos = ray_orig;
        rec.water_normal = vec3<f32>(0.0, 1.0, 0.0);
    }

    for (var i = 0; i < 150; i = i + 1) {
        // Early vertical bounds cull for performance
        if (pos.y > 64 && step_dir.y >= 0) {
            break;
        }
        if (pos.y < 0 && step_dir.y <= 0) {
            break;
        }

        let v = get_voxel(pos, mode);
        if (v == VOXEL_WATER) {
            if (!rec.water_hit) {
                rec.water_hit = true;
                rec.water_dist = d;
                rec.water_pos = ray_orig + ray_dir * d;
                rec.water_normal = -vec3<f32>(step_dir) * mask;
            }
            // March through water volume to hit underlying riverbed / terrain
        } else if (v != VOXEL_AIR) {
            rec.hit = true;
            rec.voxel_pos = pos;
            rec.voxel_type = v;
            rec.normal = -vec3<f32>(step_dir) * mask;
            rec.dist = d;
            return rec;
        }

        if (side_dist.x < side_dist.y) {
            if (side_dist.x < side_dist.z) {
                d = side_dist.x;
                side_dist.x = side_dist.x + delta_dist.x;
                pos.x = pos.x + step_dir.x;
                mask = vec3<f32>(1.0, 0.0, 0.0);
            } else {
                d = side_dist.z;
                side_dist.z = side_dist.z + delta_dist.z;
                pos.z = pos.z + step_dir.z;
                mask = vec3<f32>(0.0, 0.0, 1.0);
            }
        } else {
            if (side_dist.y < side_dist.z) {
                d = side_dist.y;
                side_dist.y = side_dist.y + delta_dist.y;
                pos.y = pos.y + step_dir.y;
                mask = vec3<f32>(0.0, 1.0, 0.0);
            } else {
                d = side_dist.z;
                side_dist.z = side_dist.z + delta_dist.z;
                pos.z = pos.z + step_dir.z;
                mask = vec3<f32>(0.0, 0.0, 1.0);
            }
        }

        if (d > max_dist) {
            break;
        }
    }

    return rec;
}

// Hard voxel shadow ray
fn cast_shadow_ray(orig: vec3<f32>, light_dir: vec3<f32>, max_steps: i32, mode: u32) -> f32 {
    var pos = vec3<i32>(floor(orig));
    let step_dir = vec3<i32>(sign(light_dir));
    let delta_dist = abs(vec3<f32>(1.0, 1.0, 1.0) / light_dir);

    var side_dist: vec3<f32>;
    let cell_frac = fract(orig);

    if (step_dir.x > 0) {
        side_dist.x = (1.0 - cell_frac.x) * delta_dist.x;
    } else {
        side_dist.x = cell_frac.x * delta_dist.x;
    }
    if (step_dir.y > 0) {
        side_dist.y = (1.0 - cell_frac.y) * delta_dist.y;
    } else {
        side_dist.y = cell_frac.y * delta_dist.y;
    }
    if (step_dir.z > 0) {
        side_dist.z = (1.0 - cell_frac.z) * delta_dist.z;
    } else {
        side_dist.z = cell_frac.z * delta_dist.z;
    }

    for (var i = 0; i < max_steps; i = i + 1) {
        let v = get_voxel(pos, mode);
        if (v != VOXEL_AIR && v != VOXEL_WATER) {
            return 0.32; // In shadow
        }

        if (side_dist.x < side_dist.y) {
            if (side_dist.x < side_dist.z) {
                side_dist.x = side_dist.x + delta_dist.x;
                pos.x = pos.x + step_dir.x;
            } else {
                side_dist.z = side_dist.z + delta_dist.z;
                pos.z = pos.z + step_dir.z;
            }
        } else {
            if (side_dist.y < side_dist.z) {
                side_dist.y = side_dist.y + delta_dist.y;
                pos.y = pos.y + step_dir.y;
            } else {
                side_dist.z = side_dist.z + delta_dist.z;
                pos.z = pos.z + step_dir.z;
            }
        }
    }
    return 1.0; // In full light
}

// -------------------------------------------------------------
// Fragment Shader
// -------------------------------------------------------------
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Generate camera rays
    let uv_ndc = (in.uv - vec2<f32>(0.5)) * 2.0;
    let aspect = pc.aspect;

    let fwd = normalize(pc.cam_dir);
    let right = normalize(cross(fwd, pc.cam_up));
    let up = cross(right, fwd);

    let half_fov = tan(radians(pc.fov * 0.5));
    let ray_dir = normalize(fwd + right * (uv_ndc.x * aspect * half_fov) - up * (uv_ndc.y * half_fov));

    // Sun light direction from angle
    let sun_rad = radians(pc.sun_angle);
    let sun_dir = normalize(vec3<f32>(cos(sun_rad) * 0.8, sin(sun_rad), sin(sun_rad) * 0.4));

    // Sky gradient & sun disc
    let sky_top = vec3<f32>(0.35, 0.62, 0.95);
    let sky_horizon = vec3<f32>(0.75, 0.85, 0.98);
    let sky_ground = vec3<f32>(0.55, 0.65, 0.75);

    var sky_color = mix(sky_horizon, sky_top, clamp(ray_dir.y * 1.5, 0.0, 1.0));
    if (ray_dir.y < 0.0) {
        sky_color = mix(sky_horizon, sky_ground, clamp(-ray_dir.y * 2.0, 0.0, 1.0));
    }

    // Add sun glare
    let sun_dot = max(dot(ray_dir, sun_dir), 0.0);
    sky_color = sky_color + vec3<f32>(1.0, 0.88, 0.6) * pow(sun_dot, 64.0) * 1.2;

    // Raymarch DDA through voxel world
    let max_render_dist = 160.0;
    let hit = raymarch_dda(pc.cam_pos, ray_dir, max_render_dist, pc.mode);

    if (!hit.hit && !hit.water_hit) {
        return vec4<f32>(sky_color, 1.0);
    }

    // Diagnostic render modes
    if (pc.render_mode == 1u) {
        // Mode 1: Ambient Occlusion Only (Clay render)
        let ao = calc_ambient_occlusion(hit.voxel_pos, hit.normal, pc.mode);
        return vec4<f32>(vec3<f32>(ao), 1.0);
    } else if (pc.render_mode == 2u) {
        // Mode 2: Surface Normal View
        let norm_col = hit.normal * 0.5 + vec3<f32>(0.5);
        return vec4<f32>(norm_col, 1.0);
    } else if (pc.render_mode == 3u) {
        // Mode 3: Depth / Distance
        let depth_val = clamp(hit.dist / max_render_dist, 0.0, 1.0);
        return vec4<f32>(vec3<f32>(1.0 - depth_val), 1.0);
    } else if (pc.render_mode == 4u) {
        // Mode 4: Voxel ID palette
        let hash = hash31(vec3<f32>(f32(hit.voxel_type) * 7.13, 13.5, 41.2));
        let id_col = vec3<f32>(fract(hash * 3.0), fract(hash * 7.0), fract(hash * 11.0));
        return vec4<f32>(id_col, 1.0);
    }

    // --- Mode 0: Full Beauty Lighting ---
    var final_color = vec3<f32>(0.0);

    // 1. Shade solid ground (whether on land or underwater)
    var ground_color = sky_color;
    if (hit.hit) {
        let albedo = get_voxel_color(hit.voxel_type, hit.normal, hit.voxel_pos);
        let ndotl = max(dot(hit.normal, sun_dir), 0.0);
        let ao = calc_ambient_occlusion(hit.voxel_pos, hit.normal, pc.mode);

        let shadow_orig = pc.cam_pos + ray_dir * hit.dist + hit.normal * 0.08;
        var shadow = 1.0;
        if (ndotl > 0.05) {
            shadow = cast_shadow_ray(shadow_orig, sun_dir, 35, pc.mode);
        } else {
            shadow = 0.3;
        }

        let ambient_light = vec3<f32>(0.28, 0.35, 0.45) * ao;
        let sun_color = vec3<f32>(1.1, 1.02, 0.90);
        let diffuse = sun_color * ndotl * shadow * ao;

        ground_color = albedo * (ambient_light + diffuse);
    }

    // 2. Handle Minecraft-style translucent water
    if (hit.water_hit) {
        // Compute water depth: distance ray traveled through water column
        let water_depth = max(hit.dist - hit.water_dist, 0.0);

        // Animated water caustics on underwater seabed
        if (hit.hit) {
            let wp = vec3<f32>(hit.voxel_pos);
            let caustic = (sin((wp.x + wp.z) * 1.2 + pc.time * 2.8) * cos((wp.x - wp.z) * 1.2 + pc.time * 2.2) + 1.0) * 0.14;
            ground_color = ground_color + vec3<f32>(0.1, 0.25, 0.3) * caustic;
        }

        // Beer-Lambert law depth absorption (water absorbs red/green faster than blue)
        let absorb = exp(-water_depth * vec3<f32>(0.16, 0.06, 0.02));
        let water_fog = vec3<f32>(0.08, 0.32, 0.58);
        let underwater_scenery = mix(water_fog, ground_color, absorb);

        // Surface wave perturbation
        var wave_norm = hit.water_normal;
        if (abs(hit.water_normal.y) > 0.5) {
            let p = hit.water_pos;
            let wx = sin(p.z * 1.6 + pc.time * 2.5) * 0.06 + sin(p.x * 2.8 + pc.time * 1.7) * 0.03;
            let wz = cos(p.x * 1.6 + pc.time * 2.3) * 0.06 + cos(p.z * 2.8 + pc.time * 1.9) * 0.03;
            wave_norm = normalize(vec3<f32>(wx, 1.0, wz));
        }

        // Fresnel reflection
        let cos_theta = clamp(-dot(ray_dir, wave_norm), 0.0, 1.0);
        let fresnel = 0.06 + 0.94 * pow(1.0 - cos_theta, 4.0);

        // Sky reflection on water
        let refl_dir = reflect(ray_dir, wave_norm);
        var refl_sky = mix(sky_horizon, sky_top, clamp(refl_dir.y * 1.5, 0.0, 1.0));
        let sun_refl_dot = max(dot(refl_dir, sun_dir), 0.0);
        refl_sky = refl_sky + vec3<f32>(1.0, 0.92, 0.75) * pow(sun_refl_dot, 64.0) * 2.0;

        // Sun specular glint
        let half_vec = normalize(sun_dir - ray_dir);
        let spec = pow(max(dot(wave_norm, half_vec), 0.0), 96.0) * 2.2;
        let specular = vec3<f32>(1.0, 0.96, 0.88) * spec;

        // Composite water: blend underwater scenery with surface sky reflection
        let water_composite = mix(underwater_scenery, refl_sky, clamp(fresnel, 0.20, 0.85)) + specular;

        // Atmospheric distance fog to water surface
        let fog_factor = clamp(hit.water_dist / max_render_dist, 0.0, 1.0);
        final_color = mix(water_composite, sky_horizon, fog_factor * fog_factor);
    } else {
        // Pure land terrain
        let fog_factor = clamp(hit.dist / max_render_dist, 0.0, 1.0);
        final_color = mix(ground_color, sky_horizon, fog_factor * fog_factor);
    }

    // Camera underwater effect
    if (pc.cam_pos.y <= 16.0 && get_voxel(vec3<i32>(floor(pc.cam_pos)), pc.mode) == VOXEL_WATER) {
        let underwater_tint = vec3<f32>(0.08, 0.35, 0.65);
        final_color = mix(final_color, underwater_tint, 0.45);
    }

    return vec4<f32>(final_color, 1.0);
}
