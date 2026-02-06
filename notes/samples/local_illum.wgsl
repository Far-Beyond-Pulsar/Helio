const PI : f32 = 3.14159265359;

/* ===== Vertex → Fragment inputs ===== */
struct FragmentInput {
    @location(0) interpNormal : vec3<f32>,
    @location(1) worldPos     : vec3<f32>,
};

/* ===== Fragment output ===== */
struct FragmentOutput {
    @location(0) fragColor : vec3<f32>,
};

/* ===== Uniforms ===== */
struct Uniforms {
    uLightDir      : vec3<f32>,
    uCameraPos     : vec3<f32>,
    uDebugView     : i32,
    uDistribution  : i32,
    uUseOrenNayar  : i32,   // bool → i32 (WGSL-friendly)
    uRoughness     : f32,
    uMetallness    : f32,
    uAlbedo        : vec3<f32>,
};

@group(0) @binding(0)
var<uniform> u : Uniforms;

/* ===== Fresnel ===== */
fn F_schlickApprox(HdotV : f32, R0 : vec3<f32>) -> vec3<f32> {
    return R0 + (vec3<f32>(1.0) - R0) * pow(1.0 - HdotV, 5.0);
}

/* ===== Beckmann NDF ===== */
fn D_beckmannDistribution(NdotH : f32, sigma2 : f32) -> f32 {
    let cosNH2 = NdotH * NdotH;
    let tanNH2 = (1.0 - cosNH2) / cosNH2;
    let cosNH4 = cosNH2 * cosNH2;
    return exp(-tanNH2 / sigma2) / (PI * sigma2 * cosNH4);
}

/* ===== Cook-Torrance G ===== */
fn G_geometricAttenuation(
    NdotL : f32,
    NdotV : f32,
    NdotH : f32,
    HdotV : f32
) -> f32 {
    return min(
        1.0,
        min(
            2.0 * NdotL * NdotH / HdotV,
            2.0 * NdotV * NdotH / HdotV
        )
    );
}

/* ===== GGX / Trowbridge-Reitz NDF ===== */
fn D_TrowbridgeReitz(NdotH : f32, sigma2 : f32) -> f32 {
    let cos2 = NdotH * NdotH;
    let sigmaCosPSin = sigma2 * cos2 + 1.0 - cos2;
    return sigma2 / (PI * sigmaCosPSin * sigmaCosPSin);
}

/* ===== GGX Lambda ===== */
fn Lambda_TrowbridgeReitz(NdotV : f32, sigma2 : f32) -> f32 {
    return (-1.0 + sqrt(1.0 + sigma2 * (1.0 - NdotV * NdotV) / (NdotV * NdotV))) * 0.5;
}

/* ===== GGX Smith G ===== */
fn G_TrowbridgeReitz(NdotL : f32, NdotV : f32, sigma2 : f32) -> f32 {
    return 1.0 / (
        1.0 +
        Lambda_TrowbridgeReitz(NdotV, sigma2) +
        Lambda_TrowbridgeReitz(NdotL, sigma2)
    );
}

/* ===== Oren–Nayar ===== */
fn orenNayarTerm(
    sigma2 : f32,
    NdotV : f32,
    NdotL : f32,
    N : vec3<f32>,
    L : vec3<f32>,
    V : vec3<f32>
) -> f32 {
    let A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.57);
    let B = 0.45 * sigma2 / (sigma2 + 0.09);

    let sinL2 = 1.0 - NdotL * NdotL;
    let sinV2 = 1.0 - NdotV * NdotV;
    let sinVsinL = sqrt(sinL2 * sinV2);

    let cosBeta = max(NdotL, NdotV);
    let sinAlphaTanBeta = sinVsinL / cosBeta;

    let VP = normalize(V - NdotV * N);
    let LP = normalize(L - NdotL * N);
    let cosVL = dot(VP, LP);

    return A + B * max(0.0, cosVL) * sinAlphaTanBeta;
}

/* ===== Debug coloring ===== */
fn debugColor(x : vec3<f32>) -> vec3<f32> {
    if (any(isNan(x))) {
        return vec3<f32>(1.0, 0.0, 1.0);
    }
    if (any(isInf(x))) {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    if (any(x < vec3<f32>(0.0))) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    if (any(x > vec3<f32>(1.0))) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return x;
}

/* ===== Principled BRDF ===== */
fn principledBRDF(
    roughness : f32,
    metallness : f32,
    albedo : vec3<f32>,
    NdotL : f32,
    NdotV : f32,
    NdotH : f32,
    HdotV : f32,
    N : vec3<f32>,
    L : vec3<f32>,
    V : vec3<f32>
) -> vec3<f32> {

    let sigma = roughness * roughness;
    let sigma2 = sigma * sigma;

    let R0 = mix(vec3<f32>(0.04), albedo, metallness);
    let F = F_schlickApprox(HdotV, R0);

    var D : f32;
    var G : f32;

    if (u.uDistribution == 0) {
        D = D_beckmannDistribution(NdotH, sigma2);
        G = G_geometricAttenuation(NdotL, NdotV, NdotH, HdotV);
    } else {
        D = D_TrowbridgeReitz(NdotH, sigma2);
        G = G_TrowbridgeReitz(NdotL, NdotV, sigma2);
    }

    let specular = F * G * D / (4.0 * NdotV * NdotL);

    var diffuse = albedo / PI;

    if (u.uUseOrenNayar != 0) {
        diffuse *= orenNayarTerm(sigma2, NdotV, NdotL, N, L, V);
    }

    diffuse *= (vec3<f32>(1.0) - F) * (1.0 - metallness);

    if (u.uDebugView == 1) { return debugColor(vec3<f32>(D)); }
    if (u.uDebugView == 2) { return debugColor(F); }
    if (u.uDebugView == 3) { return debugColor(vec3<f32>(G)); }
    if (u.uDebugView == 4) { return debugColor(specular); }
    if (u.uDebugView == 5) { return debugColor(diffuse); }
    if (u.uDebugView == 6) { return debugColor(vec3<f32>(orenNayarTerm(sigma2, NdotV, NdotL, N, L, V) / PI)); }

    return specular + diffuse;
}

/* ===== Fragment main ===== */
@fragment
fn main(input : FragmentInput) -> FragmentOutput {
    var out : FragmentOutput;

    let N = normalize(input.interpNormal);
    let L = normalize(u.uLightDir);
    let V = normalize(u.uCameraPos - input.worldPos);
    let H = normalize(V + L);

    let NdotL = max(dot(N, L), 0.0);
    let NdotV = max(dot(N, V), 0.0);
    let NdotH = max(dot(N, H), 0.0);
    let HdotV = max(dot(H, V), 0.0);

    let lightColor = vec3<f32>(1.0) * PI;

    let lighting =
        principledBRDF(
            u.uRoughness,
            u.uMetallness,
            u.uAlbedo,
            NdotL,
            NdotV,
            NdotH,
            HdotV,
            N,
            L,
            V
        )
        * NdotL
        * lightColor;

    if (u.uDebugView == 0) { out.fragColor = lighting; }
    else if (u.uDebugView == 7) { out.fragColor = 0.5 * N + 0.5; }
    else if (u.uDebugView == 8) { out.fragColor = 0.5 * V + 0.5; }
    else if (u.uDebugView == 9) { out.fragColor = 0.5 * H + 0.5; }
    else if (u.uDebugView == 10) { out.fragColor = debugColor(vec3<f32>(NdotL)); }
    else if (u.uDebugView == 11) { out.fragColor = debugColor(vec3<f32>(NdotV)); }
    else if (u.uDebugView == 12) { out.fragColor = debugColor(vec3<f32>(NdotH)); }
    else if (u.uDebugView == 13) { out.fragColor = debugColor(vec3<f32>(HdotV)); }
    else if (u.uDebugView == 14) { out.fragColor = input.worldPos; }
    else { out.fragColor = lighting; }

    return out;
}
