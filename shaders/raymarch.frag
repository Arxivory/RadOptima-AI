#version 450 core

out vec4 FragColor;
in vec3 TexCoords; 

// --- UNIFORMS ---
uniform float tf_multiplier;
uniform sampler3D volumeTexture; 
uniform sampler3D volumeTextureAI;
uniform sampler1D transferFunction;
uniform float windowWidth;
uniform float windowLevel;
uniform float stepSize = 0.002;
uniform mat4 invModel;
uniform vec3 eyePos;

uniform vec3 lensCenter;
uniform float lensRadius;
uniform bool lensEnabled;
uniform bool aiReady;

uniform bool diffMode;
uniform bool is2DView;
uniform float sliceZ;
uniform int compareMode2D;
uniform float sliderX;

// --- MATH UTILITIES ---
float pseudo_random(vec2 co) {
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

float cubic(float x) {
    float ax = abs(x);
    if (ax < 1.0) return (1.5 * ax - 2.5) * ax * ax + 1.0;
    if (ax < 2.0) return ((-0.5 * ax + 2.5) * ax - 4.0) * ax + 2.0;
    return 0.0;
}

float sampleBicubic(sampler3D tex, vec3 uvw) {
    vec3 res = vec3(textureSize(tex, 0));
    vec3 st = uvw * res - 0.5;
    vec3 iuvw = floor(st);
    vec3 fuvw = fract(st);

    float texelSum = 0.0;
    float weightSum = 0.0;

    for (int m = -1; m <= 2; m++) {
        for (int n = -1; n <= 2; n++) {
            vec3 offset = vec3(float(m), float(n), 0.0);
            vec3 samplePos = (iuvw + offset + 0.5) / res;
            float val = texture(tex, samplePos).r;
            float w = cubic(float(m) - fuvw.x) * cubic(float(n) - fuvw.y);
            texelSum += val * w;
            weightSum += w;
        }
    }
    return (weightSum > 0.0001) ? texelSum / weightSum : texture(tex, uvw).r;
}

vec3 clinicalColormap(float t) {
    vec3 c0 = vec3(0.02,  0.02,  0.03);   // near-black for air
    vec3 c1 = vec3(0.12,  0.14,  0.18);   // dark soft tissue
    vec3 c2 = vec3(0.34,  0.40,  0.48);   // muscle / organ
    vec3 c3 = vec3(0.58,  0.64,  0.70);   // fat / lighter tissue  
    vec3 c4 = vec3(0.80,  0.85,  0.88);   // bright soft tissue
    vec3 c5 = vec3(0.93,  0.95,  0.97);   // near-bone
    vec3 c6 = vec3(1.0,   1.0,   1.0);    // bone / calcification

    t = clamp(t, 0.0, 1.0);
    if (t < 0.15) return mix(c0, c1, t / 0.15);
    if (t < 0.35) return mix(c1, c2, (t - 0.15) / 0.20);
    if (t < 0.55) return mix(c2, c3, (t - 0.35) / 0.20);
    if (t < 0.72) return mix(c3, c4, (t - 0.55) / 0.17);
    if (t < 0.88) return mix(c4, c5, (t - 0.72) / 0.16);
                  return mix(c5, c6, (t - 0.88) / 0.12);
}

vec3 viridisColormap(float t) {
    vec3 c0 = vec3(0.267, 0.004, 0.329); // Dark Purple
    vec3 c1 = vec3(0.218, 0.231, 0.518); // Deep Blue
    vec3 c2 = vec3(0.127, 0.428, 0.551); // Teal
    vec3 c3 = vec3(0.145, 0.609, 0.528); // Emerald Green
    vec3 c4 = vec3(0.364, 0.767, 0.410); // Light Green
    vec3 c5 = vec3(0.993, 0.906, 0.143); // Vibrant Yellow

    t = clamp(t, 0.0, 1.0);
    if (t < 0.2) return mix(c0, c1, t / 0.2);
    if (t < 0.4) return mix(c1, c2, (t - 0.2) / 0.2);
    if (t < 0.6) return mix(c2, c3, (t - 0.4) / 0.2);
    if (t < 0.8) return mix(c3, c4, (t - 0.6) / 0.2);
    return mix(c4, c5, (t - 0.8) / 0.2);
}

vec3 hotIronColormap(float t) {
    vec3 c0 = vec3(0.0,  0.0,  0.0);   // Black (Air)
    vec3 c1 = vec3(0.5,  0.0,  0.0);   // Dark Red (Low Density)
    vec3 c2 = vec3(1.0,  0.4,  0.0);   // Orange (Soft Tissue)
    vec3 c3 = vec3(1.0,  0.9,  0.2);   // Yellow (Contrast/Fluid)
    vec3 c4 = vec3(1.0,  1.0,  1.0);   // White (Bone)

    t = clamp(t, 0.0, 1.0);
    if (t < 0.25) return mix(c0, c1, t / 0.25);
    if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
    if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
    return mix(c3, c4, (t - 0.75) / 0.25);
}

void main() {
    if (is2DView) {
        vec3 res = vec3(textureSize(volumeTexture, 0));
        vec3 samplePos = vec3(TexCoords.x, TexCoords.y, sliceZ);
        float rawSample = sampleBicubic(volumeTexture, samplePos);

        float hu = rawSample * 32767.0; 

        vec2 texelSize = 1.0 / res.xy;
        float rawLaplacian = rawSample * 4.0;
        rawLaplacian -= sampleBicubic(volumeTexture, samplePos + vec3(texelSize.x, 0, 0));
        rawLaplacian -= sampleBicubic(volumeTexture, samplePos - vec3(texelSize.x, 0, 0));
        rawLaplacian -= sampleBicubic(volumeTexture, samplePos + vec3(0, texelSize.y, 0));
        rawLaplacian -= sampleBicubic(volumeTexture, samplePos - vec3(0, texelSize.y, 0));

        float laplacian = rawLaplacian * 32767.0; 

        float diagnosticHU = hu + clamp(laplacian * 0.03, -40.0, 40.0);

        float aiHU = aiReady ? (sampleBicubic(volumeTextureAI, samplePos) * 32767.0) : hu;
        float dist = distance(samplePos, lensCenter);

        float finalHU;
        if (compareMode2D == 2 && aiReady) {
            finalHU = (TexCoords.x < sliderX) ? hu : aiHU;
        } else if (compareMode2D == 1 && aiReady) {
            finalHU = (lensEnabled && dist < lensRadius) ? aiHU : diagnosticHU;
        } else {
            finalHU = diagnosticHU;
        }

        float lowerBound = windowLevel - (windowWidth / 2.0);
        float norm = clamp((finalHU - lowerBound) / windowWidth, 0.0, 1.0);
        
        norm = norm * 0.92 + 0.02;
        norm = pow(norm, 0.85);

        FragColor = vec4(hotIronColormap(norm), 1.0);
    }
    
    else {
        vec3 objEyePos = (invModel * vec4(eyePos, 1.0f)).xyz;
        vec3 rayDir = normalize(TexCoords - objEyePos); 
    
        vec3 L = lensCenter - objEyePos;
        float t = dot(L, rayDir);
        vec3 closestPointOnRay = objEyePos + rayDir * t;
        float distToLensAxis = distance(closestPointOnRay, lensCenter);
    
        bool rayHitsLensCircle = (lensEnabled && distToLensAxis < lensRadius);

        float jitter = pseudo_random(gl_FragCoord.xy) * stepSize;
        vec3 currentPos = TexCoords + rayDir * jitter;

        vec4 accumulatedColor = vec4(0.0);
        float accumulatedOpacity = 0.0;

        float sharpAmount = 0.8;
        vec3 texelSize = 1.0 / vec3(textureSize(volumeTexture, 0));

        for (int i = 0; i < 512; i++) {
            if (accumulatedOpacity >= 0.95) break;
            if (any(greaterThan(currentPos, vec3(1.0))) || any(lessThan(currentPos, vec3(0.0)))) break;

            float rawHU = texture(volumeTexture, currentPos).r;
            float hu = rawHU * 32767.0; 

            float neighbors = 0.0;
            neighbors += texture(volumeTexture, currentPos + vec3(texelSize.x, 0, 0)).r * 32767.0;
            neighbors += texture(volumeTexture, currentPos - vec3(texelSize.x, 0, 0)).r * 32767.0;
            neighbors += texture(volumeTexture, currentPos + vec3(0, texelSize.y, 0)).r * 32767.0;
            neighbors += texture(volumeTexture, currentPos - vec3(0, texelSize.y, 0)).r * 32767.0;
            neighbors += texture(volumeTexture, currentPos + vec3(0, 0, texelSize.z)).r * 32767.0;
            neighbors += texture(volumeTexture, currentPos - vec3(0, 0, texelSize.z)).r * 32767.0;

            float edge = hu - (neighbors / 6.0);
            float diagnosticHU = hu + (edge * sharpAmount);

            float rawAIHU = aiReady ? texture(volumeTextureAI, currentPos).r : rawHU;
            float aiHU = aiReady ? rawAIHU * 32767.0 : diagnosticHU;

            float finalHU = (aiReady && rayHitsLensCircle) ?
                (diffMode ? abs(diagnosticHU - aiHU) * 5.0 : aiHU) : diagnosticHU;

            float lowerBound = windowLevel - (windowWidth / 2.0);
            float normalizedIntensity = clamp((finalHU - lowerBound) / windowWidth, 0.0, 1.0);

            vec4 tfSample = texture(transferFunction, normalizedIntensity);
            float sampleOpacity = tfSample.a * tf_multiplier;

            if (normalizedIntensity <= 0.02) { 
                sampleOpacity = 0.0;
            }

            vec3 sampleColor = tfSample.rgb;

            if (lensEnabled && i < 10) {
                 float ring = smoothstep(0.005, 0.0, abs(distToLensAxis - lensRadius));
                 sampleColor += vec3(ring * 0.8);
            }

            if (sampleOpacity > 0.01) {
                float alpha = (1.0 - accumulatedOpacity) * sampleOpacity;
                accumulatedColor.rgb += alpha * sampleColor;
                accumulatedOpacity += alpha;
            }

            currentPos += rayDir * stepSize;
        }
        FragColor = vec4(accumulatedColor.rgb, accumulatedOpacity);
    }
}