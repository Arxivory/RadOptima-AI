#version 450 core

out vec4 FragColor;
in vec3 TexCoords; 

// --- UNIFORMS ---
uniform float tf_multiplier;
uniform isampler3D volumeTexture; 
uniform isampler3D volumeTextureAI;
uniform sampler1D transferFunction;
uniform float windowWidth;
uniform float windowLevel;
uniform float stepSize = 0.002;
uniform mat4 invModel;
uniform vec3 eyePos;

uniform vec3 lensCenter;
uniform float lensRadius;
uniform bool lensEnabled;

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

// Industry-Standard Bicubic Sampler
float sampleBicubic(isampler3D tex, vec3 uvw) {
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
            float val = float(texture(tex, samplePos).r);
            float w = cubic(float(m) - fuvw.x) * cubic(float(n) - fuvw.y);
            texelSum += val * w;
            weightSum += w;
        }
    }
    return texelSum / weightSum;
}

void main() {
    if (is2DView) {
        vec3 res = vec3(textureSize(volumeTexture, 0));
        vec3 samplePos = vec3(TexCoords.x, TexCoords.y, sliceZ);

        float hu = sampleBicubic(volumeTexture, samplePos);
        
        vec2 texelSize = 1.0 / res.xy;
        float laplacian = hu * 4.0;
        laplacian -= sampleBicubic(volumeTexture, samplePos + vec3(texelSize.x, 0, 0));
        laplacian -= sampleBicubic(volumeTexture, samplePos - vec3(texelSize.x, 0, 0));
        laplacian -= sampleBicubic(volumeTexture, samplePos + vec3(0, texelSize.y, 0));
        laplacian -= sampleBicubic(volumeTexture, samplePos - vec3(0, texelSize.y, 0));

        float sharpAmount = 0.6; // Adjust for "Film" vs "Smooth" look
        float diagnosticHU = hu + (laplacian * sharpAmount);

        float aiHU = sampleBicubic(volumeTextureAI, samplePos);
        float dist = distance(samplePos, lensCenter);
        bool isInsideLens = (lensEnabled && dist < lensRadius);

        float finalHU;

        if (compareMode2D == 2) { // Slider mode
            finalHU = (TexCoords.x < sliderX) ? hu : aiHU;
        } else if (compareMode2D == 1) { // Lens mode
            float dist = distance(samplePos, lensCenter);
            finalHU = (lensEnabled && dist < lensRadius) ? aiHU : diagnosticHU;
        } else { // Raw mode
            finalHU = diagnosticHU;
        }

        /* if (isInsideLens) {
            finalHU = diffMode ? abs(diagnosticHU - aiHU) * 8.0 : aiHU;
        } else {
            finalHU = diagnosticHU;
        } */

        float lowerBound = windowLevel - (windowWidth / 2.0);
        float norm = clamp((finalHU - lowerBound) / windowWidth, 0.0, 1.0);
        
        norm = pow(norm, 1.1); 

        vec3 clinicalColor = vec3(norm);
        clinicalColor.b *= 1.08;
        clinicalColor.r *= 0.98;

        FragColor = vec4(clinicalColor, 1.0);
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

            float hu = float(texture(volumeTexture, currentPos).r);

            float neighbors = 0.0;
            neighbors += float(texture(volumeTexture, currentPos + vec3(texelSize.x, 0, 0)).r);
            neighbors += float(texture(volumeTexture, currentPos - vec3(texelSize.x, 0, 0)).r);
            neighbors += float(texture(volumeTexture, currentPos + vec3(0, texelSize.y, 0)).r);
            neighbors += float(texture(volumeTexture, currentPos - vec3(0, texelSize.y, 0)).r);
            neighbors += float(texture(volumeTexture, currentPos + vec3(0, 0, texelSize.z)).r);
            neighbors += float(texture(volumeTexture, currentPos - vec3(0, 0, texelSize.z)).r);

            float edge = hu - (neighbors / 6.0);
            float diagnosticHU = hu + (edge * sharpAmount);

            float aiHU = float(texture(volumeTextureAI, currentPos).r);

            float finalHU = rayHitsLensCircle ? 
                            (diffMode ? abs(diagnosticHU - aiHU) * 5.0 : aiHU) : diagnosticHU;

            float lowerBound = windowLevel - (windowWidth / 2.0);
            float normalizedIntensity = clamp((finalHU - lowerBound) / windowWidth, 0.0, 1.0);

            vec4 tfSample = texture(transferFunction, normalizedIntensity);
            float sampleOpacity = tfSample.a * tf_multiplier;
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