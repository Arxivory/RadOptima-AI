#version 450 core

out vec4 FragColor;
in vec3 TexCoords; 

uniform float tf_multiplier;
uniform isampler3D volumeTexture; 
uniform sampler1D transferFunction;
uniform float windowWidth;
uniform float windowLevel;
uniform float stepSize = 0.002;
uniform mat4 invModel;
uniform vec3 eyePos;

uniform vec3 lensCenter;
uniform float lensRadius;
uniform bool lensEnabled;

uniform isampler3D volumeTextureAI;

uniform bool diffMode;
uniform bool is2DView;
uniform float sliceZ;

float pseudo_random(vec2 co) {
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main() {
    // --- 2D MPR (SLICE) MODE ---
    if (is2DView) {
        vec2 uv = TexCoords.xy; 
    
        vec3 samplePos = vec3(uv.x, uv.y, sliceZ);
        float hu = float(texture(volumeTexture, samplePos).r);
        float aiHU = float(texture(volumeTextureAI, samplePos).r);

        float finalHU = diffMode ? abs(hu - aiHU) * 5.0 : aiHU;

        float lowerBound = windowLevel - (windowWidth / 2.0);
        float norm = clamp((finalHU - lowerBound) / windowWidth, 0.0, 1.0);
    
        vec3 boneColor = vec3(norm);
        boneColor.r = clamp(norm * 1.05, 0.0, 1.0);
        boneColor.b = clamp(norm * 1.15, 0.0, 1.0); 

        FragColor = vec4(boneColor, 1.0);
    } 
    
    // --- 3D RAYMARCHING MODE ---
    else {
        vec3 objEyePos = (invModel * vec4(eyePos, 1.0f)).xyz;
        vec3 rayDir = normalize(TexCoords - objEyePos); 
        
        float jitter = pseudo_random(gl_FragCoord.xy) * stepSize;
        vec3 currentPos = TexCoords + rayDir * jitter;

        vec4 accumulatedColor = vec4(0.0);
        float accumulatedOpacity = 0.0;

        for (int i = 0; i < 512; i++) {
            if (accumulatedOpacity >= 0.95) break;
            if (any(greaterThan(currentPos, vec3(1.0))) || any(lessThan(currentPos, vec3(0.0)))) break;

            float hu = float(texture(volumeTexture, currentPos).r);
            float aiHU = float(texture(volumeTextureAI, currentPos).r);

            float dist = distance(currentPos, lensCenter);
            float feather = 0.05 * lensRadius;
            float weight = 1.0 - smoothstep(lensRadius - feather, lensRadius + feather, dist);

            float finalHU;
            bool isInsideLens = (lensEnabled && dist < lensRadius);

            if (isInsideLens) {
                finalHU = diffMode ? abs(hu - aiHU) * 5.0 : aiHU;
            } else {
                finalHU = hu;
            }

            float lowerBound = windowLevel - (windowWidth / 2.0);
            float normalizedIntensity = (finalHU - lowerBound) / windowWidth;
            normalizedIntensity = clamp(normalizedIntensity, 0.0, 1.0);

            vec4 tfSample = texture(transferFunction, normalizedIntensity);
            float sampleOpacity = tfSample.a * tf_multiplier;
            vec3 sampleColor = tfSample.rgb;

            if (isInsideLens && diffMode) {
                sampleColor = mix(sampleColor, vec3(0.2, 0.6, 1.0) * normalizedIntensity, 0.5);
            }

            if (lensEnabled) {
                float ringWidth = 0.003;
                float ring = smoothstep(ringWidth, 0.0, abs(dist - lensRadius));
                sampleColor += vec3(ring);
            }

            if (sampleOpacity > 0.0) {
                float alpha = (1.0 - accumulatedOpacity) * sampleOpacity;
                accumulatedColor.rgb += alpha * sampleColor;
                accumulatedOpacity += alpha;
            }

            currentPos += rayDir * stepSize;
        }
        FragColor = vec4(accumulatedColor.rgb, accumulatedOpacity);
    }
}