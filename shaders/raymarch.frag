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

float pseudo_random(vec2 co) {
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main() {
    vec3 objEyePos = (invModel * vec4(eyePos, 1.0f)).xyz;
    vec3 rayDir = normalize(TexCoords - objEyePos); 
    
    float jitter = pseudo_random(gl_FragCoord.xy) * stepSize;
    vec3 currentPos = TexCoords + rayDir * jitter;

    vec4 accumulatedColor = vec4(0.0);
    float accumulatedOpacity = 0.0;

    for (int i = 0; i < 512; i++) {
        if (accumulatedOpacity >= 0.95) break;
        if (any(greaterThan(currentPos, vec3(1.0))) || any(lessThan(currentPos, vec3(0.0)))) break;

        int rawHU = texture(volumeTexture, currentPos).r;
        float hu = float(rawHU);

        float aiHU = float(texture(volumeTextureAI, currentPos).r);


        float dist = distance(currentPos, lensCenter);
        float feather = 0.2 * lensRadius;
        float weight = 1.0 - smoothstep(lensRadius - feather, lensRadius + feather, dist);

        float finalHU = mix(hu, aiHU, weight * float(lensEnabled));

        if (lensEnabled && dist < lensRadius) {
            hu += (600.0 * weight); 
        }

        float lowerBound = windowLevel - (windowWidth / 2.0);
        float normalizedIntensity = (finalHU - lowerBound) / windowWidth;
        normalizedIntensity = clamp(normalizedIntensity, 0.0, 1.0);

        vec4 tfSample = texture(transferFunction, normalizedIntensity);
        
        float sampleOpacity = tfSample.a * tf_multiplier;
        vec3 sampleColor = tfSample.rgb;

        if (lensEnabled && dist < lensRadius) {
            sampleColor.rgb = mix(sampleColor.rgb, sampleColor.rgb + vec3(0.0, 0.2, 0.0), weight);
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