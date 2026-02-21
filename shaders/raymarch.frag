#version 450 core

out vec4 FragColor;
in vec3 TexCoords; 


uniform float tf_opacity_power;
uniform float tf_multiplier;
uniform isampler3D volumeTexture; 
uniform float windowWidth;
uniform float windowLevel;
uniform float stepSize = 0.002; // Slightly smaller for better detail
uniform mat4 invModel;
uniform vec3 eyePos;

float pseudo_random(vec2 co) {
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main() {
    vec3 objEyePos = (invModel * vec4(eyePos, 1.0f)).xyz;

    vec3 rayDir = normalize(TexCoords - objEyePos); 
    
    float jitter = pseudo_random(TexCoords.xy) * stepSize;
    vec3 currentPos = TexCoords + rayDir * jitter;

    vec4 accumulatedColor = vec4(0.0);
    float accumulatedOpacity = 0.0;

    for (int i = 0; i < 512; i++) {
        if (accumulatedOpacity >= 0.95) break;
        if (any(greaterThan(currentPos, vec3(1.0))) || any(lessThan(currentPos, vec3(0.0)))) break;

        int rawHU = texture(volumeTexture, currentPos).r;
        float hu = float(rawHU);

        float lowerBound = windowLevel - (windowWidth / 2.0);
        float intensity = (hu - lowerBound) / windowWidth;
        intensity = clamp(intensity, 0.0, 1.0);

        float opacity = 0.0;
        if (hu > lowerBound) {
            opacity = pow(intensity, 2.0) * 0.05; 
        }

        if (opacity > 0.0) {
            vec3 color = vec3(intensity);
            accumulatedColor.rgb += (1.0 - accumulatedOpacity) * color * opacity;
            accumulatedOpacity += (1.0 - accumulatedOpacity) * opacity;
        }

        currentPos += rayDir * stepSize;
    }

    FragColor = vec4(accumulatedColor.rgb, accumulatedOpacity);
}