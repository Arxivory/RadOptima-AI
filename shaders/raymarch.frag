#version 450 core

out vec4 FragColor;
in vec3 TexCoords; 

uniform isampler3D volumeTexture; 

uniform float windowWidth;
uniform float windowLevel;
uniform float stepSize = 0.003;
uniform mat4 invModel;
uniform vec3 eyePos;

void main() {
    vec3 objEyePos = (invModel * vec4(eyePos, 1.0f)).xyz;

    vec3 rayDir = normalize(TexCoords - objEyePos); 
    vec3 currentPos = TexCoords;

    vec4 accumulatedColor = vec4(0.0);
    float accumulatedOpacity = 0.0;

    for (int i = 0; i < 256; i++) {
        if (accumulatedOpacity >= 0.95 || any(greaterThan(currentPos, vec3(1.0))) || any(lessThan(currentPos, vec3(0.0))))
            break;

        int rawHU = texture(volumeTexture, currentPos).r;
        float hu = float(rawHU);

        float lowerBound = windowLevel - (windowWidth / 2.0);
        float intensity = (hu - lowerBound) / windowWidth;
        intensity = clamp(intensity, 0.0, 1.0);

        float opacity = 0.0;
        if (hu > lowerBound) {
            opacity = intensity * 0.05;
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