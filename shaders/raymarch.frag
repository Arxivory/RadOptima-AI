#version 450 core

out vec4 FragColor;
in vec3 TexCoords;

uniform sampler3D volumeTexture;
uniform float windowWidth;
uniform float windowLevel;
uniform float stepSize = 0.005;

void main() {
	vec3 rayDir = vec3(0.0, 0.0, 1.0);
	vec3 currentPos = vec3(TexCoords.xy, 0.0);

	vec4 accumulatedColor = vec4(0.0);
	float accumulatedOpacity = 0.0;

	for (int i = 0; i < 200; i++) {
		if (accumulatedOpacity >= 0.95 || any(greaterThan(currentPos, vec3(1.0))) || any(lessThan(currentPos, vec3(0.0))))
			break;

		float hu = texture(volumeTexture, currentPos).r;

		float opacity = 0.0;
		if (hu > 400.0) {
			opacity = 0.05;
		}

		if (opacity > 0.0) {
			vec4 color = vec4(vec3(hu / 1000.0), opacity); 
            accumulatedColor.rgb += (1.0 - accumulatedOpacity) * color.rgb * color.a;
            accumulatedOpacity += (1.0 - accumulatedOpacity) * color.a;
		}

		currentPos += rayDir * stepSize;
	}

	FragColor = vec4(accumulatedColor.rgb, accumulatedOpacity);
}