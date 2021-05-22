#version 330 core

in vec3 vertexPosition_modelFrame2;
in vec3 vertexPosition_viewFrame;
in vec3 vertexNormal_viewFrame;
in vec2 texCoords;

out vec4 fragColor;

uniform vec3 lightDirection_viewFrame;  // assume is normalized to length 1
uniform int select;

void main()
{
    if (select == 0) {
        fragColor = vec4(vertexPosition_modelFrame2, 1.0);
    }
    else if (select == 1) {
        vec3 position = normalize(vertexPosition_viewFrame);
        vec3 normal = normalize(vertexNormal_viewFrame);
        float cos_incidence = clamp(dot(normal, lightDirection_viewFrame), 0.0, 1.0);  // mu0 in hapke
        float cos_emission = clamp(dot(-position, normal), 0.0, 1.0);        // mu in hapke
        float cos_phase_angle = clamp(dot(position, -lightDirection_viewFrame), -1.0, 1.0);
        fragColor = vec4(cos_incidence, cos_emission, cos_phase_angle, 1.0);
    }
    else {
        fragColor = vec4(texCoords, 1.0, 1.0);
    }
 }