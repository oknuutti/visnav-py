#version 330 core

layout(location = 0) in vec3 vertexPosition_modelFrame;
layout(location = 1) in vec3 vertexNormal_modelFrame;
layout(location = 2) in vec2 aTexCoords;

out vec3 vertexPosition_viewFrame;
out vec3 vertexNormal_viewFrame;
out vec2 texCoords;
out vec3 vertexPosition_shadowFrame;

uniform mat4 mvp;
uniform mat4 mv;
//uniform mat4 inv_mv;
uniform int reflection_model;
uniform bool use_texture;
uniform bool use_shadows;
uniform mat4 shadow_mvp;

void main()
{
//    gl_FragDepth = (mv * vertexPosition_modelFrame).z;
    gl_Position = mvp * vec4(vertexPosition_modelFrame, 1.0);

    vertexNormal_viewFrame = (mv * vec4(vertexNormal_modelFrame, 0)).xyz;
    if (reflection_model > 0) {
        vertexPosition_viewFrame = (mv * vec4(vertexPosition_modelFrame, 1.0)).xyz;
    }
    if (use_shadows) {
        vertexPosition_shadowFrame = (shadow_mvp * vec4(vertexPosition_modelFrame, 1.0)).xyz;
    }
//    gl_Position = mvp * vec4(vertexPosition, 1.0);
//    fragPos = vec3(model * vec4(pos_modelspace, 1.0));
//    vertexNormal_worldFrame = (inverseModelMatrix * vec4(vertexNormal, 0)).xyz;
    texCoords = aTexCoords;
}