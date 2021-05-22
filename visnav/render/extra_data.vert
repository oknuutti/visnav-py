#version 330 core

layout(location = 0) in vec3 vertexPosition_modelFrame;
layout(location = 1) in vec3 vertexNormal_modelFrame;
layout(location = 2) in vec2 aTexCoords;

out vec3 vertexPosition_modelFrame2;
out vec3 vertexPosition_viewFrame;
out vec3 vertexNormal_viewFrame;
out vec2 texCoords;

uniform mat4 mvp;
uniform mat4 mv;

void main()
{
    gl_Position = mvp * vec4(vertexPosition_modelFrame, 1.0);
    vertexPosition_modelFrame2 = vertexPosition_modelFrame;
    vertexNormal_viewFrame = (mv * vec4(vertexNormal_modelFrame, 0)).xyz;
    vertexPosition_viewFrame = (mv * vec4(vertexPosition_modelFrame, 1.0)).xyz;
    texCoords = aTexCoords;
}