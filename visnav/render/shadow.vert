#version 330 core

layout(location = 0) in vec3 vertexPosition_modelFrame;
uniform mat4 mvp;

void main() {
    gl_Position =  mvp * vec4(vertexPosition_modelFrame, 1);
}