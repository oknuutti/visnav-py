#version 400 core

layout(triangles) in;
layout(line_strip, max_vertices = 3) out;

void main()
{
    vec3 p1 = gl_in[0].gl_Position.xyz / gl_in[0].gl_Position.w;
    vec3 p2 = gl_in[1].gl_Position.xyz / gl_in[1].gl_Position.w;
    vec3 p3 = gl_in[2].gl_Position.xyz / gl_in[2].gl_Position.w;
    float tol = 0.01;

    if((p2.y - p1.y)*(p3.x - p2.x) - (p3.y - p2.y)*(p2.x - p1.x) > -tol)
    {
        gl_Position = gl_in[0].gl_Position;
        EmitVertex();

        gl_Position = gl_in[1].gl_Position;
        EmitVertex();

        gl_Position = gl_in[2].gl_Position;
        EmitVertex();
    }

    EndPrimitive();
}