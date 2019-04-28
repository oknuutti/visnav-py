#version 330 core

layout(triangles) in;
layout(line_strip, max_vertices = 4) out;

void main()
{
    vec3 a = gl_in[0].gl_Position.xyz / gl_in[0].gl_Position.w;
    vec3 b = gl_in[1].gl_Position.xyz / gl_in[1].gl_Position.w;
    vec3 c = gl_in[2].gl_Position.xyz / gl_in[2].gl_Position.w;

    if(b.x*c.y - b.y*c.x - a.x*c.y + a.x*b.y + a.y*c.x - a.y*b.x > 0)
    {
//        float width = 0.005;
//        vec3 nab = cross(normalize(b-a), vec3(0, 0, 1));
//        vec3 nbc = cross(normalize(c-b), vec3(0, 0, 1));
//        vec3 nca = cross(normalize(a-c), vec3(0, 0, 1));
//
//        p0 = a - nab*width;
//        p1 = a + nab*width;
//        p2 = b - nab*width;
//        p3 = b + nab*width;

        gl_Position = gl_in[0].gl_Position;
        EmitVertex();

        gl_Position = gl_in[1].gl_Position;
        EmitVertex();

        gl_Position = gl_in[2].gl_Position;
        EmitVertex();

        gl_Position = gl_in[0].gl_Position;
        EmitVertex();
    }

    EndPrimitive();
}