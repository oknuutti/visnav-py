#version 330 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 12) out;

void main()
{
    vec3 a = gl_in[0].gl_Position.xyz / gl_in[0].gl_Position.w;
    vec3 b = gl_in[1].gl_Position.xyz / gl_in[1].gl_Position.w;
    vec3 c = gl_in[2].gl_Position.xyz / gl_in[2].gl_Position.w;

    if(b.x*c.y - b.y*c.x - a.x*c.y + a.x*b.y + a.y*c.x - a.y*b.x > 0)
    {
        float width = 0.002;
        vec3 nab = cross(normalize(b-a), vec3(0, 0, 1));
        vec3 nbc = cross(normalize(c-b), vec3(0, 0, 1));
        vec3 nca = cross(normalize(a-c), vec3(0, 0, 1));

        // four corners of fat line AB
        gl_Position = vec4(a - nab*width, 1);
        EmitVertex();

        gl_Position = vec4(a + nab*width, 1);
        EmitVertex();

        gl_Position = vec4(b - nab*width, 1);
        EmitVertex();

        gl_Position = vec4(b + nab*width, 1);
        EmitVertex();


        // four corners of fat line BC
        gl_Position = vec4(b - nbc*width, 1);
        EmitVertex();

        gl_Position = vec4(b + nbc*width, 1);
        EmitVertex();

        gl_Position = vec4(c - nbc*width, 1);
        EmitVertex();

        gl_Position = vec4(c + nbc*width, 1);
        EmitVertex();


        // four corners of fat line CA
        gl_Position = vec4(c - nca*width, 1);
        EmitVertex();

        gl_Position = vec4(c + nca*width, 1);
        EmitVertex();

        gl_Position = vec4(a - nca*width, 1);
        EmitVertex();

        gl_Position = vec4(a + nca*width, 1);
        EmitVertex();
    }

    EndPrimitive();
}