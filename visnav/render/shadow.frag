#version 330 core


//layout(location = 0) out float fragmentdepth;
out vec4 fragColor;

void main(){
    // Not really needed, OpenGL does it anyway
    //fragmentdepth = gl_FragCoord.z;
    fragColor = vec4(1,1,1,1);
}