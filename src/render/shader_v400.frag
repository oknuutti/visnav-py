#version 400 core

in vec3 vertexPosition_viewFrame;
in vec3 vertexNormal_viewFrame;
//in vec2 texCoords; // aka albedo?
in vec3 vertexPosition_shadowFrame;

out vec4 fragColor;

uniform vec3 lightDirection_viewFrame; // assume is normalized to length 1
uniform float brightness_coef; // 0.4 seemed good
uniform bool lambertian;
uniform bool shadows;
uniform sampler2D shadow_map;

void main()
{
    vec3 normal = normalize(vertexNormal_viewFrame);
    float cos_incidence = clamp(dot(normal, lightDirection_viewFrame), 0.0, 1.0);
    float radiance = 1;
    // * texture(texture_diffuse1, texCoords).rgb;

    if(shadows) {
        float bias = clamp(0.02*tan(acos(cos_incidence)), 0, 0.01);
//        radiance = texture(shadow_map,
//                vec3(vertexPosition_shadowFrame.xy,
//                    (vertexPosition_shadowFrame.z - bias)/vertexPosition_shadowFrame.w));
        if(texture(shadow_map, vertexPosition_shadowFrame.xy).r < vertexPosition_shadowFrame.z - bias) {
            radiance = 0;
        }
    }

    if(radiance>0) {
        if(!lambertian) {
            vec3 position = normalize(vertexPosition_viewFrame);

            float cos_emission = clamp(dot(position, normal), 0.0, 1.0);
            float cos_phase_angle = clamp(dot(position, lightDirection_viewFrame), 0.0, 1.0);

            // from: "refinement of stereo image analysis using photometric shape recovery as an alternative to bundle adjustment"
            //   by A. Grumpe, C. Schröer, S. Kauffmann, T. Fricke, C. Wöhler, U. Mall
            //   in The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, Volume XLI-B4, 2016
            // and
            //   https://raw.githubusercontent.com/NeoGeographyToolkit/StereoPipeline/master/src/asp/Tools/sfs.cc

            float a = degrees(acos(cos_phase_angle));
            float L = 1 - 0.019*a + 2.42e-4*a*a - 1.46e-6*a*a*a; // lunar-lambert function
            radiance *= brightness_coef * (
                2*L*cos_incidence / (cos_incidence + cos_emission)
                + (1-L)*cos_incidence
            );
        }
        else {
            radiance *= brightness_coef * cos_incidence;
        }
    }

    fragColor = vec4(radiance * vec3(1, 1, 1), 1);
}