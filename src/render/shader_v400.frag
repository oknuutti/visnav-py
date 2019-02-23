#version 400 core

#define PI 3.1415926538

in vec3 vertexPosition_viewFrame;
in vec3 vertexNormal_viewFrame;
//in vec2 texCoords; // aka albedo?
in vec3 vertexPosition_shadowFrame;

out vec4 fragColor;

uniform vec3 lightDirection_viewFrame; // assume is normalized to length 1
uniform float brightness_coef; // 0.4 seemed good
uniform int reflection_model;  // 0: lambertian, 1: lunar-lambert, 2: hapke
uniform float model_coefs[10];
uniform bool shadows;
uniform sampler2D shadow_map;

void main()
{
    vec3 normal = normalize(vertexNormal_viewFrame);
    float cos_incidence = clamp(dot(normal, lightDirection_viewFrame), 0.0, 1.0);  // mu0 in hapke
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
        radiance *= brightness_coef;
        if(reflection_model == 0) {
             radiance *= cos_incidence;
        }
        else if(reflection_model == 1 || reflection_model == 2) {
            vec3 position = normalize(vertexPosition_viewFrame);

            float cos_emission = clamp(dot(-position, normal), 0.0, 1.0);        // mu in hapke
            float cos_phase_angle = clamp(dot(position, lightDirection_viewFrame), -1.0, 1.0);  //  alpha in hapke
            float g = acos(cos_phase_angle);  // phase angle in radians, alpha, g in hapke

            // from: "refinement of stereo image analysis using photometric shape recovery as an alternative to bundle adjustment"
            //   by A. Grumpe, C. Schröer, S. Kauffmann, T. Fricke, C. Wöhler, U. Mall
            //   in The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, Volume XLI-B4, 2016
            // and
            //   https://raw.githubusercontent.com/NeoGeographyToolkit/StereoPipeline/master/src/asp/Tools/sfs.cc
            if(reflection_model == 1) {
                // lunar-lambert
                float a = degrees(g);
                float a2 = a*a;
                float L = model_coefs[0]+model_coefs[1]*a+model_coefs[2]*a2+model_coefs[3]*a2*a+model_coefs[4]*a2*a2+model_coefs[5]*a2*a2*a;

                // model_coefs[6] is a scaling coef, reflectance can theoretically go to negative => clamp from below
                radiance *= model_coefs[6]*clamp((2*L*cos_incidence / (cos_incidence + cos_emission) + (1-L)*cos_incidence), 0, 1e10);
            }
            else if(reflection_model == 2) {
                // Hapke params & exact variation from article:
                //      Ciarniello et al., 2015,
                //      "Photometric properties of comet 67P/Churyumov-Gerasimenko from VIRTIS-M onboard Rosetta"
                //      https://www.aanda.org/articles/aa/pdf/2015/11/aa26307-15.pdf
                //
                // See also Fornasier et al, 2015,
                //      "Spectrophotometric properties of the nucleus of comet 67P/Churyumov-Gerasimenko
                //      from the OSIRIS instrument onboard the ROSETTA spacecraft."
                //      https://arxiv.org/pdf/1505.06888.pdf
                //
                // Details for mu0_eff, mu_eff, K and S from book by Hapke, 2012,
                //      "Theory of Reflectance and Emittance Spectroscopy", chapter 12

                // ## NOTE : DOES NOT SEEM TO WORK AS IMPOSSIBLE TO GET THE "HIGHLIGHTS"
                // ######################################################################
                float J = model_coefs[0]; 		// 600, brightness scaling
                float th_p = radians(model_coefs[1]); 	// 19, average surface slope, effective roughness, theta hat sub p
                float w  = model_coefs[2];   	// 0.052, single scattering albedo (w, omega, SSA)
                float b  = model_coefs[3];   	// -0.42, SPPF asymmetry parameter (sometimes g?)
                float c  = model_coefs[4];      // - another parameter for a more complex SPPF
                float B_SH0 = model_coefs[5];   // - or B0, amplitude of shadow-hiding opposition effect (shoe)
                float hs    = model_coefs[6];   // - or h or k, angular half width of shoe
                float B_CB0 = model_coefs[7];   // - amplitude of coherent backscatter opposition effect (cboe)
                float hc = model_coefs[8];      // - angular half width of cboe
                float mode = model_coefs[9];	 // - extra mode selection: first bit for usage of roughness correction term K

                // calculate mu0_eff, mu_eff and large-scale roughness factor S
                // >>
                float mu0_eff, mu_eff, S;

                if (th_p>0) {
                    float i = acos(cos_incidence);
                    float e = acos(cos_emission);
                    float tan_th = tan(th_p);
                    float xi = 1/sqrt(1 + PI*tan_th*tan_th);

                    float y = i < e ? i : e;  // think that y = i && cos(i)= mu0
                    float z = i < e ? e : i;  // think that z = e && cos(e)= mu
                    float cos_y = i < e ? cos_incidence : cos_emission; // think that cos(i) = cos_y = mu0
                    float cos_z = i < e ? cos_emission : cos_incidence; // think that cos(e) = cos_z = mu
                    float cot_y = 1/tan(y);
                    float cot_z = 1/tan(z);

                    float fg = exp(-2*tan(g/2));
                    float E1y = exp(-2/tan_th*cot_y/PI);
                    float E1z = exp(-2/tan_th*cot_z/PI);
                    float E2y = exp(-cot_y*cot_y/(tan_th*tan_th)/PI);
                    float E2z = exp(-cot_z*cot_z/(tan_th*tan_th)/PI);

                    // with above substitutions, can now implement formulas as if i < e all the time
                    mu0_eff = xi*(cos_y + sin(y)*tan_th*(cos_phase_angle*E2z + pow(sin(g/2),2)*E2y)/(2-E1z-(g/PI)*E1y));
                    mu_eff = xi*(cos_z + sin(z)*tan_th*(E2z - pow(sin(g/2),2)*E2y)/(2-E1z-(g/PI)*E1y));

                    // calculate S(incidence, emission, phase_angle, th), shadowing function for large-scale roughness
                    float ny = xi*(cos_y + sin(y)*tan_th*E2y/(2-E1y));
                    float nz = xi*(cos_z + sin(z)*tan_th*E2z/(2-E1z));
                    S = mu_eff*cos_incidence*xi/ny/nz/(1-fg+fg*xi*cos_y/nz);
                }
                else {
                    mu0_eff = cos_incidence;
                    mu_eff = cos_emission;
                    S = 1;
                }
                // <<
                // calculate mu0_eff, mu_eff and S

                // p(phase_angle), p(g), single particle phase function (SPPF),
                // first simple, single-term Henyey-Greenstein function
                float sppf = (1.0 - b*b) / pow(1.0 + 2.0*b*cos_phase_angle + b*b, 1.5);
                if(c != 0) {
                    // calculate more complex two term model
                    // from Hapke 2012, where sppf=p_HGF, alternative formulation for c can be 0.5*(1.0-c)*sppf + 0.5*(1.0+c)*...
                    // if c is missing, can be estimated by:
                    //     c = pow(0.05/(b-0.15), 0.75) - 1
                    //  or c = 3.29*exp(-17.4*b*b) - 0.908
                    sppf = (1.0-c)*sppf + c*(1.0 - b*b)/pow(1.0 - 2.0*b*cos_phase_angle + b*b, 1.5);
                }

				// SHOE opposition effect
				if(B_SH0 != 0) {
                    // B_SH(g), ~b(g), shadow-hiding opposition effect (shoe), as in Hapke 2002
                    float B_SH = 1.0 + B_SH0 / (1.0 + (1.0/hs) * tan(g/2.0));

                    // Chandrasekhar functions for incidence and emission angles (for multiple scattering)
                    float H_incidence = (1.0 + 2*mu0_eff) / (1.0 + 2*mu0_eff*sqrt(1.0-w));
                    float H_emission  = (1.0 + 2*mu_eff) / (1.0 + 2*mu_eff*sqrt(1.0-w));

					sppf = (sppf*B_SH + H_incidence*H_emission - 1.0);
				}

				// CBOE opposition effect, probably not needed for asteroids
				if(B_SH0 != 0) {
                    // B_CB = coherent-backscattering opposition effect (CBOE), Hapke 2002
                    // "The Coherent Backscatter Opposition Effect and Anisotropic Scattering"
                    float t0 = 1/hc*tan(g/2);
                    float B_CB = 1.0 + B_CB0 * (1 + (1-exp(-t0))/t0) / (2*(1+t0)*(1+t0));

					J *= B_CB;
				}

                // roughness correction factor, works ok for <60deg phase angles, for others use table 12.1 from Hapke 2012
				if(mod(mode, 2) > 0) {
					J *= exp(-0.32*th_p*sqrt(tan(th_p)*tan(g/2)) -0.52*th_p*tan(th_p)*tan(g/2));
				}

                // final value
                radiance *= J * w/4.0/PI * mu0_eff/(mu0_eff + mu_eff) * sppf * S;
            }
        }
    }

    fragColor = vec4(radiance * vec3(1, 1, 1), 1);
}