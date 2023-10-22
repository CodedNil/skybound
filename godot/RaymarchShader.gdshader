shader_type canvas_item;

uniform int MAX_STEPS = 100;
uniform float MAX_DIST = 20;
uniform float MIN_HIT_DIST = 0.0001;
uniform float DERIVATIVE_STEP = 0.0001;

uniform float ambientCoeff = 0.1;
uniform float diffuseCoeff = 0.9;
uniform float specularCoeff = 1.0;
uniform float specularExponent = 64.0;
uniform vec3 lightPos = vec3(0.0, 5.0, 0.0);

uniform float fov = 45.0;
uniform vec3 cameraPos = vec3(-5.0, 0.0, 0.0);
uniform vec3 front = vec3(1.0, 0.0, 0.0);
uniform vec3 up = vec3(0.0, 1.0, 0.0);

// Locations of spheres to render
uniform int sphereN = 0;
uniform vec3 sphereCenters[100000];

uniform float blendAmount = 0.2;
float smoothMin(float a, float b) {
    float h = clamp(0.5 + 0.5 * (b - a) / blendAmount, 0.0, 1.0);
    return mix(b, a, h) - blendAmount * h * (1.0 - h);
}

float sdSphere( vec3 p, vec3 c, float s )
{
    return length(c-p)-s;
}

float sdf(vec3 p) {
    float res = MAX_DIST; 
    for(int i = 0; i < sphereN; i++) {
        float d = sdSphere(p, sphereCenters[i], 0.1);
        res = i == 0 ? d : smoothMin(res, d);
    }
    return res;
}

vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        sdf(vec3(p.x + DERIVATIVE_STEP, p.y, p.z)) - sdf(vec3(p.x - DERIVATIVE_STEP, p.y, p.z)),
        sdf(vec3(p.x, p.y + DERIVATIVE_STEP, p.z)) - sdf(vec3(p.x, p.y - DERIVATIVE_STEP, p.z)),
        sdf(vec3(p.x, p.y, p.z  + DERIVATIVE_STEP)) - sdf(vec3(p.x, p.y, p.z - DERIVATIVE_STEP))
    ));
}

vec3 blinnPhong(vec3 position, vec3 lightPosition, vec3 ambient, vec3 diffuse)
{
    vec3 normal = estimateNormal(position);
    vec3 toEye = normalize(cameraPos - position);
    vec3 toLight = normalize(lightPosition - position);
    // vec3 reflection = reflect(-toLight, normal); // uncomment for Phong model
    vec3 halfwayDir = normalize(toLight + toEye);
    
    vec3 ambientFactor = ambient * ambientCoeff;
    vec3 diffuseFactor = diffuse * max(0.0, dot(normal, toLight));
    // vec3 specularFactor = diffuse * pow(max(0.0, dot(toEye, reflection)), specularExponent)
    //                     * specularCoeff; // uncomment for Phong model
    vec3 specularFactor = diffuse * pow(max(0.0, dot(normal, halfwayDir)), specularExponent)
                        * specularCoeff;
    
    return ambientFactor + diffuseFactor + specularFactor;
}

vec3 raymarch(vec3 rayDir)
{
    vec3 ambientColor = vec3(1.0, 1.0, 1.0);
	vec3 hitColor = vec3(1.0, 1.0, 1.0);
	vec3 missColor = vec3(0.0, 0.0, 0.0);
	
	float depth = 0.0;
	float minDist = MAX_DIST;
	for (int i=0; depth<MAX_DIST && i<MAX_STEPS; ++i)
	{
		vec3 pos = cameraPos + rayDir * depth;
		float dist = sdf(pos);
		minDist = min(minDist, dist);
		if (dist < MIN_HIT_DIST) {
            vec3 normal = estimateNormal(pos);
            vec3 lightningColor = blinnPhong(pos, lightPos, normal, hitColor);
            
			return lightningColor;
		}
		depth += dist;
	}
    return missColor;
}

vec3 getRayDirection(vec2 resolution, vec2 uv)
{
	float aspect = resolution.x / resolution.y;
	float fov2 = radians(fov) / 2.0;
	
	// convert coordinates from [0, 1] to [-1, 1]
	// and invert y axis to flow from bottom to top
	vec2 screenCoord = (uv - 0.5) * 2.0;
	screenCoord.x *= aspect;
	screenCoord.y = -screenCoord.y;
	
	vec2 offsets = screenCoord * tan(fov2);
	
	vec3 rayFront = normalize(front);
	vec3 rayRight = normalize(cross(rayFront, normalize(up)));
	vec3 rayUp = cross(rayRight, rayFront);
	vec3 rayDir = rayFront + rayRight * offsets.x + rayUp * offsets.y;
	
	return normalize(rayDir);
}

void fragment()
{
	vec2 resolution = 1.0 / SCREEN_PIXEL_SIZE;
	
	vec3 rayDir = getRayDirection(resolution, UV);
	vec3 rayMarch = raymarch(rayDir);
	if (rayMarch != vec3(0.0, 0.0, 0.0)) {
		COLOR = vec4(rayMarch, 1.0);
	} else {
		COLOR = vec4(0.0, 0.0, 0.0, 0.0);
	}
}
