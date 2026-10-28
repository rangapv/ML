#version 450

layout(location = 0) out vec3 fragColor;

const int   CIRCLE_SEGMENTS = 64;   // must match CIRCLE_SEGMENTS in tria.cpp
const float RADIUS          = 0.5;
const float TWO_PI          = 6.28318530718;

// Rim colors vary with angle so you can see the shape
vec3 rimColor(float a) {
    return vec3(0.5 + 0.5 * cos(a),
                0.5 + 0.5 * cos(a + 2.094),    // +120 degrees
                0.5 + 0.5 * cos(a + 4.189));   // +240 degrees
}

void main() {
    int segment = gl_VertexIndex / 3;   // which triangle of the fan
    int corner  = gl_VertexIndex % 3;   // 0 = center, 1 = rim a0, 2 = rim a1

    float a0 = TWO_PI * float(segment)     / float(CIRCLE_SEGMENTS);
    float a1 = TWO_PI * float(segment + 1) / float(CIRCLE_SEGMENTS);

    vec3 pos;
    vec3 col;

    if (corner == 0) {                  // center, white
        pos = vec3(0.0, 0.0, 0.5);
        col = vec3(1.0, 1.0, 1.0);
    } else if (corner == 1) {           // first rim point
        pos = vec3(RADIUS * cos(a0), RADIUS * sin(a0), 0.5);
        col = rimColor(a0);
    } else {                            // second rim point
        pos = vec3(RADIUS * cos(a1), RADIUS * sin(a1), 0.5);
        col = rimColor(a1);
    }

    gl_Position = vec4(pos, 1.0);
    fragColor = col;
}
