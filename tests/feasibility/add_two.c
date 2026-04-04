#include <emscripten/emscripten.h>

EMSCRIPTEN_KEEPALIVE
int add_two(int a, int b) {
    return a + b;
}

EMSCRIPTEN_KEEPALIVE
float multiply_f32(float a, float b) {
    return a * b;
}
