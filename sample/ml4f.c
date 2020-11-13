#include "ml4f.h"
#include <string.h>

int ml4f_is_valid_header(const ml4f_header_t *header) {
    if (!header || header->magic0 != ML4F_MAGIC0 || header->magic1 != ML4F_MAGIC1)
        return 0;
    if (header->input_type != ML4F_TYPE_FLOAT32 || header->output_type != ML4F_TYPE_FLOAT32)
        return 0;
    return 1;
}

typedef void (*model_fn_t)(const ml4f_header_t *model, uint8_t *arena);

int ml4f_invoke(const ml4f_header_t *model, uint8_t *arena) {
    if (!ml4f_is_valid_header(model))
        return -1;
    // +1 for Thumb mode
    model_fn_t fn = (model_fn_t)((const uint8_t *)model + model->header_size + 1);
    fn(model, arena);
    return 0;
}

#define EPS 0.00002f
static int is_near(float a, float b) {
    float diff = a - b;
    if (diff < 0)
        diff = -diff;
    if (diff < EPS)
        return 1;
    if (a < 0)
        a = -a;
    if (b < 0)
        b = -b;
    if (diff / (a + b) < EPS)
        return 1;
    return 0;
}

int ml4f_test(const ml4f_header_t *model, uint8_t *arena) {
    if (!ml4f_is_valid_header(model))
        return -1;

    if (!model->test_input_offset || !model->test_output_offset)
        return 0; // no tests

    memcpy(arena + model->input_offset, (uint8_t *)model + model->test_input_offset,
           ml4f_shape_size(ml4f_input_shape(model), model->input_type));

    ml4f_invoke(model, arena);

    float *actual = (float *)(arena + model->output_offset);
    const float *expected = (const float *)((const uint8_t *)model + model->test_output_offset);
    int elts = ml4f_shape_elements(ml4f_output_shape(model));
    for (int i = 0; i < elts; ++i) {
        if (!is_near(actual[i], expected[i]))
            return -2;
    }

    return 1; // tests OK
}

const uint32_t *ml4f_input_shape(const ml4f_header_t *model) {
    return model->input_shape;
}

const uint32_t *ml4f_output_shape(const ml4f_header_t *model) {
    const uint32_t *p = model->input_shape;
    while (*p)
        p++;
    p++;
    return p;
}

uint32_t ml4f_shape_elements(const uint32_t *shape) {
    uint32_t r = 1;
    while (*shape)
        r *= *shape++;
    return r;
}

uint32_t ml4f_shape_size(const uint32_t *shape, uint32_t type) {
    if (type != ML4F_TYPE_FLOAT32)
        return 0;
    return ml4f_shape_elements(shape) << 2;
}

// This function is just an example - you'll likely have your own tensor formats and memory
// allocation functions

#include <stdlib.h>

int ml4f_full_invoke(const ml4f_header_t *model, const float *input, float *output) {
    if (!ml4f_is_valid_header(model))
        return -1;
    uint8_t *arena = malloc(model->arena_bytes);
    memcpy(arena + model->input_offset, input,
           ml4f_shape_size(ml4f_input_shape(model), model->input_type));
    int r = ml4f_invoke(model, arena);
    memcpy(output, arena + model->output_offset,
           ml4f_shape_size(ml4f_output_shape(model), model->output_type));
    free(arena);
    return r;
}
