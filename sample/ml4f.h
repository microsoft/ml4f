#ifndef __ML4F_H
#define __ML4F_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ML4F_TYPE_FLOAT32 1

#define ML4F_MAGIC0 0x30470f62
#define ML4F_MAGIC1 0x46344c4d // "ML4F"

// All values are little endian.
// All offsets and sizes are in bytes.

typedef struct ml4f_header {
    uint32_t magic0;
    uint32_t magic1;
    uint32_t header_size;
    uint32_t object_size;
    uint32_t weights_offset;
    uint32_t test_input_offset;
    uint32_t test_output_offset;
    uint32_t arena_bytes;
    uint32_t input_offset;
    uint32_t input_type; // always ML4F_TYPE_FLOAT32
    uint32_t output_offset;
    uint32_t output_type; // always ML4F_TYPE_FLOAT32
    uint32_t reserved[4];
    // Shapes are 0-terminated, and are given in elements (not bytes).
    // Input shape is followed by output shape.
    uint32_t input_shape[0];
} ml4f_header_t;

int ml4f_is_valid_header(const ml4f_header_t *header);
int ml4f_invoke(const ml4f_header_t *model, uint8_t *arena);
int ml4f_test(const ml4f_header_t *model, uint8_t *arena);
const uint32_t *ml4f_input_shape(const ml4f_header_t *model);
const uint32_t *ml4f_output_shape(const ml4f_header_t *model);
uint32_t ml4f_shape_elements(const uint32_t *shape);
uint32_t ml4f_shape_size(const uint32_t *shape, uint32_t type);
int ml4f_argmax(float *data, uint32_t size);

int ml4f_full_invoke(const ml4f_header_t *model, const float *input, float *output);
int ml4f_full_invoke_argmax(const ml4f_header_t *model, const float *input);

#ifdef __cplusplus
}
#endif

#endif
