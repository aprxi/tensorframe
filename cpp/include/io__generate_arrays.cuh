/* Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
 * This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
*/

#include "tf_types_no.h"
#include "f_type_map.cuh"

typedef struct {
    uint8_t function_id;
    uint8_t size;
} lookup_types_dyn;


typedef void (*threadedFunc_generate)(curandGenerator_t gen, void *dest, size_t num);

void generate_uint32(curandGenerator_t gen, void *dest, size_t num);
void generate_uint64(curandGenerator_t gen, void *dest, size_t num);
void generate_fp32(curandGenerator_t gen, void *dest, size_t num);
void generate_fp64(curandGenerator_t gen, void *dest, size_t num);

threadedFunc_generate generate_num[] = {
    generate_uint32,
    generate_uint32,
    generate_uint32,
    generate_uint64,
    generate_uint32,
    generate_uint32,
    generate_uint32,
    generate_uint64,
    generate_fp32,
    generate_fp32,
    generate_fp64
};
