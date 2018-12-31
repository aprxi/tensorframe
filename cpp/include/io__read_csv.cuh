/* Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
 * This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
*/

#include "tf_types_no.h"
#include "f_type_map.cuh"

typedef struct {
    uint8_t function_id;
    uint8_t size;
} lookup_types_dyn;

typedef void (*threadedFunc)(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na);

__device__ void conv_int8(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na);
__device__ void conv_int16(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na);
__device__ void conv_int32(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na);
__device__ void conv_int64(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na);

__device__ void conv_uint8(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na);
__device__ void conv_uint16(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na);
__device__ void conv_uint32(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na);
__device__ void conv_uint64(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na);

__device__ void conv_fp16(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na);
__device__ void conv_fp32(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na);
__device__ void conv_fp64(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na);

__constant__ threadedFunc datatype_conv[] = {
    conv_int8,
    conv_int16,
    conv_int32,
    conv_int64,
    conv_uint8,
    conv_uint16,
    conv_uint32,
    conv_uint64,
    conv_fp16,
    conv_fp32,
    conv_fp64
};
