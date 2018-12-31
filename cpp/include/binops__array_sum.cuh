/* Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
 * This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
*/

#include "tf_types_no.h"
#include "f_type_map.cuh"

typedef struct {
    uint8_t function_sum;
    uint8_t size;
} lookup_types_dyn;

typedef void (*threadedFunc_sum)(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow);

__device__ void sum_int8(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow);
__device__ void sum_int16(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow);
__device__ void sum_int32(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow);
__device__ void sum_int64(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow);

__device__ void sum_uint8(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow);
__device__ void sum_uint16(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow);
__device__ void sum_uint32(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow);
__device__ void sum_uint64(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow);

__device__ void sum_fp16(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow);
__device__ void sum_fp32(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow);
__device__ void sum_fp64(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow);

__constant__ threadedFunc_sum datatype_sum[] = {
    sum_int8,
    sum_int16,
    sum_int32,
    sum_int64,
    sum_uint8,
    sum_uint16,
    sum_uint32,
    sum_uint64,
    sum_fp16,
    sum_fp32,
    sum_fp64
};
