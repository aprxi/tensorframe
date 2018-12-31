/* Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
 * This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
*/

#include <stdio.h>
#include <unistd.h>
#include <iostream>

#include <chrono>

#include <cub/block/block_reduce.cuh>

#include <cuda_runtime.h>

#include "tf_types.cuh"
#include "binops__array_sum.cuh"

// NOTE: its not safe (yet) to increase this number.
// we dont need it currently as performance is already excellent.
// TODO: nonetheless, fix the BlockSum function to do proper second round summing.
#define ITEMS_PER_BLOCK 32
#define WARPSIZE 32

using namespace std;


template<typename T, typename S>
__device__ void BlockSum(T *d_in, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow, S max){
// NOTE: this function can be still majorly improved. See NOTE regarding ITEMS_PER_BLOCK.
    uint64_cu idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    typedef cub::WarpReduce<S>WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[ITEMS_PER_BLOCK / WARPSIZE];
    __shared__ S round_1[(ITEMS_PER_BLOCK / WARPSIZE)];

    uint8_t warp_id;
    uint32_cu overflow_vote;
    S thread_data;

    S sum_1;
    S sum_2;
    
    if(idx < no_items){
        S *counter = (S*) d_out;

        warp_id = threadIdx.x / WARPSIZE;
        round_1[warp_id] = 1;

        thread_data = (S) d_in[idx];
        
        overflow_vote = __popc(__ballot_sync(__activemask(), thread_data > max));
        if(overflow_vote > 0){
            // TODO: write an escape function when near max (1/32 of max) that is more exact
            // For everything else, stick with CUBs WarpReduce to keep performance
            // Currently keep it safe and assume overflow.
            sum_1 = 0;
        }else{
           sum_1 = WarpReduce(temp_storage[warp_id]).Sum(thread_data);
        }
        if(threadIdx.x == 0){
            if(overflow_vote != 0 || ((max - atomicAdd(&counter[0], (S) sum_1)) < sum_1)){
                // Overflow detected
                atomicInc(&ctr_overflow[0], overflow_vote);
            }
        }
    }
}

__device__ void sum_int8(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow){
    BlockSum<int8_cu, uint64_cu>((int8_cu *) ptr, d_out, no_items, ctr_overflow, UINT64_MAX / WARPSIZE);
}
__device__ void sum_int16(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow){
    BlockSum<int16_cu, uint64_cu>((int16_cu *) ptr, d_out, no_items, ctr_overflow, UINT64_MAX / WARPSIZE);
}
__device__ void sum_int32(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow){
    BlockSum<int32_cu, uint64_cu>((int32_cu *) ptr, d_out, no_items, ctr_overflow, UINT64_MAX / WARPSIZE);
}
__device__ void sum_int64(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow){
    BlockSum<int64_cu, uint64_cu>((int64_cu *) ptr, d_out, no_items, ctr_overflow, UINT64_MAX / WARPSIZE);
}
__device__ void sum_uint8(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow){
    BlockSum<uint8_cu,uint64_cu>((uint8_cu *) ptr, d_out, no_items, ctr_overflow, UINT64_MAX / WARPSIZE);
}
__device__ void sum_uint16(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow){
    BlockSum<uint16_cu, uint64_cu>((uint16_cu *) ptr, d_out, no_items, ctr_overflow, UINT64_MAX / WARPSIZE);
}
__device__ void sum_uint32(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow){
    BlockSum<uint32_cu, uint64_cu>((uint32_cu *) ptr, d_out, no_items, ctr_overflow, UINT64_MAX / WARPSIZE);
}
__device__ void sum_uint64(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow){
    BlockSum<uint64_cu, uint64_cu>((uint64_cu *) ptr, d_out, no_items, ctr_overflow, UINT64_MAX / WARPSIZE);
}
__device__ void sum_fp16(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow){
    BlockSum<fp16_cu, fp64_cu>((fp16_cu *) ptr, d_out, no_items, ctr_overflow, DBL_MAX / WARPSIZE);
}
__device__ void sum_fp32(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow){
    BlockSum<fp32_cu, fp64_cu>((fp32_cu *) ptr, d_out, no_items, ctr_overflow, DBL_MAX / WARPSIZE);
}
__device__ void sum_fp64(void *ptr, void *d_out, uint64_cu no_items, uint32_cu *ctr_overflow){
    BlockSum<fp64_cu, fp64_cu>((fp64_cu *) ptr, d_out, no_items, ctr_overflow, DBL_MAX / WARPSIZE);
}

__global__ void BlockSumGlobal(void *ptr, void *d_out, uint64_cu no_items, uint8_t function_id, uint32_cu *ctr_overflow){
    datatype_sum[function_id](ptr, d_out, no_items, ctr_overflow);
}


void cu__array_sum(
        void *columns,
        void *dest,
        uint8_t stride_dest,
        uint64_t no_columns,
        uint64_t no_items
    ){


lookup_types_cumap TF_TYPES_CUDA[] = {
    { TF_INT8,      sizeof(int8_cu),    0},
    { TF_INT16,     sizeof(int16_cu),   1},
    { TF_INT32,     sizeof(int32_cu),   2},
    { TF_INT64,     sizeof(int64_cu),   3},
    { TF_UINT8,     sizeof(uint8_cu),   4},
    { TF_UINT16,    sizeof(uint16_cu),  5},
    { TF_UINT32,    sizeof(uint32_cu),  6},
    { TF_UINT64,    sizeof(uint64_cu),  7},
    { TF_FP16,      sizeof(fp16_cu),    8},
    { TF_FP32,      sizeof(fp32_cu),    9},
    { TF_FP64,      sizeof(fp64_cu),    10},
};

    int ii;
    int jj;

    void *d_dest;    
    uint64_t no_items_chunk = no_items;
    uint32_t overflow = 50;
    dim3 blocksPerGrid((uint64_cu) (no_items_chunk / ITEMS_PER_BLOCK + 1));
    tf_column_obj *columns_cu = (tf_column_obj*) columns;

    lookup_types_dyn *column_table;

    // Set device column_table
    if(!(column_table = (lookup_types_dyn*) malloc(sizeof(lookup_types_dyn) * no_columns))){ exit(1);}

    for(ii=0; ii<no_columns; ii++){
        for(jj=0; jj<pow(2, 8) - 1; jj++){ if(TF_TYPES_CUDA[jj].id == columns_cu[ii].datatype){ break;}};
        if(jj == pow(2, 8) - 1){
            printf("CANT FIND COLUMNTYPE FOR:%i\n", columns_cu[ii].datatype);
            // TODO: add c++ exceptions
            return;
        }
        column_table[ii].function_sum = TF_TYPES_CUDA[jj].function_id;
        column_table[ii].size = TF_TYPES_CUDA[jj].size;
    }

    // One counter per column
    cudaMalloc((void**)&d_dest, no_columns * stride_dest);

    // Ensure column counters start at 0
    cudaMemset(d_dest, 0, no_columns * stride_dest);

    uint32_cu *d_overflow;

    cudaMalloc((void**)&d_overflow, sizeof(uint32_cu));
    
    for(ii = 0; ii < no_columns; ii++){
        cudaMemset(d_overflow, 0, sizeof(uint32_cu));
        BlockSumGlobal<<<blocksPerGrid, ITEMS_PER_BLOCK>>>(
            columns_cu[ii].memory_ptr,
            (void*) (((uint8_t*) d_dest) + (ii * stride_dest)),       // Starting position to write counters
            (uint64_cu) no_items,
            column_table[ii].function_sum,
            d_overflow
        );

        cudaMemcpy(&overflow, d_overflow, sizeof(uint32_cu), cudaMemcpyDeviceToHost);
        if(overflow != 0){
            cudaMemset((void*) (((uint8_t*) d_dest) + (ii * stride_dest)), 0, stride_dest);
        }
    }
    
    cudaFree(d_overflow);

    // Copy results to host
    cudaMemcpy(dest, d_dest, no_columns * stride_dest, cudaMemcpyDeviceToHost);
    cudaFree(d_dest);

    // Counters are either in uint64_t or double (fp64)
    // If source colum is of signed integer type, a conversion from unsigned to signed is needed
    int64_cu *d_i64 = (int64_cu*) dest;
    for(ii = 0; ii < no_columns; ii++){
        if(columns_cu[ii].datatype < TF_INT8 || columns_cu[ii].datatype > TF_INT64){ continue;};
        d_i64[ii] = (int64_cu) d_i64[ii];   //re-sign
    }

    free(column_table);
}

