/* Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
 * This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
*/

#include <stdio.h>
#include <unistd.h>
#include <iostream>

#include <chrono>

#include <cuda_runtime.h>
#include <curand.h>

#include <cub/block/block_reduce.cuh>
#include "tf_types.cuh"
#include "io__generate_arrays.cuh"

using namespace std;

void generate_uint32(curandGenerator_t gen, void *dest, size_t num){
    uint32_cu *dst_ptr = (uint32_cu*) dest;
    curandGenerate(gen, dst_ptr, num);
}

void generate_uint64(curandGenerator_t gen, void *dest, size_t num){
    uint32_cu *dst_ptr = (uint32_cu*) dest;
    curandGenerate(gen,dst_ptr, num);
    // NOTE: we dont use the 64bit version below yet as it requires a 32bit generator.
    // while the latter can be easily fixed in code, performance appears less from early testing.
    // current hack is to simply write double the number of items to get sufficient random bits.
    // uint64_cu *dst_ptr = (uint64_cu*) dest;
    // curandGenerateLongLong(gen, dst_ptr, num);
}

void generate_fp32(curandGenerator_t gen, void *dest, size_t num){
    fp32_cu *dst_ptr = (fp32_cu*) dest;
    curandGenerateUniform(gen, dst_ptr, num);
}
void generate_fp64(curandGenerator_t gen, void *dest, size_t num){
    // NOTE: see note at generate_uint64
    // fp64_cu *dst_ptr = (fp64_cu*) dest;
    // curandGenerateUniformDouble(gen, dst_ptr, num);
    fp32_cu *dst_ptr = (fp32_cu*) dest;
    curandGenerateUniform(gen, dst_ptr, num);
}

// TODO: move this to a library/ include section
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

void cu__generate_arrays(
        void *columns_meta,
        void *dest,
        uint64_t no_columns,
        uint64_t no_items
    ){
    int ii;
    int jj;

    void *d_dest;

    size_t items_generated;

    tf_column_meta *columns_meta_cu = (tf_column_meta*) columns_meta;

    lookup_types_dyn *column_table;

    // Set device column_table
    if(!(column_table = (lookup_types_dyn*) malloc(sizeof(lookup_types_dyn) * no_columns))){ exit(1);}

    for(ii=0; ii<no_columns; ii++){
        for(jj=0; jj<pow(2, 8) - 1; jj++){ if(TF_TYPES_CUDA[jj].id == columns_meta_cu[ii].datatype){ break;}};
        if(jj == pow(2, 8) - 1){
            printf("CANT FIND COLUMNTYPE FOR:%i\n", columns_meta_cu[ii].datatype);
            // TODO: add c++ exceptions
            return;
        }
        column_table[ii].function_id = TF_TYPES_CUDA[jj].function_id;
        column_table[ii].size = TF_TYPES_CUDA[jj].size;
    }

    // create pseudo-random number generator 32 bits
    curandGenerator_t gen32;
    curandCreateGenerator(&gen32, CURAND_RNG_PSEUDO_DEFAULT);

    // 64 bits version
    // curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64);

    // seed
    // NOTE: for testing and verication purposes we keep this fixed so we get the same "random" ( :') ) numbers
    // TODO: when auto sum functions are fully implemented we can change this to a slightly more random one
    curandSetPseudoRandomGeneratorSeed(gen32, 15588238ULL);
    //curandSetPseudoRandomGeneratorSeed(gen32, cpp_random_long());

    // TODO: simple column loop implementation that can easily be improved noticably
    // despite this performance is very impressive
    for(ii = 0; ii < no_columns; ii++){

        if(column_table[ii].size < 4){
            items_generated = no_items;
            //(u)int8, (u)int16, fp16. Generator still fills up 4 bytes per item
            // correct no_items to adjust for this

            // round up such that generated_items is a sum of 4 (making it divisble)
            if(( no_items % 4) != 0){ 
                items_generated = no_items + (4 - (no_items % 4));
            }else{
                items_generated = no_items;
            }

            if(column_table[ii].size == 2){ items_generated = items_generated / 2;}
            else if(column_table[ii].size == 1){ items_generated = items_generated / 1;}
            else{
                // should not happen since we control input
                // TODO: improve error checking output here
                printf("incorrect column size\n");
                exit(1);
            };
        }else if(column_table[ii].size == 8){
            items_generated = no_items * 2;

        }else{
            items_generated = no_items;
        }

        generate_num[column_table[ii].function_id](
            gen32,  // For now, use 32 bits generator for each datatype.
            (void*) (((uint8_t*) dest) + columns_meta_cu[ii].offset),       // Starting position to write counters
            (size_t) items_generated
        );
        cudaMemset((void*) (((uint8_t*) dest) + columns_meta_cu[ii].offset_na), 0xffffff, (no_items / 32) * 4 + 4);
    }

    curandDestroyGenerator(gen32);

    free(column_table);
}

