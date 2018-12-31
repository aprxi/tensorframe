/* Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
 * This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
*/

#include <stdio.h>
#include <unistd.h>
#include <iostream>

#include <chrono>

#include <cub/device/device_radix_sort.cuh>

#include <cuda_runtime.h>

#include "tf_types.cuh"
#include "io__read_csv.cuh"

#define COLUMNS_PER_ROUND 1
#define ROWS_PER_BLOCK 256
#define COLUMN_LENGTH 32   // 24 to hold float64, +8 to hold separators
#define WARPSIZE 32

using namespace std;


__device__ unsigned int __lanemask(){
    unsigned int lanemask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask));
    return lanemask;
}


__global__ void linecount(char separator, char* data, uint64_cu *ctr_row, unsigned long fsize){ 
    unsigned long idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    unsigned int active_threads;;
    unsigned int thread_rank;
    
    uint64_cu mask;

    if(idx < fsize){
        active_threads = __activemask();
        mask = __popc(__ballot_sync(active_threads, data[idx] == separator));
        thread_rank = __popc(active_threads & __lanemask());

        if(thread_rank == 0){ 
            atomicAdd(&ctr_row[0], mask);
        }
    }
};


template<typename T_1, typename T_2>
__device__ void conv_float(char *buf, int array_length, double float_max, uint8_t float_max_remainder, T_1 *results, uint64_cu offset, uint8_t *na){
    int8_t d;
    uint8_t n;
    uint8_t length = 0;

    // Calculate in double precision regardless ot float type.
    // Re-adjust precision when writing to memory (based on typename).
    double rez = 0.;
    double fact = 1.;

	// remove lead spacing and determine fact.
    for(n=0; n<array_length && buf[n] == ' '; n++);
    if(n < array_length && buf[n] == '-'){
        fact = -1.; // numbers starting with - are negative
        float_max_remainder += 1;
        n++;
    }

    for(; n<array_length; n++, length++){
        d = buf[n] - '0';
        if(d < 0 || d > 9){
            // reset length to 0 to signal NaN. First '.' accounts as valid.
            if(buf[n] != '.'){ length = 0;}
            else{ n++; length++;};
            break;
        }
        rez = rez * 10 + d;
    }

    if(length > 0){
    	for(; n<array_length; n++){
    		d = buf[n] - '0';
            if(d < 0 || d > 9){
                length = 0;
                break;
            }
    	    fact /= 10.0f;
    	    rez = rez * 10.0f + d;
    	}
    }
    
    // zero matched equals NaN
    if(length == 0){ *na = 1; rez = nanf("");}

    results[offset] = (T_1) (rez * fact);
}


template<typename T>
__device__ void conv_uint(char *buf, int array_length, T int_max, uint8_t int_max_remainder,  T *results, uint64_cu offset, uint8_t *na){
    int8_t d;
    uint8_t n;
    uint8_t length = 0;

    T rez = 0;

    // remove empty lead space
    for(n=0; n<array_length && buf[n] == ' '; n++);

    for(; n<array_length; n++, length++){
        d = buf[n] - '0';
        if(d < 0 || d > 9){
            // reset length to 0 to signal NaN. First '.' accounts as valid.
            if(buf[n] != '.'){ length = 0;}
            else{ length++;};
            break;
        }
        if(rez > int_max || (rez == int_max && d > int_max_remainder)){
            // overflow
            length = 0;
            break;
        }
        rez = rez * 10 + d;
    }
    if(length == 0){ *na = 1; rez = 0;}   // zero matched sets NaN
    results[offset] = (T) rez;
}


template<typename T>
__device__ void conv_int(char *buf, int array_length, T int_max, uint8_t int_max_remainder,  T *results, uint64_cu offset, uint8_t *na){
    int8_t d;
    uint8_t n;
    uint8_t length = 0;
    int8_t fact = 1;

    T rez = 0;

    // remove empty lead space
    for(n=0; n<array_length && buf[n] == ' '; n++);
    if(n < array_length && buf[n] == '-'){  
        // numbers starting with - are negative
        fact = -1;
        int_max_remainder += 1;     //
        n++;
    }    

    for(; n<array_length; n++, length++){
        d = buf[n] - '0';
        if(d < 0 || d > 9){
            // reset length to 0 to signal NaN. First '.' accounts as valid.
            if(buf[n] != '.'){ length = 0;}
            else{ length++;};
            break;
        }
        if( rez > int_max || (rez == int_max && d > int_max_remainder)){
            // overflow
            length = 0;
            break;
        }

        rez = rez * 10 + d;
    }
    if(length == 0){ *na = 1; rez = 0;}   // zero matched sets NaN
    results[offset] = (T) rez * fact;
}

__device__ void conv_fp16(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na){
    fp64_cu float_max = 110. / 10.;  //FLT_MAX / 10;  //INT16_MAX / 10;
    uint8_t int_max_remainder = INT16_MAX - (INT16_MAX / 10 * 10) - 1;
    conv_float<fp16_cu,int16_cu>(buf, array_length, float_max, int_max_remainder, (fp16_cu*) results, offset, na);
}
__device__ void conv_fp32(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na){
    fp64_cu float_max = FLT_MAX / 10;
    uint8_t int_max_remainder = 1;  //FLT_MAX - (FLT_MAX / 10 * 10) - 1;
    conv_float<fp32_cu,int32_cu>(buf, array_length, float_max, int_max_remainder, (fp32_cu*) results, offset, na);
}
__device__ void conv_fp64(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na){
    fp64_cu float_max = DBL_MAX / 10;
    uint8_t int_max_remainder = 1;  //DBL_MAX - (DBL_MAX / 10 * 10) - 1;
    conv_float<fp64_cu,int64_cu>(buf, array_length, float_max, int_max_remainder, (fp64_cu*) results, offset, na);
}

__device__ void conv_int8(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na){
    int8_cu int_max = INT8_MAX / 10;
    uint8_t int_max_remainder = INT8_MAX - (INT8_MAX / 10 * 10) - 1;
    conv_int<int8_cu>(buf, array_length, int_max, int_max_remainder, (int8_cu*) results, offset, na);
}
__device__ void conv_int16(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na){
    int16_cu int_max = INT16_MAX / 10;
    uint8_t int_max_remainder = INT16_MAX - (INT16_MAX / 10 * 10) - 1;
    conv_int<int16_cu>(buf, array_length, int_max, int_max_remainder, (int16_cu*) results, offset, na);
}
__device__ void conv_int32(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na){
    int32_cu int_max = INT32_MAX / 10;
    uint8_t int_max_remainder = INT32_MAX - (INT32_MAX / 10 * 10) - 1;  // (2 << 30) - ((2 << 30) / 10 * 10) - 1;
    conv_int<int32_cu>(buf, array_length, int_max, int_max_remainder, (int32_cu*) results, offset, na);
}
__device__ void conv_int64(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na){
    int64_cu int_max = INT64_MAX / 10;  //(2 << 62) / 10;
    uint8_t int_max_remainder = INT64_MAX - ((INT64_MAX - 1) / 10 * 10);    // (2 << 62) - ((2 << 62) / 10 * 10) - 1;
    conv_int<int64_cu>(buf, array_length, int_max, int_max_remainder, (int64_cu*) results, offset, na);
}
__device__ void conv_uint8(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na){
    uint8_cu int_max = UINT8_MAX / 10;
    uint8_t int_max_remainder = UINT8_MAX - (UINT8_MAX / 10 * 10) - 1;
    conv_uint<uint8_cu>(buf, array_length, int_max, int_max_remainder, (uint8_cu*) results, offset, na);
}
__device__ void conv_uint16(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na){
    uint16_cu int_max = UINT16_MAX / 10;
    uint8_t int_max_remainder = UINT16_MAX - (UINT16_MAX / 10 * 10) - 1;
    conv_uint<uint16_cu>(buf, array_length, int_max, int_max_remainder, (uint16_cu*) results, offset, na);
}
__device__ void conv_uint32(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na){
    uint32_cu int_max = UINT32_MAX / 10;    // ((2 << 31) - 1) / 10;
    uint8_t int_max_remainder = UINT32_MAX - ((UINT32_MAX - 1) / 10 * 10);  //((2 << 31) - 1) - (((2 << 31) - 1) / 10 * 10);
    conv_uint<uint32_cu>(buf, array_length, int_max, int_max_remainder, (uint32_cu*) results, offset, na);
}
__device__ void conv_uint64(char *buf, int array_length, void *results, uint64_cu offset, uint8_t *na){
    uint64_cu int_max = UINT64_MAX / 10;    //(2 << 63) / 10;
    uint8_t int_max_remainder = UINT64_MAX - ((UINT64_MAX - 1) / 10 * 10);  // ((2 << 63) - 1) - (((2 << 63) - 1) / 10 * 10);
    conv_uint<uint64_cu>(buf, array_length, int_max, int_max_remainder, (uint64_cu*) results, offset, na);
}


__global__ void parseColumns(
        char* data,
        uint32_t *column_map,
        void *results,
        uint64_cu no_rows_chunk,
        uint64_cu no_rows,
        uint64_cu no_columns,
        uint64_cu rows_offset,
        lookup_types_dyn *column_table
    ){

    uint64_cu idx = blockIdx.x*blockDim.x+threadIdx.x;

    __shared__ char local_data[ROWS_PER_BLOCK * COLUMN_LENGTH];  //~8K Bytes

    uint64_cu column_map_offset;

    uint64_cu begin_pos;
    uint64_cu end_pos;
    int array_length;

    uint64_cu column_offset;
    uint64_cu thread_offset;
    uint64_cu thread_offset_na_bytes;
    uint64_cu idx_offset;

    int j;


    int column_size;
    int column_size_sub;
    
    uint64_cu offset_round; 
    uint64_cu column_na_size;

    uint8_t rez_na;
    unsigned int thread_rank;
    uint32_t mask_na;
    uint32_t *na_int;

    // Eeach thread parses one row
    if(idx < no_rows_chunk){
        // pointer to start first column of a row
        column_map_offset = idx * no_columns;
        idx_offset = 0;

        thread_rank = __popc(__activemask() & __lanemask());

        offset_round = 0;
        column_size_sub = 0;

        if(rows_offset > 0){ idx_offset = rows_offset / 32 + 1;}
                
        na_int = (uint32_t*) results;

        //Fixed size for NaN. Ensure its a multiple of 8 (=size of largest supported datatype)
        column_na_size = (no_rows / WARPSIZE) * 4;
        column_na_size += (8 - (column_na_size % 8));

        for(int column_no = 0; column_no < no_columns; column_no++){
            begin_pos = column_map[column_map_offset + column_no] + 1;
            end_pos = column_map[column_map_offset + column_no + 1];
        
            array_length = end_pos - begin_pos;
        	if(array_length > COLUMN_LENGTH){   array_length = COLUMN_LENGTH;}

            // Note. In theory this is slow due to misaligned. However tests give similar results for this and a (much) more complex version
            // i.e. more productive to stick to this simple approach for now
	        for(j=0; j<array_length; j++){
	            local_data[threadIdx.x * COLUMN_LENGTH + j] = data[begin_pos + j];
	        }
       
            // wait till threads have finished fillup local_data
            __syncthreads();

            column_size = column_table[column_no].size;

            column_offset = 
                (column_size_sub * no_rows)
                + column_no * column_na_size
                + (rows_offset * column_size)
                + offset_round;
            
            thread_offset = (column_offset + idx * column_size) / column_size;

            rez_na = 0;     // assume valid number
            datatype_conv[column_table[column_no].function_id](
                &local_data[threadIdx.x * COLUMN_LENGTH],
                array_length,
                results,
                thread_offset,
                &rez_na
            );

            // offset_round ensures all fsize of a column is a multiple of 8, to prevent NaN_bitstring to overlap.
            offset_round += (8 - ((column_size * no_rows) % 8));

            thread_offset_na_bytes = 
                (column_size_sub * no_rows)
                + column_no * column_na_size
                + (no_rows * column_size)
                + ((idx_offset + (idx / WARPSIZE)) * 4)
                + offset_round;

            mask_na = __ballot_sync(__activemask(), rez_na == 0);   // collect NaN values from threads
            if(thread_rank == 0){
                na_int[thread_offset_na_bytes / sizeof(uint32_t)] = mask_na;
            }

            column_size_sub += column_size;
        }
    }
}


__global__ void indexColumns(char separator, char* data, uint32_t *map, uint64_cu *ctr_col, uint64_cu *ctr_row, unsigned long fsize){
    unsigned long idx = blockIdx.x*blockDim.x+threadIdx.x;

    if(idx < fsize){
    	char p = data[idx];

        int leader;
        int warp_res;
        bool newline = false;

        unsigned int active;
        unsigned int rank;
        uint64_cu change_col;
        uint64_cu change_row;

        if(p == separator || p == '\n'){
            if(p == '\n'){ newline = true;}     // count newlines separately

            active = __activemask();
            leader = __ffs(active) - 1;
            change_col = __popc(active);
            rank = __popc(active & __lanemask());
            change_row = __popc(__ballot_sync(active, newline));

            if(rank == 0){
                warp_res = atomicAdd(ctr_col, change_col);
                atomicAdd(ctr_row, change_row);
            }

            warp_res = __shfl_sync(active, warp_res, leader);
            map[warp_res + rank] = idx;
        }
    }
};

void cuda_parse_csv(
        char *data,
        unsigned long fsize,
        unsigned long *offsets,
        unsigned long chunks,
        unsigned long *dims,
        void *dest,
        uint8_t *column_types
    ){

// TODO: move to library/ include space
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
    char *d_data;

    unsigned long chunksize;
    char separator = ',';    // TODO: fixed for now, pass through function argument

    int ii, jj;

    lookup_types_dyn *column_table;
    lookup_types_dyn *d_column_table;

    unsigned long chunksize_max = 0;
    uint64_cu total_items;

    uint64_cu no_rows_chunk;
    uint64_cu no_rows;
    uint64_cu num_items;
    uint64_cu no_columns;

	int threads_per_block = ROWS_PER_BLOCK;;
	dim3 blocksPerGrid((uint64_cu) (fsize / threads_per_block + 1) );
	dim3 threadsPerBlock(threads_per_block);

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    size_t   temp_storage_bytes_2 = 0;
    
    uint32_t *d_key_alt_buf; 
    uint32_t *d_column_index;
    uint32_t *d_column_map;

    uint64_cu *d_ctr_col;
    uint64_cu *d_ctr_row;

    // get column count
    for(ii=0; column_types[ii] < pow(2, 8) - 1; ii++){ no_columns = ii + 1;}
  
    // set device column_table
    if(!(column_table = (lookup_types_dyn*) malloc(sizeof(lookup_types_dyn) * no_columns))){ exit(1);}
    cudaMalloc((void**) &d_column_table, sizeof(lookup_types_dyn) * no_columns);

    for(ii=0; ii<no_columns; ii++){
        for(jj=0; jj<pow(2, 8) - 1; jj++){ if(TF_TYPES_CUDA[jj].id == column_types[ii]){ break;}};
        if(jj == pow(2, 8) - 1){
            printf("CANT FIND COLUMNTYPE FOR:%i\n", column_types[ii]);
            // TODO: implement proper erroring using C++ exceptions
            return;
        }
        column_table[ii].function_id = TF_TYPES_CUDA[jj].function_id;
        column_table[ii].size = TF_TYPES_CUDA[jj].size;
    }
    cudaMemcpy(d_column_table, column_table, sizeof(lookup_types_dyn) * no_columns, cudaMemcpyHostToDevice);
    free(column_table);

    // find longest chunksize. Round up in 32 increments + 1 to ensure all data fits
    for(ii=0; ii<chunks; ii++){
        chunksize = (unsigned long) offsets[ii+1] - offsets[ii];
        if(chunksize > chunksize_max){  chunksize_max = (((unsigned long) chunksize / 32) + 1) * 32 + 1;}
    }

    cudaMalloc((void**)&d_data, chunksize_max * sizeof(char));  // * 123);
    cudaMalloc((void**)&d_ctr_col, sizeof(uint64_cu));
    cudaMalloc((void**)&d_ctr_row, sizeof(uint64_cu));

    no_rows = 0;
    no_rows_chunk = 0;

    cudaMemset(d_ctr_row, 0, sizeof(uint64_cu));
	for(ii=0; ii<chunks; ii++){
        chunksize = (unsigned long) offsets[ii+1] - offsets[ii];
        linecount<<<blocksPerGrid,threadsPerBlock>>>('\n', &data[offsets[ii]], d_ctr_row, chunksize);
        cudaMemcpy(&total_items, d_ctr_row, sizeof(uint64_t), cudaMemcpyDeviceToHost);

        // Store largest number of rows to use for allocating minimal amount of required memory.
        if((total_items - no_rows) > no_rows_chunk){ no_rows_chunk = total_items - no_rows;}
        no_rows = total_items;
    }

    // Allocate memory based on max expected within 1 chunk
    cudaMalloc((void**)&d_column_index, (no_rows_chunk * no_columns + 2) * sizeof(uint32_t));
    cudaMalloc((void**)&d_column_map, (no_rows_chunk * no_columns + 2) * sizeof(uint32_t));

    cudaMemset(d_data, '\n', sizeof(char));     //Insert newline position at start

    total_items = 0;
	for(ii=0; ii< chunks; ii++){

        // Difference between offsets is chunksize for this round.
        chunksize = (unsigned long) offsets[ii+1] - offsets[ii];    

        // Start copying past first byte (inserted newline)
        cudaMemcpy(&d_data[1], &data[offsets[ii]], chunksize * sizeof(char), cudaMemcpyHostToDevice);

        // Add 1 to chunksize to account for inserted newline at chunkstart
        chunksize += 1;

        // Reset column and row counters
        cudaMemset(d_ctr_col, 0, sizeof(uint64_cu));
        cudaMemset(d_ctr_row, 0, sizeof(uint64_cu));

        // Create an index for end-of-column positions
    	indexColumns<<<blocksPerGrid,threadsPerBlock>>>(separator, d_data, d_column_index, d_ctr_col, d_ctr_row, chunksize);

        // Copy counters to host and verify
        cudaMemcpy(&num_items, d_ctr_col, sizeof(uint64_cu), cudaMemcpyDeviceToHost);
        cudaMemcpy(&no_rows_chunk, d_ctr_row, sizeof(uint64_cu), cudaMemcpyDeviceToHost);

        no_rows_chunk -= 1; // Deduct one to exclude inserted newline at chunkstart.
        num_items -= 1;     // Idem.

        if(no_columns != (num_items / no_rows_chunk)){
            total_items = 0;
            break;
        }

        d_temp_storage = NULL;
        cub::DeviceRadixSort::SortKeys(NULL, temp_storage_bytes, d_column_index, d_column_map, num_items + 1);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_column_index, d_column_map, num_items + 1);
        cudaFree(d_temp_storage);

	    dim3 blocksPerGrid((uint64_cu) (no_rows_chunk / threads_per_block + 1));
        parseColumns<<<blocksPerGrid,threadsPerBlock>>>(
            d_data, d_column_map, dest, no_rows_chunk, no_rows, no_columns, total_items / no_columns, d_column_table
        );
        total_items += num_items;
    }

    dims[0] = total_items / no_columns;
    dims[1] = no_columns;

    cudaFree(d_data);
    cudaFree(d_column_index); 
    cudaFree(d_column_map);
    cudaFree(d_column_table);
    cudaFree(d_ctr_col);
    cudaFree(d_ctr_row);
}
